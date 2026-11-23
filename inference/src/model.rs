use std::{
    collections::{BTreeSet, HashMap},
    path::Path,
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::qwen3::{Config, ModelForCausalLM},
};
use inference::{
    GenerationEvent, GenerationFailure, Job,
    prefix_cache::{PREFILL_BLOCK_SIZE, PrefixCache},
    worker::Generator,
};
use tokenizers::Tokenizer;

pub struct Engine {
    model: ModelForCausalLM,
    device: Device,
    tokenizer: Tokenizer,
    max_context: usize,
    eos: BTreeSet<u32>,
    prefix_cache: PrefixCache<PrefixState>,
}

#[derive(Clone)]
struct PrefixState {
    model: ModelForCausalLM,
    logits: Tensor,
}

#[derive(serde::Deserialize)]
struct WeightIndex {
    weight_map: HashMap<String, String>,
}

impl Engine {
    pub fn load(path: &Path, max_context: usize, prefix_cache_tokens: usize) -> Result<Self> {
        let started = Instant::now();
        let mut config: Config = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
        ensure!(
            max_context <= config.max_position_embeddings,
            "contexte supérieur à celui du modèle"
        );
        config.max_position_embeddings = max_context;
        let index: WeightIndex =
            serde_json::from_slice(&std::fs::read(path.join("model.safetensors.index.json"))?)?;
        ensure!(!index.weight_map.is_empty(), "index des poids vide");
        let files: BTreeSet<_> = index
            .weight_map
            .values()
            .map(|filename| path.join(filename))
            .collect();
        let files: Vec<_> = files.into_iter().collect();
        let tokenizer =
            Tokenizer::from_file(path.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let eos = ["<|endoftext|>", "<|im_end|>"]
            .into_iter()
            .map(|token| tokenizer.token_to_id(token).context("token de fin absent"))
            .collect::<Result<_>>()?;
        let device = Device::new_cuda(0).context("initialisation CUDA ; vérifier nvidia-smi")?;
        // Les fichiers locaux sont mappés en lecture seule et restent ouverts
        // pendant la construction ; les tenseurs sont ensuite copiés sur le GPU.
        let builder = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F16, &device)? };
        let model = ModelForCausalLM::new(&config, builder)?;
        println!(
            "Qwen3 chargé sur CUDA en float16 ({:.2}s)",
            started.elapsed().as_secs_f64()
        );
        Ok(Self {
            model,
            device,
            tokenizer,
            max_context,
            eos,
            prefix_cache: PrefixCache::new(prefix_cache_tokens.min(max_context)),
        })
    }

    fn generate_inner(&mut self, job: &Job) -> Result<()> {
        if job.events.is_closed() {
            return Ok(());
        }
        let encoded = self
            .tokenizer
            .encode(job.prompt.as_str(), true)
            .map_err(anyhow::Error::msg)?;
        let prompt = encoded.get_ids();
        let budget = match job
            .request
            .completion_budget(prompt.len(), self.max_context)
        {
            Ok(budget) => budget,
            Err(message) => {
                job.emit(GenerationEvent::Failed(GenerationFailure::invalid_input(
                    format!(
                        "{message} : {} tokens de prompt, contexte {}",
                        prompt.len(),
                        self.max_context
                    ),
                )));
                return Ok(());
            }
        };
        let cached = self
            .prefix_cache
            .lookup(prompt)
            .map(|(length, state)| (length, state.clone()));
        // Pour les très longues entrées, consomme un préfixe existant mais ne
        // garde pas une seconde copie du KV : cette marge revient au contexte.
        let retain_prefix = prompt.len() <= self.max_context / 2;
        if !retain_prefix {
            self.prefix_cache.clear();
        }
        let (cached_tokens, mut logits) = if let Some((length, state)) = cached {
            self.model = state.model;
            (length, Some(state.logits))
        } else {
            (0, None)
        };
        if !job.emit(GenerationEvent::Started {
            tokens: prompt.len(),
            cached_tokens,
        }) {
            return Ok(());
        }
        let started = Instant::now();
        // Les snapshots de ConcatKvCache partagent des tenseurs immuables : les
        // append de Candle créent de nouveaux tenseurs et ne modifient pas le préfixe.
        let cache_target = if retain_prefix {
            self.prefix_cache.target_len(prompt.len())
        } else {
            0
        };
        for (index, chunk) in prompt[cached_tokens..]
            .chunks(PREFILL_BLOCK_SIZE)
            .enumerate()
        {
            if job.events.is_closed() {
                return Ok(());
            }
            let input = Tensor::new(chunk, &self.device)?.unsqueeze(0)?;
            let offset = cached_tokens + index * PREFILL_BLOCK_SIZE;
            let next_logits = self.model.forward(&input, offset)?;
            if offset + chunk.len() == cache_target && cache_target > cached_tokens {
                self.prefix_cache.store(
                    &prompt[..cache_target],
                    PrefixState {
                        model: self.model.clone(),
                        logits: next_logits.clone(),
                    },
                );
            }
            logits = Some(next_logits);
            // Borne aussi le travail CUDA déjà soumis : l'annulation d'un long
            // prompt ne doit pas laisser tout son préremplissage dans la file GPU.
            if (index + 1).is_multiple_of(2) {
                self.device.synchronize()?;
            }
        }
        let mut logits = logits.context("prompt vide")?;
        let mut sampler = LogitsProcessor::new(
            42,
            (job.request.temperature > 0.0).then_some(job.request.temperature),
            None,
        );
        let mut decoder = self.tokenizer.decode_stream(true);
        let mut reason = "length";
        let mut count = 0;
        for step in 0..budget {
            if job.events.is_closed() {
                return Ok(());
            }
            let scores = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let token = sampler.sample(&scores)?;
            count += 1;
            if self.eos.contains(&token) {
                reason = "stop";
                break;
            }
            if let Some(delta) = decoder.step(token).map_err(anyhow::Error::msg)?
                && !job.emit(GenerationEvent::Delta(delta))
            {
                return Ok(());
            }
            if step + 1 < budget {
                let input = Tensor::new(&[token], &self.device)?.unsqueeze(0)?;
                logits = self.model.forward(&input, prompt.len() + step)?;
            }
        }
        job.emit(GenerationEvent::Finished {
            tokens: count,
            reason,
        });
        println!(
            "Génération : {} tokens, {:.2}s, fin={reason}, préfixe réutilisé={cached_tokens}",
            count,
            started.elapsed().as_secs_f64()
        );
        Ok(())
    }
}

impl Generator for Engine {
    fn generate(&mut self, job: &Job) -> std::result::Result<(), GenerationFailure> {
        self.model.clear_kv_cache();
        let result = self.generate_inner(job);
        self.model.clear_kv_cache();
        result.map_err(|error| GenerationFailure::engine(error.to_string()))
    }

    fn recover(&mut self) -> std::result::Result<(), String> {
        self.model.clear_kv_cache();
        self.prefix_cache.clear();
        self.device
            .synchronize()
            .map_err(|error| format!("contexte CUDA indisponible après récupération : {error}"))
    }
}
