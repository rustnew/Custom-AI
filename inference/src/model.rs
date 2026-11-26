use std::{
    collections::{BTreeSet, HashMap},
    path::Path,
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config, ModelForCausalLM};
use inference::{
    GenerationEvent, GenerationFailure, Job,
    prefix_cache::{PrefixCache, prefill_block_size},
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
    block_size: usize,
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
        let block_size = prefill_block_size(prompt.len());
        let cached = self
            .prefix_cache
            .lookup(prompt)
            // Un préfixe calculé en blocs de 512 ne remplace pas un calcul
            // en blocs de 128 : leurs arrondis FP16 peuvent différer.
            .filter(|(_, state)| state.block_size == block_size)
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
        let mut queued_tokens = 0;
        for (index, chunk) in prompt[cached_tokens..].chunks(block_size).enumerate() {
            if job.events.is_closed() {
                return Ok(());
            }
            let input = Tensor::new(chunk, &self.device)?.unsqueeze(0)?;
            let offset = cached_tokens + index * block_size;
            let next_logits = self.model.forward(&input, offset)?;
            if offset + chunk.len() == cache_target && cache_target > cached_tokens {
                self.prefix_cache.store(
                    &prompt[..cache_target],
                    PrefixState {
                        model: self.model.clone(),
                        logits: next_logits.clone(),
                        block_size,
                    },
                );
            }
            logits = Some(next_logits);
            // Borne aussi le travail CUDA déjà soumis : l'annulation d'un long
            // prompt ne doit pas laisser tout son préremplissage dans la file GPU.
            queued_tokens += chunk.len();
            if queued_tokens >= 256 {
                self.device.synchronize()?;
                queued_tokens = 0;
            }
        }
        let mut logits = logits.context("prompt vide")?;
        let temperature = job.request.temperature;
        if temperature > 0.0 {
            // La graine interne reste fixe par requête, comme avec l'ancien sampler.
            // Le nouvel algorithme conserve une loi de tirage équivalente, pas la
            // même séquence de tokens que WeightedIndex sur CPU.
            self.device.set_seed(42)?;
        }
        let mut decoder = self.tokenizer.decode_stream(true);
        let mut reason = "length";
        let mut count = 0;
        for step in 0..budget {
            if job.events.is_closed() {
                return Ok(());
            }
            let scores = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let token = sample_on_device(&scores, temperature)?;
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

/// Choisit le prochain token sans copier tout le vocabulaire vers le CPU : un seul entier revient du GPU.
/// À température nulle, c'est l'argmax (déjà sur GPU avec Candle). Sinon, le
/// tirage Gumbel évite le transfert du vecteur de probabilités vers le CPU.
fn sample_on_device(scores: &Tensor, temperature: f64) -> Result<u32> {
    let chosen = if temperature > 0.0 {
        let minus_gumbel = minus_gumbel(scores)?;
        // argmax(logits/T + G) = argmax(logits + T*G), sans division des logits.
        if temperature == 1.0 {
            (scores - minus_gumbel)?.argmax(0)?
        } else {
            (scores + (minus_gumbel * -temperature)?)?.argmax(0)?
        }
    } else {
        scores.argmax(0)?
    };
    Ok(chosen.to_scalar::<u32>()?)
}

fn minus_gumbel(scores: &Tensor) -> Result<Tensor> {
    // cuRAND inclut 1 dans son intervalle. La borne supérieure doit rester
    // strictement sous 1 pour éviter log(-log(1)) = -inf.
    let upper = f32::from_bits(1.0_f32.to_bits() - 1);
    let uniform = Tensor::rand(1e-9_f32, upper, scores.shape(), scores.device())?;
    Ok(uniform.log()?.neg()?.log()?)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_temperature_keeps_the_existing_greedy_choice() -> Result<()> {
        let scores = Tensor::new(&[-10_f32, 2., 0.5], &Device::Cpu)?;
        let mut old = candle_transformers::generation::LogitsProcessor::new(42, None, None);
        assert_eq!(sample_on_device(&scores, 0.)?, old.sample(&scores)?);
        Ok(())
    }

    #[test]
    fn positive_temperature_follows_the_expected_categorical_probabilities() -> Result<()> {
        let scores = Tensor::new(&[0_f32, 3_f32.ln()], &Device::Cpu)?;
        for (temperature, expected) in [(1., 0.75), (0.5, 0.9)] {
            let mut second = 0;
            for _ in 0..4096 {
                second += usize::from(sample_on_device(&scores, temperature)? == 1);
            }
            let observed = second as f64 / 4096.;
            assert!(
                (observed - expected).abs() < 0.05,
                "T={temperature}: {observed}, attendu {expected}"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requiert un GPU CUDA ; exécuté explicitement pendant la validation"]
    fn cuda_gumbel_noise_stays_finite_across_large_vocabulary_draws() -> Result<()> {
        let device = Device::new_cuda(0)?;
        device.set_seed(299792458)?;
        let count = 1usize << 24;
        let scores = Tensor::zeros(count, DType::F32, &device)?;
        for _ in 0..8 {
            let noise = minus_gumbel(&scores)?;
            let finite = noise
                .abs()?
                .lt(f32::INFINITY)?
                .to_dtype(DType::U32)?
                .sum_all()?
                .to_scalar::<u32>()?;
            assert_eq!(finite as usize, count);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requiert un GPU CUDA ; exécuté explicitement pendant la validation"]
    fn cuda_sampling_repeats_when_the_request_seed_is_reset() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let scores = Tensor::new(&[0_f32, 1., 2., 3.], &device)?;
        let sequence = || -> Result<Vec<u32>> {
            device.set_seed(42)?;
            (0..64).map(|_| sample_on_device(&scores, 0.7)).collect()
        };
        assert_eq!(sequence()?, sequence()?);
        Ok(())
    }
}
