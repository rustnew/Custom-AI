use serde::Deserialize;
use serde_json::{Value, json};

use crate::protocol::{ChatRequest, FunctionCall, ToolCall};

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// Retient les balises réparties entre tokens, sans retarder le texte ordinaire.
/// Seuls les appels JSON complets et les noms d'outils proposés sont exposés au client.
pub(crate) struct ToolOutput {
    parse_tools: bool,
    names: Vec<String>,
    required: bool,
    parallel: bool,
    forced: Option<String>,
    id: String,
    pending: String,
    in_call: bool,
    pub content: String,
    pub calls: Vec<ToolCall>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelCall {
    name: String,
    arguments: Value,
}

impl ToolOutput {
    pub fn set_id(&mut self, id: &str) {
        self.id = id.to_owned();
    }

    pub fn new(request: &ChatRequest, id: &str) -> Self {
        Self {
            parse_tools: !request.tools.is_empty(),
            names: if request.tools_enabled() {
                request
                    .tools
                    .iter()
                    .map(|tool| tool.function.name.clone())
                    .collect()
            } else {
                Vec::new()
            },
            required: request.required_tool(),
            parallel: request.parallel_tool_calls != Some(false),
            forced: request.forced_tool().map(str::to_owned),
            id: id.to_owned(),
            pending: String::new(),
            in_call: false,
            content: String::new(),
            calls: Vec::new(),
        }
    }

    fn text(&mut self, text: String) -> Value {
        self.content.push_str(&text);
        json!({"content": text})
    }

    pub fn push(&mut self, text: &str) -> Result<Vec<Value>, &'static str> {
        if !self.parse_tools {
            return Ok(vec![self.text(text.to_owned())]);
        }
        self.pending.push_str(text);
        let mut deltas = Vec::new();
        loop {
            if self.in_call {
                // Le JSON peut lui-même contenir </tool_call> dans une chaîne.
                // On cherche donc la balise après la fin de l'objet JSON.
                let mut values =
                    serde_json::Deserializer::from_str(&self.pending).into_iter::<ModelCall>();
                let call = match values.next() {
                    None => break,
                    Some(Ok(call)) => call,
                    Some(Err(error)) if error.is_eof() => break,
                    Some(Err(_)) => {
                        return Err("le modèle a produit un appel d'outil JSON invalide");
                    }
                };
                let rest = self.pending[values.byte_offset()..].trim_start();
                if !rest.starts_with(CLOSE) {
                    if CLOSE.starts_with(rest) {
                        break;
                    }
                    return Err("balise de fin d'appel d'outil invalide");
                }
                let consumed = self.pending.len() - rest.len() + CLOSE.len();
                let arguments = match call.arguments {
                    Value::String(text) => serde_json::from_str(&text)
                        .map_err(|_| "les arguments d'outil générés ne sont pas un objet JSON")?,
                    value => value,
                };
                if !arguments.is_object() {
                    return Err("les arguments d'outil générés ne sont pas un objet JSON");
                }
                if !self.names.contains(&call.name)
                    || self.forced.as_ref().is_some_and(|name| name != &call.name)
                {
                    return Err("le modèle a demandé un outil inconnu ou interdit");
                }
                if !self.parallel && !self.calls.is_empty() {
                    return Err(
                        "le modèle a produit plusieurs appels malgré parallel_tool_calls=false",
                    );
                }
                let index = self.calls.len();
                let call = ToolCall {
                    id: format!("call_{}_{index}", self.id),
                    kind: "function".to_owned(),
                    function: FunctionCall {
                        name: call.name,
                        arguments: arguments.to_string(),
                    },
                };
                deltas.push(json!({"tool_calls": [{
                    "index": index, "id": call.id, "type": "function", "function": call.function
                }]}));
                self.calls.push(call);
                self.pending.drain(..consumed);
                self.in_call = false;
            } else if let Some(start) = self.pending.find(OPEN) {
                let text = self.pending[..start].to_owned();
                self.pending.drain(..start + OPEN.len());
                if !text.is_empty() {
                    deltas.push(self.text(text));
                }
                self.in_call = true;
            } else {
                let keep = (1..OPEN.len())
                    .rev()
                    .find(|&size| self.pending.ends_with(&OPEN[..size]))
                    .unwrap_or(0);
                let emit = self.pending.len() - keep;
                if emit > 0 {
                    let text: String = self.pending.drain(..emit).collect();
                    deltas.push(self.text(text));
                }
                break;
            }
        }
        Ok(deltas)
    }

    pub fn finish(&mut self) -> Result<Vec<Value>, &'static str> {
        if self.in_call {
            return Err("la génération s'est arrêtée avant la fin d'un appel d'outil");
        }
        if self.required && self.calls.is_empty() {
            return Err("le modèle n'a pas produit l'appel d'outil imposé");
        }
        let text = std::mem::take(&mut self.pending);
        Ok(if text.is_empty() {
            Vec::new()
        } else {
            vec![self.text(text)]
        })
    }

    pub fn finish_reason(&self, reason: &'static str) -> &'static str {
        if reason == "stop" && !self.calls.is_empty() {
            "tool_calls"
        } else {
            reason
        }
    }

    pub fn message(&self) -> Value {
        if self.calls.is_empty() {
            json!({"role": "assistant", "content": self.content})
        } else {
            json!({"role": "assistant", "content": if self.content.is_empty() { None } else { Some(&self.content) }, "tool_calls": self.calls})
        }
    }
}
