use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default, alias = "max_completion_tokens")]
    pub max_tokens: Option<usize>,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub tools: Vec<Tool>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// Paramètre de compatibilité OpenCode. Ce modèle Instruct ne possède pas
    /// de mode de raisonnement réglable ; l'effort ne change pas son prompt.
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

fn default_temperature() -> f64 {
    0.7
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: Option<Content>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<TextPart>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum TextPart {
    #[serde(rename = "text")]
    Text { text: String },
}

impl Content {
    fn text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .map(|TextPart::Text { text }| text.as_str())
                .collect(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),
    Function {
        #[serde(rename = "type")]
        kind: String,
        function: NamedFunction,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamedFunction {
    pub name: String,
}

impl ChatRequest {
    pub fn tools_enabled(&self) -> bool {
        !self.tools.is_empty()
            && !matches!(&self.tool_choice, Some(ToolChoice::Mode(mode)) if mode == "none")
    }

    pub fn required_tool(&self) -> bool {
        matches!(&self.tool_choice, Some(ToolChoice::Mode(mode)) if mode == "required")
            || self.forced_tool().is_some()
    }

    pub fn forced_tool(&self) -> Option<&str> {
        match &self.tool_choice {
            Some(ToolChoice::Function { function, .. }) => Some(&function.name),
            _ => None,
        }
    }

    pub fn prompt(&self, model: &str, max_context: usize) -> Result<String, &'static str> {
        if self.model != model {
            return Err("modèle inconnu");
        }
        if self.messages.is_empty() {
            return Err("messages ne doit pas être vide");
        }
        if self
            .max_tokens
            .is_some_and(|tokens| tokens == 0 || tokens >= max_context)
        {
            return Err("max_tokens doit être positif et inférieur au contexte");
        }
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) {
            return Err("temperature doit être comprise entre 0 et 2");
        }
        let mut names = HashSet::new();
        for tool in &self.tools {
            if tool.kind != "function"
                || tool.function.name.is_empty()
                || !names.insert(tool.function.name.as_str())
                || !tool.function.parameters.is_object()
            {
                return Err(
                    "tools doit contenir des fonctions uniques avec un schéma de paramètres",
                );
            }
            if tool.function.strict == Some(true) {
                return Err("les schémas d'outils stricts ne sont pas pris en charge");
            }
        }
        match &self.tool_choice {
            Some(ToolChoice::Mode(mode))
                if !matches!(mode.as_str(), "none" | "auto" | "required") =>
            {
                return Err("tool_choice doit être none, auto, required ou une fonction");
            }
            Some(ToolChoice::Function { kind, function })
                if kind != "function" || !names.contains(function.name.as_str()) =>
            {
                return Err("tool_choice désigne une fonction inconnue");
            }
            _ => {}
        }
        if self.required_tool() && self.tools.is_empty() {
            return Err("tool_choice impose un outil mais tools est vide");
        }

        // Reproduit le chat_template du tokenizer Qwen3-4B-Instruct-2507.
        let mut prompt = String::new();
        let first_is_system = matches!(self.messages[0].role.as_str(), "system" | "developer");
        if self.tools_enabled() {
            prompt.push_str("<|im_start|>system\n");
            if first_is_system {
                let content = self.messages[0]
                    .content
                    .as_ref()
                    .ok_or("un message system doit contenir du texte")?;
                prompt.push_str(&content.text());
                prompt.push_str("\n\n");
            }
            prompt.push_str("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
            for tool in &self.tools {
                prompt.push('\n');
                prompt.push_str(&serde_json::to_string(tool).expect("outil JSON sérialisable"));
            }
            prompt.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
            if let Some(name) = self.forced_tool() {
                prompt.push_str(&format!("\nYou must call the function {name}."));
            } else if self.required_tool() {
                prompt.push_str("\nYou must call at least one of these functions.");
            }
            if self.parallel_tool_calls == Some(false) {
                prompt.push_str("\nCall only one function in this response.");
            }
            prompt.push_str("<|im_end|>\n");
        }

        let mut pending = HashSet::new();
        let mut seen = HashSet::new();
        for (index, message) in self.messages.iter().enumerate() {
            let content = message.content.as_ref().map(Content::text);
            if message.role != "tool" && !pending.is_empty() {
                return Err("chaque tool_call doit recevoir un résultat avant le message suivant");
            }
            if message.role != "assistant" && !message.tool_calls.is_empty() {
                return Err("seul assistant peut fournir tool_calls");
            }
            if message.role != "tool" && message.tool_call_id.is_some() {
                return Err("tool_call_id est réservé aux résultats d'outils");
            }
            match message.role.as_str() {
                "system" | "developer" | "user" => {
                    let content = content.ok_or("le message doit contenir du texte")?;
                    if index == 0 && first_is_system && self.tools_enabled() {
                        continue;
                    }
                    let role = if message.role == "developer" {
                        "system"
                    } else {
                        &message.role
                    };
                    prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
                }
                "assistant" => {
                    if content.is_none() && message.tool_calls.is_empty() {
                        return Err("assistant doit contenir du texte ou des tool_calls");
                    }
                    let content = content.unwrap_or_default();
                    prompt.push_str("<|im_start|>assistant\n");
                    prompt.push_str(&content);
                    for (call_index, call) in message.tool_calls.iter().enumerate() {
                        if call.kind != "function"
                            || call.function.name.is_empty()
                            || call.id.is_empty()
                            || !seen.insert(call.id.as_str())
                        {
                            return Err("tool_call invalide ou identifiant dupliqué");
                        }
                        let arguments: Value = serde_json::from_str(&call.function.arguments)
                            .map_err(|_| "les arguments d'outil doivent être un objet JSON")?;
                        if !arguments.is_object() {
                            return Err("les arguments d'outil doivent être un objet JSON");
                        }
                        pending.insert(call.id.as_str());
                        if call_index > 0 || !content.is_empty() {
                            prompt.push('\n');
                        }
                        let function = json!({"name": call.function.name, "arguments": arguments});
                        prompt.push_str(&format!("<tool_call>\n{function}\n</tool_call>"));
                    }
                    prompt.push_str("<|im_end|>\n");
                }
                "tool" => {
                    let id = message
                        .tool_call_id
                        .as_deref()
                        .ok_or("tool_call_id manquant")?;
                    if !pending.remove(id) {
                        return Err("résultat d'outil sans appel correspondant ou déjà reçu");
                    }
                    let content = content.ok_or("le résultat d'outil doit contenir du texte")?;
                    if index == 0 || self.messages[index - 1].role != "tool" {
                        prompt.push_str("<|im_start|>user");
                    }
                    prompt.push_str(&format!("\n<tool_response>\n{content}\n</tool_response>"));
                    if self
                        .messages
                        .get(index + 1)
                        .is_none_or(|next| next.role != "tool")
                    {
                        prompt.push_str("<|im_end|>\n");
                    }
                }
                _ => {
                    return Err(
                        "les rôles acceptés sont system, developer, user, assistant et tool",
                    );
                }
            }
        }
        if !pending.is_empty() {
            return Err("résultat d'outil manquant");
        }
        prompt.push_str("<|im_start|>assistant\n");
        Ok(prompt)
    }

    /// Un budget explicite reste strict ; sans budget, utilise tout le contexte restant.
    pub fn completion_budget(
        &self,
        prompt_tokens: usize,
        max_context: usize,
    ) -> Result<usize, &'static str> {
        let remaining = max_context
            .checked_sub(prompt_tokens)
            .filter(|remaining| *remaining > 0)
            .ok_or("le prompt ne laisse aucune place pour une réponse")?;
        match self.max_tokens {
            Some(tokens) if tokens == 0 || tokens > remaining => {
                Err("prompt et réponse demandée dépassent le contexte disponible")
            }
            Some(tokens) => Ok(tokens),
            None => Ok(remaining),
        }
    }
}
