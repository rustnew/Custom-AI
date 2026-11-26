use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use futures_util::StreamExt;
use inference::{ChatRequest, GenerationEvent, Job};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tower::ServiceExt;

fn payload(stream: bool) -> Value {
    // Format observé avec opencode v2.2.6, runtime openai-compatible.
    json!({
        "model":"qwen3-4b-instruct",
        "messages":[{"role":"system","content":"Tu es un assistant."},{"role":"user","content":"Lis le fichier."}],
        "tools":[{"type":"function","function":{
            "name":"read", "description":"Lire un fichier", "strict":false,
            "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
        }}],
        "stream":stream, "stream_options":{"include_usage":true}, "max_completion_tokens":2048
    })
}

fn prompt(value: Value) -> Result<String, &'static str> {
    serde_json::from_value::<ChatRequest>(value)
        .unwrap()
        .prompt("qwen3-4b-instruct", 15360)
}

fn app() -> (axum::Router, mpsc::Receiver<Job>) {
    let (sender, jobs) = mpsc::channel(1);
    (
        inference::router(sender, "qwen3-4b-instruct".into(), 15360),
        jobs,
    )
}

fn request(value: &Value) -> Request<Body> {
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(value.to_string()))
        .unwrap()
}

async fn generated(value: Value, text: &str) -> (StatusCode, String) {
    let (app, mut jobs) = app();
    let text = text.to_owned();
    let worker = tokio::spawn(async move {
        let job = jobs.recv().await.unwrap();
        if job
            .events
            .send(GenerationEvent::Started {
                tokens: 42,
                cached_tokens: 16,
            })
            .await
            .is_err()
        {
            return;
        }
        // Toutes les balises, chaînes JSON et lettres UTF-8 sont coupées entre deltas.
        for character in text.chars() {
            if job
                .events
                .send(GenerationEvent::Delta(character.to_string()))
                .await
                .is_err()
            {
                return;
            }
        }
        let _ = job
            .events
            .send(GenerationEvent::Finished {
                tokens: 25,
                reason: "stop",
            })
            .await;
    });
    let response = app.oneshot(request(&value)).await.unwrap();
    let status = response.status();
    let body = String::from_utf8(
        to_bytes(response.into_body(), 100_000)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    worker.await.unwrap();
    (status, body)
}

fn chunks(body: &str) -> Vec<Value> {
    assert!(body.ends_with("data: [DONE]\n\n"));
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

#[test]
fn qwen_template_keeps_tools_assistant_calls_and_grouped_tool_results() {
    let mut value = payload(false);
    value["messages"].as_array_mut().unwrap().extend([
        json!({"role":"assistant","content":null,"tool_calls":[
            {"id":"call_1","type":"function","function":{"name":"read","arguments":"{\"path\":\"é.txt\"}"}},
            {"id":"call_2","type":"function","function":{"name":"read","arguments":"{\"path\":\"b.txt\"}"}}
        ]}),
        json!({"role":"tool","tool_call_id":"call_1","content":[{"type":"text","text":"Résultat un"}]}),
        json!({"role":"tool","tool_call_id":"call_2","content":"Résultat deux"}),
    ]);
    let text = prompt(value).unwrap();
    assert!(text.starts_with("<|im_start|>system\nTu es un assistant.\n\n# Tools\n"));
    assert!(text.contains("<tools>\n{\"type\":\"function\""));
    assert!(text.contains("<|im_start|>assistant\n<tool_call>\n{\"arguments\":{\"path\":\"é.txt\"},\"name\":\"read\"}\n</tool_call>"));
    assert!(text.ends_with("<|im_start|>user\n<tool_response>\nRésultat un\n</tool_response>\n<tool_response>\nRésultat deux\n</tool_response><|im_end|>\n<|im_start|>assistant\n"));
}

#[test]
fn invalid_history_and_unsupported_controls_fail_before_generation() {
    for extra in [
        json!({"role":"tool","content":"résultat","tool_call_id":"missing"}),
        json!({"role":"user","content":null}),
        json!({"role":"user","content":"x","tool_calls":[{"id":"x","type":"function","function":{"name":"read","arguments":"{}"}}]}),
        json!({"role":"assistant","content":null,"tool_calls":[{"id":"x","type":"function","function":{"name":"read","arguments":"{}"}}]}),
    ] {
        let mut value = payload(false);
        value["messages"].as_array_mut().unwrap().push(extra);
        assert!(prompt(value).is_err());
    }
    let mut value = payload(false);
    value["tool_choice"] = json!({"type":"function","function":{"name":"missing"}});
    assert!(prompt(value).is_err());
    let mut value = payload(false);
    value["tools"][0]["function"]["strict"] = json!(true);
    assert!(prompt(value).is_err());
    let mut value = payload(false);
    value["tools"][0]["type"] = json!("custom");
    assert!(prompt(value).is_err());
    let mut value = payload(false);
    value["response_format"] = json!({"type":"json_object"});
    assert!(serde_json::from_value::<ChatRequest>(value).is_err());
    let mut value = payload(false);
    value["messages"][1]["content"] = json!([{"type":"image_url","image_url":{"url":"x"}}]);
    assert!(serde_json::from_value::<ChatRequest>(value).is_err());
}

#[test]
fn tool_choice_none_omits_definitions_and_forced_choices_reach_the_prompt() {
    let mut value = payload(false);
    value["tool_choice"] = json!("none");
    assert!(!prompt(value.clone()).unwrap().contains("<tools>"));
    value["tool_choice"] = json!({"type":"function","function":{"name":"read"}});
    value["parallel_tool_calls"] = json!(false);
    let text = prompt(value).unwrap();
    assert!(text.contains("You must call the function read."));
    assert!(text.contains("Call only one function in this response."));
}

#[tokio::test]
async fn json_tool_calls_have_openai_ids_arguments_and_finish_reason() {
    let (status, body) = generated(payload(false), "Je lis.\n<tool_call>\n{\"name\":\"read\",\"arguments\":{\"path\":\"é.txt\"}}\n</tool_call>").await;
    assert_eq!(status, StatusCode::OK);
    let body: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    let message = &body["choices"][0]["message"];
    assert_eq!(message["content"], "Je lis.\n");
    assert!(
        message["tool_calls"][0]["id"]
            .as_str()
            .unwrap()
            .starts_with("call_")
    );
    assert_eq!(message["tool_calls"][0]["function"]["name"], "read");
    assert_eq!(
        serde_json::from_str::<Value>(
            message["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap()
        )
        .unwrap(),
        json!({"path":"é.txt"})
    );
    assert_eq!(body["usage"]["total_tokens"], 67);
}

#[tokio::test]
async fn streaming_parses_split_tags_and_preserves_usage_and_parallel_indexes() {
    let text = "<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"é.txt\"}}</tool_call>\n<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"b.txt\"}}</tool_call>";
    let (status, body) = generated(payload(true), text).await;
    assert_eq!(status, StatusCode::OK);
    let chunks = chunks(&body);
    let calls: Vec<_> = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["tool_calls"][0].as_object())
        .collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["index"], 0);
    assert_eq!(calls[1]["index"], 1);
    assert_ne!(calls[0]["id"], calls[1]["id"]);
    assert_eq!(calls[0]["function"]["arguments"], "{\"path\":\"é.txt\"}");
    assert_eq!(
        chunks[chunks.len() - 2]["choices"][0]["finish_reason"],
        "tool_calls"
    );
    assert_eq!(chunks.last().unwrap()["choices"], json!([]));
    assert_eq!(
        chunks.last().unwrap()["usage"]["prompt_tokens_details"]["cached_tokens"],
        16
    );
    assert_eq!(chunks.last().unwrap()["usage"]["total_tokens"], 67);
    let text: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(text, "\n");
}

#[tokio::test]
async fn tool_arguments_can_contain_xml_delimiters_without_being_truncated() {
    let (_, body) = generated(payload(false), "<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"</tool_call><tool_call>\"}}</tool_call>").await;
    let body: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        "{\"path\":\"</tool_call><tool_call>\"}"
    );
}

#[tokio::test]
async fn malformed_unknown_and_truncated_calls_are_errors_in_json_and_sse() {
    for text in [
        "<tool_call>{broken}</tool_call>",
        "<tool_call>{\"name\":\"bash\",\"arguments\":{}}</tool_call>",
        "<tool_call>{\"name\":\"read\",\"arguments\":[]}</tool_call>",
        "<tool_call>{\"name\":\"read\",\"arguments\":{}}",
    ] {
        let (status, body) = generated(payload(false), text).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY, "{body}");
        assert!(serde_json::from_str::<Value>(&body).unwrap()["error"].is_object());
        let (_, body) = generated(payload(true), text).await;
        let chunks = chunks(&body);
        assert!(chunks.iter().any(|c| c["error"].is_object()));
        assert!(
            !chunks
                .iter()
                .any(|c| c["choices"][0]["finish_reason"] == "tool_calls")
        );
    }
}

#[tokio::test]
async fn required_none_and_single_call_choices_are_enforced_on_model_output() {
    let call = "<tool_call>{\"name\":\"read\",\"arguments\":{}}</tool_call>";
    let mut value = payload(false);
    value["tool_choice"] = json!("none");
    assert_eq!(generated(value, call).await.0, StatusCode::BAD_GATEWAY);
    let mut value = payload(false);
    value["tool_choice"] = json!("required");
    assert_eq!(
        generated(value.clone(), "Bonjour.").await.0,
        StatusCode::BAD_GATEWAY
    );
    assert_eq!(generated(value, call).await.0, StatusCode::OK);
    let mut value = payload(false);
    value["parallel_tool_calls"] = json!(false);
    assert_eq!(
        generated(value, &format!("{call}\n{call}")).await.0,
        StatusCode::BAD_GATEWAY
    );
}

#[tokio::test]
async fn every_native_tool_name_uses_the_same_generic_function_call_protocol() {
    let names = [
        "edit",
        "experimental_image_generate",
        "glob",
        "grep",
        "question",
        "read",
        "shell",
        "skill",
        "subagent",
        "webfetch",
        "websearch",
        "write",
        "execute",
    ];
    let mut value = payload(true);
    value["tools"] = json!(
        names
            .iter()
            .map(|name| json!({
                "type":"function", "function":{"name":name,"parameters":{"type":"object"}}
            }))
            .collect::<Vec<_>>()
    );
    let arguments =
        json!({"nested":[{"text":"é\n\"citation\"","flag":true}],"number":42,"empty":null});
    let text = names
        .iter()
        .map(|name| {
            format!(
                "<tool_call>{}</tool_call>",
                json!({"name":name,"arguments":arguments})
            )
        })
        .collect::<String>();
    let (status, body) = generated(value, &text).await;
    assert_eq!(status, StatusCode::OK);
    let chunks = chunks(&body);
    let calls: Vec<_> = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["tool_calls"][0].as_object())
        .collect();
    assert_eq!(calls.len(), names.len());
    for (call, name) in calls.iter().zip(names) {
        assert_eq!(call["function"]["name"], name);
        assert_eq!(
            serde_json::from_str::<Value>(call["function"]["arguments"].as_str().unwrap()).unwrap(),
            arguments
        );
    }
}

#[tokio::test]
async fn opencode_reasoning_effort_is_accepted_without_changing_the_instruct_prompt() {
    let original = prompt(payload(false)).unwrap();
    for effort in [
        Value::Null,
        json!("none"),
        json!("minimal"),
        json!("low"),
        json!("medium"),
        json!("high"),
        json!("xhigh"),
        json!("max"),
    ] {
        let mut value = payload(true);
        value["reasoning_effort"] = effort;
        assert_eq!(prompt(value.clone()).unwrap(), original);
        let (status, body) = generated(value, "BONJOUR_LOCAL").await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let chunks = chunks(&body);
        assert!(
            chunks
                .iter()
                .any(|chunk| chunk["choices"][0]["finish_reason"] == "stop")
        );
        assert!(!chunks.iter().any(|chunk| chunk["error"].is_object()));
    }
}

#[tokio::test]
async fn ordinary_text_stays_incremental_when_tools_are_available() {
    let (app, mut jobs) = app();
    let (release, ready) = tokio::sync::oneshot::channel();
    let worker = tokio::spawn(async move {
        let job = jobs.recv().await.unwrap();
        job.events
            .send(GenerationEvent::Started {
                tokens: 1,
                cached_tokens: 0,
            })
            .await
            .unwrap();
        job.events
            .send(GenerationEvent::Delta("Bonjour é".into()))
            .await
            .unwrap();
        ready.await.unwrap();
        job.events
            .send(GenerationEvent::Finished {
                tokens: 2,
                reason: "stop",
            })
            .await
            .unwrap();
    });
    let response = app.oneshot(request(&payload(true))).await.unwrap();
    let mut stream = response.into_body().into_data_stream();
    let _role = stream.next().await.unwrap().unwrap();
    let text = tokio::time::timeout(std::time::Duration::from_secs(1), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(
        String::from_utf8(text.to_vec())
            .unwrap()
            .contains("Bonjour é")
    );
    release.send(()).unwrap();
    while stream.next().await.is_some() {}
    worker.await.unwrap();
}
