#![cfg(unix)]

use std::{
    fs,
    net::TcpListener,
    os::unix::fs::PermissionsExt,
    path::PathBuf,
    process::{Command, Output, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
    thread,
    time::{Duration, Instant},
};

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

struct ModelFixture(PathBuf);

impl ModelFixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "custom-ai-serve-test-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        for file in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            fs::write(path.join(file), "{}").unwrap();
        }
        fs::write(
            path.join("model.safetensors.index.json"),
            r#"{"weight_map":{"layer.weight":"weights.safetensors"}}"#,
        )
        .unwrap();
        fs::write(path.join("weights.safetensors"), "test-only-placeholder").unwrap();
        Self(path)
    }

    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_cli"));
        command.args(["serve", "--model"]).arg(&self.0);
        command
    }
}

impl Drop for ModelFixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn finish(mut command: Command) -> Output {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            child.kill().unwrap();
            let output = child.wait_with_output().unwrap();
            panic!("la CLI ne s'arrête pas : {output:?}");
        }
        thread::sleep(Duration::from_millis(10));
    }
    child.wait_with_output().unwrap()
}

#[test]
fn missing_weights_are_rejected_before_engine_start() {
    let fixture = ModelFixture::new();
    fs::remove_file(fixture.0.join("weights.safetensors")).unwrap();
    let output = finish(fixture.command());
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("fichier de poids absent ou vide"));
}

#[test]
fn engine_failure_stops_the_local_service_and_is_reported() {
    let fixture = ModelFixture::new();
    let engine = fixture.0.join("fake-vllm");
    fs::write(&engine, "#!/bin/sh\nexit 42\n").unwrap();
    fs::set_permissions(&engine, fs::Permissions::from_mode(0o700)).unwrap();
    let available_port = TcpListener::bind("127.0.0.1:0").unwrap();
    let upstream_port = available_port.local_addr().unwrap().port();
    drop(available_port);
    let mut command = fixture.command();
    command
        .args(["--bind", "127.0.0.1:0", "--upstream-port"])
        .arg(upstream_port.to_string())
        .arg("--vllm-bin")
        .arg(engine);
    let output = finish(command);
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("vLLM s'est arrêté"), "{error}");
    assert!(error.contains("42"), "{error}");
    // Le listener de la gateway ne doit pas survivre à la mort du moteur.
    assert!(String::from_utf8_lossy(&output.stdout).contains("Gateway : http://127.0.0.1:"));
}

#[test]
fn occupied_engine_port_is_reported_before_spawning_a_second_engine() {
    let fixture = ModelFixture::new();
    let occupied = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut command = fixture.command();
    command
        .args(["--bind", "127.0.0.1:0", "--upstream-port"])
        .arg(occupied.local_addr().unwrap().port().to_string());
    let output = finish(command);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("port vLLM déjà occupé"));
}
