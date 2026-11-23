use inference::{
    ChatRequest, GenerationEvent, GenerationFailure, Job,
    worker::{Generator, run_worker},
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::sync::mpsc;

struct FaultyGenerator {
    calls: usize,
    panic_first: bool,
    resets: Arc<AtomicUsize>,
    fatal_reset: bool,
}
impl Generator for FaultyGenerator {
    fn generate(&mut self, job: &Job) -> Result<(), GenerationFailure> {
        self.calls += 1;
        if self.calls == 1 {
            assert!(!self.panic_first, "panne simulée du générateur");
            return Err(GenerationFailure::engine("allocation GPU simulée refusée"));
        }
        let _ = job.events.blocking_send(GenerationEvent::Started {
            tokens: 2,
            cached_tokens: 0,
        });
        let _ = job.events.blocking_send(GenerationEvent::Finished {
            tokens: 1,
            reason: "stop",
        });
        Ok(())
    }
    fn recover(&mut self) -> Result<(), String> {
        self.resets.fetch_add(1, Ordering::SeqCst);
        if self.fatal_reset {
            Err("contexte perdu".into())
        } else {
            Ok(())
        }
    }
}

fn job() -> (Job, mpsc::Receiver<GenerationEvent>) {
    let (events, receiver) = mpsc::channel(4);
    let request: ChatRequest =
        serde_json::from_str(r#"{"model":"qwen","messages":[{"role":"user","content":"hello"}]}"#)
            .unwrap();
    (
        Job {
            request,
            prompt: "hello".into(),
            events,
        },
        receiver,
    )
}

#[tokio::test]
async fn recoverable_error_and_panic_do_not_kill_the_next_request() {
    for panic_first in [false, true] {
        let resets = Arc::new(AtomicUsize::new(0));
        let mut engine = FaultyGenerator {
            calls: 0,
            panic_first,
            resets: resets.clone(),
            fatal_reset: false,
        };
        let (sender, mut queue) = mpsc::channel(4);
        let worker = tokio::task::spawn_blocking(move || {
            run_worker(
                &mut engine,
                &mut queue,
                &inference::worker::WorkerStatus::default(),
            )
        });
        let (first, mut first_events) = job();
        sender.send(first).await.unwrap();
        match first_events.recv().await.unwrap() {
            GenerationEvent::Failed(failure) => assert_eq!(
                failure.status,
                axum::http::StatusCode::INTERNAL_SERVER_ERROR
            ),
            _ => panic!("le client doit recevoir l'erreur"),
        }
        let (next, mut next_events) = job();
        sender.send(next).await.unwrap();
        assert!(matches!(
            next_events.recv().await,
            Some(GenerationEvent::Started { .. })
        ));
        assert!(matches!(
            next_events.recv().await,
            Some(GenerationEvent::Finished { .. })
        ));
        drop(sender);
        worker.await.unwrap().unwrap();
        assert_eq!(resets.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn unrecoverable_device_failure_stops_processing_instead_of_reporting_false_health() {
    let mut engine = FaultyGenerator {
        calls: 0,
        panic_first: false,
        resets: Arc::new(AtomicUsize::new(0)),
        fatal_reset: true,
    };
    let (sender, mut queue) = mpsc::channel(4);
    let worker = tokio::task::spawn_blocking(move || {
        run_worker(
            &mut engine,
            &mut queue,
            &inference::worker::WorkerStatus::default(),
        )
    });
    let (first, mut events) = job();
    sender.send(first).await.unwrap();
    assert!(matches!(
        events.recv().await,
        Some(GenerationEvent::Failed(_))
    ));
    assert_eq!(worker.await.unwrap(), Err("contexte perdu".into()));
    assert!(sender.is_closed());
}

#[tokio::test]
async fn cancelled_queued_job_never_calls_the_gpu() {
    let resets = Arc::new(AtomicUsize::new(0));
    let mut engine = FaultyGenerator {
        calls: 0,
        panic_first: false,
        resets: resets.clone(),
        fatal_reset: false,
    };
    let (sender, mut queue) = mpsc::channel(4);
    let (cancelled, receiver) = job();
    drop(receiver);
    sender.send(cancelled).await.unwrap();
    drop(sender);
    tokio::task::spawn_blocking(move || {
        run_worker(
            &mut engine,
            &mut queue,
            &inference::worker::WorkerStatus::default(),
        )
        .unwrap();
        assert_eq!(engine.calls, 0);
    })
    .await
    .unwrap();
    assert_eq!(resets.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn slow_consumer_cannot_block_the_gpu_sender_forever() {
    let (mut job, _receiver) = job();
    let (sender, _slow_reader) = mpsc::channel(1);
    sender
        .send(GenerationEvent::Delta("full".into()))
        .await
        .unwrap();
    job.events = sender;
    let sent = tokio::task::spawn_blocking(move || {
        job.emit_with_timeout(
            GenerationEvent::Delta("blocked".into()),
            std::time::Duration::from_millis(20),
        )
    });
    assert!(
        !tokio::time::timeout(std::time::Duration::from_secs(1), sent)
            .await
            .unwrap()
            .unwrap()
    );
}

#[tokio::test]
async fn readiness_is_false_until_error_recovery_has_finished() {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;
    struct Recovering {
        entered: Option<tokio::sync::oneshot::Sender<()>>,
        release: std::sync::mpsc::Receiver<()>,
    }
    impl Generator for Recovering {
        fn generate(&mut self, _: &Job) -> Result<(), GenerationFailure> {
            Err(GenerationFailure::engine("simulated OOM"))
        }
        fn recover(&mut self) -> Result<(), String> {
            self.entered.take().unwrap().send(()).unwrap();
            self.release
                .recv_timeout(std::time::Duration::from_secs(2))
                .map_err(|error| error.to_string())
        }
    }
    let (entered, waiting) = tokio::sync::oneshot::channel();
    let (release, receiver) = std::sync::mpsc::channel();
    let mut engine = Recovering {
        entered: Some(entered),
        release: receiver,
    };
    let status = inference::worker::WorkerStatus::default();
    let worker_status = status.clone();
    let (sender, mut queue) = mpsc::channel(4);
    let api = inference::router_with_status(sender.clone(), "qwen".into(), 32768, status.clone());
    let worker =
        tokio::task::spawn_blocking(move || run_worker(&mut engine, &mut queue, &worker_status));
    let (first, _events) = job();
    sender.send(first).await.unwrap();
    waiting.await.unwrap();
    assert!(status.is_recovering());
    let response = api
        .clone()
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    release.send(()).unwrap();
    drop(api);
    drop(sender);
    worker.await.unwrap().unwrap();
    assert!(!status.is_recovering());
}
