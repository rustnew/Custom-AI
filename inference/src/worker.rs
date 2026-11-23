use crate::{GenerationEvent, GenerationFailure, Job};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::sync::mpsc;

pub trait Generator {
    fn generate(&mut self, job: &Job) -> Result<(), GenerationFailure>;
    /// Vide les états partiels et vérifie que le moteur peut traiter la requête suivante.
    fn recover(&mut self) -> Result<(), String>;
}

#[derive(Clone, Default)]
pub struct WorkerStatus(Arc<AtomicBool>);

impl WorkerStatus {
    pub fn is_recovering(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

pub fn run_worker(
    generator: &mut impl Generator,
    queue: &mut mpsc::Receiver<Job>,
    status: &WorkerStatus,
) -> Result<(), String> {
    while let Some(job) = queue.blocking_recv() {
        if job.events.is_closed() {
            continue;
        }
        let result = catch_unwind(AssertUnwindSafe(|| generator.generate(&job)));
        let failure = match result {
            Ok(Ok(())) => continue,
            Ok(Err(failure)) => failure,
            Err(_) => {
                GenerationFailure::engine("la génération a été interrompue par une erreur interne")
            }
        };
        // Le client reçoit une erreur explicite. La reprise doit réussir avant
        // d'accepter un autre travail ; un contexte CUDA perdu reste fatal.
        status.0.store(true, Ordering::Release);
        job.emit(GenerationEvent::Failed(failure));
        generator.recover()?;
        status.0.store(false, Ordering::Release);
    }
    Ok(())
}
