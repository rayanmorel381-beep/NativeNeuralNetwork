mod helpers;
mod add_neuron;
mod add_layer;
mod split_neuron;

pub use add_neuron::evolve_add_neuron;
pub use split_neuron::evolve_split_neuron;
pub use add_layer::evolve_add_layer;

use crate::format::model_config::TrainingMetrics;
use crate::api::rnn_api::core_api::{map_engine_error, RnnApiError};

#[derive(Clone, Copy)]
pub struct EvolveConfig {
    pub max_evolves: usize,
}

fn evolve_policy(metrics: &TrainingMetrics) -> f64 {
    if metrics.last_loss <= 0.0 {
        1.0
    } else if metrics.avg_loss > 0.0 {
        1.0 / (1.0 + metrics.avg_loss / metrics.last_loss.max(1.0))
    } else {
        1.0
    }
}

pub fn do_evolve(
    model_bytes: &mut [u8],
    current_enc_len: usize,
    evolve_idx: usize,
    iteration_seed: usize,
) -> Result<usize, RnnApiError> {
    let (_loss, new_enc_len) = crate::engine::train::trainer::apply_evolve_step(
        model_bytes,
        current_enc_len,
        evolve_idx,
        iteration_seed,
    )
    .map_err(map_engine_error)?;
    Ok(new_enc_len)
}

pub struct PhasePlan {
    pub phase_idx: usize,
    pub phase_seconds: u64,
    pub phase_max_iterations: Option<usize>,
    pub accumulated_iterations: usize,
    pub accumulated_elapsed_ns: u64,
}

pub struct PhaseResult {
    pub iterations: usize,
    pub elapsed_ns: u64,
    pub avg_loss: f64,
    pub last_loss: f64,
}

pub enum PhaseEvent {
    Phase(PhasePlan),
    Evolve { evolve_idx: usize, iteration_seed: usize },
    Done,
}

pub struct PhaseScheduler {
    n_phases: usize,
    phase_seconds: u64,
    max_iterations: Option<usize>,
    evolves_done: usize,
    max_evolves: usize,
    prev_loss: f64,
    current_phase: usize,
    accumulated_iters: usize,
    accumulated_elapsed_ns: u64,
    last_loss: f64,
    last_avg_loss: f64,
    pending_evolve: bool,
    improvement_ratio: f64,
}

impl PhaseScheduler {
    pub fn new(
        evolve_cfg: EvolveConfig,
        train_seconds: u64,
        max_iterations: Option<usize>,
        start_iteration: usize,
        elapsed_offset_ns: u64,
    ) -> Self {
        let n_phases = evolve_cfg.max_evolves + 1;
        let phase_seconds_raw = train_seconds / n_phases as u64;
        let phase_seconds = if phase_seconds_raw < 3 { 3 } else { phase_seconds_raw };
        Self {
            n_phases,
            phase_seconds,
            max_iterations,
            evolves_done: 0,
            max_evolves: evolve_cfg.max_evolves,
            prev_loss: f64::MAX,
            current_phase: 0,
            accumulated_iters: start_iteration,
            accumulated_elapsed_ns: elapsed_offset_ns,
            last_loss: 0.0,
            last_avg_loss: 0.0,
            pending_evolve: false,
            improvement_ratio: 0.95,
        }
    }

    pub fn next_event(&mut self) -> PhaseEvent {
        if self.pending_evolve {
            self.pending_evolve = false;
            let evt = PhaseEvent::Evolve {
                evolve_idx: self.evolves_done,
                iteration_seed: self.accumulated_iters,
            };
            self.evolves_done += 1;
            return evt;
        }
        if self.current_phase >= self.n_phases {
            return PhaseEvent::Done;
        }
        let phase_max = self.max_iterations.map(|m| self.accumulated_iters + m / self.n_phases);
        PhaseEvent::Phase(PhasePlan {
            phase_idx: self.current_phase,
            phase_seconds: self.phase_seconds,
            phase_max_iterations: phase_max,
            accumulated_iterations: self.accumulated_iters,
            accumulated_elapsed_ns: self.accumulated_elapsed_ns,
        })
    }

    pub fn record(&mut self, result: &PhaseResult) {
        self.accumulated_iters = result.iterations;
        self.accumulated_elapsed_ns = result.elapsed_ns;
        let is_last_phase = self.current_phase + 1 >= self.n_phases;
        let evolve_budget_left = self.evolves_done < self.max_evolves;
        let stagnated = result.last_loss >= self.prev_loss * self.improvement_ratio;
        let metrics = TrainingMetrics {
            iterations: result.iterations,
            avg_loss: result.avg_loss,
            last_loss: result.last_loss,
            elapsed_ns: result.elapsed_ns,
            cpu_temp: None,
            logical_cores: 1,
            current_workers: 1,
            max_workers: 1,
            eval_mse: result.avg_loss,
            eval_mae: result.avg_loss,
            eval_accuracy: 0.0,
            eval_cross_entropy: result.avg_loss,
            eval_confidence: 0.0,
            eval_ops_per_second: 0.0,
            eval_bytes_per_second: 0.0,
            eval_arithmetic_intensity: 0.0,
            eval_memory_heavy: false,
        };
        let policy_score = evolve_policy(&metrics);
        self.pending_evolve = !is_last_phase && evolve_budget_left && stagnated && policy_score > 0.5;
        self.prev_loss = result.last_loss;
        self.last_loss = result.last_loss;
        self.last_avg_loss = result.avg_loss;
        self.current_phase += 1;
    }

    pub fn final_iterations(&self) -> usize { self.accumulated_iters }
    pub fn final_elapsed_ns(&self) -> u64 { self.accumulated_elapsed_ns }
    pub fn final_last_loss(&self) -> f64 { self.last_loss }
    pub fn final_avg_loss(&self) -> f64 { self.last_avg_loss }
    pub fn evolves_done(&self) -> usize { self.evolves_done }
}

