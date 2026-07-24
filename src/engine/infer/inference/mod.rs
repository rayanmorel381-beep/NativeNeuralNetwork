mod forward_ops;
mod postprocess;
mod scratch;
mod activations;
pub(crate) mod transformer_block;
mod graph;
pub mod transformer_forward;
mod generate;
mod backward;
mod lm_run;
mod lm_text;
mod lm_chat;
mod run;
pub(crate) mod prefill;

pub use forward_ops::*;
pub use postprocess::*;
pub use generate::{generate, generate_lora, GenerateArgs, LoraModel, TokenSink};
pub use lm_run::{run_lm, LmRunBufs, LmRunInputs, LoraSpec};
pub use lm_text::{run_lm_text, LmTextBufs, LmTextInputs};
pub use lm_chat::{run_lm_chat, LmChatBufs, LmChatInputs};
pub(crate) use run::run_inference;
pub use scratch::ForwardScratch;
pub use activations::{BlockActivations, TrainActivations};
pub use backward::{transformer_block_backward, mlp_head_backward, BackwardScratch};
pub(crate) use backward::rmsnorm_backward;
pub use transformer_block::{transformer_block_forward, BlockQuant};
pub(crate) use graph::mlp_graph_forward;
pub use crate::engine::train::trainer::batched_train::{
    batched_work_count, block_param_count, checkpoint_count, forward_loss_batched,
    train_batched, TrainBatchedBuffers,
};
#[cfg(feature = "publisher-trust-service")]
pub use crate::engine::train::trainer::batched_train::forward_predict_batched;
