use alloc::{string::String, vec::Vec};
use super::level::NNNLogLevel;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NNNLogContext {
    pub run_id: Option<String>,
    pub event_id: Option<String>,
    pub parent_id: Option<String>,
    pub stage: Option<String>,
    pub operation: Option<String>,
    pub phase: Option<String>,
    pub protocol: Option<String>,
    pub command_id: Option<String>,
    pub device: Option<String>,
    pub partition: Option<String>,
    pub image: Option<String>,
    pub outcome: Option<String>,
    pub duration_ms: Option<u64>,
    pub progress_completed: Option<u64>,
    pub progress_total: Option<u64>,
    pub progress_unit: Option<String>,
    pub error_kind: Option<String>,
    pub exit_code: Option<i32>,
    pub retry_count: Option<u32>,
    pub attributes: Vec<(String, String)>,
}

#[derive(Clone, Debug)]
pub struct NNNLogEvent {
    pub level: NNNLogLevel,
    pub component: String,
    pub code: String,
    pub detail: String,
    pub context: NNNLogContext,
}

impl NNNLogEvent {
    pub fn new(level: NNNLogLevel, component: &str, code: &str, detail: &str) -> Self {
        Self {
            level,
            component: component.into(),
            code: code.into(),
            detail: detail.into(),
            context: NNNLogContext::default(),
        }
    }

    pub fn with_context(mut self, context: NNNLogContext) -> Self {
        self.context = context;
        self
    }

    pub fn command(level: NNNLogLevel, component: &str, command_id: &str, detail: &str) -> Self {
        Self::new(level, component, "command", detail).with_context(NNNLogContext::default().with_command(command_id))
    }

    pub fn progress(component: &str, completed: u64, total: Option<u64>, unit: &str) -> Self {
        Self::new(NNNLogLevel::Info, component, "progress", "operation progress")
            .with_context(NNNLogContext::default().with_progress(completed, total, unit))
    }

    pub fn error(component: &str, code: &str, detail: &str, kind: &str) -> Self {
        Self::new(NNNLogLevel::Error, component, code, detail)
            .with_context(NNNLogContext::default().with_error_kind(kind))
    }
}

impl NNNLogContext {
    pub fn with_command(mut self, command_id: &str) -> Self { self.command_id = Some(command_id.into()); self }
    pub fn with_stage(mut self, stage: &str) -> Self { self.stage = Some(stage.into()); self }
    pub fn with_operation(mut self, operation: &str) -> Self { self.operation = Some(operation.into()); self }
    pub fn with_phase(mut self, phase: &str) -> Self { self.phase = Some(phase.into()); self }
    pub fn with_protocol(mut self, protocol: &str) -> Self { self.protocol = Some(protocol.into()); self }
    pub fn with_progress(mut self, completed: u64, total: Option<u64>, unit: &str) -> Self {
        self.progress_completed = Some(completed); self.progress_total = total; self.progress_unit = Some(unit.into()); self
    }
    pub fn with_error_kind(mut self, kind: &str) -> Self { self.error_kind = Some(kind.into()); self }
    pub fn with_attribute(mut self, name: &str, value: &str) -> Self { self.attributes.push((name.into(), value.into())); self }
}
