use alloc::{format, string::{String, ToString}, vec::Vec};
use super::{event::NNNLogEvent, NNNLogFormat};

pub(crate) fn format(event: &NNNLogEvent, output: NNNLogFormat) -> String {
    match output {
        NNNLogFormat::Cli => {
            let context = cli_context(&event.context);
            format!("{} {:<12} {} {}{}", event.level.as_str(), event.component, event.code, super::redact(&event.detail), context)
        }
        NNNLogFormat::Uix => format!(
            "{{\"level\":{},\"component\":{},\"code\":{},\"detail\":{},\"context\":{}}}",
            json_string(event.level.as_str()),
            json_string(&super::redact(&event.component)),
            json_string(&super::redact(&event.code)),
            json_string(&super::redact(&event.detail)),
            context_json(&event.context),
        ),
    }
}

fn context_json(context: &super::event::NNNLogContext) -> String {
    let mut fields = Vec::new();
    for (name, value) in [
        ("run_id", context.run_id.as_deref()),
        ("event_id", context.event_id.as_deref()),
        ("parent_id", context.parent_id.as_deref()),
        ("stage", context.stage.as_deref()),
        ("operation", context.operation.as_deref()),
        ("phase", context.phase.as_deref()),
        ("protocol", context.protocol.as_deref()),
        ("command_id", context.command_id.as_deref()),
        ("device", context.device.as_deref()),
        ("partition", context.partition.as_deref()),
        ("image", context.image.as_deref()),
        ("outcome", context.outcome.as_deref()),
        ("progress_unit", context.progress_unit.as_deref()),
        ("error_kind", context.error_kind.as_deref()),
    ] {
        if let Some(value) = value {
            fields.push(format!("{}:{}", json_string(name), json_string(&super::redact(value))));
        }
    }
    if let Some(duration_ms) = context.duration_ms {
        fields.push(format!("\"duration_ms\":{duration_ms}"));
    }
    if let Some(progress_completed) = context.progress_completed { fields.push(format!("\"progress_completed\":{progress_completed}")); }
    if let Some(progress_total) = context.progress_total { fields.push(format!("\"progress_total\":{progress_total}")); }
    if let Some(exit_code) = context.exit_code { fields.push(format!("\"exit_code\":{exit_code}")); }
    if let Some(retry_count) = context.retry_count { fields.push(format!("\"retry_count\":{retry_count}")); }
    for (name, value) in &context.attributes {
        fields.push(format!("{}:{}", json_string(name), json_string(&super::redact(value))));
    }
    format!("{{{}}}", fields.join(","))
}

fn cli_context(context: &super::event::NNNLogContext) -> String {
    let mut values = Vec::new();
    for (name, value) in [("stage", context.stage.as_deref()), ("operation", context.operation.as_deref()), ("phase", context.phase.as_deref()), ("command", context.command_id.as_deref()), ("protocol", context.protocol.as_deref()), ("error", context.error_kind.as_deref())] {
        if let Some(value) = value { values.push(format!("{name}={}", super::redact(value))); }
    }
    if let Some(completed) = context.progress_completed { values.push(format!("progress={completed}/{}", context.progress_total.map_or_else(|| "?".into(), |total| total.to_string()))); }
    if values.is_empty() { String::new() } else { format!(" [{}]", values.join(" ")) }
}

fn json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            '\u{08}' => escaped.push_str("\\b"),
            '\u{0c}' => escaped.push_str("\\f"),
            character if character.is_control() => {
                escaped.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => escaped.push(character),
        }
    }
    escaped.push('"');
    escaped
}
