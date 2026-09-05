#![no_std]

extern crate alloc;

mod event;
mod file;
mod formatter;
mod level;
mod logger;

pub use event::{NNNLogContext, NNNLogEvent};
pub use file::NNNFileLogger;
pub use level::NNNLogLevel;
pub use logger::{FanoutLogger, NNNLogger, LogFilter, Logger};

use alloc::{borrow::ToOwned, string::String};

pub fn redact(value: &str) -> String {
	let mut redacted = value.to_owned();
	for key in ["serial", "serialno", "device_serial", "adb_serial"] {
		redacted = redact_key(&redacted, key);
	}
	redacted
}

fn redact_key(value: &str, key: &str) -> String {
	let mut output = String::with_capacity(value.len());
	for part in value.split_whitespace() {
		if let Some((name, _)) = part.split_once('=') {
			if name.eq_ignore_ascii_case(key) {
				output.push_str(name);
				output.push_str("=<redacted> ");
				continue;
			}
		}
		output.push_str(part);
		output.push(' ');
	}
	output.trim_end().to_owned()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NNNLogFormat { Cli, Uix }

pub fn format(event: &NNNLogEvent, output: NNNLogFormat) -> String {
	formatter::format(event, output)
}
