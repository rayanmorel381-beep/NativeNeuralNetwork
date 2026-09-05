use alloc::{string::String, vec::Vec};
use super::{event::NNNLogEvent, formatter, NNNLogFormat};

pub struct NNNFileLogger {
    pub path: String,
    output: NNNLogFormat,
    last_error: Option<String>,
    lines: Vec<String>,
}
impl NNNFileLogger {
    pub fn new(path: impl Into<String>, output: NNNLogFormat) -> Self {
        Self { path: path.into(), output, last_error: None, lines: Vec::new() }
    }

    pub fn write(&mut self, event: &NNNLogEvent, output: NNNLogFormat) -> Result<(), String> {
        self.lines.push(formatter::format(event, output));
        Ok(())
    }

    pub fn sync(&self) -> Result<(), String> {
        Ok(())
    }

    pub fn last_error(&self) -> Option<&str> { self.last_error.as_deref() }

    pub fn lines(&self) -> &[String] { &self.lines }
}

impl NNNFileLogger {
    pub fn try_emit(&mut self, event: &NNNLogEvent) -> Result<(), String> {
        self.write(event, self.output)
    }

    pub fn take_error(&mut self) -> Option<String> {
        self.last_error.take()
    }
}

impl super::logger::Logger for NNNFileLogger {
    fn emit(&mut self, event: NNNLogEvent) {
        if let Err(error) = self.try_emit(&event) {
            self.last_error = Some(error);
        }
    }
}
