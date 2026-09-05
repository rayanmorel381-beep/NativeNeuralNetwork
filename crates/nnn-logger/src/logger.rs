use alloc::vec::Vec;
use super::event::NNNLogEvent;
use super::level::NNNLogLevel;

pub trait Logger {
    fn emit(&mut self, event: NNNLogEvent);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogFilter {
    pub minimum_level: NNNLogLevel,
}

impl Default for LogFilter {
    fn default() -> Self { Self { minimum_level: NNNLogLevel::Trace } }
}

impl LogFilter {
    pub fn allows(&self, event: &NNNLogEvent) -> bool { event.level >= self.minimum_level }
}

pub struct FanoutLogger<'a> {
    sinks: Vec<&'a mut dyn Logger>,
}

impl<'a> FanoutLogger<'a> {
    pub fn new() -> Self { Self { sinks: Vec::new() } }

    pub fn add_sink(&mut self, sink: &'a mut dyn Logger) {
        self.sinks.push(sink);
    }
}

impl Logger for FanoutLogger<'_> {
    fn emit(&mut self, event: NNNLogEvent) {
        for sink in &mut self.sinks {
            sink.emit(event.clone());
        }
    }
}

pub struct NNNLogger {
    events: Vec<NNNLogEvent>,
    filter: LogFilter,
}

impl NNNLogger {
    pub fn new() -> Self { Self::default() }
    pub fn with_filter(filter: LogFilter) -> Self { Self { events: Vec::new(), filter } }
    pub fn filter(&self) -> LogFilter { self.filter }
    pub fn events(&self) -> &[NNNLogEvent] { &self.events }
}

impl Default for NNNLogger {
    fn default() -> Self { Self { events: Vec::new(), filter: LogFilter::default() } }
}

impl Logger for NNNLogger {
    fn emit(&mut self, event: NNNLogEvent) {
        if self.filter.allows(&event) { self.events.push(event); }
    }
}
