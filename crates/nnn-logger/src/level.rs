#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum NNNLogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl NNNLogLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Trace => "Trace",
            Self::Debug => "Debug",
            Self::Info => "Info",
            Self::Warn => "Warn",
            Self::Error => "Error",
        }
    }
}
