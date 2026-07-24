#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationKind {
    Identity,
    Relu,
    Sigmoid,
    Tanh,
    SoftmaxOutput,
}

impl ActivationKind {
    pub fn to_u8(self) -> u8 {
        match self {
            ActivationKind::Identity => 0,
            ActivationKind::Relu => 1,
            ActivationKind::Sigmoid => 2,
            ActivationKind::Tanh => 3,
            ActivationKind::SoftmaxOutput => 4,
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(ActivationKind::Identity),
            1 => Some(ActivationKind::Relu),
            2 => Some(ActivationKind::Sigmoid),
            3 => Some(ActivationKind::Tanh),
            4 => Some(ActivationKind::SoftmaxOutput),
            _ => None,
        }
    }

    pub fn apply(self, x: f32) -> f32 {
        match self {
            ActivationKind::Identity => x,
            ActivationKind::SoftmaxOutput => x,
            ActivationKind::Relu => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            ActivationKind::Sigmoid => {
                if x >= 0.0 {
                    let z = crate::base::math::expf(-x);
                    1.0 / (1.0 + z)
                } else {
                    let z = crate::base::math::expf(x);
                    z / (1.0 + z)
                }
            }
            ActivationKind::Tanh => crate::base::math::tanhf(x),
        }
    }

    pub fn apply_f64(self, x: f64) -> f64 {
        match self {
            ActivationKind::Identity => x,
            ActivationKind::SoftmaxOutput => x,
            ActivationKind::Relu => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            ActivationKind::Sigmoid => {
                if x >= 0.0 {
                    let z = crate::base::math::expd(-x);
                    1.0 / (1.0 + z)
                } else {
                    let z = crate::base::math::expd(x);
                    z / (1.0 + z)
                }
            }
            ActivationKind::Tanh => crate::base::math::tanhd(x),
        }
    }

    pub fn apply_in_place_f64(self, values: &mut [f64]) {
        for value in values {
            *value = self.apply_f64(*value);
        }
    }

    pub fn derivative_from_output(self, output: f32) -> f32 {
        match self {
            ActivationKind::Identity => 1.0,
            ActivationKind::SoftmaxOutput => output * (1.0 - output),
            ActivationKind::Relu => {
                if output > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationKind::Sigmoid => output * (1.0 - output),
            ActivationKind::Tanh => 1.0 - output * output,
        }
    }

    pub fn derivative_from_output_f64(self, output: f64) -> f64 {
        match self {
            ActivationKind::Identity => 1.0,
            ActivationKind::SoftmaxOutput => output * (1.0 - output),
            ActivationKind::Relu => {
                if output > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationKind::Sigmoid => output * (1.0 - output),
            ActivationKind::Tanh => 1.0 - output * output,
        }
    }

    pub fn apply_in_place(self, values: &mut [f32]) {
        for value in values {
            *value = self.apply(*value);
        }
    }

    pub fn apply_t<T: crate::base::math::Float>(self, x: T) -> T {
        match self {
            ActivationKind::Identity => x,
            ActivationKind::SoftmaxOutput => x,
            ActivationKind::Relu => {
                if x > T::ZERO {
                    x
                } else {
                    T::ZERO
                }
            }
            ActivationKind::Sigmoid => {
                if x >= T::ZERO {
                    let z = (-x).exp();
                    T::ONE / (T::ONE + z)
                } else {
                    let z = x.exp();
                    z / (T::ONE + z)
                }
            }
            ActivationKind::Tanh => x.tanh(),
        }
    }
}