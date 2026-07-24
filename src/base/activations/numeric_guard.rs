use super::ActivationKind;

pub fn validate_activation_outputs(kind: ActivationKind, values: &[f32]) -> bool {
    for &v in values {
        let y = kind.apply(v);
        if !y.is_finite() {
            return false;
        }
    }
    true
}

pub fn validate_activation_outputs_f64(kind: ActivationKind, values: &[f64]) -> bool {
    for &v in values {
        let y = kind.apply_f64(v);
        if !y.is_finite() {
            return false;
        }
    }
    true
}

pub fn derivative_is_finite(kind: ActivationKind, outputs: &[f32]) -> bool {
    for &y in outputs {
        let d = kind.derivative_from_output(y);
        if !d.is_finite() {
            return false;
        }
    }
    true
}

pub fn derivative_is_finite_f64(kind: ActivationKind, outputs: &[f64]) -> bool {
    for &y in outputs {
        let d = kind.derivative_from_output_f64(y);
        if !d.is_finite() {
            return false;
        }
    }
    true
}
