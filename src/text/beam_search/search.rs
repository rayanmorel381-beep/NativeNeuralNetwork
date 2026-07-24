#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeamError {
    ShapeMismatch,
    Empty,
}

pub fn select_top_beams(
    scores: &[f32],
    beam_width: usize,
    out_indices: &mut [usize],
) -> Result<usize, BeamError> {
    if scores.is_empty() || beam_width == 0 {
        return Err(BeamError::Empty);
    }
    if out_indices.len() < beam_width {
        return Err(BeamError::ShapeMismatch);
    }

    let filled = beam_width.min(scores.len());

    for (slot, idx) in out_indices.iter_mut().take(filled).zip(0..filled) {
        *slot = idx;
    }

    for i in filled..scores.len() {
        let mut worst_slot = 0usize;
        for s in 1..filled {
            if scores[out_indices[s]] < scores[out_indices[worst_slot]] {
                worst_slot = s;
            }
        }

        if scores[i] > scores[out_indices[worst_slot]] {
            out_indices[worst_slot] = i;
        }
    }

    out_indices[..filled].sort_unstable_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    Ok(filled)
}
