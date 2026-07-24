use crate::observability::visualization::errors::VisualizeError;
use crate::observability::visualization::view::NetworkView;
use core::f32;

const VERTEX_FLOATS: usize = 8;
const LAYER_SPACING: f32 = 2.0;
const NEURON_SPACING: f32 = 1.0;
const DEFAULT_NEURON_QUAD_SIZE: f32 = 0.4;
const CONNECTION_THICKNESS: f32 = 0.05;

struct SmallUsizeVec<const N: usize> {
    len: usize,
    data: [usize; N],
}

impl<const N: usize> SmallUsizeVec<N> {
    const fn new() -> Self {
        Self {
            len: 0,
            data: [0usize; N],
        }
    }

    fn push(&mut self, v: usize) -> Result<(), ()> {
        if self.len >= N {
            return Err(());
        }
        self.data[self.len] = v;
        self.len += 1;
        Ok(())
    }

    fn get(&self, i: usize) -> Option<&usize> {
        if i < self.len {
            Some(&self.data[i])
        } else {
            None
        }
    }
}

pub fn mesh_required_buffers_from_bytes(bytes: &[u8]) -> Result<(usize, usize), VisualizeError> {
    let view = NetworkView::from_rnn_bytes(bytes)?;

    let mut total_neurons = 0usize;
    if view.layer_count() == 0 {
        return Ok((0, 0));
    }
    let first = view.layer(0)?;
    total_neurons = total_neurons.saturating_add(first.input_count()?);
    let mut total_connections = 0usize;
    for i in 0..view.layer_count() {
        let layer = view.layer(i)?;
        let input_size = layer.input_count()?;
        let output_size = layer.neuron_count()?;
        total_neurons = total_neurons.saturating_add(output_size);
        total_connections =
            total_connections.saturating_add(input_size.saturating_mul(output_size));
    }

    let vertex_count = total_neurons;
    let index_count = total_connections.saturating_mul(2);
    Ok((vertex_count, index_count))
}

pub fn fill_mesh_from_bytes(
    bytes: &[u8],
    vertex_buf: &mut [f32],
    index_buf: &mut [u32],
) -> Result<(usize, usize), VisualizeError> {
    let view = NetworkView::from_rnn_bytes(bytes)?;
    if view.layer_count() == 0 {
        return Ok((0, 0));
    }

    let (vertex_count, index_count) = mesh_required_buffers_from_bytes(bytes)?;
    if vertex_buf.len()
        < vertex_count
            .checked_mul(VERTEX_FLOATS)
            .ok_or(VisualizeError::InvalidFormat)?
    {
        return Err(VisualizeError::OutOfBounds);
    }
    if index_buf.len() < index_count {
        return Err(VisualizeError::OutOfBounds);
    }

    let mut bases: SmallUsizeVec<32> = SmallUsizeVec::new();
    let mut next_base = 0usize;
    let first = view.layer(0)?;
    let first_input = first.input_count()?;
    bases
        .push(next_base)
        .map_err(|_| VisualizeError::InvalidFormat)?;
    next_base = next_base.saturating_add(first_input);
    for i in 0..view.layer_count() {
        let layer = view.layer(i)?;
        bases
            .push(next_base)
            .map_err(|_| VisualizeError::InvalidFormat)?;
        next_base = next_base.saturating_add(layer.neuron_count()?);
    }

    let mut v_idx = 0usize;
    let input_size = first_input;
    for j in 0..input_size {
        let vx = 0f32;
        let vy = j as f32 * NEURON_SPACING;
        write_vertex(vertex_buf, v_idx, vx, vy);
        v_idx += 1;
    }
    for i in 0..view.layer_count() {
        let layer = view.layer(i)?;
        let output_size = layer.neuron_count()?;
        let layer_x = (i + 1) as f32 * LAYER_SPACING;
        for j in 0..output_size {
            let vx = layer_x;
            let vy = j as f32 * NEURON_SPACING;
            write_vertex(vertex_buf, v_idx, vx, vy);
            v_idx += 1;
        }
    }

    let mut idx_write = 0usize;
    for i in 0..view.layer_count() {
        let layer = view.layer(i)?;
        let input_size = layer.input_count()?;
        let output_size = layer.neuron_count()?;
        let src_base = bases.get(i).copied().ok_or(VisualizeError::InvalidFormat)?;
        let dst_base = bases
            .get(i + 1)
            .copied()
            .ok_or(VisualizeError::InvalidFormat)?;
        for out_j in 0..output_size {
            let dst_idx = (dst_base + out_j) as u32;
            for in_k in 0..input_size {
                let src_idx = (src_base + in_k) as u32;
                if idx_write + 2 > index_buf.len() {
                    return Err(VisualizeError::OutOfBounds);
                }
                index_buf[idx_write] = src_idx;
                idx_write += 1;
                index_buf[idx_write] = dst_idx;
                idx_write += 1;
            }
        }
    }

    Ok((vertex_count, idx_write))
}

fn write_vertex(buf: &mut [f32], vertex_index: usize, x: f32, y: f32) {
    let base = vertex_index * VERTEX_FLOATS;
    if base + VERTEX_FLOATS > buf.len() {
        return;
    }
    buf[base] = x;
    buf[base + 1] = y;
    buf[base + 2] = DEFAULT_NEURON_QUAD_SIZE;
    buf[base + 3] = 0.0;
    buf[base + 4] = 0.0;
    buf[base + 5] = 1.0;
    buf[base + 6] = CONNECTION_THICKNESS;
    buf[base + 7] = 0.0;
}
