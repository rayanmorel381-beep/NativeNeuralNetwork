use crate::engine::rnn_flow::{blob_range, rnn_dtype, RnnFlowError, MAX_LAYERS};
use crate::format::model_config::{model_config, Precision};
use crate::format::model_format::{model_format, BLOB_BIASES, BLOB_WEIGHTS};
use super::helpers::*;

pub fn evolve_add_neuron(
    bytes: &[u8],
    layer_idx: usize,
    seed: u64,
    out: &mut [u8],
) -> Result<usize, RnnFlowError> {
    let dtype = rnn_dtype(bytes)?;
    let (topology, topo_len) = parse_topology(bytes)?;
    let topo = &topology[..topo_len];

    if layer_idx == 0 || layer_idx >= topo_len - 1 {
        return Err(RnnFlowError::InvalidTopology);
    }

    let model_name = extract_model_name(bytes);

    let (w_start, w_end) = blob_range(bytes, BLOB_WEIGHTS)?;
    let (b_start, b_end) = blob_range(bytes, BLOB_BIASES)?;
    let old_w_bytes = &bytes[w_start..w_end];
    let old_b_bytes = &bytes[b_start..b_end];

    let mut new_topo = [0usize; MAX_LAYERS];
    new_topo[..topo_len].copy_from_slice(topo);
    new_topo[layer_idx] += 1;
    let new_topo_slice = &new_topo[..topo_len];

    let ctx = EvolveCtx {
        model_name, old_topo: topo, new_topo: new_topo_slice,
        old_w_bytes, old_b_bytes, seed,
    };
    match dtype {
        0 => evolve_add_neuron_f32_inner(&ctx, layer_idx, out),
        1 => evolve_add_neuron_f64_inner(&ctx, layer_idx, out),
        _ => Err(RnnFlowError::BadBytes),
    }
}

fn evolve_add_neuron_f32_inner(
    ctx: &EvolveCtx,
    layer_idx: usize,
    out: &mut [u8],
) -> Result<usize, RnnFlowError> {
    let (old_wc, old_bc) = crate::base::initializers::expected_parameter_counts(ctx.old_topo)
        .ok_or(RnnFlowError::InvalidTopology)?;
    let (new_wc, new_bc) = crate::base::initializers::expected_parameter_counts(ctx.new_topo)
        .ok_or(RnnFlowError::InvalidTopology)?;

    if ctx.old_w_bytes.len() != old_wc * 4 || ctx.old_b_bytes.len() != old_bc * 4 {
        return Err(RnnFlowError::BadBytes);
    }

    let new_w_size = new_wc * 4;
    let new_b_size = new_bc * 4;
    let workspace_needed = new_w_size + new_b_size;
    let third = out.len() / 3;
    if workspace_needed > third || third == 0 {
        return Err(RnnFlowError::CapacityTooSmall);
    }

    let workspace_total = workspace_needed.saturating_add(third);
    let split = out.len().saturating_sub(workspace_total) & !7;
    let (rnn_out, workspace) = out.split_at_mut(split);
    let ws_total = workspace.len();
    if ws_total < workspace_needed {
        return Err(RnnFlowError::CapacityTooSmall);
    }
    let remainder = ws_total - workspace_needed;
    let metadata_cap = (remainder / 5).max(1).min(remainder.saturating_sub(1));
    let rmd1_cap = remainder - metadata_cap;

    if rmd1_cap == 0 || metadata_cap == 0 {
        return Err(RnnFlowError::CapacityTooSmall);
    }

    let (w_area, rest) = workspace.split_at_mut(new_w_size);
    let (b_area, rest2) = rest.split_at_mut(new_b_size);
    let (rmd1_area, metadata_area) = rest2.split_at_mut(rmd1_cap);

    build_new_weights_f32(ctx.old_topo, ctx.new_topo, layer_idx, ctx.old_w_bytes, w_area, ctx.seed);
    build_new_biases_f32(ctx.old_topo, ctx.new_topo, layer_idx, ctx.old_b_bytes, b_area, ctx.seed);

    let w_f32 = bytes_as_f32(w_area);
    let b_f32 = bytes_as_f32(b_area);

    let precision = Precision::F32 { weights: w_f32, biases: b_f32, runtime_input: None, conv_spec: None };
    let rmd1_used = model_config(ctx.new_topo, &precision, rmd1_area)
        .map_err(|_| RnnFlowError::Model)?;

    model_format(
        ctx.model_name, ctx.new_topo, &rmd1_area[..rmd1_used],
        None, &precision, metadata_area, rnn_out,
    )
}

fn evolve_add_neuron_f64_inner(
    ctx: &EvolveCtx,
    layer_idx: usize,
    out: &mut [u8],
) -> Result<usize, RnnFlowError> {
    let (old_wc, old_bc) = crate::base::initializers::expected_parameter_counts(ctx.old_topo)
        .ok_or(RnnFlowError::InvalidTopology)?;
    let (new_wc, new_bc) = crate::base::initializers::expected_parameter_counts(ctx.new_topo)
        .ok_or(RnnFlowError::InvalidTopology)?;

    if ctx.old_w_bytes.len() != old_wc * 8 || ctx.old_b_bytes.len() != old_bc * 8 {
        return Err(RnnFlowError::BadBytes);
    }

    let new_w_size = new_wc * 8;
    let new_b_size = new_bc * 8;
    let workspace_needed = new_w_size + new_b_size;
    let third = out.len() / 3;
    if workspace_needed > third || third == 0 {
        return Err(RnnFlowError::CapacityTooSmall);
    }

    let workspace_total = workspace_needed.saturating_add(third);
    let split = out.len().saturating_sub(workspace_total) & !7;
    let (rnn_out, workspace) = out.split_at_mut(split);
    let ws_total = workspace.len();
    if ws_total < workspace_needed {
        return Err(RnnFlowError::CapacityTooSmall);
    }
    let remainder = ws_total - workspace_needed;
    let metadata_cap = (remainder / 5).max(1).min(remainder.saturating_sub(1));
    let rmd1_cap = remainder - metadata_cap;

    if rmd1_cap == 0 || metadata_cap == 0 {
        return Err(RnnFlowError::CapacityTooSmall);
    }

    let (w_area, rest) = workspace.split_at_mut(new_w_size);
    let (b_area, rest2) = rest.split_at_mut(new_b_size);
    let (rmd1_area, metadata_area) = rest2.split_at_mut(rmd1_cap);

    build_new_weights_f64(ctx.old_topo, ctx.new_topo, layer_idx, ctx.old_w_bytes, w_area, ctx.seed);
    build_new_biases_f64(ctx.old_topo, ctx.new_topo, layer_idx, ctx.old_b_bytes, b_area, ctx.seed);

    let w_f64 = bytes_as_f64(w_area);
    let b_f64 = bytes_as_f64(b_area);

    let precision = Precision::F64 { weights: w_f64, biases: b_f64, runtime_input: None };
    let rmd1_used = model_config(ctx.new_topo, &precision, rmd1_area)
        .map_err(|_| RnnFlowError::Model)?;

    model_format(
        ctx.model_name, ctx.new_topo, &rmd1_area[..rmd1_used],
        None, &precision, metadata_area, rnn_out,
    )
}

fn build_new_weights_f32(
    old_topo: &[usize],
    new_topo: &[usize],
    layer_idx: usize,
    old_w_bytes: &[u8],
    new_w_bytes: &mut [u8],
    seed: u64,
) {
    let mut rng = SplitMix64::new(seed);
    let layer_count = old_topo.len() - 1;
    let mut old_off = 0usize;
    let mut new_off = 0usize;

    for l in 0..layer_count {
        let old_in = old_topo[l];
        let old_out = old_topo[l + 1];
        let new_in = new_topo[l];
        let new_out = new_topo[l + 1];

        if l + 1 == layer_idx {
            let limit = xavier_limit_f32(new_in, new_out);
            for r in 0..old_in {
                for c in 0..old_out {
                    let v = read_f32(old_w_bytes, (old_off + r * old_out + c) * 4);
                    write_f32(new_w_bytes, (new_off + r * new_out + c) * 4, v);
                }
                let v = (rng.next_f32() * 2.0 - 1.0) * limit;
                write_f32(new_w_bytes, (new_off + r * new_out + old_out) * 4, v);
            }
        } else if l == layer_idx {
            let limit = xavier_limit_f32(new_in, new_out);
            for r in 0..old_in {
                for c in 0..old_out {
                    let v = read_f32(old_w_bytes, (old_off + r * old_out + c) * 4);
                    write_f32(new_w_bytes, (new_off + r * new_out + c) * 4, v);
                }
            }
            for c in 0..new_out {
                let v = (rng.next_f32() * 2.0 - 1.0) * limit;
                write_f32(new_w_bytes, (new_off + old_in * new_out + c) * 4, v);
            }
        } else {
            let block = old_in * old_out * 4;
            new_w_bytes[new_off * 4..new_off * 4 + block]
                .copy_from_slice(&old_w_bytes[old_off * 4..old_off * 4 + block]);
        }

        old_off += old_in * old_out;
        new_off += new_in * new_out;
    }
}

fn build_new_biases_f32(
    old_topo: &[usize],
    new_topo: &[usize],
    layer_idx: usize,
    old_b_bytes: &[u8],
    new_b_bytes: &mut [u8],
    seed: u64,
) {
    let mut rng = SplitMix64::new(seed.wrapping_add(0xDEADBEEF));
    let layer_count = old_topo.len() - 1;
    let mut old_off = 0usize;
    let mut new_off = 0usize;

    for l in 0..layer_count {
        let old_size = old_topo[l + 1];
        let new_size = new_topo[l + 1];

        if l + 1 == layer_idx {
            new_b_bytes[new_off * 4..new_off * 4 + old_size * 4]
                .copy_from_slice(&old_b_bytes[old_off * 4..old_off * 4 + old_size * 4]);
            let v = rng.next_f32() * 0.01;
            write_f32(new_b_bytes, (new_off + old_size) * 4, v);
        } else {
            new_b_bytes[new_off * 4..new_off * 4 + old_size * 4]
                .copy_from_slice(&old_b_bytes[old_off * 4..old_off * 4 + old_size * 4]);
        }

        old_off += old_size;
        new_off += new_size;
    }
}

fn build_new_weights_f64(
    old_topo: &[usize],
    new_topo: &[usize],
    layer_idx: usize,
    old_w_bytes: &[u8],
    new_w_bytes: &mut [u8],
    seed: u64,
) {
    let mut rng = SplitMix64::new(seed);
    let layer_count = old_topo.len() - 1;
    let mut old_off = 0usize;
    let mut new_off = 0usize;

    for l in 0..layer_count {
        let old_in = old_topo[l];
        let old_out = old_topo[l + 1];
        let new_in = new_topo[l];
        let new_out = new_topo[l + 1];

        if l + 1 == layer_idx {
            let limit = xavier_limit_f64(new_in, new_out);
            for r in 0..old_in {
                for c in 0..old_out {
                    let v = read_f64(old_w_bytes, (old_off + r * old_out + c) * 8);
                    write_f64(new_w_bytes, (new_off + r * new_out + c) * 8, v);
                }
                let v = (rng.next_f64() * 2.0 - 1.0) * limit;
                write_f64(new_w_bytes, (new_off + r * new_out + old_out) * 8, v);
            }
        } else if l == layer_idx {
            let limit = xavier_limit_f64(new_in, new_out);
            for r in 0..old_in {
                for c in 0..old_out {
                    let v = read_f64(old_w_bytes, (old_off + r * old_out + c) * 8);
                    write_f64(new_w_bytes, (new_off + r * new_out + c) * 8, v);
                }
            }
            for c in 0..new_out {
                let v = (rng.next_f64() * 2.0 - 1.0) * limit;
                write_f64(new_w_bytes, (new_off + old_in * new_out + c) * 8, v);
            }
        } else {
            let block = old_in * old_out * 8;
            new_w_bytes[new_off * 8..new_off * 8 + block]
                .copy_from_slice(&old_w_bytes[old_off * 8..old_off * 8 + block]);
        }

        old_off += old_in * old_out;
        new_off += new_in * new_out;
    }
}

fn build_new_biases_f64(
    old_topo: &[usize],
    new_topo: &[usize],
    layer_idx: usize,
    old_b_bytes: &[u8],
    new_b_bytes: &mut [u8],
    seed: u64,
) {
    let mut rng = SplitMix64::new(seed.wrapping_add(0xDEADBEEF));
    let layer_count = old_topo.len() - 1;
    let mut old_off = 0usize;
    let mut new_off = 0usize;

    for l in 0..layer_count {
        let old_size = old_topo[l + 1];
        let new_size = new_topo[l + 1];

        if l + 1 == layer_idx {
            new_b_bytes[new_off * 8..new_off * 8 + old_size * 8]
                .copy_from_slice(&old_b_bytes[old_off * 8..old_off * 8 + old_size * 8]);
            let v = rng.next_f64() * 0.01;
            write_f64(new_b_bytes, (new_off + old_size) * 8, v);
        } else {
            new_b_bytes[new_off * 8..new_off * 8 + old_size * 8]
                .copy_from_slice(&old_b_bytes[old_off * 8..old_off * 8 + old_size * 8]);
        }

        old_off += old_size;
        new_off += new_size;
    }
}
