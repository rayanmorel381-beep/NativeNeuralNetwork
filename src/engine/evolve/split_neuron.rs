use crate::engine::rnn_flow::{blob_range, rnn_dtype, RnnFlowError, MAX_LAYERS};
use crate::format::model_config::{model_config, Precision};
use crate::format::model_format::{model_format, BLOB_BIASES, BLOB_WEIGHTS};
use super::helpers::*;

pub fn evolve_split_neuron(
    bytes: &[u8],
    layer_idx: usize,
    neuron_idx: usize,
    seed: u64,
    out: &mut [u8],
) -> Result<usize, RnnFlowError> {
    let dtype = rnn_dtype(bytes)?;
    let (topology, topo_len) = parse_topology(bytes)?;
    let topo = &topology[..topo_len];

    if layer_idx == 0 || layer_idx >= topo_len - 1 {
        return Err(RnnFlowError::InvalidTopology);
    }
    if neuron_idx >= topo[layer_idx] {
        return Err(RnnFlowError::InvalidTopology);
    }

    let model_name = extract_model_name(bytes);
    let (w_start, w_end) = blob_range(bytes, BLOB_WEIGHTS)?;
    let (b_start, b_end) = blob_range(bytes, BLOB_BIASES)?;

    let mut new_topo = [0usize; MAX_LAYERS];
    new_topo[..topo_len].copy_from_slice(topo);
    new_topo[layer_idx] += 1;
    let new_topo_slice = &new_topo[..topo_len];

    let ctx = EvolveCtx {
        model_name, old_topo: topo, new_topo: new_topo_slice,
        old_w_bytes: &bytes[w_start..w_end],
        old_b_bytes: &bytes[b_start..b_end],
        seed,
    };
    match dtype {
        0 => evolve_split_f32(&ctx, layer_idx, neuron_idx, out),
        1 => evolve_split_f64(&ctx, layer_idx, neuron_idx, out),
        _ => Err(RnnFlowError::BadBytes),
    }
}

fn evolve_split_f32(
    ctx: &EvolveCtx,
    layer_idx: usize,
    neuron_idx: usize,
    out: &mut [u8],
) -> Result<usize, RnnFlowError> {
    let (new_wc, new_bc) = crate::base::initializers::expected_parameter_counts(ctx.new_topo)
        .ok_or(RnnFlowError::InvalidTopology)?;

    let new_w_size = new_wc * 4;
    let new_b_size = new_bc * 4;
    let workspace_needed = new_w_size + new_b_size;
    let third = out.len() / 3;
    if workspace_needed > third || third == 0 {
        return Err(RnnFlowError::CapacityTooSmall);
    }

    let split_point = (out.len() - workspace_needed - third) & !7;
    let (rnn_out, workspace) = out.split_at_mut(split_point);
    let ws_total = workspace.len();
    let remainder = ws_total - workspace_needed;
    let metadata_cap = (remainder / 5).max(1).min(remainder.saturating_sub(1));
    let rmd1_cap = remainder - metadata_cap;

    if rmd1_cap == 0 {
        return Err(RnnFlowError::CapacityTooSmall);
    }

    let (w_area, rest) = workspace.split_at_mut(new_w_size);
    let (b_area, rest2) = rest.split_at_mut(new_b_size);
    let (rmd1_area, metadata_area) = rest2.split_at_mut(rmd1_cap);

    let mut rng = SplitMix64::new(ctx.seed);
    let layer_count = ctx.old_topo.len() - 1;
    let mut old_w_off = 0usize;
    let mut new_w_off = 0usize;
    let mut old_b_off = 0usize;
    let mut new_b_off = 0usize;

    for l in 0..layer_count {
        let old_in = ctx.old_topo[l];
        let old_out = ctx.old_topo[l + 1];
        let new_in = ctx.new_topo[l];
        let new_out = ctx.new_topo[l + 1];

        if l + 1 == layer_idx {
            for r in 0..old_in {
                for c in 0..old_out {
                    let v = read_f32(ctx.old_w_bytes, (old_w_off + r * old_out + c) * 4);
                    write_f32(w_area, (new_w_off + r * new_out + c) * 4, v);
                    if c == neuron_idx {
                        let noise = (rng.next_f32() * 2.0 - 1.0) * v.abs() * 0.01;
                        write_f32(w_area, (new_w_off + r * new_out + old_out) * 4, v + noise);
                    }
                }
            }
        } else if l == layer_idx {
            for r in 0..old_in {
                for c in 0..old_out {
                    let v = read_f32(ctx.old_w_bytes, (old_w_off + r * old_out + c) * 4);
                    if r == neuron_idx {
                        write_f32(w_area, (new_w_off + r * new_out + c) * 4, v * 0.5);
                    } else {
                        write_f32(w_area, (new_w_off + r * new_out + c) * 4, v);
                    }
                }
            }
            for c in 0..old_out {
                let v = read_f32(ctx.old_w_bytes, (old_w_off + neuron_idx * old_out + c) * 4);
                let noise = (rng.next_f32() * 2.0 - 1.0) * v.abs() * 0.01;
                write_f32(w_area, (new_w_off + old_in * new_out + c) * 4, v * 0.5 + noise);
            }
        } else {
            let block = old_in * old_out * 4;
            w_area[new_w_off * 4..new_w_off * 4 + block]
                .copy_from_slice(&ctx.old_w_bytes[old_w_off * 4..old_w_off * 4 + block]);
        }

        if l + 1 == layer_idx {
            b_area[new_b_off * 4..new_b_off * 4 + old_out * 4]
                .copy_from_slice(&ctx.old_b_bytes[old_b_off * 4..old_b_off * 4 + old_out * 4]);
            let orig_bias = read_f32(ctx.old_b_bytes, (old_b_off + neuron_idx) * 4);
            write_f32(b_area, (new_b_off + old_out) * 4, orig_bias);
        } else {
            b_area[new_b_off * 4..new_b_off * 4 + old_out * 4]
                .copy_from_slice(&ctx.old_b_bytes[old_b_off * 4..old_b_off * 4 + old_out * 4]);
        }

        old_w_off += old_in * old_out;
        new_w_off += new_in * new_out;
        old_b_off += old_out;
        new_b_off += new_out;
    }

    let w_f32 = bytes_as_f32(&w_area[..new_w_size]);
    let b_f32 = bytes_as_f32(&b_area[..new_b_size]);

    let precision = Precision::F32 { weights: w_f32, biases: b_f32, runtime_input: None, conv_spec: None };
    let rmd1_used = model_config(ctx.new_topo, &precision, rmd1_area)
        .map_err(|_| RnnFlowError::Model)?;

    model_format(ctx.model_name, ctx.new_topo, &rmd1_area[..rmd1_used], None, &precision, metadata_area, rnn_out)
}

fn evolve_split_f64(
    ctx: &EvolveCtx,
    layer_idx: usize,
    neuron_idx: usize,
    out: &mut [u8],
) -> Result<usize, RnnFlowError> {
    let (new_wc, new_bc) = crate::base::initializers::expected_parameter_counts(ctx.new_topo)
        .ok_or(RnnFlowError::InvalidTopology)?;

    let new_w_size = new_wc * 8;
    let new_b_size = new_bc * 8;
    let workspace_needed = new_w_size + new_b_size;
    let third = out.len() / 3;
    if workspace_needed > third || third == 0 {
        return Err(RnnFlowError::CapacityTooSmall);
    }

    let split_point = (out.len() - workspace_needed - third) & !7;
    let (rnn_out, workspace) = out.split_at_mut(split_point);
    let ws_total = workspace.len();
    let remainder = ws_total - workspace_needed;
    let metadata_cap = (remainder / 5).max(1).min(remainder.saturating_sub(1));
    let rmd1_cap = remainder - metadata_cap;

    if rmd1_cap == 0 {
        return Err(RnnFlowError::CapacityTooSmall);
    }

    let (w_area, rest) = workspace.split_at_mut(new_w_size);
    let (b_area, rest2) = rest.split_at_mut(new_b_size);
    let (rmd1_area, metadata_area) = rest2.split_at_mut(rmd1_cap);

    let mut rng = SplitMix64::new(ctx.seed);
    let layer_count = ctx.old_topo.len() - 1;
    let mut old_w_off = 0usize;
    let mut new_w_off = 0usize;
    let mut old_b_off = 0usize;
    let mut new_b_off = 0usize;

    for l in 0..layer_count {
        let old_in = ctx.old_topo[l];
        let old_out = ctx.old_topo[l + 1];
        let new_in = ctx.new_topo[l];
        let new_out = ctx.new_topo[l + 1];

        if l + 1 == layer_idx {
            for r in 0..old_in {
                for c in 0..old_out {
                    let v = read_f64(ctx.old_w_bytes, (old_w_off + r * old_out + c) * 8);
                    write_f64(w_area, (new_w_off + r * new_out + c) * 8, v);
                    if c == neuron_idx {
                        let noise = (rng.next_f64() * 2.0 - 1.0) * v.abs() * 0.01;
                        write_f64(w_area, (new_w_off + r * new_out + old_out) * 8, v + noise);
                    }
                }
            }
        } else if l == layer_idx {
            for r in 0..old_in {
                for c in 0..old_out {
                    let v = read_f64(ctx.old_w_bytes, (old_w_off + r * old_out + c) * 8);
                    if r == neuron_idx {
                        write_f64(w_area, (new_w_off + r * new_out + c) * 8, v * 0.5);
                    } else {
                        write_f64(w_area, (new_w_off + r * new_out + c) * 8, v);
                    }
                }
            }
            for c in 0..old_out {
                let v = read_f64(ctx.old_w_bytes, (old_w_off + neuron_idx * old_out + c) * 8);
                let noise = (rng.next_f64() * 2.0 - 1.0) * v.abs() * 0.01;
                write_f64(w_area, (new_w_off + old_in * new_out + c) * 8, v * 0.5 + noise);
            }
        } else {
            let block = old_in * old_out * 8;
            w_area[new_w_off * 8..new_w_off * 8 + block]
                .copy_from_slice(&ctx.old_w_bytes[old_w_off * 8..old_w_off * 8 + block]);
        }

        if l + 1 == layer_idx {
            b_area[new_b_off * 8..new_b_off * 8 + old_out * 8]
                .copy_from_slice(&ctx.old_b_bytes[old_b_off * 8..old_b_off * 8 + old_out * 8]);
            let orig_bias = read_f64(ctx.old_b_bytes, (old_b_off + neuron_idx) * 8);
            write_f64(b_area, (new_b_off + old_out) * 8, orig_bias);
        } else {
            b_area[new_b_off * 8..new_b_off * 8 + old_out * 8]
                .copy_from_slice(&ctx.old_b_bytes[old_b_off * 8..old_b_off * 8 + old_out * 8]);
        }

        old_w_off += old_in * old_out;
        new_w_off += new_in * new_out;
        old_b_off += old_out;
        new_b_off += new_out;
    }

    let w_f64 = bytes_as_f64(&w_area[..new_w_size]);
    let b_f64 = bytes_as_f64(&b_area[..new_b_size]);

    let precision = Precision::F64 { weights: w_f64, biases: b_f64, runtime_input: None };
    let rmd1_used = model_config(ctx.new_topo, &precision, rmd1_area)
        .map_err(|_| RnnFlowError::Model)?;

    model_format(ctx.model_name, ctx.new_topo, &rmd1_area[..rmd1_used], None, &precision, metadata_area, rnn_out)
}
