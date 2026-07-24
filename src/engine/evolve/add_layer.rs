use crate::engine::rnn_flow::{blob_range, rnn_dtype, RnnFlowError, MAX_LAYERS};
use crate::format::model_config::{model_config, Precision};
use crate::format::model_format::{model_format, BLOB_BIASES, BLOB_WEIGHTS};
use super::helpers::*;

pub fn evolve_add_layer(
    bytes: &[u8],
    position: usize,
    size: usize,
    seed: u64,
    out: &mut [u8],
) -> Result<usize, RnnFlowError> {
    let dtype = rnn_dtype(bytes)?;
    let (topology, topo_len) = parse_topology(bytes)?;
    let topo = &topology[..topo_len];

    if position == 0 || position >= topo_len || size == 0 || topo_len + 1 > MAX_LAYERS {
        return Err(RnnFlowError::InvalidTopology);
    }

    let model_name = extract_model_name(bytes);

    let (w_start, w_end) = blob_range(bytes, BLOB_WEIGHTS)?;
    let (b_start, b_end) = blob_range(bytes, BLOB_BIASES)?;
    let old_w_bytes = &bytes[w_start..w_end];
    let old_b_bytes = &bytes[b_start..b_end];

    let mut new_topo = [0usize; MAX_LAYERS];
    let mut new_len = 0usize;
    for (i, &val) in topo.iter().enumerate() {
        if i == position {
            new_topo[new_len] = size;
            new_len += 1;
        }
        new_topo[new_len] = val;
        new_len += 1;
    }
    let new_topo_slice = &new_topo[..new_len];

    let ctx = EvolveCtx {
        model_name, old_topo: topo, new_topo: new_topo_slice,
        old_w_bytes, old_b_bytes, seed,
    };
    match dtype {
        0 => evolve_add_layer_f32_inner(&ctx, position, size, out),
        1 => evolve_add_layer_f64_inner(&ctx, position, size, out),
        _ => Err(RnnFlowError::BadBytes),
    }
}

fn evolve_add_layer_f32_inner(
    ctx: &EvolveCtx,
    position: usize,
    size: usize,
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

    build_weights_with_new_layer_f32(ctx.old_topo, position, size, ctx.old_w_bytes, w_area, ctx.seed);
    build_biases_with_new_layer_f32(ctx.old_topo, position, size, ctx.old_b_bytes, b_area, ctx.seed);

    let w_f32 = bytes_as_f32(w_area);
    let b_f32 = bytes_as_f32(b_area);

    let precision = Precision::F32 { weights: w_f32, biases: b_f32, runtime_input: None, conv_spec: None };
    let rmd1_used = model_config(ctx.new_topo, &precision, rmd1_area)
        .map_err(|_| RnnFlowError::Model)?;

    model_format(ctx.model_name, ctx.new_topo, &rmd1_area[..rmd1_used], None, &precision, metadata_area, rnn_out)
}

fn evolve_add_layer_f64_inner(
    ctx: &EvolveCtx,
    position: usize,
    size: usize,
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

    build_weights_with_new_layer_f64(ctx.old_topo, position, size, ctx.old_w_bytes, w_area, ctx.seed);
    build_biases_with_new_layer_f64(ctx.old_topo, position, size, ctx.old_b_bytes, b_area, ctx.seed);

    let w_f64 = bytes_as_f64(w_area);
    let b_f64 = bytes_as_f64(b_area);

    let precision = Precision::F64 { weights: w_f64, biases: b_f64, runtime_input: None };
    let rmd1_used = model_config(ctx.new_topo, &precision, rmd1_area)
        .map_err(|_| RnnFlowError::Model)?;

    model_format(ctx.model_name, ctx.new_topo, &rmd1_area[..rmd1_used], None, &precision, metadata_area, rnn_out)
}

fn build_weights_with_new_layer_f32(
    old_topo: &[usize],
    position: usize,
    size: usize,
    old_w_bytes: &[u8],
    new_w_bytes: &mut [u8],
    seed: u64,
) {
    let mut rng = SplitMix64::new(seed);
    let old_layer_count = old_topo.len() - 1;
    let mut old_off = 0usize;
    let mut new_off = 0usize;

    let split_layer = position - 1;

    for l in 0..old_layer_count {
        let in_s = old_topo[l];
        let out_s = old_topo[l + 1];

        if l == split_layer {
            let limit_a = xavier_limit_f32(in_s, size);
            for i in 0..(in_s * size) {
                let v = (rng.next_f32() * 2.0 - 1.0) * limit_a;
                write_f32(new_w_bytes, (new_off + i) * 4, v);
            }
            new_off += in_s * size;

            let limit_b = xavier_limit_f32(size, out_s);
            for i in 0..(size * out_s) {
                let v = (rng.next_f32() * 2.0 - 1.0) * limit_b;
                write_f32(new_w_bytes, (new_off + i) * 4, v);
            }
            new_off += size * out_s;
            old_off += in_s * out_s;
        } else {
            let block = in_s * out_s;
            new_w_bytes[new_off * 4..(new_off + block) * 4]
                .copy_from_slice(&old_w_bytes[old_off * 4..(old_off + block) * 4]);
            new_off += block;
            old_off += block;
        }
    }
}

fn build_biases_with_new_layer_f32(
    old_topo: &[usize],
    position: usize,
    size: usize,
    old_b_bytes: &[u8],
    new_b_bytes: &mut [u8],
    seed: u64,
) {
    let mut rng = SplitMix64::new(seed.wrapping_add(0xCAFEBABE));
    let old_layer_count = old_topo.len() - 1;
    let mut old_off = 0usize;
    let mut new_off = 0usize;

    for l in 0..old_layer_count {
        let out_s = old_topo[l + 1];

        if l + 1 == position {
            for i in 0..size {
                let v = rng.next_f32() * 0.01;
                write_f32(new_b_bytes, (new_off + i) * 4, v);
            }
            new_off += size;
        }

        new_b_bytes[new_off * 4..(new_off + out_s) * 4]
            .copy_from_slice(&old_b_bytes[old_off * 4..(old_off + out_s) * 4]);
        new_off += out_s;
        old_off += out_s;
    }
}

fn build_weights_with_new_layer_f64(
    old_topo: &[usize],
    position: usize,
    size: usize,
    old_w_bytes: &[u8],
    new_w_bytes: &mut [u8],
    seed: u64,
) {
    let mut rng = SplitMix64::new(seed);
    let old_layer_count = old_topo.len() - 1;
    let mut old_off = 0usize;
    let mut new_off = 0usize;

    for l in 0..old_layer_count {
        let in_s = old_topo[l];
        let out_s = old_topo[l + 1];

        if l == position - 1 {
            let limit_a = xavier_limit_f64(in_s, size);
            for i in 0..(in_s * size) {
                let v = (rng.next_f64() * 2.0 - 1.0) * limit_a;
                write_f64(new_w_bytes, (new_off + i) * 8, v);
            }
            new_off += in_s * size;

            let limit_b = xavier_limit_f64(size, out_s);
            for i in 0..(size * out_s) {
                let v = (rng.next_f64() * 2.0 - 1.0) * limit_b;
                write_f64(new_w_bytes, (new_off + i) * 8, v);
            }
            new_off += size * out_s;
            old_off += in_s * out_s;
        } else {
            let block = in_s * out_s;
            new_w_bytes[new_off * 8..(new_off + block) * 8]
                .copy_from_slice(&old_w_bytes[old_off * 8..(old_off + block) * 8]);
            new_off += block;
            old_off += block;
        }
    }
}

fn build_biases_with_new_layer_f64(
    old_topo: &[usize],
    position: usize,
    size: usize,
    old_b_bytes: &[u8],
    new_b_bytes: &mut [u8],
    seed: u64,
) {
    let mut rng = SplitMix64::new(seed.wrapping_add(0xCAFEBABE));
    let old_layer_count = old_topo.len() - 1;
    let mut old_off = 0usize;
    let mut new_off = 0usize;

    for l in 0..old_layer_count {
        let out_s = old_topo[l + 1];

        if l + 1 == position {
            for i in 0..size {
                let v = rng.next_f64() * 0.01;
                write_f64(new_b_bytes, (new_off + i) * 8, v);
            }
            new_off += size;
        }

        new_b_bytes[new_off * 8..(new_off + out_s) * 8]
            .copy_from_slice(&old_b_bytes[old_off * 8..(old_off + out_s) * 8]);
        new_off += out_s;
        old_off += out_s;
    }
}
