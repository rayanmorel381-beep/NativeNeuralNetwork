use crate::base::activations::ActivationKind;
use crate::engine::rnn_flow::MAX_LAYERS;
use crate::graph::net::layers::{chain_widths, layer_chain_is_compatible, layout_spec, LayerDesc, LayerPlan, LayerPlanF64, LayerSpec};
use crate::api::rnn_api::core_api::RnnApiError;
use crate::engine::train::trainer::ParsedLayout;

pub fn single_forward_f32(
    tmp: &[u8],
    layout: &ParsedLayout,
    output_size: usize,
    input: &[f32],
    out: Option<&mut [f32]>,
) -> Result<(f64, usize), RnnApiError> {
    let topo = &layout.topology[..layout.topo_len];
    let w_bytes = &tmp[layout.w_start..layout.w_end];
    let b_bytes = &tmp[layout.b_start..layout.b_end];
    if !w_bytes.len().is_multiple_of(4) || !b_bytes.len().is_multiple_of(4) {
        return Err(RnnApiError::BadBytes);
    }
    let weights = unsafe { core::slice::from_raw_parts(w_bytes.as_ptr() as *const f32, w_bytes.len() / 4) };
    let biases = unsafe { core::slice::from_raw_parts(b_bytes.as_ptr() as *const f32, b_bytes.len() / 4) };

    let mut specs = [LayerSpec::Dense(LayerDesc {
        input_size: 1, output_size: 1, weight_offset: 0, bias_offset: 0,
        activation: ActivationKind::Identity,
    }); MAX_LAYERS];
    let network = crate::graph::net::network::NeuralNetwork::from_parts(topo, weights, biases)
        .ok_or(RnnApiError::BadBytes)?;
    let lc = network
        .build_layer_specs(layout.hidden_activation, layout.output_activation, &mut specs)
        .ok_or(RnnApiError::BadBytes)?;

    let mut widths = [0usize; MAX_LAYERS + 1];
    let width_len = chain_widths(&specs[..lc], &mut widths).map_err(|_| RnnApiError::BadBytes)?;
    if width_len != lc + 1 || !layer_chain_is_compatible(&specs[..lc]) {
        return Err(RnnApiError::BadBytes);
    }

    let plan = LayerPlan { layers: &specs[..lc], weights, biases };
    if plan.output_size() != Some(output_size) {
        return Err(RnnApiError::BadBytes);
    }
    crate::engine::validate_forward_plan(&plan).map_err(|_| RnnApiError::BadBytes)?;
    let scratch_len = crate::engine::required_single_infer_scratch(&plan)
        .ok_or(RnnApiError::BadBytes)?;

    let input_size = plan.input_size().ok_or(RnnApiError::BadBytes)?;
    let mut conv_ptr: *mut u8 = core::ptr::null_mut();
    let mut conv_bytes = 0usize;
    let dense_input: &[f32] = match crate::engine::rnn_flow::blob_range(
        tmp,
        crate::format::model_format::BLOB_CONV_SPEC,
    ) {
        Ok((s, e)) => {
            let spec = &tmp[s..e];
            let want_in = crate::graph::conv::conv_net::conv_net_input_len(spec).map_err(|_| RnnApiError::BadBytes)?;
            let want_out = crate::graph::conv::conv_net::conv_net_output_len(spec).map_err(|_| RnnApiError::BadBytes)?;
            if input.len() != want_in || want_out != input_size {
                return Err(RnnApiError::BadBytes);
            }
            conv_bytes = input_size * 4;
            conv_ptr = crate::engine::runtime::hardware::mmap_shared_anon(conv_bytes);
            if conv_ptr.is_null() {
                return Err(RnnApiError::CapacityTooSmall);
            }
            let conv_out = unsafe { core::slice::from_raw_parts_mut(conv_ptr as *mut f32, input_size) };
            match crate::graph::conv::conv_net::conv_net_forward_f32(spec, input, conv_out) {
                Ok(n) if n == input_size => &conv_out[..],
                _ => {
                    crate::engine::runtime::hardware::munmap(conv_ptr, conv_bytes);
                    return Err(RnnApiError::Model);
                }
            }
        }
        Err(_) => input,
    };

    let buf_bytes = (output_size + scratch_len) * 4;
    let buf_ptr = crate::engine::runtime::hardware::mmap_shared_anon(buf_bytes);
    if buf_ptr.is_null() {
        if !conv_ptr.is_null() {
            crate::engine::runtime::hardware::munmap(conv_ptr, conv_bytes);
        }
        return Err(RnnApiError::CapacityTooSmall);
    }

    let out_slice = unsafe { core::slice::from_raw_parts_mut(buf_ptr as *mut f32, output_size) };
    let scratch = unsafe { core::slice::from_raw_parts_mut(buf_ptr.add(output_size * 4) as *mut f32, scratch_len) };

    let res: Result<(), ()> = match plan.validate() {
        Ok(()) => {
            let half = scratch.len() / 2;
            let (hidden, norm_buf) = scratch.split_at_mut(half);
            crate::engine::infer::inference::mlp_graph_forward(
                &specs[..lc],
                weights,
                biases,
                dense_input,
                hidden,
                norm_buf,
                out_slice,
            )
            .map_err(|_| ())
        }
        Err(_) => Err(()),
    };
    let val = if res.is_ok() { out_slice[0] as f64 } else { 0.0 };
    if res.is_ok() {
        if let Some(o) = out {
            let n = output_size.min(o.len());
            o[..n].copy_from_slice(&out_slice[..n]);
        }
    }

    crate::engine::runtime::hardware::munmap(buf_ptr, buf_bytes);
    if !conv_ptr.is_null() {
        crate::engine::runtime::hardware::munmap(conv_ptr, conv_bytes);
    }
    res.map_err(|_| RnnApiError::Model)?;
    Ok((val, output_size))
}

pub fn single_forward_f64(
    tmp: &[u8],
    layout: &ParsedLayout,
    input_size: usize,
    output_size: usize,
    input_f32: &[f32],
    input_f64: Option<&[f64]>,
    out: Option<&mut [f64]>,
) -> Result<(f64, usize), RnnApiError> {
    let topo = &layout.topology[..layout.topo_len];
    let w_bytes = &tmp[layout.w_start..layout.w_end];
    let b_bytes = &tmp[layout.b_start..layout.b_end];
    if !w_bytes.len().is_multiple_of(8) || !b_bytes.len().is_multiple_of(8) {
        return Err(RnnApiError::BadBytes);
    }
    let weights = unsafe { core::slice::from_raw_parts(w_bytes.as_ptr() as *const f64, w_bytes.len() / 8) };
    let biases = unsafe { core::slice::from_raw_parts(b_bytes.as_ptr() as *const f64, b_bytes.len() / 8) };

    let mut specs = [LayerSpec::Dense(LayerDesc {
        input_size: 1, output_size: 1, weight_offset: 0, bias_offset: 0,
        activation: ActivationKind::Identity,
    }); MAX_LAYERS];
    let lc = layout_spec(topo, layout.hidden_activation, layout.output_activation, &mut specs)
        .ok_or(RnnApiError::BadBytes)?;

    let mut widths = [0usize; MAX_LAYERS + 1];
    let width_len = chain_widths(&specs[..lc], &mut widths).map_err(|_| RnnApiError::BadBytes)?;
    if width_len != lc + 1 || !layer_chain_is_compatible(&specs[..lc]) {
        return Err(RnnApiError::BadBytes);
    }

    let plan = LayerPlanF64 { layers: &specs[..lc], weights, biases };
    if plan.input_size() != Some(input_size) || plan.output_size() != Some(output_size) {
        return Err(RnnApiError::BadBytes);
    }
    let max_w = plan.max_width().ok_or(RnnApiError::BadBytes)?;
    let scratch_len = max_w * 2;
    let conv_bytes = match input_f64 {
        Some(_) => 0,
        None => input_size * 8,
    };
    let buf_bytes = (output_size + scratch_len) * 8 + conv_bytes;
    let buf_ptr = crate::engine::runtime::hardware::mmap_shared_anon(buf_bytes);
    if buf_ptr.is_null() { return Err(RnnApiError::CapacityTooSmall); }

    let input_ref: &[f64] = match input_f64 {
        Some(fi) => fi,
        None => {
            let conv = unsafe { core::slice::from_raw_parts_mut(buf_ptr as *mut f64, input_size) };
            for i in 0..input_size { conv[i] = input_f32[i] as f64; }
            unsafe { core::slice::from_raw_parts(buf_ptr as *const f64, input_size) }
        }
    };
    let out_slice = unsafe { core::slice::from_raw_parts_mut(buf_ptr.add(conv_bytes) as *mut f64, output_size) };
    let scratch = unsafe { core::slice::from_raw_parts_mut(buf_ptr.add(conv_bytes + output_size * 8) as *mut f64, scratch_len) };

    let res: Result<(), ()> = match plan.validate() {
        Ok(()) => {
            let half = scratch.len() / 2;
            let (hidden, norm_buf) = scratch.split_at_mut(half);
            crate::engine::infer::inference::mlp_graph_forward(
                &specs[..lc],
                weights,
                biases,
                input_ref,
                hidden,
                norm_buf,
                out_slice,
            )
            .map_err(|_| ())
        }
        Err(_) => Err(()),
    };
    let val = if res.is_ok() { out_slice[0] } else { 0.0 };
    if res.is_ok() {
        if let Some(o) = out {
            let n = output_size.min(o.len());
            o[..n].copy_from_slice(&out_slice[..n]);
        }
    }

    crate::engine::runtime::hardware::munmap(buf_ptr, buf_bytes);
    res.map_err(|_| RnnApiError::Model)?;
    Ok((val, output_size))
}
