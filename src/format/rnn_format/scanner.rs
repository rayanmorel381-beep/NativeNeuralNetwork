use super::bounds::{all_blob_bounds_valid, all_blob_digests_valid};
use super::lookup::find_blob_index;
use super::parser::{parse_rnn_from_bytes, RnnHandle};
use crate::security::crypto::{
    constant_time_eq, extract_rnn_crypto_timestamp, extract_rnn_issue_counter,
    extract_rnn_key_version, extract_rnn_public_has_benchmark, extract_rnn_public_has_model_name,
    extract_rnn_public_has_model_precision, extract_rnn_public_header_summary, is_encrypted_rnn,
    RNN_MIN_ACCEPTED_KEY_VERSION,
};
use crate::format::model_format::{
    header_tlv_payload, BLOB_BIASES, BLOB_LAYER_META, BLOB_WEIGHTS, LAYER_META_SIZE, RNN0_MAGIC,
    RNN0_VERSION, TLV_HEADER_BENCHMARK, TLV_HEADER_NETWORK_SUMMARY,
};
use crate::base::scratch::Scratch;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Truncated,
    BadMagic,
    BadVersion,
    BadHeader,
    BadBounds,
    CapacityTooSmall,
    InvalidPayload,
    InvalidContainer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeaderSummary {
    pub dtype: u8,
    pub layer_count: u32,
    pub total_neurons: u32,
    pub weights_len: u32,
    pub biases_len: u32,
    pub blob_count: u32,
    pub flags: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanReport {
    pub encrypted: bool,
    pub header: Option<HeaderSummary>,
    pub has_benchmark: Option<bool>,
    pub has_model_name: Option<bool>,
    pub has_model_precision: Option<bool>,
    pub key_version: Option<u32>,
    pub issue_counter: Option<u64>,
    pub crypto_timestamp: Option<u64>,
}

const MODEL_NAME_BLOB_DATA: &str = "model.name";
const MODEL_PRECISION_BLOB_DATA: &str = "model.precision";

fn encoded_payload_size(
    dtype: u8,
    layer_count: usize,
    weights_len: usize,
    biases_len: usize,
) -> Option<usize> {
    let elem_size = match dtype {
        0 => 4usize,
        1 => 8usize,
        _ => return None,
    };
    let layers_bytes = layer_count.checked_mul(LAYER_META_SIZE)?;
    let weights_bytes = weights_len.checked_mul(elem_size)?;
    let biases_bytes = biases_len.checked_mul(elem_size)?;
    20usize
        .checked_add(layers_bytes)?
        .checked_add(weights_bytes)?
        .checked_add(biases_bytes)
}

pub(crate) fn extract_header_summary(bytes: &[u8]) -> Result<HeaderSummary, Error> {
    let payload =
        header_tlv_payload(bytes, TLV_HEADER_NETWORK_SUMMARY).map_err(|_| Error::InvalidPayload)?;
    if payload.len() >= 16 && constant_time_eq(&payload[0..4], b"VIZ\x00") {
        return decode_compact_viz_summary(payload);
    }
    decode_header_summary(payload)
}

pub fn validate(bytes: &[u8], device_id: Option<&[u8]>) -> Result<ScanReport, Error> {
    validate_scan_backend(bytes, device_id)
}

pub(crate) fn validate_scan_backend(
    bytes: &[u8],
    device_id: Option<&[u8]>,
) -> Result<ScanReport, Error> {
    validate_inner(bytes, device_id)
}

fn validate_inner(bytes: &[u8], device_id: Option<&[u8]>) -> Result<ScanReport, Error> {
    if bytes.len() < 12 {
        return Err(Error::Truncated);
    }
    if !constant_time_eq(&bytes[0..4], RNN0_MAGIC) {
        return Err(Error::BadMagic);
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != RNN0_VERSION {
        return Err(Error::BadVersion);
    }

    let mut scratch_buf = [0u8; 16384];
    let mut scratch = Scratch::new(&mut scratch_buf);
    let handle = match parse_rnn_from_bytes(bytes, &mut scratch) {
        Ok(handle) => handle,
        Err(_) => {
            if is_encrypted_rnn(bytes) {
                return validate_encrypted_container(bytes, device_id);
            }
            return Err(Error::InvalidContainer);
        }
    };

    if !all_blob_bounds_valid(&handle) {
        return Err(Error::BadBounds);
    }
    if !all_blob_digests_valid(&handle) {
        return Err(Error::BadBounds);
    }

    if find_blob_index(&handle, crate::format::model_format::LMLP_CONFIG_BLOB_DATA).is_some() {
        return validate_lm_inner(bytes, &handle);
    }

    let layer_meta = find_layer_meta_blob(&handle).ok_or(Error::InvalidPayload)?;
    let weights = find_weights_blob(&handle).ok_or(Error::InvalidPayload)?;
    let biases = find_biases_blob(&handle).ok_or(Error::InvalidPayload)?;

    let layer_meta_idx = find_blob_index(&handle, BLOB_LAYER_META).ok_or(Error::InvalidPayload)?;
    let weights_idx = find_blob_index(&handle, BLOB_WEIGHTS).ok_or(Error::InvalidPayload)?;
    let biases_idx = find_blob_index(&handle, BLOB_BIASES).ok_or(Error::InvalidPayload)?;

    let layer_meta_desc = handle
        .blobs
        .get(layer_meta_idx)
        .ok_or(Error::InvalidPayload)?;
    let weights_desc = handle.blobs.get(weights_idx).ok_or(Error::InvalidPayload)?;
    let biases_desc = handle.blobs.get(biases_idx).ok_or(Error::InvalidPayload)?;

    if layer_meta.is_empty() || weights.is_empty() || biases.is_empty() {
        return Err(Error::InvalidPayload);
    }
    if !layer_meta.len().is_multiple_of(LAYER_META_SIZE) {
        return Err(Error::InvalidPayload);
    }
    let benchmark = crate::format::model_format::header_tlv_payload(bytes, TLV_HEADER_BENCHMARK)
        .map_err(|_| Error::InvalidPayload)?;
    if benchmark.is_empty() {
        return Err(Error::InvalidPayload);
    }

    if layer_meta_desc.dtype != weights_desc.dtype || weights_desc.dtype != biases_desc.dtype {
        return Err(Error::InvalidPayload);
    }

    let layer_count = layer_meta.len() / LAYER_META_SIZE;
    match weights_desc.dtype {
        0 => {
            if !weights.len().is_multiple_of(4) || !biases.len().is_multiple_of(4) {
                return Err(Error::InvalidPayload);
            }
            let weights_len = weights.len() / 4;
            let biases_len = biases.len() / 4;
            encoded_payload_size(weights_desc.dtype, layer_count, weights_len, biases_len)
                .ok_or(Error::InvalidPayload)?;
            validate_constellation_f32(layer_meta, weights, biases)?;
        }
        1 => {
            if !weights.len().is_multiple_of(8) || !biases.len().is_multiple_of(8) {
                return Err(Error::InvalidPayload);
            }
            let weights_len = weights.len() / 8;
            let biases_len = biases.len() / 8;
            encoded_payload_size(weights_desc.dtype, layer_count, weights_len, biases_len)
                .ok_or(Error::InvalidPayload)?;
        }
        _ => return Err(Error::InvalidPayload),
    }

    if let Some(conv_idx) = find_blob_index(&handle, crate::format::model_format::BLOB_CONV_SPEC) {
        if weights_desc.dtype != 0 {
            return Err(Error::InvalidPayload);
        }
        let conv_desc = handle.blobs.get(conv_idx).ok_or(Error::InvalidPayload)?;
        let start = usize::try_from(conv_desc.offset).map_err(|_| Error::InvalidPayload)?;
        let len = usize::try_from(conv_desc.length).map_err(|_| Error::InvalidPayload)?;
        let end = start.checked_add(len).ok_or(Error::InvalidPayload)?;
        if end > bytes.len() {
            return Err(Error::BadBounds);
        }
        let spec = &bytes[start..end];
        crate::graph::conv::conv_net::conv_net_validate(spec).map_err(|_| Error::InvalidPayload)?;
        let dense_input =
            u32::from_le_bytes([layer_meta[0], layer_meta[1], layer_meta[2], layer_meta[3]]) as usize;
        let conv_in = crate::graph::conv::conv_net::conv_net_input_len(spec).map_err(|_| Error::InvalidPayload)?;
        let conv_out = crate::graph::conv::conv_net::conv_net_output_len(spec).map_err(|_| Error::InvalidPayload)?;
        if conv_in == 0 || conv_out != dense_input {
            return Err(Error::InvalidPayload);
        }
    }

    let has_benchmark =
        crate::format::model_format::header_tlv_payload(bytes, TLV_HEADER_BENCHMARK).is_ok();
    let has_model_name = find_blob_index(&handle, MODEL_NAME_BLOB_DATA).is_some();
    let has_model_precision = find_blob_index(&handle, MODEL_PRECISION_BLOB_DATA).is_some();
    let header = extract_header_summary(bytes).ok();

    Ok(ScanReport {
        encrypted: false,
        header,
        has_benchmark: Some(has_benchmark),
        has_model_name: Some(has_model_name),
        has_model_precision: Some(has_model_precision),
        key_version: None,
        issue_counter: None,
        crypto_timestamp: None,
    })
}

fn validate_constellation_f32(
    meta: &[u8],
    weights_bytes: &[u8],
    biases_bytes: &[u8],
) -> Result<(), Error> {
    if !meta.len().is_multiple_of(LAYER_META_SIZE) || meta.len() < LAYER_META_SIZE {
        return Err(Error::InvalidPayload);
    }
    if !weights_bytes.len().is_multiple_of(4) || !biases_bytes.len().is_multiple_of(4) {
        return Err(Error::InvalidPayload);
    }
    let layer_count = meta.len() / LAYER_META_SIZE;
    let mut topology = [0usize; 128];
    let mut topo_len = 0usize;
    for i in 0..layer_count {
        let off = i * LAYER_META_SIZE;
        let inp = u32::from_le_bytes([meta[off], meta[off + 1], meta[off + 2], meta[off + 3]])
            as usize;
        let out = u32::from_le_bytes([meta[off + 4], meta[off + 5], meta[off + 6], meta[off + 7]])
            as usize;
        if i == 0 {
            if topo_len >= topology.len() {
                return Err(Error::InvalidPayload);
            }
            topology[topo_len] = inp;
            topo_len += 1;
        }
        if topo_len >= topology.len() {
            return Err(Error::InvalidPayload);
        }
        topology[topo_len] = out;
        topo_len += 1;
    }
    if topo_len < 2 {
        return Err(Error::InvalidPayload);
    }

    let meta_end = 20 + meta.len();
    let w_end = meta_end + weights_bytes.len();
    let rmd1_len = w_end + biases_bytes.len();
    let rmd1_ptr = crate::engine::runtime::hardware::mmap_shared_anon(rmd1_len);
    if rmd1_ptr.is_null() {
        return Err(Error::CapacityTooSmall);
    }
    let rmd1 = unsafe { core::slice::from_raw_parts_mut(rmd1_ptr, rmd1_len) };
    rmd1[0..4].copy_from_slice(b"RMD1");
    rmd1[4..6].copy_from_slice(&1u16.to_le_bytes());
    rmd1[6] = 0;
    rmd1[7] = 0;
    rmd1[8..12].copy_from_slice(&(layer_count as u32).to_le_bytes());
    rmd1[12..16].copy_from_slice(&((weights_bytes.len() / 4) as u32).to_le_bytes());
    rmd1[16..20].copy_from_slice(&((biases_bytes.len() / 4) as u32).to_le_bytes());
    rmd1[20..meta_end].copy_from_slice(meta);
    rmd1[meta_end..w_end].copy_from_slice(weights_bytes);
    rmd1[w_end..rmd1_len].copy_from_slice(biases_bytes);

    let result = constellation_inner(rmd1, &topology[..topo_len], meta_end, w_end);
    crate::engine::runtime::hardware::munmap(rmd1_ptr, rmd1_len);
    result
}

struct ConstellationOffsets {
    pts_off: usize,
    nb_off: usize,
    vtx_off: usize,
    idx_off: usize,
    biases_off: usize,
    total_neurons: usize,
    vertex_count: usize,
    index_count: usize,
}

fn constellation_inner(
    rmd1: &[u8],
    topology: &[usize],
    weights_off: usize,
    biases_off: usize,
) -> Result<(), Error> {
    let weights_len = u32::from_le_bytes([rmd1[12], rmd1[13], rmd1[14], rmd1[15]]) as usize;
    let biases_len = u32::from_le_bytes([rmd1[16], rmd1[17], rmd1[18], rmd1[19]]) as usize;
    let weights = unsafe {
        core::slice::from_raw_parts(rmd1.as_ptr().add(weights_off) as *const f32, weights_len)
    };
    let biases = unsafe {
        core::slice::from_raw_parts(rmd1.as_ptr().add(biases_off) as *const f32, biases_len)
    };
    let expected_weights = crate::graph::net::network::NeuralNetwork::expected_weights_count(topology)
        .ok_or(Error::InvalidPayload)?;
    if expected_weights != weights_len {
        return Err(Error::InvalidPayload);
    }
    let network = crate::graph::net::network::NeuralNetwork::from_parts(topology, weights, biases)
        .ok_or(Error::InvalidPayload)?;

    let stats = crate::graph::net::network::network_stats(&network).ok_or(Error::InvalidPayload)?;
    if stats.total_weights != weights_len || stats.total_biases != biases_len {
        return Err(Error::InvalidPayload);
    }

    let view = crate::observability::visualization::get_network_view(rmd1).map_err(|_| Error::InvalidPayload)?;
    if view.layer_count() != network.layer_count() {
        return Err(Error::InvalidPayload);
    }
    let first_layer = view.layer(0).map_err(|_| Error::InvalidPayload)?;
    let last_layer = view
        .layer(network.layer_count().saturating_sub(1))
        .map_err(|_| Error::InvalidPayload)?;
    if first_layer.input_count().map_err(|_| Error::InvalidPayload)? != stats.input_size
        || last_layer.neuron_count().map_err(|_| Error::InvalidPayload)? != stats.output_size
    {
        return Err(Error::InvalidPayload);
    }
    let (vertex_count, index_count) =
        crate::observability::visualization::mesh_required_buffers_from_bytes(rmd1)
            .map_err(|_| Error::InvalidPayload)?;

    let total_neurons = topology.iter().fold(0usize, |a, &x| a.saturating_add(x));
    if total_neurons == 0 || vertex_count == 0 {
        return Err(Error::InvalidPayload);
    }

    let point_size = core::mem::size_of::<crate::graph::conv::sphere5d::NeuronPoint>();
    let pts_bytes = total_neurons.saturating_mul(point_size);
    let nb_bytes = total_neurons.saturating_mul(core::mem::size_of::<usize>());
    let vtx_bytes = vertex_count.saturating_mul(8 * 4);
    let idx_bytes = index_count.saturating_mul(4);
    let pts_off = 0usize;
    let nb_off = pts_off + pts_bytes;
    let vtx_off = nb_off + nb_bytes;
    let idx_off = vtx_off + vtx_bytes;
    let total = idx_off + idx_bytes;
    let scratch_ptr = crate::engine::runtime::hardware::mmap_shared_anon(total);
    if scratch_ptr.is_null() {
        return Err(Error::CapacityTooSmall);
    }
    let offsets = ConstellationOffsets {
        pts_off,
        nb_off,
        vtx_off,
        idx_off,
        biases_off,
        total_neurons,
        vertex_count,
        index_count,
    };
    let res = constellation_validate(&network, rmd1, scratch_ptr, offsets);
    crate::engine::runtime::hardware::munmap(scratch_ptr, total);
    res
}

fn constellation_validate(
    network: &crate::graph::net::network::NeuralNetwork<'_>,
    rmd1: &[u8],
    scratch_ptr: *mut u8,
    off: ConstellationOffsets,
) -> Result<(), Error> {
    let points = unsafe {
        core::slice::from_raw_parts_mut(
            scratch_ptr.add(off.pts_off) as *mut crate::graph::conv::sphere5d::NeuronPoint,
            off.total_neurons,
        )
    };
    let mut sphere = network
        .conceptualize_5d(points, 1.0)
        .map_err(|_| Error::InvalidPayload)?;
    if sphere.is_empty()
        || sphere.len() != off.total_neurons
        || sphere.capacity() < off.total_neurons
    {
        return Err(Error::InvalidPayload);
    }
    let view = crate::observability::visualization::get_network_view(rmd1).map_err(|_| Error::InvalidPayload)?;
    if view.header_size() >= rmd1.len() {
        return Err(Error::InvalidPayload);
    }
    if view.biases_offset() != off.biases_off {
        return Err(Error::InvalidPayload);
    }
    {
        let pts = sphere.as_mut_slice();
        for p in pts.iter_mut() {
            if p.layer == 0 {
                continue;
            }
            let lv = view.layer(p.layer - 1).map_err(|_| Error::InvalidPayload)?;
            let act = lv.activation().map_err(|_| Error::InvalidPayload)?;
            p.activation = act as f32;
        }
    }
    let slice = sphere.as_slice();
    for (i, item) in slice.iter().enumerate() {
        let (idx, dist) = sphere
            .nearest(item.position)
            .ok_or(Error::InvalidPayload)?;
        if idx != i || dist != 0.0 {
            return Err(Error::InvalidPayload);
        }
        if item.layer != 0 && !item.activation.is_finite() {
            return Err(Error::InvalidPayload);
        }
    }
    if let Some(first) = slice.first() {
        let neighbors = unsafe {
            core::slice::from_raw_parts_mut(
                scratch_ptr.add(off.nb_off) as *mut usize,
                off.total_neurons,
            )
        };
        let found = sphere.neighbors_within(first.position, sphere.radius() * 2.0, neighbors);
        if found == 0 {
            return Err(Error::InvalidPayload);
        }
    }

    for li in 0..view.layer_count() {
        let lv = view.layer(li).map_err(|_| Error::InvalidPayload)?;
        if lv.layer_meta_size() == 0 || lv.layer_meta_size() > rmd1.len() {
            return Err(Error::InvalidPayload);
        }
        let nc = lv.neuron_count().map_err(|_| Error::InvalidPayload)?;
        if nc == 0 {
            return Err(Error::InvalidPayload);
        }
        let nv = lv.neuron(0).map_err(|_| Error::InvalidPayload)?;
        if nv.weight_count() == 0 || nv.weight_bytes().len() < 4 {
            return Err(Error::InvalidPayload);
        }
        let w = nv.weight_at(0).map_err(|_| Error::InvalidPayload)?;
        if !w.is_finite() {
            return Err(Error::InvalidPayload);
        }
    }

    let vertex_buf = unsafe {
        core::slice::from_raw_parts_mut(
            scratch_ptr.add(off.vtx_off) as *mut f32,
            off.vertex_count * 8,
        )
    };
    let index_buf = unsafe {
        core::slice::from_raw_parts_mut(scratch_ptr.add(off.idx_off) as *mut u32, off.index_count)
    };
    let (vc, written) = crate::observability::visualization::fill_mesh_from_bytes(rmd1, vertex_buf, index_buf)
        .map_err(|_| Error::InvalidPayload)?;
    if vc != off.vertex_count || written != off.index_count {
        return Err(Error::InvalidPayload);
    }
    Ok(())
}

fn validate_lm_inner(
    bytes: &[u8],
    handle: &RnnHandle<'_, '_>,
) -> Result<ScanReport, Error> {
    let config_blob = crate::engine::rnn_flow::blob_payload_by_name(
        bytes,
        handle,
        crate::format::model_format::LMLP_CONFIG_BLOB_DATA,
    )
    .ok_or(Error::InvalidPayload)?;
    let cfg = crate::graph::lm::lm_config_decode::<f32>(config_blob).map_err(|_| Error::InvalidPayload)?;

    let weights_idx = find_blob_index(handle, crate::format::model_format::LMLP_WEIGHTS_BLOB_DATA)
        .ok_or(Error::InvalidPayload)?;
    let weights_desc = handle.blobs.get(weights_idx).ok_or(Error::InvalidPayload)?;
    let bytes_per_param: usize = if weights_desc.dtype == 1 { 8 } else { 4 };
    let weights = crate::engine::rnn_flow::blob_payload_by_name(bytes, handle, crate::format::model_format::LMLP_WEIGHTS_BLOB_DATA)
        .ok_or(Error::InvalidPayload)?;
    let param_count = crate::graph::lm::lm_weights_param_count(&cfg);
    let expected = param_count.checked_mul(bytes_per_param).ok_or(Error::InvalidPayload)?;
    if weights.is_empty() || weights.len() != expected {
        return Err(Error::InvalidPayload);
    }

    let header = extract_header_summary(bytes).ok();
    let has_model_name = find_blob_index(handle, MODEL_NAME_BLOB_DATA).is_some();
    let has_model_precision = find_blob_index(handle, MODEL_PRECISION_BLOB_DATA).is_some();

    Ok(ScanReport {
        encrypted: false,
        header,
        has_benchmark: Some(true),
        has_model_name: Some(has_model_name),
        has_model_precision: Some(has_model_precision),
        key_version: None,
        issue_counter: None,
        crypto_timestamp: None,
    })
}

fn validate_encrypted_container(
    bytes: &[u8],
    key_material: Option<&[u8]>,
) -> Result<ScanReport, Error> {
    if let Some(material) = key_material {
        if material.is_empty() {
            return Err(Error::InvalidContainer);
        }
    }
    if !is_encrypted_rnn(bytes) {
        return Err(Error::BadMagic);
    }
    let key_version = extract_rnn_key_version(bytes).ok_or(Error::BadHeader)?;
    let issue_counter = extract_rnn_issue_counter(bytes).ok_or(Error::BadHeader)?;
    let crypto_timestamp = extract_rnn_crypto_timestamp(bytes).ok_or(Error::BadHeader)?;
    if key_version < RNN_MIN_ACCEPTED_KEY_VERSION || issue_counter == 0 || crypto_timestamp == 0 {
        return Err(Error::InvalidContainer);
    }
    let header = extract_rnn_public_header_summary(bytes).map(
        |(dtype, layer_count, total_neurons, weights_len, biases_len, blob_count, flags)| {
            HeaderSummary {
                dtype,
                layer_count,
                total_neurons,
                weights_len,
                biases_len,
                blob_count,
                flags,
            }
        },
    );
    let has_benchmark = extract_rnn_public_has_benchmark(bytes);
    Ok(ScanReport {
        encrypted: true,
        header,
        has_benchmark,
        has_model_name: extract_rnn_public_has_model_name(bytes),
        has_model_precision: extract_rnn_public_has_model_precision(bytes),
        key_version: Some(key_version),
        issue_counter: Some(issue_counter),
        crypto_timestamp: Some(crypto_timestamp),
    })
}

pub(crate) fn blob_bytes<'bytes, 'scratch>(
    handle: &RnnHandle<'bytes, 'scratch>,
    index: usize,
) -> Option<&'bytes [u8]> {
    let meta = handle.blobs.get(index)?;
    let start = usize::try_from(meta.offset).ok()?;
    let len = usize::try_from(meta.length).ok()?;
    let end = start.checked_add(len)?;
    handle.bytes.get(start..end)
}

fn find_blob_bytes<'bytes, 'scratch>(
    handle: &RnnHandle<'bytes, 'scratch>,
    name: &str,
) -> Option<&'bytes [u8]> {
    let idx = find_blob_index(handle, name)?;
    blob_bytes(handle, idx)
}

fn find_layer_meta_blob<'bytes, 'scratch>(
    handle: &RnnHandle<'bytes, 'scratch>,
) -> Option<&'bytes [u8]> {
    find_blob_bytes(handle, BLOB_LAYER_META)
}

fn find_weights_blob<'bytes, 'scratch>(
    handle: &RnnHandle<'bytes, 'scratch>,
) -> Option<&'bytes [u8]> {
    find_blob_bytes(handle, BLOB_WEIGHTS)
}

fn find_biases_blob<'bytes, 'scratch>(
    handle: &RnnHandle<'bytes, 'scratch>,
) -> Option<&'bytes [u8]> {
    find_blob_bytes(handle, BLOB_BIASES)
}

fn decode_header_summary(payload: &[u8]) -> Result<HeaderSummary, Error> {
    if payload.len() < 2 {
        return Err(Error::BadHeader);
    }
    let summary_version = payload[0];
    let codec = payload[1];
    if summary_version != 1 || codec != 1 {
        return Err(Error::BadHeader);
    }

    let compressed = &payload[2..];
    if !compressed.len().is_multiple_of(2) {
        return Err(Error::BadHeader);
    }

    let mut plain = [0u8; 30];
    let mut plain_len = 0usize;
    let mut idx = 0usize;
    while idx < compressed.len() {
        let run = compressed[idx] as usize;
        let value = compressed[idx + 1];
        if run == 0 {
            return Err(Error::BadHeader);
        }
        plain_len = plain_len.checked_add(run).ok_or(Error::BadHeader)?;
        if plain_len <= 30 {
            let start = plain_len - run;
            let end = plain_len;
            plain[start..end].fill(value);
        } else if plain_len - run < 30 {
            let start = plain_len - run;
            let end = 30usize;
            plain[start..end].fill(value);
        }
        idx += 2;
    }

    if plain_len < 30 {
        return Err(Error::BadHeader);
    }
    if !constant_time_eq(&plain[0..4], b"S5D0") {
        return Err(Error::BadHeader);
    }
    if plain[4] != 1 {
        return Err(Error::BadHeader);
    }

    let dtype = plain[5];
    let layer_count = u32::from_le_bytes([plain[6], plain[7], plain[8], plain[9]]);
    let total_neurons = u32::from_le_bytes([plain[10], plain[11], plain[12], plain[13]]);
    let weights_len = u32::from_le_bytes([plain[14], plain[15], plain[16], plain[17]]);
    let biases_len = u32::from_le_bytes([plain[18], plain[19], plain[20], plain[21]]);
    let blob_count = u32::from_le_bytes([plain[22], plain[23], plain[24], plain[25]]);
    let flags = u32::from_le_bytes([plain[26], plain[27], plain[28], plain[29]]);

    Ok(HeaderSummary {
        dtype,
        layer_count,
        total_neurons,
        weights_len,
        biases_len,
        blob_count,
        flags,
    })
}

fn decode_compact_viz_summary(payload: &[u8]) -> Result<HeaderSummary, Error> {
    if payload.len() < 16 {
        return Err(Error::BadHeader);
    }
    if !constant_time_eq(&payload[0..4], b"VIZ\x00") {
        return Err(Error::BadHeader);
    }
    if payload[3] != 0 {
        return Err(Error::BadHeader);
    }

    let dtype = payload[4];
    let layer_count = u16::from_le_bytes([payload[5], payload[6]]) as u32;
    let total_neurons = u32::from_le_bytes([payload[7], payload[8], payload[9], payload[10]]);
    let weights_len = u16::from_le_bytes([payload[11], payload[12]]) as u32;
    let biases_len = u16::from_le_bytes([payload[13], payload[14]]) as u32;
    let blob_count = payload[15] as u32;

    Ok(HeaderSummary {
        dtype,
        layer_count,
        total_neurons,
        weights_len,
        biases_len,
        blob_count,
        flags: 0,
    })
}
