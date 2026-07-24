pub(crate) const RNN0_MAGIC: &[u8; 4] = b"RNN\x00";
pub(crate) const RNN0_VERSION: u16 = 1;
pub(crate) const RMD1_MAGIC: &[u8; 4] = b"RMD1";
pub(crate) const RMD1_VERSION: u16 = 1;
pub(crate) const RMD1_HEADER_SIZE: usize = 20;

pub(crate) const LAYER_META_SIZE: usize = 20;
pub(crate) const NEURON_POSITION_COLS: usize = 7;

pub(crate) const TLV_BLOB_TABLE: u8 = 0x03;
pub(crate) const TLV_HEADER_MODEL_NAME: u8 = 0x04;
pub(crate) const TLV_HEADER_NETWORK_SUMMARY: u8 = 0x05;
pub(crate) const TLV_HEADER_BENCHMARK: u8 = 0x06;
pub(crate) const TLV_VIZ_INDEX: u8 = 0x07;

pub(crate) const BLOB_NEURON_POSITIONS: &str = "neuron_positions";
pub(crate) const BLOB_LAYER_META: &str = "layer_meta";
pub(crate) const BLOB_WEIGHTS: &str = "weights";
pub(crate) const BLOB_BIASES: &str = "biases";
pub(crate) const BLOB_RUNTIME_INPUT: &str = "runtime.input";
pub(crate) const BLOB_TRAINING_LOG: &str = "training.log";
pub(crate) const LMLP_CONFIG_BLOB_DATA: &str = "lmlp.config";
pub(crate) const TOKENIZER_VOCAB_BLOB_DATA: &str = "tokenizer.vocab";
pub(crate) const TOKENIZER_MERGES_BLOB_DATA: &str = "tokenizer.merges";
pub(crate) const BLOB_CONV_SPEC: &str = "conv.spec";
pub(crate) const LMLP_WEIGHTS_BLOB_DATA: &str = "lmlp.weights";
pub(crate) const TRAIN_CONFIG_BLOB_DATA: &str = "train.config";
pub(crate) const LMLP_GRAPH_BLOB_DATA: &str = "lmlp.graph";
pub(crate) const LMLP_TENSORS_BLOB_DATA: &str = "lmlp.tensors";

pub(crate) const GRAPH_OPS_BLOB_DATA: &str = "graph.ops";
pub(crate) const OPTIMIZER_STATE_BLOB_DATA: &str = "optimizer.state";
pub(crate) const TENSORS_SNAPSHOT_BLOB_DATA: &str = "tensors.snapshot";

const SUMMARY_MAGIC: &[u8; 4] = b"S5D0";
const SUMMARY_VERSION: u8 = 1;
const SUMMARY_CODEC_RLE: u8 = 1;
const SUMMARY_PLAIN_LEN: usize = 30;
const SUMMARY_MAX_COMPRESSED_LEN: usize = 2 + (SUMMARY_PLAIN_LEN * 2);
pub(crate) const COMPACT_BENCHMARK_LEN: usize = 32;
const COMPACT_NETWORK_SUMMARY_LEN: usize = 16;

const SUMMARY_FLAG_HAS_LAYER_META: u32 = 1 << 0;
const SUMMARY_FLAG_HAS_WEIGHTS: u32 = 1 << 1;
const SUMMARY_FLAG_HAS_BIASES: u32 = 1 << 2;
const SUMMARY_FLAG_HAS_NEURON_POSITIONS: u32 = 1 << 3;
const SUMMARY_FLAG_HAS_TOKENIZER_VOCAB: u32 = 1 << 4;
const SUMMARY_FLAG_HAS_TOKENIZER_MERGES: u32 = 1 << 5;
const SUMMARY_FLAG_HAS_GRAPH_OPS: u32 = 1 << 6;
const SUMMARY_FLAG_HAS_OPTIMIZER_STATE: u32 = 1 << 7;
const SUMMARY_FLAG_HAS_TENSORS_SNAPSHOT: u32 = 1 << 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RnnProtocolError {
    Truncated,
    BadMagic,
    BadVersion,
    BadHeader,
    CapacityTooSmall,
    InvalidPayload,
}

#[derive(Clone, Copy)]
pub(crate) struct BlobDesc<'a> {
    pub name: &'a str,
    pub dtype: u8,
    pub dims: [u32; 2],
    pub ndim: u8,
    pub payload: &'a [u8],
}

#[derive(Clone, Copy)]
pub(crate) struct Payload<'a> {
    pub dtype: u8,
    pub layer_count: usize,
    pub weights_len: usize,
    pub biases_len: usize,
    pub layer_meta: &'a [u8],
    pub weights: &'a [u8],
    pub biases: &'a [u8],
}

#[derive(Clone, Copy)]
struct SummaryBlob {
    bytes: [u8; SUMMARY_MAX_COMPRESSED_LEN],
    len: usize,
}

impl SummaryBlob {
    fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    fn len(&self) -> usize {
        self.len
    }
}

pub(crate) fn parse_payload<'a>(bytes: &'a [u8]) -> Result<Payload<'a>, RnnProtocolError> {
    if bytes.len() < RMD1_HEADER_SIZE {
        return Err(RnnProtocolError::Truncated);
    }
    if &bytes[0..4] != RMD1_MAGIC {
        return Err(RnnProtocolError::BadMagic);
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != RMD1_VERSION {
        return Err(RnnProtocolError::BadVersion);
    }

    let dtype = bytes[6];
    if dtype != 0 && dtype != 1 {
        return Err(RnnProtocolError::BadHeader);
    }
    let layer_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let weights_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    let biases_len = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;

    let layer_meta_bytes = layer_count
        .checked_mul(LAYER_META_SIZE)
        .ok_or(RnnProtocolError::BadHeader)?;
    let elem_size = if dtype == 0 { 4usize } else { 8usize };
    let weights_bytes = weights_len
        .checked_mul(elem_size)
        .ok_or(RnnProtocolError::BadHeader)?;
    let biases_bytes = biases_len
        .checked_mul(elem_size)
        .ok_or(RnnProtocolError::BadHeader)?;

    let layers_start = RMD1_HEADER_SIZE;
    let layers_end = layers_start
        .checked_add(layer_meta_bytes)
        .ok_or(RnnProtocolError::BadHeader)?;
    let weights_end = layers_end
        .checked_add(weights_bytes)
        .ok_or(RnnProtocolError::BadHeader)?;
    let biases_end = weights_end
        .checked_add(biases_bytes)
        .ok_or(RnnProtocolError::BadHeader)?;

    if biases_end > bytes.len() {
        return Err(RnnProtocolError::Truncated);
    }

    Ok(Payload {
        dtype,
        layer_count,
        weights_len,
        biases_len,
        layer_meta: &bytes[layers_start..layers_end],
        weights: &bytes[layers_end..weights_end],
        biases: &bytes[weights_end..biases_end],
    })
}

pub(crate) fn header_tlv_payload(bytes: &[u8], wanted_type: u8) -> Result<&[u8], RnnProtocolError> {
    if bytes.len() < 12 {
        return Err(RnnProtocolError::Truncated);
    }
    if &bytes[0..4] != RNN0_MAGIC {
        return Err(RnnProtocolError::BadMagic);
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != RNN0_VERSION {
        return Err(RnnProtocolError::BadVersion);
    }

    let header_size = u32::from_le_bytes(
        bytes[8..12]
            .try_into()
            .map_err(|_| RnnProtocolError::BadHeader)?,
    ) as usize;
    if header_size > bytes.len() || header_size < 12 {
        return Err(RnnProtocolError::BadHeader);
    }

    if header_size >= crate::format::model_config::CANONICAL_HEADER_SIZE && bytes.len() >= crate::format::model_config::CANONICAL_HEADER_SIZE {
        let bmk_off = bytes[0x0C] as usize;
        let viz_off = bytes[0x0D] as usize;
        let tbl_off = bytes[0x0E] as usize;
        if tbl_off < crate::format::model_config::CANONICAL_HEADER_SIZE && bytes[tbl_off] == 0xC1 {
            match wanted_type {
                TLV_HEADER_MODEL_NAME => {
                    let region = &bytes[0x10..0x20];
                    let end = region.iter().position(|&b| b == 0).unwrap_or(region.len());
                    return Ok(&region[..end]);
                }
                TLV_HEADER_BENCHMARK => {
                    let end = (bmk_off + COMPACT_BENCHMARK_LEN).min(viz_off).min(crate::format::model_config::CANONICAL_HEADER_SIZE);
                    return Ok(&bytes[bmk_off..end]);
                }
                TLV_HEADER_NETWORK_SUMMARY => {
                    let end = (viz_off + COMPACT_NETWORK_SUMMARY_LEN).min(tbl_off).min(crate::format::model_config::CANONICAL_HEADER_SIZE);
                    return Ok(&bytes[viz_off..end]);
                }
                TLV_BLOB_TABLE => return Ok(&bytes[tbl_off..crate::format::model_config::CANONICAL_HEADER_SIZE]),
                _ => return Err(RnnProtocolError::InvalidPayload),
            }
        }
    }

    let mut cursor = 12usize;
    while cursor < header_size {
        if bytes[cursor] == 0 {
            cursor += 1;
            continue;
        }
        if cursor + 5 > header_size {
            return Err(RnnProtocolError::BadHeader);
        }
        let tlv_type = bytes[cursor];
        cursor += 1;
        let tlv_len = u32::from_le_bytes(
            bytes[cursor..cursor + 4]
                .try_into()
                .map_err(|_| RnnProtocolError::BadHeader)?,
        ) as usize;
        cursor += 4;
        let end = cursor
            .checked_add(tlv_len)
            .ok_or(RnnProtocolError::BadHeader)?;
        if end > header_size {
            return Err(RnnProtocolError::BadHeader);
        }
        if tlv_type == wanted_type {
            return Ok(&bytes[cursor..end]);
        }
        cursor = end;
    }

    Err(RnnProtocolError::InvalidPayload)
}

pub(crate) fn total_neurons_from_layer_meta(layer_meta: &[u8]) -> Result<usize, RnnProtocolError> {
    if layer_meta.len() < LAYER_META_SIZE || !layer_meta.len().is_multiple_of(LAYER_META_SIZE) {
        return Err(RnnProtocolError::InvalidPayload);
    }
    let layer_count = layer_meta.len() / LAYER_META_SIZE;
    let input0 =
        u32::from_le_bytes([layer_meta[0], layer_meta[1], layer_meta[2], layer_meta[3]]) as usize;
    let mut total = input0;
    for idx in 0..layer_count {
        let base = idx * LAYER_META_SIZE;
        let out = u32::from_le_bytes([
            layer_meta[base + 4],
            layer_meta[base + 5],
            layer_meta[base + 6],
            layer_meta[base + 7],
        ]) as usize;
        total = total.checked_add(out).ok_or(RnnProtocolError::BadHeader)?;
    }
    Ok(total)
}

pub(crate) fn neuron_positions_blob_size(
    dtype: u8,
    layer_meta: &[u8],
) -> Result<usize, RnnProtocolError> {
    if (dtype != 0 && dtype != 1)
        || layer_meta.len() < LAYER_META_SIZE
        || !layer_meta.len().is_multiple_of(LAYER_META_SIZE)
    {
        return Err(RnnProtocolError::InvalidPayload);
    }
    let total_neurons = total_neurons_from_layer_meta(layer_meta)?;
    let elem_size = if dtype == 0 { 4usize } else { 8usize };
    let rows = total_neurons
        .checked_add(1)
        .ok_or(RnnProtocolError::BadHeader)?;
    let out_len = rows
        .checked_mul(NEURON_POSITION_COLS)
        .and_then(|v| v.checked_mul(elem_size))
        .ok_or(RnnProtocolError::BadHeader)?;
    Ok(out_len)
}

pub(crate) fn neuron_positions_rows(layer_meta: &[u8]) -> Result<usize, RnnProtocolError> {
    total_neurons_from_layer_meta(layer_meta)?
        .checked_add(1)
        .ok_or(RnnProtocolError::BadHeader)
}

fn largest_layer_size(layer_meta: &[u8]) -> Result<usize, RnnProtocolError> {
    let layer_count = layer_meta.len() / LAYER_META_SIZE;
    let mut max = u32::from_le_bytes([
        layer_meta[0],
        layer_meta[1],
        layer_meta[2],
        layer_meta[3],
    ]) as usize;
    for idx in 0..layer_count {
        let base = idx * LAYER_META_SIZE;
        let sz = u32::from_le_bytes([
            layer_meta[base + 4],
            layer_meta[base + 5],
            layer_meta[base + 6],
            layer_meta[base + 7],
        ]) as usize;
        if sz > max {
            max = sz;
        }
    }
    if max == 0 {
        return Err(RnnProtocolError::InvalidPayload);
    }
    Ok(max)
}

struct ConstellationScratch {
    base: *mut u8,
    pts_bytes: usize,
    kern_bytes: usize,
    idx_bytes: usize,
}

pub(crate) fn neuron_positions_blob_into(
    dtype: u8,
    layer_meta: &[u8],
    biases: &[u8],
    out: &mut [u8],
) -> Result<usize, RnnProtocolError> {
    let out_len = neuron_positions_blob_size(dtype, layer_meta)?;
    if out.len() < out_len {
        return Err(RnnProtocolError::CapacityTooSmall);
    }
    if dtype == 0 {
        neuron_positions_blob_f32(layer_meta, biases, &mut out[..out_len])?;
    } else {
        neuron_positions_blob_f64(layer_meta, biases, &mut out[..out_len])?;
    }
    Ok(out_len)
}

fn neuron_positions_blob_f32(
    layer_meta: &[u8],
    biases: &[u8],
    out: &mut [u8],
) -> Result<(), RnnProtocolError> {
    let total_neurons = total_neurons_from_layer_meta(layer_meta)?;
    if total_neurons == 0 {
        return Err(RnnProtocolError::InvalidPayload);
    }
    let pt_size = core::mem::size_of::<crate::graph::conv::sphere5d::NeuronPoint>();
    let kern_size = core::mem::size_of::<crate::graph::conv::sphere5d::NeuronKernel>();
    let idx_size = core::mem::size_of::<usize>();
    let pts_bytes = total_neurons
        .checked_mul(pt_size)
        .ok_or(RnnProtocolError::BadHeader)?;
    let kern_bytes = total_neurons
        .checked_mul(kern_size)
        .ok_or(RnnProtocolError::BadHeader)?;
    let idx_bytes = total_neurons
        .checked_mul(idx_size)
        .ok_or(RnnProtocolError::BadHeader)?;
    let scratch_bytes = pts_bytes
        .checked_add(kern_bytes)
        .and_then(|v| v.checked_add(idx_bytes))
        .and_then(|v| v.checked_add(idx_bytes))
        .ok_or(RnnProtocolError::BadHeader)?;
    let base = crate::engine::runtime::hardware::mmap_shared_anon(scratch_bytes);
    if base.is_null() {
        return Err(RnnProtocolError::CapacityTooSmall);
    }
    let scratch = ConstellationScratch {
        base,
        pts_bytes,
        kern_bytes,
        idx_bytes,
    };
    let res = neuron_positions_blob_f32_inner(layer_meta, biases, out, total_neurons, &scratch);
    crate::engine::runtime::hardware::munmap(base, scratch_bytes);
    res
}

fn neuron_positions_blob_f32_inner(
    layer_meta: &[u8],
    biases: &[u8],
    out: &mut [u8],
    total_neurons: usize,
    scratch: &ConstellationScratch,
) -> Result<(), RnnProtocolError> {
    let pts = unsafe {
        core::slice::from_raw_parts_mut(
            scratch.base as *mut crate::graph::conv::sphere5d::NeuronPoint,
            total_neurons,
        )
    };
    let kernels = unsafe {
        core::slice::from_raw_parts_mut(
            scratch.base.add(scratch.pts_bytes) as *mut crate::graph::conv::sphere5d::NeuronKernel,
            total_neurons,
        )
    };
    let member = unsafe {
        core::slice::from_raw_parts_mut(
            scratch.base.add(scratch.pts_bytes + scratch.kern_bytes) as *mut usize,
            total_neurons,
        )
    };
    let assign = unsafe {
        core::slice::from_raw_parts_mut(
            scratch
                .base
                .add(scratch.pts_bytes + scratch.kern_bytes + scratch.idx_bytes) as *mut usize,
            total_neurons,
        )
    };

    let mut sphere = crate::graph::conv::sphere5d::Sphere5D::new(pts, 1.0)
        .map_err(|_| RnnProtocolError::InvalidPayload)?;
    fill_sphere_f32(&mut sphere, layer_meta, biases)?;

    let centroid = crate::graph::conv::sphere5d::centroid(sphere.as_slice());
    let mean_radius = crate::graph::conv::sphere5d::mean_radius(sphere.as_slice(), centroid);

    let max_kernel_size = largest_layer_size(layer_meta)?;
    let max_distance = if mean_radius > 0.0 {
        mean_radius * 2.0
    } else {
        sphere.radius()
    };
    let stats = crate::graph::conv::sphere5d::aggregate_constellation_to_kernels(
        sphere.as_slice(),
        max_distance,
        max_kernel_size,
        kernels,
        member,
        assign,
    )
    .map_err(|_| RnnProtocolError::InvalidPayload)?;
    if stats.points_used != total_neurons {
        return Err(RnnProtocolError::InvalidPayload);
    }
    let last_layer = layer_meta.len() / LAYER_META_SIZE;
    if crate::graph::conv::sphere5d::layer_bounds(sphere.as_slice(), last_layer).is_none()
        || !crate::graph::conv::sphere5d::neuron_exists(sphere.as_slice(), 0, 0)
    {
        return Err(RnnProtocolError::InvalidPayload);
    }

    write_blob_header_f32(out, &centroid, mean_radius, stats.kernels_used);
    let mut cursor = NEURON_POSITION_COLS * 4;
    for p in sphere.as_slice() {
        let row = [
            p.position[0],
            p.position[1],
            p.position[2],
            p.position[3],
            p.position[4],
            p.bias,
            p.activation,
        ];
        for v in row {
            out[cursor..cursor + 4].copy_from_slice(&v.to_le_bytes());
            cursor += 4;
        }
    }
    Ok(())
}

fn write_blob_header_f32(
    out: &mut [u8],
    centroid: &[f32; 5],
    mean_radius: f32,
    kernel_count: usize,
) {
    let header = [
        centroid[0],
        centroid[1],
        centroid[2],
        centroid[3],
        centroid[4],
        mean_radius,
        kernel_count as f32,
    ];
    let mut cursor = 0usize;
    for v in header {
        out[cursor..cursor + 4].copy_from_slice(&v.to_le_bytes());
        cursor += 4;
    }
}

fn fill_sphere_f32(
    sphere: &mut crate::graph::conv::sphere5d::Sphere5D<'_>,
    layer_meta: &[u8],
    biases: &[u8],
) -> Result<(), RnnProtocolError> {
    let layer_count = layer_meta.len() / LAYER_META_SIZE;
    let input0 =
        u32::from_le_bytes([layer_meta[0], layer_meta[1], layer_meta[2], layer_meta[3]]) as usize;
    let mut bias_cursor = 0usize;
    for layer_idx in 0..=layer_count {
        let (layer_size, activation) = if layer_idx == 0 {
            (input0, 0u8)
        } else {
            let base = (layer_idx - 1) * LAYER_META_SIZE;
            let sz = u32::from_le_bytes([
                layer_meta[base + 4],
                layer_meta[base + 5],
                layer_meta[base + 6],
                layer_meta[base + 7],
            ]) as usize;
            (sz, layer_meta[base + 16])
        };
        for neuron_idx in 0..layer_size {
            let bias = if layer_idx == 0 {
                0.0f32
            } else {
                let off = bias_cursor
                    .checked_mul(4)
                    .ok_or(RnnProtocolError::BadHeader)?;
                if off + 4 > biases.len() {
                    return Err(RnnProtocolError::InvalidPayload);
                }
                let b = f32::from_le_bytes([
                    biases[off],
                    biases[off + 1],
                    biases[off + 2],
                    biases[off + 3],
                ]);
                bias_cursor += 1;
                b
            };
            sphere
                .add_neuron(layer_idx, neuron_idx, bias, activation as f32)
                .ok_or(RnnProtocolError::CapacityTooSmall)?;
        }
    }
    Ok(())
}

struct ConstellationScratchF64 {
    base: *mut u8,
    pts_bytes: usize,
    pos_bytes: usize,
    kern_bytes: usize,
    idx_bytes: usize,
}

fn neuron_positions_blob_f64(
    layer_meta: &[u8],
    biases: &[u8],
    out: &mut [u8],
) -> Result<(), RnnProtocolError> {
    let total_neurons = total_neurons_from_layer_meta(layer_meta)?;
    if total_neurons == 0 {
        return Err(RnnProtocolError::InvalidPayload);
    }
    let pt_size = core::mem::size_of::<crate::graph::conv::sphere5d::NeuronPointF64>();
    let pos_size = core::mem::size_of::<[f64; 5]>();
    let kern_size = core::mem::size_of::<crate::graph::conv::sphere5d::NeuronKernelF64>();
    let idx_size = core::mem::size_of::<usize>();
    let pts_bytes = total_neurons
        .checked_mul(pt_size)
        .ok_or(RnnProtocolError::BadHeader)?;
    let pos_bytes = total_neurons
        .checked_mul(pos_size)
        .ok_or(RnnProtocolError::BadHeader)?;
    let kern_bytes = total_neurons
        .checked_mul(kern_size)
        .ok_or(RnnProtocolError::BadHeader)?;
    let idx_bytes = total_neurons
        .checked_mul(idx_size)
        .ok_or(RnnProtocolError::BadHeader)?;
    let scratch_bytes = pts_bytes
        .checked_add(pos_bytes)
        .and_then(|v| v.checked_add(kern_bytes))
        .and_then(|v| v.checked_add(idx_bytes))
        .and_then(|v| v.checked_add(idx_bytes))
        .ok_or(RnnProtocolError::BadHeader)?;
    let base = crate::engine::runtime::hardware::mmap_shared_anon(scratch_bytes);
    if base.is_null() {
        return Err(RnnProtocolError::CapacityTooSmall);
    }
    let scratch = ConstellationScratchF64 {
        base,
        pts_bytes,
        pos_bytes,
        kern_bytes,
        idx_bytes,
    };
    let res = neuron_positions_blob_f64_inner(layer_meta, biases, out, total_neurons, &scratch);
    crate::engine::runtime::hardware::munmap(base, scratch_bytes);
    res
}

fn neuron_positions_blob_f64_inner(
    layer_meta: &[u8],
    biases: &[u8],
    out: &mut [u8],
    total_neurons: usize,
    scratch: &ConstellationScratchF64,
) -> Result<(), RnnProtocolError> {
    let pts = unsafe {
        core::slice::from_raw_parts_mut(
            scratch.base as *mut crate::graph::conv::sphere5d::NeuronPointF64,
            total_neurons,
        )
    };
    let positions = unsafe {
        core::slice::from_raw_parts_mut(
            scratch.base.add(scratch.pts_bytes) as *mut [f64; 5],
            total_neurons,
        )
    };
    let kernels = unsafe {
        core::slice::from_raw_parts_mut(
            scratch.base.add(scratch.pts_bytes + scratch.pos_bytes)
                as *mut crate::graph::conv::sphere5d::NeuronKernelF64,
            total_neurons,
        )
    };
    let member = unsafe {
        core::slice::from_raw_parts_mut(
            scratch
                .base
                .add(scratch.pts_bytes + scratch.pos_bytes + scratch.kern_bytes)
                as *mut usize,
            total_neurons,
        )
    };
    let assign = unsafe {
        core::slice::from_raw_parts_mut(
            scratch
                .base
                .add(scratch.pts_bytes + scratch.pos_bytes + scratch.kern_bytes + scratch.idx_bytes)
                as *mut usize,
            total_neurons,
        )
    };

    let filled = fill_points_f64(pts, layer_meta, biases)?;
    if filled != total_neurons {
        return Err(RnnProtocolError::InvalidPayload);
    }
    for (i, p) in pts.iter().enumerate() {
        positions[i] = p.position;
    }

    let centroid = crate::graph::conv::sphere5d::centroid_f64(positions);
    let mean_radius = crate::graph::conv::sphere5d::mean_radius_f64(positions, centroid);

    let max_kernel_size = largest_layer_size(layer_meta)?;
    let max_distance = if mean_radius > 0.0 {
        mean_radius * 2.0
    } else {
        1.0
    };
    let stats = crate::graph::conv::sphere5d::aggregate_constellation_to_kernels_f64(
        pts,
        max_distance,
        max_kernel_size,
        kernels,
        member,
        assign,
    )
    .map_err(|_| RnnProtocolError::InvalidPayload)?;
    if stats.points_used != total_neurons {
        return Err(RnnProtocolError::InvalidPayload);
    }
    if !crate::graph::conv::sphere5d::neuron_exists_f64(pts, 0, 0) {
        return Err(RnnProtocolError::InvalidPayload);
    }

    write_blob_header_f64(out, &centroid, mean_radius, stats.kernels_used);
    let mut cursor = NEURON_POSITION_COLS * 8;
    for p in pts.iter() {
        let row = [
            p.position[0],
            p.position[1],
            p.position[2],
            p.position[3],
            p.position[4],
            p.bias,
            p.activation,
        ];
        for v in row {
            out[cursor..cursor + 8].copy_from_slice(&v.to_le_bytes());
            cursor += 8;
        }
    }
    Ok(())
}

fn write_blob_header_f64(
    out: &mut [u8],
    centroid: &[f64; 5],
    mean_radius: f64,
    kernel_count: usize,
) {
    let header = [
        centroid[0],
        centroid[1],
        centroid[2],
        centroid[3],
        centroid[4],
        mean_radius,
        kernel_count as f64,
    ];
    let mut cursor = 0usize;
    for v in header {
        out[cursor..cursor + 8].copy_from_slice(&v.to_le_bytes());
        cursor += 8;
    }
}

fn fill_points_f64(
    pts: &mut [crate::graph::conv::sphere5d::NeuronPointF64],
    layer_meta: &[u8],
    biases: &[u8],
) -> Result<usize, RnnProtocolError> {
    let layer_count = layer_meta.len() / LAYER_META_SIZE;
    let input0 =
        u32::from_le_bytes([layer_meta[0], layer_meta[1], layer_meta[2], layer_meta[3]]) as usize;
    let mut bias_cursor = 0usize;
    let mut written = 0usize;
    for layer_idx in 0..=layer_count {
        let (layer_size, activation) = if layer_idx == 0 {
            (input0, 0u8)
        } else {
            let base = (layer_idx - 1) * LAYER_META_SIZE;
            let sz = u32::from_le_bytes([
                layer_meta[base + 4],
                layer_meta[base + 5],
                layer_meta[base + 6],
                layer_meta[base + 7],
            ]) as usize;
            (sz, layer_meta[base + 16])
        };
        for neuron_idx in 0..layer_size {
            let bias = if layer_idx == 0 {
                0.0f64
            } else {
                let off = bias_cursor
                    .checked_mul(8)
                    .ok_or(RnnProtocolError::BadHeader)?;
                if off + 8 > biases.len() {
                    return Err(RnnProtocolError::InvalidPayload);
                }
                let b = f64::from_le_bytes([
                    biases[off],
                    biases[off + 1],
                    biases[off + 2],
                    biases[off + 3],
                    biases[off + 4],
                    biases[off + 5],
                    biases[off + 6],
                    biases[off + 7],
                ]);
                bias_cursor += 1;
                b
            };
            if written >= pts.len() {
                return Err(RnnProtocolError::CapacityTooSmall);
            }
            let seed = crate::graph::conv::sphere5d::mix_seed(layer_idx as u64, neuron_idx as u64);
            let position = crate::graph::conv::sphere5d::sphere_pos_from_seed_f64(seed, 1.0);
            pts[written] = crate::graph::conv::sphere5d::NeuronPointF64 {
                layer: layer_idx,
                neuron: neuron_idx,
                bias,
                activation: activation as f64,
                position,
            };
            written += 1;
        }
    }
    Ok(written)
}

pub(crate) fn encode_blob_payloads_with_header_metadata(
    records: &[BlobDesc<'_>],
    model_name_header: Option<&str>,
    benchmark_header: Option<&[u8]>,
    out: &mut [u8],
) -> Result<usize, RnnProtocolError> {
    encode_blob_records_impl(records, model_name_header, benchmark_header, out, true, true)
}

pub(crate) fn encode_blob_payloads_no_digest(
    records: &[BlobDesc<'_>],
    model_name_header: Option<&str>,
    benchmark_header: Option<&[u8]>,
    out: &mut [u8],
) -> Result<usize, RnnProtocolError> {
    encode_blob_records_impl(records, model_name_header, benchmark_header, out, true, false)
}

fn encode_blob_records_impl(
    records: &[BlobDesc<'_>],
    model_name_header: Option<&str>,
    benchmark_header: Option<&[u8]>,
    out: &mut [u8],
    copy_payloads: bool,
    compute_digests: bool,
) -> Result<usize, RnnProtocolError> {
    let compact_eligible = records.len() <= 5
        && records.iter().all(|rec| {
            rec.ndim > 0
                && rec.ndim <= 2
                && rec.payload.len() <= u32::MAX as usize
        });

    if compact_eligible && payload_total_len(records)? <= u32::MAX as usize {
        let payload_len = payload_total_len_aligned(records)?;
        let header_size = crate::format::model_config::CANONICAL_HEADER_SIZE;
        let total_size = (header_size
            .checked_add(payload_len)
            .ok_or(RnnProtocolError::BadHeader)?
            + 15) & !15;
        if out.len() < total_size {
            return Err(RnnProtocolError::CapacityTooSmall);
        }

        let model_name = model_name_header.unwrap_or("");
        let benchmark_raw = benchmark_header.unwrap_or(b"BMK\x01");
        let benchmark_compact = compact_benchmark_for_header(benchmark_raw);
        let visual = compact_network_summary_blob(records);
        let header_bytes = crate::format::model_config::ingest::build_canonical_header(model_name, &benchmark_compact, &visual);
        if header_bytes.len() != header_size {
            return Err(RnnProtocolError::BadHeader);
        }
        out[..header_size].copy_from_slice(&header_bytes);

        let mut payload_cursor = header_size;
        let entry_size = 15usize;
        let mut entry_cursor = 3usize;
        let table_start = out[0x0E] as usize;
        out[table_start + 1] = 1u8;
        out[table_start + 2] = u8::try_from(records.len()).unwrap_or(u8::MAX);
        for rec in records {
            let blob_id = match rec.name {
                BLOB_NEURON_POSITIONS => 1u8,
                BLOB_LAYER_META => 2u8,
                BLOB_WEIGHTS => 3u8,
                BLOB_BIASES => 4u8,
                BLOB_RUNTIME_INPUT => 5u8,
                BLOB_TRAINING_LOG => 6u8,
                LMLP_CONFIG_BLOB_DATA => 7u8,
                TOKENIZER_VOCAB_BLOB_DATA => 8u8,
                TOKENIZER_MERGES_BLOB_DATA => 9u8,
                BLOB_CONV_SPEC => 10u8,
                LMLP_WEIGHTS_BLOB_DATA => 11u8,
                TRAIN_CONFIG_BLOB_DATA => 12u8,
                crate::format::model_config::ingest::MODEL_PRECISION_BLOB_DATA => 13u8,
                LMLP_GRAPH_BLOB_DATA => 14u8,
                LMLP_TENSORS_BLOB_DATA => 15u8,
                _ => return Err(RnnProtocolError::BadHeader),
            };
            if table_start + entry_cursor + entry_size > crate::format::model_config::CANONICAL_HEADER_SIZE {
                return Err(RnnProtocolError::BadHeader);
            }
            if rec.ndim == 0 || rec.ndim > 2 {
                return Err(RnnProtocolError::BadHeader);
            }
            if payload_cursor > u32::MAX as usize || rec.payload.len() > u32::MAX as usize {
                return Err(RnnProtocolError::BadHeader);
            }

            let d0_clamped = rec.dims[0].min(u16::MAX as u32) as u16;
            let d1_clamped = rec.dims[1].min(u16::MAX as u32) as u16;

            out[table_start + entry_cursor] = blob_id;
            out[table_start + entry_cursor + 1] = rec.dtype;
            out[table_start + entry_cursor + 2] = rec.ndim;
            out[table_start + entry_cursor + 3..table_start + entry_cursor + 5]
                .copy_from_slice(&d0_clamped.to_le_bytes());
            out[table_start + entry_cursor + 5..table_start + entry_cursor + 7]
                .copy_from_slice(&d1_clamped.to_le_bytes());
            out[table_start + entry_cursor + 7..table_start + entry_cursor + 11]
                .copy_from_slice(&(payload_cursor as u32).to_le_bytes());
            out[table_start + entry_cursor + 11..table_start + entry_cursor + 15]
                .copy_from_slice(&(rec.payload.len() as u32).to_le_bytes());
            entry_cursor += entry_size;

            let next_payload = payload_cursor
                .checked_add(rec.payload.len())
                .ok_or(RnnProtocolError::BadHeader)?;
            if copy_payloads {
                crate::engine::runtime::parallel_memcpy(&mut out[payload_cursor..next_payload], rec.payload);
            } else {
                out[payload_cursor..next_payload].fill(0);
            }
            payload_cursor = (next_payload + 15) & !15;
        }

        if payload_cursor < total_size {
            out[payload_cursor..total_size].fill(0);
        }

        return Ok(total_size);
    }

    let desc_len = descriptor_len(records)?;
    let payload_len = payload_total_len_aligned(records)?;
    let header_summary = header_network_summary_blob(records);
    let summary_len = header_summary.as_ref().map_or(0usize, |b| b.len());

    let viz_content_size = {
        let mut v = 6usize;
        for rec in records {
            v = v
                .checked_add(26usize + rec.name.len())
                .ok_or(RnnProtocolError::BadHeader)?;
        }
        v
    };

    let header_size = {
        let mut h = 12usize;
        if let Some(name) = model_name_header {
            h = h.checked_add(5 + name.len()).ok_or(RnnProtocolError::BadHeader)?;
        }
        if let Some(bmk) = benchmark_header {
            h = h.checked_add(5 + bmk.len()).ok_or(RnnProtocolError::BadHeader)?;
        }
        if summary_len > 0 {
            h = h.checked_add(5 + summary_len).ok_or(RnnProtocolError::BadHeader)?;
        }
        h = h.checked_add(5 + viz_content_size).ok_or(RnnProtocolError::BadHeader)?;
        h = h.checked_add(5 + desc_len).ok_or(RnnProtocolError::BadHeader)?;
        (h + 3) & !3
    };
    let total_size = (header_size
        .checked_add(payload_len)
        .ok_or(RnnProtocolError::BadHeader)?
        + 15) & !15;

    if out.len() < total_size {
        return Err(RnnProtocolError::CapacityTooSmall);
    }

    out[0..4].copy_from_slice(RNN0_MAGIC);
    out[4..6].copy_from_slice(&RNN0_VERSION.to_le_bytes());
    out[6..8].copy_from_slice(&0u16.to_le_bytes());
    out[8..12].copy_from_slice(&(header_size as u32).to_le_bytes());

    let mut cursor = if header_size == crate::format::model_config::CANONICAL_HEADER_SIZE {
        let model_name = model_name_header.unwrap_or("");
        let benchmark = benchmark_header.unwrap_or(b"BMK\x01");
        let visual = header_summary
            .as_ref()
            .map(|b| b.as_slice())
            .map(|_s| compact_network_summary_blob(records))
            .unwrap_or(compact_network_summary_blob(records));
        let header_bytes = crate::format::model_config::ingest::build_canonical_header(model_name, benchmark, &visual);
        if header_bytes.len() != crate::format::model_config::CANONICAL_HEADER_SIZE {
            return Err(RnnProtocolError::BadHeader);
        }
        out[..crate::format::model_config::CANONICAL_HEADER_SIZE].copy_from_slice(&header_bytes);
        let tbl2 = out[0x0E] as usize;
        out[tbl2 + 1] = 1u8;
        out[tbl2 + 2] = u8::try_from(records.len()).unwrap_or(u8::MAX);
        header_size
    } else {
        12usize
    };

    let mut viz_start = 0usize;

    if header_size != crate::format::model_config::CANONICAL_HEADER_SIZE {
        if let Some(model_name) = model_name_header {
            let name_bytes = model_name.as_bytes();
            out[cursor] = TLV_HEADER_MODEL_NAME;
            cursor += 1;
            out[cursor..cursor + 4].copy_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            cursor += 4;
            out[cursor..cursor + name_bytes.len()].copy_from_slice(name_bytes);
            cursor += name_bytes.len();
        }

        if let Some(benchmark) = benchmark_header {
            out[cursor] = TLV_HEADER_BENCHMARK;
            cursor += 1;
            out[cursor..cursor + 4].copy_from_slice(&(benchmark.len() as u32).to_le_bytes());
            cursor += 4;
            out[cursor..cursor + benchmark.len()].copy_from_slice(benchmark);
            cursor += benchmark.len();
        }

        if let Some(summary) = header_summary.as_ref() {
            out[cursor] = TLV_HEADER_NETWORK_SUMMARY;
            cursor += 1;
            out[cursor..cursor + 4].copy_from_slice(&(summary.len() as u32).to_le_bytes());
            cursor += 4;
            out[cursor..cursor + summary.len()].copy_from_slice(summary.as_slice());
            cursor += summary.len();
        }

        viz_start = cursor;
        cursor += 5 + viz_content_size;

        out[cursor] = TLV_BLOB_TABLE;
        cursor += 1;
        out[cursor..cursor + 4].copy_from_slice(&(desc_len as u32).to_le_bytes());
        cursor += 4;
    } else {
        out[header_size..header_size + 1].copy_from_slice(&[TLV_BLOB_TABLE]);
        out[header_size + 1..header_size + 5].copy_from_slice(&(desc_len as u32).to_le_bytes());
        cursor = header_size + 5;
    }

    let mut payload_cursor = header_size;

    const MAX_VIZ_BLOBS: usize = 16;
    let mut viz_file_offsets = [0u64; MAX_VIZ_BLOBS];
    let mut viz_orig_sizes = [0u64; MAX_VIZ_BLOBS];
    let mut viz_stored_sizes = [0u64; MAX_VIZ_BLOBS];
    let mut viz_blob_flags = [0u8; MAX_VIZ_BLOBS];

    if header_size == crate::format::model_config::CANONICAL_HEADER_SIZE {
        let table_start = out[0x0E] as usize;
        let entry_size = 15usize;
        let mut entry_cursor = 3usize;
        for rec in records {
            let blob_id = match rec.name {
                BLOB_NEURON_POSITIONS => 1u8,
                BLOB_LAYER_META => 2u8,
                BLOB_WEIGHTS => 3u8,
                BLOB_BIASES => 4u8,
                BLOB_RUNTIME_INPUT => 5u8,
                BLOB_TRAINING_LOG => 6u8,
                LMLP_CONFIG_BLOB_DATA => 7u8,
                TOKENIZER_VOCAB_BLOB_DATA => 8u8,
                TOKENIZER_MERGES_BLOB_DATA => 9u8,
                BLOB_CONV_SPEC => 10u8,
                LMLP_WEIGHTS_BLOB_DATA => 11u8,
                TRAIN_CONFIG_BLOB_DATA => 12u8,
                crate::format::model_config::ingest::MODEL_PRECISION_BLOB_DATA => 13u8,
                LMLP_GRAPH_BLOB_DATA => 14u8,
                LMLP_TENSORS_BLOB_DATA => 15u8,
                _ => 0u8,
            };
            if table_start + entry_cursor + entry_size > crate::format::model_config::CANONICAL_HEADER_SIZE {
                return Err(RnnProtocolError::BadHeader);
            }

            out[table_start + entry_cursor] = blob_id;
            out[table_start + entry_cursor + 1] = rec.dtype;
            out[table_start + entry_cursor + 2] = rec.ndim;
            out[table_start + entry_cursor + 3..table_start + entry_cursor + 5]
                .copy_from_slice(&(rec.dims[0] as u16).to_le_bytes());
            out[table_start + entry_cursor + 5..table_start + entry_cursor + 7]
                .copy_from_slice(&(rec.dims[1] as u16).to_le_bytes());
            out[table_start + entry_cursor + 7..table_start + entry_cursor + 11]
                .copy_from_slice(&(payload_cursor as u32).to_le_bytes());
            out[table_start + entry_cursor + 11..table_start + entry_cursor + 15]
                .copy_from_slice(&(rec.payload.len() as u32).to_le_bytes());
            entry_cursor += entry_size;

            let next_payload = payload_cursor
                .checked_add(rec.payload.len())
                .ok_or(RnnProtocolError::BadHeader)?;
            if copy_payloads {
                crate::engine::runtime::parallel_memcpy(&mut out[payload_cursor..next_payload], rec.payload);
            } else {
                out[payload_cursor..next_payload].fill(0);
            }
            payload_cursor = (next_payload + 15) & !15;
        }

        let table_used = 3usize
            .checked_add(
                records
                    .len()
                    .checked_mul(entry_size)
                    .ok_or(RnnProtocolError::BadHeader)?,
            )
            .ok_or(RnnProtocolError::BadHeader)?;
        let table_body_start = table_start + table_used;
        if table_body_start + 12 <= crate::format::model_config::CANONICAL_HEADER_SIZE {
            out[table_body_start..table_body_start + 4]
                .copy_from_slice(&(payload_len as u32).to_le_bytes());
            out[table_body_start + 4..table_body_start + 8]
                .copy_from_slice(&(header_size as u32).to_le_bytes());
            out[table_body_start + 8..table_body_start + 12]
                .copy_from_slice(&(payload_cursor as u32).to_le_bytes());
        }

        if payload_cursor < total_size {
            out[payload_cursor..total_size].fill(0);
        }

        return Ok(total_size);
    } else {
        for (blob_idx, rec) in records.iter().enumerate() {
            let name_bytes = rec.name.as_bytes();
            if name_bytes.len() > u16::MAX as usize {
                return Err(RnnProtocolError::BadHeader);
            }

            let is_compressed = rec.payload.len() >= 8
                && rec.payload[0..4] == LZ77_MAGIC;
            let stored_len = rec.payload.len();
            let orig_size = if is_compressed {
                lz77_decompressed_len(rec.payload)
                    .unwrap_or(stored_len) as u64
            } else {
                stored_len as u64
            };

            let next_payload = payload_cursor
                .checked_add(stored_len)
                .ok_or(RnnProtocolError::BadHeader)?;
            if copy_payloads {
                crate::engine::runtime::parallel_memcpy(&mut out[payload_cursor..next_payload], rec.payload);
            } else {
                out[payload_cursor..next_payload].fill(0);
            }

            if blob_idx < MAX_VIZ_BLOBS {
                viz_file_offsets[blob_idx] = payload_cursor as u64;
                viz_orig_sizes[blob_idx] = orig_size;
                viz_stored_sizes[blob_idx] = stored_len as u64;
                viz_blob_flags[blob_idx] = if is_compressed { 1u8 } else { 0u8 };
            }

            out[cursor..cursor + 2].copy_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            cursor += 2;
            out[cursor..cursor + name_bytes.len()].copy_from_slice(name_bytes);
            cursor += name_bytes.len();

            out[cursor] = rec.dtype;
            cursor += 1;
            out[cursor] = rec.ndim;
            cursor += 1;

            for dim in rec.dims.iter().take(rec.ndim as usize) {
                out[cursor..cursor + 4].copy_from_slice(&dim.to_le_bytes());
                cursor += 4;
            }

            out[cursor..cursor + 8].copy_from_slice(&(payload_cursor as u64).to_le_bytes());
            cursor += 8;
            out[cursor..cursor + 8].copy_from_slice(&(stored_len as u64).to_le_bytes());
            cursor += 8;

            if compute_digests {
                let mut digest = [0u8; 32];
                crate::format::model_config::compute_sha256(
                    &out[payload_cursor..payload_cursor + stored_len],
                    &mut digest,
                );
                out[cursor..cursor + 32].copy_from_slice(&digest);
            } else {
                out[cursor..cursor + 32].fill(0);
            }
            cursor += 32;

            payload_cursor = ((payload_cursor + stored_len) + 15) & !15;
        }

        out[viz_start] = TLV_VIZ_INDEX;
        out[viz_start + 1..viz_start + 5]
            .copy_from_slice(&(viz_content_size as u32).to_le_bytes());
        let mut vp = viz_start + 5;
        out[vp..vp + 4].copy_from_slice(b"VIZX");
        vp += 4;
        out[vp] = 1u8;
        vp += 1;
        out[vp] = records.len().min(255) as u8;
        vp += 1;
        for (i, rec) in records.iter().enumerate().take(255) {
            let nb = rec.name.as_bytes();
            out[vp] = nb.len() as u8;
            vp += 1;
            out[vp..vp + nb.len()].copy_from_slice(nb);
            vp += nb.len();
            out[vp..vp + 8].copy_from_slice(&viz_file_offsets[i].to_le_bytes());
            vp += 8;
            out[vp..vp + 8].copy_from_slice(&viz_orig_sizes[i].to_le_bytes());
            vp += 8;
            out[vp..vp + 8].copy_from_slice(&viz_stored_sizes[i].to_le_bytes());
            vp += 8;
            out[vp] = viz_blob_flags[i];
            vp += 1;
        }

        if cursor < header_size {
            out[cursor..header_size].fill(0);
        }
    }

    if payload_cursor < total_size {
        out[payload_cursor..total_size].fill(0);
    }

    Ok(payload_cursor)
}

fn header_network_summary_blob(records: &[BlobDesc<'_>]) -> Option<SummaryBlob> {
    let mut flags = 0u32;
    let mut dtype = 2u8;
    let mut layer_count = 0u32;
    let mut total_neurons = 0u32;
    let mut weights_len = 0u32;
    let mut biases_len = 0u32;

    for rec in records {
        match rec.name {
            BLOB_LAYER_META => {
                flags |= SUMMARY_FLAG_HAS_LAYER_META;
                dtype = rec.dtype;
                layer_count = rec.dims[0];
            }
            BLOB_WEIGHTS => {
                flags |= SUMMARY_FLAG_HAS_WEIGHTS;
                dtype = rec.dtype;
                weights_len = rec.dims[0];
            }
            BLOB_BIASES => {
                flags |= SUMMARY_FLAG_HAS_BIASES;
                dtype = rec.dtype;
                biases_len = rec.dims[0];
            }
            BLOB_NEURON_POSITIONS => {
                flags |= SUMMARY_FLAG_HAS_NEURON_POSITIONS;
                dtype = rec.dtype;
                total_neurons = rec.dims[0].saturating_sub(1);
            }
            TOKENIZER_VOCAB_BLOB_DATA => flags |= SUMMARY_FLAG_HAS_TOKENIZER_VOCAB,
            TOKENIZER_MERGES_BLOB_DATA => flags |= SUMMARY_FLAG_HAS_TOKENIZER_MERGES,
            GRAPH_OPS_BLOB_DATA => flags |= SUMMARY_FLAG_HAS_GRAPH_OPS,
            OPTIMIZER_STATE_BLOB_DATA => flags |= SUMMARY_FLAG_HAS_OPTIMIZER_STATE,
            TENSORS_SNAPSHOT_BLOB_DATA => flags |= SUMMARY_FLAG_HAS_TENSORS_SNAPSHOT,
            _ => {}
        }
    }

    if (flags & SUMMARY_FLAG_HAS_NEURON_POSITIONS) == 0 {
        let compact = compact_network_summary_blob(records);
        let mut out = [0u8; SUMMARY_MAX_COMPRESSED_LEN];
        let copy_len = compact.len().min(SUMMARY_MAX_COMPRESSED_LEN);
        out[..copy_len].copy_from_slice(&compact[..copy_len]);
        return Some(SummaryBlob {
            bytes: out,
            len: copy_len,
        });
    }

    let mut plain = [0u8; SUMMARY_PLAIN_LEN];
    plain[0..4].copy_from_slice(SUMMARY_MAGIC);
    plain[4] = SUMMARY_VERSION;
    plain[5] = dtype;
    plain[6..10].copy_from_slice(&layer_count.to_le_bytes());
    plain[10..14].copy_from_slice(&total_neurons.to_le_bytes());
    plain[14..18].copy_from_slice(&weights_len.to_le_bytes());
    plain[18..22].copy_from_slice(&biases_len.to_le_bytes());
    plain[22..26].copy_from_slice(&(records.len() as u32).to_le_bytes());
    plain[26..30].copy_from_slice(&flags.to_le_bytes());

    let mut out = [0u8; SUMMARY_MAX_COMPRESSED_LEN];
    out[0] = SUMMARY_VERSION;
    out[1] = SUMMARY_CODEC_RLE;
    let used = rle_encode_bytes(&plain, &mut out[2..])?;
    Some(SummaryBlob {
        bytes: out,
        len: 2 + used,
    })
}

pub(crate) fn compact_benchmark_for_header(full_bmk: &[u8]) -> [u8; COMPACT_BENCHMARK_LEN] {
    let mut out = [0u8; COMPACT_BENCHMARK_LEN];
    out[0..4].copy_from_slice(b"BMK\x01");
    out[4..6].copy_from_slice(&4u16.to_le_bytes());
    if full_bmk.len() >= 168 {
        out[24..28].copy_from_slice(&full_bmk[36..40]);
        out[28..32].copy_from_slice(&full_bmk[16..20]);
        out[20..24].copy_from_slice(&full_bmk[48..52]);
        out[16..20].copy_from_slice(&full_bmk[48..52]);
        out[7] = full_bmk[56];
        out[8..12].copy_from_slice(&full_bmk[60..64]);
        out[12..16].copy_from_slice(&full_bmk[64..68]);
        out[6] = full_bmk[68];
    }
    out
}

fn compact_network_summary_blob(records: &[BlobDesc<'_>]) -> [u8; COMPACT_NETWORK_SUMMARY_LEN] {
    let mut dtype = 2u8;
    let mut layer_count = 0u16;
    let mut total_neurons = 0u32;
    let mut weights_len = 0u16;
    let mut biases_len = 0u16;

    for rec in records {
        match rec.name {
            BLOB_LAYER_META => {
                dtype = rec.dtype;
                layer_count = u16::try_from(rec.dims[0]).unwrap_or(u16::MAX);
            }
            BLOB_WEIGHTS => {
                dtype = rec.dtype;
                weights_len = u16::try_from(rec.dims[0]).unwrap_or(u16::MAX);
            }
            BLOB_BIASES => {
                dtype = rec.dtype;
                biases_len = u16::try_from(rec.dims[0]).unwrap_or(u16::MAX);
            }
            BLOB_NEURON_POSITIONS => {
                dtype = rec.dtype;
                total_neurons = rec.dims[0].saturating_sub(1);
            }
            _ => {}
        }
    }

    let mut out = [0u8; COMPACT_NETWORK_SUMMARY_LEN];
    out[0..4].copy_from_slice(b"VIZ\x00");
    out[4] = dtype;
    out[5..7].copy_from_slice(&layer_count.to_le_bytes());
    out[7..11].copy_from_slice(&total_neurons.to_le_bytes());
    out[11..13].copy_from_slice(&weights_len.to_le_bytes());
    out[13..15].copy_from_slice(&biases_len.to_le_bytes());
    out[15] = u8::try_from(records.len()).unwrap_or(u8::MAX);
    out
}

fn descriptor_len(records: &[BlobDesc<'_>]) -> Result<usize, RnnProtocolError> {
    let mut total = 0usize;
    for rec in records {
        if rec.name.len() > u16::MAX as usize {
            return Err(RnnProtocolError::BadHeader);
        }
        if rec.ndim == 0 || rec.ndim > 2 {
            return Err(RnnProtocolError::BadHeader);
        }
        total = total
            .checked_add(2)
            .and_then(|v| v.checked_add(rec.name.len()))
            .and_then(|v| v.checked_add(1 + 1))
            .and_then(|v| v.checked_add((rec.ndim as usize).checked_mul(4)?))
            .and_then(|v| v.checked_add(8 + 8 + 32))
            .ok_or(RnnProtocolError::BadHeader)?;
    }
    Ok(total)
}

fn payload_total_len(records: &[BlobDesc<'_>]) -> Result<usize, RnnProtocolError> {
    let mut total = 0usize;
    for rec in records {
        total = total
            .checked_add(rec.payload.len())
            .ok_or(RnnProtocolError::BadHeader)?;
    }
    Ok(total)
}

fn payload_total_len_aligned(records: &[BlobDesc<'_>]) -> Result<usize, RnnProtocolError> {
    let mut cursor = 0usize;
    for rec in records {
        cursor = cursor
            .checked_add(rec.payload.len())
            .ok_or(RnnProtocolError::BadHeader)?;
        cursor = (cursor + 15) & !15;
    }
    Ok(cursor)
}

fn rle_encode_bytes(input: &[u8], out: &mut [u8]) -> Option<usize> {
    if input.is_empty() {
        return Some(0);
    }
    let mut idx = 0usize;
    let mut cursor = 0usize;
    while idx < input.len() {
        let value = input[idx];
        let mut run = 1usize;
        while idx + run < input.len() && input[idx + run] == value && run < 255 {
            run += 1;
        }
        if cursor + 2 > out.len() {
            return None;
        }
        out[cursor] = run as u8;
        out[cursor + 1] = value;
        cursor += 2;
        idx += run;
    }
    Some(cursor)
}

pub(crate) const LZ77_MAGIC: [u8; 4] = *b"LZ77";
const LZ77_HEADER_LEN: usize = 8;
const LZ77_HASH_SIZE: usize = 512;
const LZ77_WINDOW: usize = 4095;
const LZ77_MIN_MATCH: usize = 3;
const LZ77_MAX_MATCH: usize = 18;

pub(crate) fn lz77_compressed_bound(input_len: usize) -> usize {
    LZ77_HEADER_LEN + input_len + (input_len / 8) + 2
}

fn lz77_hash3(a: u8, b: u8, c: u8) -> usize {
    let h = (a as usize)
        .wrapping_mul(2654435761)
        .wrapping_add((b as usize).wrapping_mul(2246822519))
        .wrapping_add((c as usize).wrapping_mul(2246822313));
    h & (LZ77_HASH_SIZE - 1)
}

pub(crate) fn lz77_compress(input: &[u8], output: &mut [u8]) -> Option<usize> {
    if output.len() < LZ77_HEADER_LEN {
        return None;
    }
    output[0..4].copy_from_slice(&LZ77_MAGIC);
    output[4..8].copy_from_slice(&(input.len() as u32).to_le_bytes());

    if input.is_empty() {
        return Some(LZ77_HEADER_LEN);
    }

    let out_cap = output.len();
    let mut out_pos = LZ77_HEADER_LEN;
    let mut in_pos = 0usize;
    let mut table = [u32::MAX; LZ77_HASH_SIZE];

    while in_pos < input.len() {
        let flag_pos = out_pos;
        out_pos += 1;
        if out_pos > out_cap {
            return None;
        }

        let mut flags = 0u8;
        let mut tokens = 0u8;

        while tokens < 8 && in_pos < input.len() {
            let best = if in_pos + LZ77_MIN_MATCH <= input.len() {
                let b0 = input[in_pos];
                let b1 = input[in_pos + 1];
                let b2 = input[in_pos + 2];
                let h = lz77_hash3(b0, b1, b2);
                let prev = table[h] as usize;
                table[h] = in_pos as u32;
                if prev != u32::MAX as usize
                    && in_pos >= prev
                    && in_pos - prev >= 1
                    && in_pos - prev <= LZ77_WINDOW
                {
                    let offset = in_pos - prev;
                    let max_len = (input.len() - in_pos).min(LZ77_MAX_MATCH);
                    let mut len = 0usize;
                    while len < max_len && input[prev + len] == input[in_pos + len] {
                        len += 1;
                    }
                    if len >= LZ77_MIN_MATCH {
                        Some((offset, len))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            if let Some((offset, len)) = best {
                flags |= 1 << tokens;
                if out_pos + 2 > out_cap {
                    return None;
                }
                let enc = ((offset as u16 & 0x0FFF) << 4) | ((len - LZ77_MIN_MATCH) as u16 & 0x0F);
                output[out_pos] = (enc >> 8) as u8;
                output[out_pos + 1] = (enc & 0xFF) as u8;
                out_pos += 2;
                in_pos += len;
            } else {
                if out_pos >= out_cap {
                    return None;
                }
                output[out_pos] = input[in_pos];
                out_pos += 1;
                in_pos += 1;
            }
            tokens += 1;
        }

        output[flag_pos] = flags;
    }

    Some(out_pos)
}

pub(crate) fn lz77_decompressed_len(input: &[u8]) -> Option<usize> {
    if input.len() < LZ77_HEADER_LEN {
        return None;
    }
    if input[0..4] != LZ77_MAGIC {
        return None;
    }
    Some(u32::from_le_bytes([input[4], input[5], input[6], input[7]]) as usize)
}

pub(crate) fn lz77_decompress(input: &[u8], output: &mut [u8]) -> bool {
    if input.len() < LZ77_HEADER_LEN {
        return false;
    }
    if input[0..4] != LZ77_MAGIC {
        return false;
    }
    let orig_len = u32::from_le_bytes([input[4], input[5], input[6], input[7]]) as usize;
    if output.len() < orig_len {
        return false;
    }

    let mut in_pos = LZ77_HEADER_LEN;
    let mut out_pos = 0usize;

    while in_pos < input.len() && out_pos < orig_len {
        if in_pos >= input.len() {
            break;
        }
        let flags = input[in_pos];
        in_pos += 1;

        let mut bit = 0u8;
        while bit < 8 {
            if out_pos >= orig_len {
                break;
            }
            if in_pos >= input.len() {
                break;
            }
            if (flags >> bit) & 1 == 1 {
                if in_pos + 2 > input.len() {
                    return false;
                }
                let enc = ((input[in_pos] as u16) << 8) | (input[in_pos + 1] as u16);
                in_pos += 2;
                let offset = (enc >> 4) as usize;
                let len = ((enc & 0x0F) as usize) + LZ77_MIN_MATCH;
                if offset == 0 || out_pos < offset {
                    return false;
                }
                let start = out_pos - offset;
                let mut j = 0usize;
                while j < len && out_pos < orig_len {
                    output[out_pos] = output[start + j];
                    out_pos += 1;
                    j += 1;
                }
            } else {
                if in_pos >= input.len() {
                    return false;
                }
                output[out_pos] = input[in_pos];
                in_pos += 1;
                out_pos += 1;
            }
            bit += 1;
        }
    }

    out_pos == orig_len
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neuron_positions_rows_matches_layer_count() {
        let mut layer_meta = [0u8; 2 * LAYER_META_SIZE];
        layer_meta[0..4].copy_from_slice(&2u32.to_le_bytes());
        layer_meta[4..8].copy_from_slice(&3u32.to_le_bytes());
        layer_meta[LAYER_META_SIZE..LAYER_META_SIZE + 4].copy_from_slice(&3u32.to_le_bytes());
        layer_meta[LAYER_META_SIZE + 4..LAYER_META_SIZE + 8].copy_from_slice(&4u32.to_le_bytes());
        assert_eq!(neuron_positions_rows(&layer_meta).unwrap(), 10);
    }

    #[test]
    fn six_blob_container_digests_roundtrip() {
        let layer_meta = [7u8; 60];
        let weights = [3u8; 512];
        let biases = [9u8; 64];
        let lm_config = [1u8; 48];
        let lm_weights = [5u8; 1024];
        let vocab = [2u8; 96];

        let records = [
            BlobDesc { name: BLOB_LAYER_META, dtype: 0, dims: [3, 5], ndim: 2, payload: &layer_meta },
            BlobDesc { name: BLOB_WEIGHTS, dtype: 0, dims: [128, 0], ndim: 1, payload: &weights },
            BlobDesc { name: BLOB_BIASES, dtype: 0, dims: [16, 0], ndim: 1, payload: &biases },
            BlobDesc { name: LMLP_CONFIG_BLOB_DATA, dtype: 2, dims: [48, 0], ndim: 1, payload: &lm_config },
            BlobDesc { name: LMLP_WEIGHTS_BLOB_DATA, dtype: 0, dims: [256, 0], ndim: 1, payload: &lm_weights },
            BlobDesc { name: TOKENIZER_VOCAB_BLOB_DATA, dtype: 2, dims: [96, 0], ndim: 1, payload: &vocab },
        ];

        let mut out = [0u8; 8192];
        let used = encode_blob_payloads_with_header_metadata(&records, Some("small.rnn"), Some(b"BMK\x01"), &mut out)
            .expect("encode");

        let mut scratch_buf = [0u8; 16384];
        let mut scratch = crate::base::scratch::Scratch::new(&mut scratch_buf);
        let handle = crate::format::rnn_format::parser::parse_rnn_from_bytes(&out[..used], &mut scratch)
            .expect("parse");

        assert!(crate::format::rnn_format::bounds::all_blob_bounds_valid(&handle), "bounds");
        assert!(crate::format::rnn_format::bounds::all_blob_digests_valid(&handle), "digests");
    }

    #[test]
    fn seven_blob_no_digest_container_roundtrip() {
        let layer_meta = [7u8; 60];
        let weights = [3u8; 512];
        let biases = [9u8; 64];
        let lm_config = [1u8; 48];
        let lm_weights = [5u8; 1024];
        let vocab = [2u8; 96];
        let train_config = [4u8; 32];

        let records = [
            BlobDesc { name: BLOB_LAYER_META, dtype: 0, dims: [3, 5], ndim: 2, payload: &layer_meta },
            BlobDesc { name: BLOB_WEIGHTS, dtype: 0, dims: [128, 0], ndim: 1, payload: &weights },
            BlobDesc { name: BLOB_BIASES, dtype: 0, dims: [16, 0], ndim: 1, payload: &biases },
            BlobDesc { name: LMLP_CONFIG_BLOB_DATA, dtype: 2, dims: [48, 0], ndim: 1, payload: &lm_config },
            BlobDesc { name: LMLP_WEIGHTS_BLOB_DATA, dtype: 0, dims: [256, 0], ndim: 1, payload: &lm_weights },
            BlobDesc { name: TOKENIZER_VOCAB_BLOB_DATA, dtype: 2, dims: [96, 0], ndim: 1, payload: &vocab },
            BlobDesc { name: TRAIN_CONFIG_BLOB_DATA, dtype: 2, dims: [32, 0], ndim: 1, payload: &train_config },
        ];

        let mut out = [0u8; 8192];
        let used = encode_blob_payloads_no_digest(&records, Some("small.rnn"), Some(b"BMK\x01"), &mut out)
            .expect("encode");

        let mut scratch_buf = [0u8; 16384];
        let mut scratch = crate::base::scratch::Scratch::new(&mut scratch_buf);
        let handle = crate::format::rnn_format::parser::parse_rnn_from_bytes(&out[..used], &mut scratch)
            .expect("parse");

        assert!(crate::format::rnn_format::bounds::all_blob_bounds_valid(&handle), "bounds");
        assert!(crate::format::rnn_format::bounds::all_blob_digests_valid(&handle), "digests");
    }
}

