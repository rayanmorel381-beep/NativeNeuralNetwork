use crate::observability::visualization::errors::VisualizeError;
use core::convert::TryInto;

pub struct NetworkView<'a> {
    bytes: &'a [u8],
    header_size: usize,
    layer_count: usize,
    layer_meta_size: usize,
    layer_meta_offset: usize,
    weights_offset: usize,
    biases_offset: usize,
}

pub struct LayerView<'a> {
    bytes: &'a [u8],
    meta_offset: usize,
    layer_meta_size: usize,
    weights_offset: usize,
}

pub struct NeuronView<'a> {
    bytes: &'a [u8],
    weight_start: usize,
    weight_count: usize,
}

impl<'a> NetworkView<'a> {
    pub(crate) fn from_rnn_bytes(bytes: &'a [u8]) -> Result<Self, VisualizeError> {
        if bytes.len() < 20 {
            return Err(VisualizeError::InvalidFormat);
        }

        let layer_count = u32::from_le_bytes(
            bytes[8..12]
                .try_into()
                .map_err(|_| VisualizeError::InvalidFormat)?,
        ) as usize;
        let weights_len = u32::from_le_bytes(
            bytes[12..16]
                .try_into()
                .map_err(|_| VisualizeError::InvalidFormat)?,
        ) as usize;
        let biases_len = u32::from_le_bytes(
            bytes[16..20]
                .try_into()
                .map_err(|_| VisualizeError::InvalidFormat)?,
        ) as usize;

        let header_size = 20usize;
        let layer_meta_size = 20usize;

        let needed_meta = layer_count
            .checked_mul(layer_meta_size)
            .ok_or(VisualizeError::InvalidFormat)?;
        let layers_meta_end = header_size
            .checked_add(needed_meta)
            .ok_or(VisualizeError::InvalidFormat)?;
        if layers_meta_end > bytes.len() {
            return Err(VisualizeError::InvalidFormat);
        }

        let weights_bytes = weights_len
            .checked_mul(4)
            .ok_or(VisualizeError::InvalidFormat)?;
        let weights_offset = layers_meta_end;
        let weights_end = weights_offset
            .checked_add(weights_bytes)
            .ok_or(VisualizeError::InvalidFormat)?;
        if weights_end > bytes.len() {
            return Err(VisualizeError::InvalidFormat);
        }

        let biases_offset = weights_end;
        let biases_bytes = biases_len
            .checked_mul(4)
            .ok_or(VisualizeError::InvalidFormat)?;
        let biases_end = biases_offset
            .checked_add(biases_bytes)
            .ok_or(VisualizeError::InvalidFormat)?;
        if biases_end > bytes.len() {
            return Err(VisualizeError::InvalidFormat);
        }

        Ok(NetworkView {
            bytes,
            header_size,
            layer_count,
            layer_meta_size,
            layer_meta_offset: header_size,
            weights_offset,
            biases_offset,
        })
    }

    pub fn layer_count(&self) -> usize {
        self.layer_count
    }

    pub fn header_size(&self) -> usize {
        self.header_size
    }

    pub fn biases_offset(&self) -> usize {
        self.biases_offset
    }

    pub fn layer(&self, idx: usize) -> Result<LayerView<'a>, VisualizeError> {
        if idx >= self.layer_count {
            return Err(VisualizeError::OutOfBounds);
        }
        let meta_offset = self.layer_meta_offset + idx * self.layer_meta_size;
        if meta_offset + self.layer_meta_size > self.bytes.len() {
            return Err(VisualizeError::OutOfBounds);
        }
        Ok(LayerView {
            bytes: self.bytes,
            meta_offset,
            layer_meta_size: self.layer_meta_size,
            weights_offset: self.weights_offset,
        })
    }
}

impl<'a> LayerView<'a> {
    pub fn neuron_count(&self) -> Result<usize, VisualizeError> {
        let start = self.meta_offset + 4;
        let end = start + 4;
        if end > self.bytes.len() {
            return Err(VisualizeError::OutOfBounds);
        }
        let output_size = u32::from_le_bytes(
            self.bytes[start..end]
                .try_into()
                .map_err(|_| VisualizeError::InvalidFormat)?,
        ) as usize;
        Ok(output_size)
    }

    pub fn layer_meta_size(&self) -> usize {
        self.layer_meta_size
    }

    pub fn activation(&self) -> Result<u8, VisualizeError> {
        let pos = self.meta_offset + 16;
        if pos >= self.bytes.len() {
            return Err(VisualizeError::OutOfBounds);
        }
        Ok(self.bytes[pos])
    }

    pub fn input_count(&self) -> Result<usize, VisualizeError> {
        let start = self.meta_offset;
        let end = start + 4;
        if end > self.bytes.len() {
            return Err(VisualizeError::OutOfBounds);
        }
        let input_size = u32::from_le_bytes(
            self.bytes[start..end]
                .try_into()
                .map_err(|_| VisualizeError::InvalidFormat)?,
        ) as usize;
        Ok(input_size)
    }

    pub fn neuron(&self, j: usize) -> Result<NeuronView<'a>, VisualizeError> {
        let output_size = self.neuron_count()?;
        if j >= output_size {
            return Err(VisualizeError::OutOfBounds);
        }
        let wo_start = self.meta_offset + 8;
        let wo_end = wo_start + 4;
        if wo_end > self.bytes.len() {
            return Err(VisualizeError::OutOfBounds);
        }
        let weight_base = u32::from_le_bytes(
            self.bytes[wo_start..wo_end]
                .try_into()
                .map_err(|_| VisualizeError::InvalidFormat)?,
        ) as usize;

        let weight_start = self
            .weights_offset
            .checked_add(weight_base)
            .ok_or(VisualizeError::InvalidFormat)?;
        if weight_start > self.bytes.len() {
            return Err(VisualizeError::OutOfBounds);
        }

        let remaining_bytes = self.bytes.len().saturating_sub(weight_start);
        let weight_count = remaining_bytes / 4;

        Ok(NeuronView {
            bytes: self.bytes,
            weight_start,
            weight_count,
        })
    }
}

impl<'a> NeuronView<'a> {
    pub fn weight_count(&self) -> usize {
        self.weight_count
    }

    pub fn weight_at(&self, idx: usize) -> Result<f32, VisualizeError> {
        if idx >= self.weight_count {
            return Err(VisualizeError::OutOfBounds);
        }
        let bstart = self.weight_start + idx * 4;
        let bend = bstart + 4;
        if bend > self.bytes.len() {
            return Err(VisualizeError::OutOfBounds);
        }
        let w = f32::from_le_bytes(
            self.bytes[bstart..bend]
                .try_into()
                .map_err(|_| VisualizeError::InvalidFormat)?,
        );
        Ok(w)
    }

    pub fn weight_bytes(&self) -> &'a [u8] {
        let end = core::cmp::min(self.weight_start + self.weight_count * 4, self.bytes.len());
        &self.bytes[self.weight_start..end]
    }
}
