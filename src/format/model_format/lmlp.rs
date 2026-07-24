pub(crate) const LMLP_GRAPH_VERSION: u16 = 1;
pub(crate) const LMLP_HEADER_SIZE: usize = 12;
pub(crate) const LMLP_NODE_SIZE: usize = 20;
pub(crate) const LMLP_EDGE_SIZE: usize = 32;
pub(crate) const LMLP_IO_HEADER_SIZE: usize = 12;

#[derive(Clone, Copy, PartialEq)]
pub(crate) struct LmlpNodeView {
    pub kind_transform: u8,
    pub memory_flags: u8,
    pub norm_flags: u8,
    pub activation: u8,
    pub in_dim: u32,
    pub out_dim: u32,
    pub theta_offset: u32,
    pub theta_len: u32,
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) struct LmlpEdgeView {
    pub from: u32,
    pub to: u32,
    pub default_mode: u8,
    pub w_offset: u32,
    pub w_len: u32,
    pub delay_hint: u32,
    pub phase: f32,
    pub decay: f32,
}

pub(crate) struct LmlpGraphView<'a> {
    pub version: u16,
    pub dtype_tag: u8,
    pub node_count: u32,
    pub edge_count: u32,
    pub vocab_size: u32,
    nodes: &'a [u8],
    edges: &'a [u8],
    in_ids: &'a [u8],
    out_ids: &'a [u8],
}

fn read_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

fn read_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn read_f32(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

pub(crate) fn parse_lmlp_graph(bytes: &[u8]) -> Option<LmlpGraphView<'_>> {
    if bytes.len() < LMLP_HEADER_SIZE {
        return None;
    }
    let version = read_u16(bytes, 0);
    let dtype_tag = bytes[2];
    if dtype_tag > 1 {
        return None;
    }
    let node_count = read_u32(bytes, 4);
    let edge_count = read_u32(bytes, 8);

    let nodes_len = (node_count as usize).checked_mul(LMLP_NODE_SIZE)?;
    let edges_len = (edge_count as usize).checked_mul(LMLP_EDGE_SIZE)?;

    let nodes_start = LMLP_HEADER_SIZE;
    let nodes_end = nodes_start.checked_add(nodes_len)?;
    let edges_end = nodes_end.checked_add(edges_len)?;
    let io_end = edges_end.checked_add(LMLP_IO_HEADER_SIZE)?;
    if bytes.len() < io_end {
        return None;
    }

    let in_count = read_u32(bytes, edges_end);
    let out_count = read_u32(bytes, edges_end + 4);
    let vocab_size = read_u32(bytes, edges_end + 8);

    let in_len = (in_count as usize).checked_mul(4)?;
    let out_len = (out_count as usize).checked_mul(4)?;
    let in_start = io_end;
    let in_end = in_start.checked_add(in_len)?;
    let out_end = in_end.checked_add(out_len)?;
    if bytes.len() < out_end {
        return None;
    }

    Some(LmlpGraphView {
        version,
        dtype_tag,
        node_count,
        edge_count,
        vocab_size,
        nodes: &bytes[nodes_start..nodes_end],
        edges: &bytes[nodes_end..edges_end],
        in_ids: &bytes[in_start..in_end],
        out_ids: &bytes[in_end..out_end],
    })
}

impl<'a> LmlpGraphView<'a> {
    pub(crate) fn node(&self, index: usize) -> Option<LmlpNodeView> {
        let off = index.checked_mul(LMLP_NODE_SIZE)?;
        let end = off.checked_add(LMLP_NODE_SIZE)?;
        if end > self.nodes.len() {
            return None;
        }
        let n = &self.nodes[off..end];
        Some(LmlpNodeView {
            kind_transform: n[0],
            memory_flags: n[1],
            norm_flags: n[2],
            activation: n[3],
            in_dim: read_u32(n, 4),
            out_dim: read_u32(n, 8),
            theta_offset: read_u32(n, 12),
            theta_len: read_u32(n, 16),
        })
    }

    pub(crate) fn edge(&self, index: usize) -> Option<LmlpEdgeView> {
        let off = index.checked_mul(LMLP_EDGE_SIZE)?;
        let end = off.checked_add(LMLP_EDGE_SIZE)?;
        if end > self.edges.len() {
            return None;
        }
        let e = &self.edges[off..end];
        Some(LmlpEdgeView {
            from: read_u32(e, 0),
            to: read_u32(e, 4),
            default_mode: e[8],
            w_offset: read_u32(e, 12),
            w_len: read_u32(e, 16),
            delay_hint: read_u32(e, 20),
            phase: read_f32(e, 24),
            decay: read_f32(e, 28),
        })
    }

    pub(crate) fn in_id(&self, index: usize) -> Option<u32> {
        let off = index.checked_mul(4)?;
        let end = off.checked_add(4)?;
        if end > self.in_ids.len() {
            return None;
        }
        Some(read_u32(self.in_ids, off))
    }

    pub(crate) fn out_id(&self, index: usize) -> Option<u32> {
        let off = index.checked_mul(4)?;
        let end = off.checked_add(4)?;
        if end > self.out_ids.len() {
            return None;
        }
        Some(read_u32(self.out_ids, off))
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lmlp_graph_into(
    version: u16,
    dtype_tag: u8,
    nodes: &[LmlpNodeView],
    edges: &[LmlpEdgeView],
    in_ids: &[u32],
    out_ids: &[u32],
    vocab_size: u32,
    out: &mut [u8],
) -> Option<usize> {
    let node_count = u32::try_from(nodes.len()).ok()?;
    let edge_count = u32::try_from(edges.len()).ok()?;
    let in_count = u32::try_from(in_ids.len()).ok()?;
    let out_count = u32::try_from(out_ids.len()).ok()?;

    let nodes_len = nodes.len().checked_mul(LMLP_NODE_SIZE)?;
    let edges_len = edges.len().checked_mul(LMLP_EDGE_SIZE)?;
    let in_len = in_ids.len().checked_mul(4)?;
    let out_len = out_ids.len().checked_mul(4)?;
    let total = LMLP_HEADER_SIZE
        .checked_add(nodes_len)?
        .checked_add(edges_len)?
        .checked_add(LMLP_IO_HEADER_SIZE)?
        .checked_add(in_len)?
        .checked_add(out_len)?;
    if out.len() < total {
        return None;
    }

    out[0..2].copy_from_slice(&version.to_le_bytes());
    out[2] = dtype_tag;
    out[3] = 0;
    out[4..8].copy_from_slice(&node_count.to_le_bytes());
    out[8..12].copy_from_slice(&edge_count.to_le_bytes());

    let mut cur = LMLP_HEADER_SIZE;
    for n in nodes {
        let s = &mut out[cur..cur + LMLP_NODE_SIZE];
        s[0] = n.kind_transform;
        s[1] = n.memory_flags;
        s[2] = n.norm_flags;
        s[3] = n.activation;
        s[4..8].copy_from_slice(&n.in_dim.to_le_bytes());
        s[8..12].copy_from_slice(&n.out_dim.to_le_bytes());
        s[12..16].copy_from_slice(&n.theta_offset.to_le_bytes());
        s[16..20].copy_from_slice(&n.theta_len.to_le_bytes());
        cur += LMLP_NODE_SIZE;
    }
    for e in edges {
        let s = &mut out[cur..cur + LMLP_EDGE_SIZE];
        s[0..4].copy_from_slice(&e.from.to_le_bytes());
        s[4..8].copy_from_slice(&e.to.to_le_bytes());
        s[8] = e.default_mode;
        s[9] = 0;
        s[10] = 0;
        s[11] = 0;
        s[12..16].copy_from_slice(&e.w_offset.to_le_bytes());
        s[16..20].copy_from_slice(&e.w_len.to_le_bytes());
        s[20..24].copy_from_slice(&e.delay_hint.to_le_bytes());
        s[24..28].copy_from_slice(&e.phase.to_le_bytes());
        s[28..32].copy_from_slice(&e.decay.to_le_bytes());
        cur += LMLP_EDGE_SIZE;
    }
    out[cur..cur + 4].copy_from_slice(&in_count.to_le_bytes());
    out[cur + 4..cur + 8].copy_from_slice(&out_count.to_le_bytes());
    out[cur + 8..cur + 12].copy_from_slice(&vocab_size.to_le_bytes());
    cur += LMLP_IO_HEADER_SIZE;
    for &id in in_ids {
        out[cur..cur + 4].copy_from_slice(&id.to_le_bytes());
        cur += 4;
    }
    for &id in out_ids {
        out[cur..cur + 4].copy_from_slice(&id.to_le_bytes());
        cur += 4;
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lmlp_graph_round_trip() {
        let nodes = [
            LmlpNodeView {
                kind_transform: 1,
                memory_flags: 0,
                norm_flags: 0,
                activation: 2,
                in_dim: 32,
                out_dim: 64,
                theta_offset: 0,
                theta_len: 2048,
            },
            LmlpNodeView {
                kind_transform: 4,
                memory_flags: 1,
                norm_flags: 1,
                activation: 0,
                in_dim: 64,
                out_dim: 8,
                theta_offset: 2048,
                theta_len: 512,
            },
        ];
        let edges = [LmlpEdgeView {
            from: 0,
            to: 1,
            default_mode: 0,
            w_offset: 0,
            w_len: 2048,
            delay_hint: 0,
            phase: 0.0,
            decay: 1.0,
        }];
        let in_ids = [0u32];
        let out_ids = [1u32];

        let mut buf = [0u8; 256];
        let used =
            encode_lmlp_graph_into(LMLP_GRAPH_VERSION, 0, &nodes, &edges, &in_ids, &out_ids, 100, &mut buf)
                .unwrap();

        let view = parse_lmlp_graph(&buf[..used]).unwrap();
        assert_eq!(view.version, LMLP_GRAPH_VERSION);
        assert_eq!(view.dtype_tag, 0);
        assert_eq!(view.node_count, 2);
        assert_eq!(view.edge_count, 1);
        assert_eq!(view.vocab_size, 100);
        assert!(view.node(0).unwrap() == nodes[0]);
        assert!(view.node(1).unwrap() == nodes[1]);
        assert!(view.edge(0).unwrap() == edges[0]);
        assert_eq!(view.in_id(0).unwrap(), 0);
        assert_eq!(view.out_id(0).unwrap(), 1);
        assert!(view.node(2).is_none());
        assert!(parse_lmlp_graph(&buf[..used - 1]).is_none());
    }
}
