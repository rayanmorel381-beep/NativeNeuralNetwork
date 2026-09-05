use crate::x86::linux::syscall;
use super::drm::{iowr, DRM_RADEON_CS, RADEON_GEM_DOMAIN_GTT};

const RADEON_CHUNK_ID_RELOCS: u32 = 0x01;
const RADEON_CHUNK_ID_IB: u32 = 0x02;
const RADEON_CHUNK_ID_FLAGS: u32 = 0x03;

const RADEON_CS_USE_VM: u32 = 0x02;
const RADEON_CS_RING_GFX: u32 = 0;

const RELOC_DWORDS: u32 = 4;
const MAX_RELOCS: usize = 8;

#[repr(C)]
struct CsChunk {
    chunk_id: u32,
    length_dw: u32,
    chunk_data: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CsReloc {
    handle: u32,
    read_domains: u32,
    write_domain: u32,
    flags: u32,
}

#[repr(C)]
struct DrmRadeonCs {
    num_chunks: u32,
    cs_id: u32,
    chunks: u64,
    gart_limit: u64,
    vram_limit: u64,
}

pub(crate) struct BufferRef {
    pub handle: u32,
    pub write: bool,
    pub domain: u32,
}

pub(crate) fn submit(fd: i64, ib_words: &[u32], buffers: &[BufferRef]) -> bool {
    if buffers.len() > MAX_RELOCS {
        return false;
    }
    let mut relocs = [CsReloc {
        handle: 0,
        read_domains: 0,
        write_domain: 0,
        flags: 0,
    }; MAX_RELOCS];
    let mut i = 0;
    while i < buffers.len() {
        let write = buffers[i].write;
        let domain = buffers[i].domain;
        relocs[i] = CsReloc {
            handle: buffers[i].handle,
            read_domains: if write { 0 } else { domain },
            write_domain: if write { domain } else { 0 },
            flags: 0,
        };
        i += 1;
    }
    let flags = [RADEON_CS_USE_VM, RADEON_CS_RING_GFX];

    let chunks = [
        CsChunk {
            chunk_id: RADEON_CHUNK_ID_RELOCS,
            length_dw: buffers.len() as u32 * RELOC_DWORDS,
            chunk_data: relocs.as_ptr() as u64,
        },
        CsChunk {
            chunk_id: RADEON_CHUNK_ID_IB,
            length_dw: ib_words.len() as u32,
            chunk_data: ib_words.as_ptr() as u64,
        },
        CsChunk {
            chunk_id: RADEON_CHUNK_ID_FLAGS,
            length_dw: flags.len() as u32,
            chunk_data: flags.as_ptr() as u64,
        },
    ];
    let chunk_ptrs = [
        &chunks[0] as *const CsChunk as u64,
        &chunks[1] as *const CsChunk as u64,
        &chunks[2] as *const CsChunk as u64,
    ];

    let mut cs = DrmRadeonCs {
        num_chunks: chunks.len() as u32,
        cs_id: 0,
        chunks: chunk_ptrs.as_ptr() as u64,
        gart_limit: 0,
        vram_limit: 0,
    };
    let code = iowr(DRM_RADEON_CS, core::mem::size_of::<DrmRadeonCs>());
    syscall::sys_ioctl(fd, code, &mut cs as *mut DrmRadeonCs as usize) == 0
}

pub(crate) fn submit_compute(
    fd: i64,
    ib_words: &[u32],
    shader_handle: u32,
    input_handle: u32,
    output_handle: u32,
) -> bool {
    let buffers = [
        BufferRef { handle: shader_handle, write: false, domain: RADEON_GEM_DOMAIN_GTT },
        BufferRef { handle: input_handle, write: false, domain: RADEON_GEM_DOMAIN_GTT },
        BufferRef { handle: output_handle, write: true, domain: RADEON_GEM_DOMAIN_GTT },
    ];
    submit(fd, ib_words, &buffers)
}
