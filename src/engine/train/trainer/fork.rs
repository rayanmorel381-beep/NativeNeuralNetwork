use crate::engine::rnn_flow::RnnFlowError;
use super::entry::ParsedLayout;
use super::train_sgd::sgd_on_decrypted;
use core::sync::atomic::{fence, AtomicU32, Ordering};

pub(crate) struct SgdPoolConfig {
    pub(crate) learning_rate: f32,
    pub(crate) gradient_clip: Option<f32>,
    pub(crate) filler: fn(usize, usize, &mut [f32], &mut [f32]),
    pub(crate) input_size: usize,
    pub(crate) output_size: usize,
    pub(crate) chunk_size: usize,
    pub(crate) core_mask: usize,
}

const CTRL_STRIDE: usize = 64;
const STATE_IDLE: u32 = 0;
const STATE_WORK: u32 = 1;
const STATE_DONE: u32 = 2;
const STATE_EXIT: u32 = 0xFF;
const RESULT_SLOT: usize = 12;

unsafe fn futex_wait(addr: *const u32, expected: u32) {
    let atomic = &*(addr as *const AtomicU32);
    while atomic.load(Ordering::Acquire) == expected {
        core::hint::spin_loop();
    }
}

unsafe fn futex_wake(_addr: *const u32, _count: u32) {}

unsafe fn vst(ptr: *mut u32, val: u32) { core::ptr::write_volatile(ptr, val); }
unsafe fn vld(ptr: *const u32) -> u32 { core::ptr::read_volatile(ptr) }

pub(crate) struct SgdWorkerPool {
    ctrl: *mut u8,
    ctrl_size: usize,
    arena: *mut u8,
    arena_size: usize,
    pids: [i64; 12],
    n_workers: usize,
    slot_size: usize,
    model_len: usize,
}

impl SgdWorkerPool {
    pub(crate) fn empty() -> Self {
        Self {
            ctrl: core::ptr::null_mut(), ctrl_size: 0,
            arena: core::ptr::null_mut(), arena_size: 0,
            pids: [0; 12], n_workers: 0, slot_size: 0, model_len: 0,
        }
    }
    pub(crate) fn is_active(&self) -> bool { self.n_workers > 0 }
}

fn worker_loop(
    idx: usize,
    ctrl: *mut u8,
    arena: *mut u8,
    slot_size: usize,
    model_len: usize,
    layout: &ParsedLayout,
    cfg: &SgdPoolConfig,
) -> ! {
    let state_ptr = unsafe { ctrl.add(idx * CTRL_STRIDE) as *mut u32 };
    let sample_entry_bytes = (cfg.input_size + cfg.output_size) * 4;
    let ref_entry_size = core::mem::size_of::<(&[f32], &[f32])>();
    let sample_base = unsafe { arena.add(idx * slot_size + model_len + RESULT_SLOT) };
    let raw_ref_off = model_len + RESULT_SLOT + cfg.chunk_size * sample_entry_bytes;
    let ref_off = (raw_ref_off + 15) & !15;
    let ref_base = unsafe { arena.add(idx * slot_size + ref_off) };
    let lane_offset = idx * cfg.chunk_size;

    loop {
        let st = unsafe { vld(state_ptr) };
        if st == STATE_EXIT { crate::engine::runtime::hardware::exit(0); }
        if st != STATE_WORK {
            unsafe { futex_wait(state_ptr, st); }
            continue;
        }
        fence(Ordering::Acquire);
        let base_iter = unsafe { vld(ctrl.add(idx * CTRL_STRIDE + 4) as *const u32) } as usize;
        let s_count = unsafe { vld(ctrl.add(idx * CTRL_STRIDE + 8) as *const u32) } as usize;

        for lane in 0..s_count {
            let offset = lane * sample_entry_bytes;
            unsafe {
                let inp = core::slice::from_raw_parts_mut(sample_base.add(offset) as *mut f32, cfg.input_size);
                let tgt = core::slice::from_raw_parts_mut(sample_base.add(offset + cfg.input_size * 4) as *mut f32, cfg.output_size);
                (cfg.filler)(base_iter, lane_offset + lane, inp, tgt);
            }
        }

        for i in 0..s_count {
            let offset = i * sample_entry_bytes;
            unsafe {
                let inp = core::slice::from_raw_parts(sample_base.add(offset) as *const f32, cfg.input_size);
                let tgt = core::slice::from_raw_parts(sample_base.add(offset + cfg.input_size * 4) as *const f32, cfg.output_size);
                let slot_ptr = ref_base.add(i * ref_entry_size) as *mut (&[f32], &[f32]);
                core::ptr::write(slot_ptr, (inp, tgt));
            }
        }

        let samples = unsafe { core::slice::from_raw_parts(ref_base as *const (&[f32], &[f32]), s_count) };
        let slot = unsafe { core::slice::from_raw_parts_mut(arena.add(idx * slot_size), model_len) };

        let result = sgd_on_decrypted(slot, samples, cfg.learning_rate, cfg.gradient_clip, layout);

        let res = unsafe { arena.add(idx * slot_size + model_len) };
        if let Ok((loss, cnt)) = result {
            unsafe {
                core::ptr::copy_nonoverlapping(loss.to_le_bytes().as_ptr(), res, 8);
                core::ptr::copy_nonoverlapping((cnt as u32).to_le_bytes().as_ptr(), res.add(8), 4);
            }
        }

        fence(Ordering::Release);
        unsafe {
            vst(state_ptr, STATE_DONE);
            futex_wake(state_ptr, 1);
        }
    }
}

pub(crate) fn sgd_pool_create(
    n_workers: usize,
    model_len: usize,
    layout: &ParsedLayout,
    cfg: &SgdPoolConfig,
    initial_model: &[u8],
) -> SgdWorkerPool {
    if n_workers < 2 {
        return SgdWorkerPool::empty();
    }

    let sample_entry_bytes = (cfg.input_size + cfg.output_size) * 4;
    let ref_entry_size = core::mem::size_of::<(&[f32], &[f32])>();
    let sample_buf_size = cfg.chunk_size * sample_entry_bytes;
    let ref_buf_size = cfg.chunk_size * ref_entry_size;
    let raw_ref_off = model_len + RESULT_SLOT + sample_buf_size;
    let aligned_ref_off = (raw_ref_off + 15) & !15;
    let slot_size = aligned_ref_off + ref_buf_size;

    let guard = crate::engine::runtime::ConsumptionGuard::detect();
    let budget = guard.ram_budget_bytes();
    let mut n_workers = n_workers;
    if budget > 0 && slot_size > 0 {
        let fit = (budget / slot_size).max(1);
        n_workers = n_workers.min(fit);
    }
    if n_workers < 2 {
        return SgdWorkerPool::empty();
    }

    let arena_size = slot_size * n_workers;

    let arena = crate::engine::runtime::hardware::mmap_shared_anon(arena_size);
    if arena.is_null() { return SgdWorkerPool::empty(); }

    let ctrl_size = n_workers * CTRL_STRIDE;
    let ctrl = crate::engine::runtime::hardware::mmap_shared_anon(ctrl_size);
    if ctrl.is_null() {
        crate::engine::runtime::hardware::munmap(arena, arena_size);
        return SgdWorkerPool::empty();
    }
    unsafe { core::ptr::write_bytes(ctrl, 0, ctrl_size); }

    for i in 0..n_workers {
        unsafe {
            core::ptr::copy_nonoverlapping(initial_model.as_ptr(), arena.add(i * slot_size), model_len);
        }
    }

    let mut pool = SgdWorkerPool {
        ctrl, ctrl_size, arena, arena_size,
        pids: [0i64; 12], n_workers: 0, slot_size, model_len,
    };

    for i in 0..n_workers {
        let pid = crate::engine::runtime::hardware::fork();
        if pid < 0 {
            for j in 0..pool.n_workers {
                unsafe {
                    vst(ctrl.add(j * CTRL_STRIDE) as *mut u32, STATE_EXIT);
                    futex_wake(ctrl.add(j * CTRL_STRIDE) as *const u32, 1);
                }
            }
            for j in 0..pool.n_workers { crate::engine::runtime::hardware::waitpid(pool.pids[j]); }
            crate::engine::runtime::hardware::munmap(ctrl, ctrl_size);
            crate::engine::runtime::hardware::munmap(arena, arena_size);
            return SgdWorkerPool::empty();
        }
        if pid == 0 {
            if cfg.core_mask != 0 {
                crate::engine::runtime::hardware::set_affinity(cfg.core_mask);
            }
            worker_loop(i, ctrl, arena, slot_size, model_len, layout, cfg);
        }
        pool.pids[i] = pid;
        pool.n_workers += 1;
    }

    pool
}

pub(crate) fn sgd_pool_start_n(
    pool: &SgdWorkerPool,
    base_iteration: usize,
    samples_per_worker: usize,
    active: usize,
) {
    let n = active.min(pool.n_workers).max(1);
    for i in 0..n {
        unsafe {
            core::ptr::write_bytes(pool.arena.add(i * pool.slot_size + pool.model_len), 0, RESULT_SLOT);
            vst(pool.ctrl.add(i * CTRL_STRIDE + 4) as *mut u32, base_iteration as u32);
            vst(pool.ctrl.add(i * CTRL_STRIDE + 8) as *mut u32, samples_per_worker as u32);
            fence(Ordering::Release);
            vst(pool.ctrl.add(i * CTRL_STRIDE) as *mut u32, STATE_WORK);
            futex_wake(pool.ctrl.add(i * CTRL_STRIDE) as *const u32, 1);
        }
    }
}

pub(crate) fn sgd_pool_finish_n(
    pool: &SgdWorkerPool,
    bytes: &mut [u8],
    layout: &ParsedLayout,
    active: usize,
) -> Result<(f64, usize), RnnFlowError> {
    let model_len = pool.model_len;
    let n_workers = active.min(pool.n_workers).max(1);
    let slot_size = pool.slot_size;
    let arena = pool.arena;

    for i in 0..n_workers {
        let addr = unsafe { pool.ctrl.add(i * CTRL_STRIDE) as *const u32 };
        loop {
            let st = unsafe { vld(addr) };
            if st == STATE_DONE { break; }
            unsafe { futex_wait(addr, STATE_WORK); }
        }
    }

    let mut loss_acc = crate::engine::eval::metrics::RunningMeanF64::new();
    for i in 0..n_workers {
        let res = unsafe { arena.add(i * slot_size + model_len) };
        let loss = f64::from_le_bytes(unsafe {
            [*res, *res.add(1), *res.add(2), *res.add(3),
             *res.add(4), *res.add(5), *res.add(6), *res.add(7)]
        });
        let cnt = u32::from_le_bytes(unsafe {
            [*res.add(8), *res.add(9), *res.add(10), *res.add(11)]
        }) as usize;
        loss_acc.merge(crate::engine::eval::metrics::RunningMeanF64 { sum: loss, count: cnt as u64 });
    }

    match layout.dtype {
        0 => {
            for byte_off in (layout.w_start..layout.w_end).step_by(4) {
                let mut sum = 0.0f64;
                for w in 0..n_workers {
                    let p = unsafe { arena.add(w * slot_size + byte_off) };
                    let v = f32::from_le_bytes(unsafe { [*p, *p.add(1), *p.add(2), *p.add(3)] });
                    sum += v as f64;
                }
                let avg = (sum / n_workers as f64) as f32;
                bytes[byte_off..byte_off + 4].copy_from_slice(&avg.to_le_bytes());
            }
            for byte_off in (layout.b_start..layout.b_end).step_by(4) {
                let mut sum = 0.0f64;
                for w in 0..n_workers {
                    let p = unsafe { arena.add(w * slot_size + byte_off) };
                    let v = f32::from_le_bytes(unsafe { [*p, *p.add(1), *p.add(2), *p.add(3)] });
                    sum += v as f64;
                }
                let avg = (sum / n_workers as f64) as f32;
                bytes[byte_off..byte_off + 4].copy_from_slice(&avg.to_le_bytes());
            }
        }
        1 => {
            for byte_off in (layout.w_start..layout.w_end).step_by(8) {
                let mut sum = 0.0f64;
                for w in 0..n_workers {
                    let p = unsafe { arena.add(w * slot_size + byte_off) };
                    let v = f64::from_le_bytes(unsafe {
                        [*p, *p.add(1), *p.add(2), *p.add(3),
                         *p.add(4), *p.add(5), *p.add(6), *p.add(7)]
                    });
                    sum += v;
                }
                let avg = sum / n_workers as f64;
                bytes[byte_off..byte_off + 8].copy_from_slice(&avg.to_le_bytes());
            }
            for byte_off in (layout.b_start..layout.b_end).step_by(8) {
                let mut sum = 0.0f64;
                for w in 0..n_workers {
                    let p = unsafe { arena.add(w * slot_size + byte_off) };
                    let v = f64::from_le_bytes(unsafe {
                        [*p, *p.add(1), *p.add(2), *p.add(3),
                         *p.add(4), *p.add(5), *p.add(6), *p.add(7)]
                    });
                    sum += v;
                }
                let avg = sum / n_workers as f64;
                bytes[byte_off..byte_off + 8].copy_from_slice(&avg.to_le_bytes());
            }
        }
        _ => {}
    }

    for i in 0..pool.n_workers {
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), arena.add(i * slot_size), model_len);
            vst(pool.ctrl.add(i * CTRL_STRIDE) as *mut u32, STATE_IDLE);
        }
    }

    Ok((loss_acc.sum, loss_acc.count as usize))
}

pub(crate) fn sgd_pool_start(
    pool: &SgdWorkerPool,
    base_iteration: usize,
    samples_per_worker: usize,
) {
    sgd_pool_start_n(pool, base_iteration, samples_per_worker, pool.n_workers);
}

pub(crate) fn sgd_pool_finish(
    pool: &SgdWorkerPool,
    bytes: &mut [u8],
    layout: &ParsedLayout,
) -> Result<(f64, usize), RnnFlowError> {
    sgd_pool_finish_n(pool, bytes, layout, pool.n_workers)
}

pub(crate) fn sgd_pool_destroy(pool: &SgdWorkerPool) {
    if !pool.is_active() { return; }
    for i in 0..pool.n_workers {
        unsafe {
            vst(pool.ctrl.add(i * CTRL_STRIDE) as *mut u32, STATE_EXIT);
            futex_wake(pool.ctrl.add(i * CTRL_STRIDE) as *const u32, 1);
        }
    }
    for i in 0..pool.n_workers {
        crate::engine::runtime::hardware::waitpid(pool.pids[i]);
    }
    crate::engine::runtime::hardware::munmap(pool.ctrl, pool.ctrl_size);
    crate::engine::runtime::hardware::munmap(pool.arena, pool.arena_size);
}
