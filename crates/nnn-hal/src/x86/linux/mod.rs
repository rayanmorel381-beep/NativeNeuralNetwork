pub(super) mod cpu;
pub(super) mod gpu;
pub(super) mod lpu;
pub(super) mod ram;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub(crate) mod syscall;
pub(super) mod tpu;

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn live_cpu_count() -> usize {
    let mut mask = [0usize; 16];
    let rc = syscall::sys_sched_getaffinity(0, &mut mask);
    if rc <= 0 {
        return cpu_count_proc().unwrap_or(1);
    }
    let mut count = 0usize;
    for w in mask.iter() {
        count = count.saturating_add(w.count_ones() as usize);
    }
    count.max(1)
}

#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn live_cpu_count() -> usize { 1 }

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
fn cpu_count_proc() -> Option<usize> {
    let path = b"/sys/devices/system/cpu/online\0";
    let mut buf = [0u8; 256];
    let n = read_small_file(path, &mut buf)?;
    Some(parse_cpu_list(&buf[..n]))
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
fn parse_cpu_list(bytes: &[u8]) -> usize {
    let mut count = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let mut start = 0usize;
        let mut has = false;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            start = start.saturating_mul(10).saturating_add((bytes[i] - b'0') as usize);
            has = true;
            i += 1;
        }
        if !has { i += 1; continue; }
        if i < bytes.len() && bytes[i] == b'-' {
            i += 1;
            let mut end = 0usize;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                end = end.saturating_mul(10).saturating_add((bytes[i] - b'0') as usize);
                i += 1;
            }
            count = count.saturating_add(end.saturating_sub(start).saturating_add(1));
        } else {
            count = count.saturating_add(1);
        }
        while i < bytes.len() && (bytes[i] == b',' || bytes[i] == b' ' || bytes[i] == b'\n') { i += 1; }
    }
    count
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn live_ram() -> (u64, u64, usize) {
    let (mem_total_kb, mem_available_kb) = parse_meminfo();
    if mem_total_kb > 0 {
        let available = if mem_available_kb > 0 {
            mem_available_kb.saturating_mul(1024)
        } else {
            sysinfo_available_bytes()
        };
        return (mem_total_kb.saturating_mul(1024), available, 4096);
    }
    let mut info = new_sysinfo();
    let rc = syscall::sys_sysinfo(&mut info);
    if rc < 0 || info.mem_unit == 0 { return (0, 0, 4096); }
    let unit = info.mem_unit as u64;
    let total = (info.totalram as u64).saturating_mul(unit);
    let free = (info.freeram as u64).saturating_mul(unit);
    let buffers = (info.bufferram as u64).saturating_mul(unit);
    (total, free.saturating_add(buffers), 4096)
}

#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn live_ram() -> (u64, u64, usize) { (0, 0, 4096) }

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
fn new_sysinfo() -> syscall::SysInfo {
    syscall::SysInfo {
        uptime: 0, loads: [0; 3], totalram: 0, freeram: 0, sharedram: 0,
        bufferram: 0, totalswap: 0, freeswap: 0, procs: 0, _pad1: 0,
        totalhigh: 0, freehigh: 0, mem_unit: 0, _pad2: 0,
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
fn sysinfo_available_bytes() -> u64 {
    let mut info = new_sysinfo();
    let rc = syscall::sys_sysinfo(&mut info);
    if rc < 0 || info.mem_unit == 0 { return 0; }
    let unit = info.mem_unit as u64;
    let free = (info.freeram as u64).saturating_mul(unit);
    let buffers = (info.bufferram as u64).saturating_mul(unit);
    free.saturating_add(buffers)
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
fn parse_meminfo() -> (u64, u64) {
    let path = b"/proc/meminfo\0";
    let mut buf = [0u8; 4096];
    let n = match read_small_file(path, &mut buf) { Some(v) => v, None => return (0, 0) };
    let bytes = &buf[..n];
    (find_kb_field(bytes, b"MemTotal:"), find_kb_field(bytes, b"MemAvailable:"))
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
fn find_kb_field(bytes: &[u8], key: &[u8]) -> u64 {
    let mut i = 0usize;
    while i + key.len() <= bytes.len() {
        if &bytes[i..i + key.len()] == key {
            let mut j = i + key.len();
            while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') { j += 1; }
            let mut value: u64 = 0;
            let mut has = false;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                value = value.saturating_mul(10).saturating_add((bytes[j] - b'0') as u64);
                has = true;
                j += 1;
            }
            return if has { value } else { 0 };
        }
        while i < bytes.len() && bytes[i] != b'\n' { i += 1; }
        i += 1;
    }
    0
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
fn read_small_file(path: &[u8], buf: &mut [u8]) -> Option<usize> {
    let fd = syscall::sys_open(path, syscall::O_RDONLY, 0);
    if fd < 0 { return None; }
    let mut total = 0usize;
    while total < buf.len() {
        let n = syscall::sys_pread64_fd(fd, &mut buf[total..], total as u64);
        if n < 0 { let _ = syscall::sys_close(fd); return None; }
        if n == 0 { break; }
        total = total.saturating_add(n as usize);
    }
    let _ = syscall::sys_close(fd);
    Some(total)
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
fn read_cpufreq_khz(cpu: usize, field: &[u8]) -> u64 {
    let mut path = [0u8; 96];
    let prefix = b"/sys/devices/system/cpu/cpu";
    let mid = b"/cpufreq/";
    let mut w = 0usize;
    for &b in prefix { path[w] = b; w += 1; }
    if cpu == 0 {
        path[w] = b'0';
        w += 1;
    } else {
        let mut digits = [0u8; 20];
        let mut d = 0usize;
        let mut n = cpu;
        while n > 0 {
            digits[d] = b'0' + (n % 10) as u8;
            d += 1;
            n /= 10;
        }
        while d > 0 {
            d -= 1;
            path[w] = digits[d];
            w += 1;
        }
    }
    for &b in mid { path[w] = b; w += 1; }
    for &b in field { path[w] = b; w += 1; }
    path[w] = 0;
    let mut buf = [0u8; 32];
    let n = match read_small_file(&path[..=w], &mut buf) {
        Some(v) => v,
        None => return 0,
    };
    let mut value = 0u64;
    let mut has = false;
    for &b in &buf[..n] {
        if b.is_ascii_digit() {
            value = value.saturating_mul(10).saturating_add((b - b'0') as u64);
            has = true;
        } else {
            break;
        }
    }
    if has { value } else { 0 }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn live_cpu_freq_mhz(cores: usize) -> (u32, u32) {
    let n = cores.clamp(1, 256);
    let mut sum_khz = 0u64;
    let mut counted = 0u64;
    let mut max_khz = 0u64;
    for c in 0..n {
        let cur = read_cpufreq_khz(c, b"scaling_cur_freq");
        if cur > 0 {
            sum_khz = sum_khz.saturating_add(cur);
            counted += 1;
        }
        let cmax = read_cpufreq_khz(c, b"cpuinfo_max_freq");
        if cmax > max_khz {
            max_khz = cmax;
        }
    }
    let avg_mhz = if counted > 0 {
        (sum_khz / counted / 1000) as u32
    } else {
        0
    };
    let max_mhz = if max_khz > 0 {
        (max_khz / 1000) as u32
    } else {
        avg_mhz
    };
    (avg_mhz, max_mhz)
}

#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn live_cpu_freq_mhz(_cores: usize) -> (u32, u32) { (0, 0) }

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn gpu_device_present() -> bool {
    const CANDIDATES: &[&[u8]] = &[
        b"/dev/dri/renderD128\0",
        b"/dev/dri/renderD129\0",
        b"/dev/dri/card0\0",
        b"/dev/dri/card1\0",
        b"/dev/nvidia0\0",
        b"/dev/nvidiactl\0",
        b"/dev/kfd\0",
    ];
    for path in CANDIDATES {
        let fd = syscall::sys_open(path, syscall::O_RDONLY, 0);
        if fd >= 0 { let _ = syscall::sys_close(fd); return true; }
    }
    false
}

#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn gpu_device_present() -> bool { false }

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn radeon_probe() -> Option<(u32, u32, u32)> {
    gpu::amd::probe::probe().map(|p| (p.device_id, p.num_backends, p.num_tile_pipes))
}

#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn radeon_probe() -> Option<(u32, u32, u32)> { None }

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn radeon_gem_selftest(elements: usize) -> Option<(u32, usize, bool, bool)> {
    let fd = gpu::amd::drm::open_render_node()?;
    let bytes = elements * core::mem::size_of::<u32>();
    let result = (|| {
        let mut buf = gpu::amd::gem::alloc_gtt(fd, bytes)?;
        let words = buf.cpu_ptr as *mut u32;
        for i in 0..elements {
            unsafe { core::ptr::write_volatile(words.add(i), (i as u32).wrapping_mul(2654435761)); }
        }
        let mut ok = true;
        for i in 0..elements {
            let got = unsafe { core::ptr::read_volatile(words.add(i)) };
            if got != (i as u32).wrapping_mul(2654435761) {
                ok = false;
                break;
            }
        }
        let va_ok = buf.bind_va(0x100_0000);
        Some((buf.handle, buf.size, ok, va_ok))
    })();
    let _ = syscall::sys_close(fd);
    result
}

#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn radeon_gem_selftest(_elements: usize) -> Option<(u32, usize, bool, bool)> { None }

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn radeon_gpu_compute_selftest(elements: usize) -> Option<(usize, f32, f32, bool)> {
    const SHADER_VA: u64 = 0x100_0000;
    const INPUT_VA: u64 = 0x200_0000;
    const OUTPUT_VA: u64 = 0x300_0000;
    const SCALE: f32 = 3.0;

    let fd = gpu::amd::drm::open_render_node()?;
    let result = (|| {
        let shader_words = &gpu::amd::shader::KERNEL_SCALE_VEC;
        let shader_bytes = shader_words.len() * core::mem::size_of::<u32>();
        let mut shader_buf = gpu::amd::gem::alloc_gtt(fd, shader_bytes)?;
        let shader_dst = shader_buf.cpu_ptr as *mut u32;
        for (i, &word) in shader_words.iter().enumerate() {
            unsafe { core::ptr::write_volatile(shader_dst.add(i), word); }
        }
        if !shader_buf.bind_va(SHADER_VA) {
            return None;
        }

        let data_bytes = elements * core::mem::size_of::<f32>();
        let mut input_buf = gpu::amd::gem::alloc_gtt(fd, data_bytes)?;
        let input_dst = input_buf.cpu_ptr as *mut f32;
        for i in 0..elements {
            unsafe { core::ptr::write_volatile(input_dst.add(i), i as f32 + 1.0); }
        }
        if !input_buf.bind_va(INPUT_VA) {
            return None;
        }

        let mut output_buf = gpu::amd::gem::alloc_gtt(fd, data_bytes)?;
        let output_dst = output_buf.cpu_ptr as *mut f32;
        for i in 0..elements {
            unsafe { core::ptr::write_volatile(output_dst.add(i), 0.0); }
        }
        if !output_buf.bind_va(OUTPUT_VA) {
            return None;
        }

        let params = gpu::amd::pm4::DispatchParams {
            shader_va: shader_buf.va,
            input_desc: gpu::amd::shader::build_buffer_descriptor(input_buf.va, data_bytes as u32),
            output_desc: gpu::amd::shader::build_buffer_descriptor(output_buf.va, data_bytes as u32),
            scale_bits: SCALE.to_bits(),
            threads_x: elements as u32,
        };
        let ib = gpu::amd::pm4::build_dispatch_ib(&params);

        let submitted = gpu::amd::cs::submit_compute(
            fd,
            &ib.words[..ib.len],
            shader_buf.handle,
            input_buf.handle,
            output_buf.handle,
        );
        if !submitted {
            return None;
        }
        output_buf.wait_idle();

        let mut all_match = true;
        let mut first_in = 0.0f32;
        let mut first_out = 0.0f32;
        for i in 0..elements {
            let inp = unsafe { core::ptr::read_volatile(input_dst.add(i)) };
            let out = unsafe { core::ptr::read_volatile(output_dst.add(i)) };
            if i == 0 {
                first_in = inp;
                first_out = out;
            }
            if out != inp * SCALE {
                all_match = false;
            }
        }
        Some((elements, first_in, first_out, all_match))
    })();
    let _ = syscall::sys_close(fd);
    result
}

#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn radeon_gpu_compute_selftest(_elements: usize) -> Option<(usize, f32, f32, bool)> { None }

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn detected_frame_budget_us() -> u64 {
    const PATHS: &[&[u8]] = &[
        b"/sys/class/graphics/fb0/modes\0",
        b"/sys/class/graphics/fb1/modes\0",
    ];
    let mut buf = [0u8; 64];
    for path in PATHS {
        if let Some(n) = read_small_file(path, &mut buf) {
            let s = &buf[..n];
            let mut i = 0usize;
            while i < s.len() && s[i] != b'-' { i += 1; }
            if i < s.len() { i += 1; }
            let mut hz: u64 = 0;
            let mut has = false;
            while i < s.len() && s[i].is_ascii_digit() {
                hz = hz.saturating_mul(10).saturating_add((s[i] - b'0') as u64);
                has = true;
                i += 1;
            }
            if has && hz > 0 {
                return 1_000_000u64 / hz;
            }
        }
    }
    8_333
}

#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn detected_frame_budget_us() -> u64 { 8_333 }

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
fn va_align_up(value: u64) -> u64 {
    (value + 0xFFF) & !0xFFF
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_matmul_dispatch(
    precision: nnn_config::Precision,
    src: &[u8],
    dst: &mut [u8],
    batch_size: usize,
    stride: usize,
    in_size: usize,
    out_size: usize,
    weights: &[u8],
    biases: &[u8],
    activation: u32,
) -> bool {
    match gpu::vendor::default_config().vendor {
        gpu::vendor::Vendor::Amd => radeon_matmul(
            precision, src, dst, batch_size, stride, in_size, out_size, weights, biases, activation,
        ),
        gpu::vendor::Vendor::Intel | gpu::vendor::Vendor::Apple => false,
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn gpu_gemm_dispatch(
    elem_size: usize,
    x: &[u8],
    y: &mut [u8],
    rows: usize,
    in_d: usize,
    out_d: usize,
    weights: &[u8],
) -> bool {
    match gpu::vendor::default_config().vendor {
        gpu::vendor::Vendor::Amd => radeon_gemm(elem_size, x, y, rows, in_d, out_d, weights),
        gpu::vendor::Vendor::Intel | gpu::vendor::Vendor::Apple => false,
    }
}

#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn gpu_gemm_dispatch(
    _elem_size: usize,
    _x: &[u8],
    _y: &mut [u8],
    _rows: usize,
    _in_d: usize,
    _out_d: usize,
    _weights: &[u8],
) -> bool {
    false
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn radeon_matmul(
    precision: nnn_config::Precision,
    src: &[u8],
    dst: &mut [u8],
    batch_size: usize,
    stride: usize,
    in_size: usize,
    out_size: usize,
    weights: &[u8],
    biases: &[u8],
    activation: u32,
) -> bool {
    let elem_size = match precision {
        nnn_config::Precision::F32 => 4usize,
        nnn_config::Precision::F64 => 8usize,
        nnn_config::Precision::F16 | nnn_config::Precision::BF16 => return false,
    };

    if batch_size == 0 || in_size == 0 || out_size == 0 {
        return false;
    }
    if weights.len() < out_size * in_size * elem_size {
        return false;
    }
    if batch_size > 1 && stride < out_size {
        return false;
    }
    let last_base = (batch_size - 1) * stride;
    if src.len() < (last_base + in_size) * elem_size || dst.len() < (last_base + out_size) * elem_size {
        return false;
    }

    let shader: &[u32] = match precision {
        nnn_config::Precision::F32 => &gpu::amd::shader::KERNEL_MATMUL_F32,
        nnn_config::Precision::F64 => &gpu::amd::shader::KERNEL_MATMUL_F64,
        nnn_config::Precision::F16 | nnn_config::Precision::BF16 => return false,
    };
    let rsrc1 = match precision {
        nnn_config::Precision::F32 => gpu::amd::pm4::COMPUTE_PGM_RSRC1_MATMUL_F32_VALUE,
        nnn_config::Precision::F64 => gpu::amd::pm4::COMPUTE_PGM_RSRC1_MATMUL_F64_VALUE,
        nnn_config::Precision::F16 | nnn_config::Precision::BF16 => return false,
    };
    let rsrc2 = match precision {
        nnn_config::Precision::F32 => gpu::amd::pm4::COMPUTE_PGM_RSRC2_MATMUL_F32_VALUE,
        nnn_config::Precision::F64 => gpu::amd::pm4::COMPUTE_PGM_RSRC2_MATMUL_F64_VALUE,
        nnn_config::Precision::F16 | nnn_config::Precision::BF16 => return false,
    };

    let src_elems = last_base + in_size;
    let dst_elems = last_base + out_size;
    let wb_elems = out_size * in_size + out_size;
    let has_bias = biases.len() >= out_size * elem_size;

    let fd = match gpu::amd::drm::open_render_node() {
        Some(fd) => fd,
        None => return false,
    };

    let ok = (|| {
        let shader_bytes = core::mem::size_of_val(shader);
        let mut shader_buf = gpu::amd::gem::alloc_gtt(fd, shader_bytes)?;
        let shader_words = shader_buf.cpu_ptr as *mut u32;
        unsafe { core::ptr::copy_nonoverlapping(shader.as_ptr(), shader_words, shader.len()); }

        let src_bytes = src_elems * elem_size;
        let mut src_buf = gpu::amd::gem::alloc_vram(fd, src_bytes)?;
        let src_dst = src_buf.cpu_ptr;
        unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), src_dst, src_bytes); }

        let wb_bytes = wb_elems * elem_size;
        let mut wb_buf = gpu::amd::gem::alloc_vram(fd, wb_bytes)?;
        let wb_dst = wb_buf.cpu_ptr;
        let weights_bytes = out_size * in_size * elem_size;
        unsafe { core::ptr::copy_nonoverlapping(weights.as_ptr(), wb_dst, weights_bytes); }
        let bias_base = weights_bytes;
        let bias_bytes = out_size * elem_size;
        if has_bias {
            unsafe { core::ptr::copy_nonoverlapping(biases.as_ptr(), wb_dst.add(bias_base), bias_bytes); }
        } else {
            unsafe { core::ptr::write_bytes(wb_dst.add(bias_base), 0u8, bias_bytes); }
        }

        let dst_bytes = dst_elems * elem_size;
        let mut dst_buf = gpu::amd::gem::alloc_gtt(fd, dst_bytes)?;
        let dst_ptr = dst_buf.cpu_ptr;
        unsafe { core::ptr::write_bytes(dst_ptr, 0u8, dst_elems * elem_size); }

        let shader_va = 0x100_0000u64;
        let src_va = va_align_up(shader_va + shader_bytes as u64);
        let wb_va = va_align_up(src_va + src_bytes as u64);
        let dst_va = va_align_up(wb_va + wb_bytes as u64);

        if !shader_buf.bind_va(shader_va) { return None; }
        if !src_buf.bind_va(src_va) { return None; }
        if !wb_buf.bind_va(wb_va) { return None; }
        if !dst_buf.bind_va(dst_va) { return None; }

        let params = gpu::amd::pm4::MatmulParams {
            shader_va: shader_buf.va,
            rsrc1,
            rsrc2,
            src_desc: gpu::amd::shader::build_buffer_descriptor(src_buf.va, src_bytes as u32),
            weights_desc: gpu::amd::shader::build_buffer_descriptor(wb_buf.va, wb_bytes as u32),
            dst_desc: gpu::amd::shader::build_buffer_descriptor(dst_buf.va, dst_bytes as u32),
            in_size: in_size as u32,
            out_size: out_size as u32,
            stride: stride as u32,
            activation,
            batch_size: batch_size as u32,
        };
        let ib = gpu::amd::pm4::build_matmul_ib(&params);

        let buffers = [
            gpu::amd::cs::BufferRef { handle: shader_buf.handle, write: false, domain: shader_buf.domain },
            gpu::amd::cs::BufferRef { handle: src_buf.handle, write: false, domain: src_buf.domain },
            gpu::amd::cs::BufferRef { handle: wb_buf.handle, write: false, domain: wb_buf.domain },
            gpu::amd::cs::BufferRef { handle: dst_buf.handle, write: true, domain: dst_buf.domain },
        ];
        if !gpu::amd::cs::submit(fd, &ib.words[..ib.len], &buffers) {
            return None;
        }
        if !dst_buf.wait_idle() {
            return None;
        }

        let row_bytes = out_size * elem_size;
        let mut b = 0;
        while b < batch_size {
            let base = b * stride * elem_size;
            unsafe { core::ptr::copy_nonoverlapping(dst_ptr.add(base), dst.as_mut_ptr().add(base), row_bytes); }
            b += 1;
        }
        Some(())
    })();

    let _ = syscall::sys_close(fd);
    ok.is_some()
}

#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn radeon_matmul(
    _src: &[u8],
    _dst: &mut [u8],
    _batch_size: usize,
    _stride: usize,
    _in_size: usize,
    _out_size: usize,
    _weights: &[u8],
    _biases: &[u8],
    _activation: u32,
) -> bool {
    false
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn radeon_gemm(
    elem_size: usize,
    x: &[u8],
    y: &mut [u8],
    rows: usize,
    in_d: usize,
    out_d: usize,
    weights: &[u8],
) -> bool {
    let precision = match elem_size {
        4 => nnn_config::Precision::F32,
        8 => nnn_config::Precision::F64,
        _ => return false,
    };

    if rows == 0 || in_d == 0 || out_d == 0 {
        return false;
    }
    if weights.len() < out_d * in_d * elem_size {
        return false;
    }
    if x.len() < rows * in_d * elem_size || y.len() < rows * out_d * elem_size {
        return false;
    }

    let stride = if in_d > out_d { in_d } else { out_d };

    let shader: &[u32] = match precision {
        nnn_config::Precision::F32 => &gpu::amd::shader::KERNEL_MATMUL_F32,
        nnn_config::Precision::F64 => &gpu::amd::shader::KERNEL_MATMUL_F64,
        nnn_config::Precision::F16 | nnn_config::Precision::BF16 => return false,
    };
    let rsrc1 = match precision {
        nnn_config::Precision::F32 => gpu::amd::pm4::COMPUTE_PGM_RSRC1_MATMUL_F32_VALUE,
        nnn_config::Precision::F64 => gpu::amd::pm4::COMPUTE_PGM_RSRC1_MATMUL_F64_VALUE,
        nnn_config::Precision::F16 | nnn_config::Precision::BF16 => return false,
    };
    let rsrc2 = match precision {
        nnn_config::Precision::F32 => gpu::amd::pm4::COMPUTE_PGM_RSRC2_MATMUL_F32_VALUE,
        nnn_config::Precision::F64 => gpu::amd::pm4::COMPUTE_PGM_RSRC2_MATMUL_F64_VALUE,
        nnn_config::Precision::F16 | nnn_config::Precision::BF16 => return false,
    };

    let last_base = (rows - 1) * stride;
    let src_elems = last_base + in_d;
    let dst_elems = last_base + out_d;
    let wb_elems = out_d * in_d + out_d;

    let fd = match gpu::amd::drm::open_render_node() {
        Some(fd) => fd,
        None => return false,
    };

    let ok = (|| {
        let shader_bytes = core::mem::size_of_val(shader);
        let mut shader_buf = gpu::amd::gem::alloc_gtt(fd, shader_bytes)?;
        let shader_words = shader_buf.cpu_ptr as *mut u32;
        unsafe { core::ptr::copy_nonoverlapping(shader.as_ptr(), shader_words, shader.len()); }

        let src_bytes = src_elems * elem_size;
        let mut src_buf = gpu::amd::gem::alloc_vram(fd, src_bytes)?;
        let src_dst = src_buf.cpu_ptr;
        let row_in_bytes = in_d * elem_size;
        let mut r = 0;
        while r < rows {
            let src_off = r * stride * elem_size;
            let x_off = r * in_d * elem_size;
            unsafe { core::ptr::copy_nonoverlapping(x.as_ptr().add(x_off), src_dst.add(src_off), row_in_bytes); }
            r += 1;
        }

        let wb_bytes = wb_elems * elem_size;
        let mut wb_buf = gpu::amd::gem::alloc_vram(fd, wb_bytes)?;
        let wb_dst = wb_buf.cpu_ptr;
        let weights_bytes = out_d * in_d * elem_size;
        unsafe { core::ptr::copy_nonoverlapping(weights.as_ptr(), wb_dst, weights_bytes); }
        let bias_bytes = out_d * elem_size;
        unsafe { core::ptr::write_bytes(wb_dst.add(weights_bytes), 0u8, bias_bytes); }

        let dst_bytes = dst_elems * elem_size;
        let mut dst_buf = gpu::amd::gem::alloc_gtt(fd, dst_bytes)?;
        let dst_ptr = dst_buf.cpu_ptr;
        unsafe { core::ptr::write_bytes(dst_ptr, 0u8, dst_bytes); }

        let shader_va = 0x100_0000u64;
        let src_va = va_align_up(shader_va + shader_bytes as u64);
        let wb_va = va_align_up(src_va + src_bytes as u64);
        let dst_va = va_align_up(wb_va + wb_bytes as u64);

        if !shader_buf.bind_va(shader_va) { return None; }
        if !src_buf.bind_va(src_va) { return None; }
        if !wb_buf.bind_va(wb_va) { return None; }
        if !dst_buf.bind_va(dst_va) { return None; }

        let params = gpu::amd::pm4::MatmulParams {
            shader_va: shader_buf.va,
            rsrc1,
            rsrc2,
            src_desc: gpu::amd::shader::build_buffer_descriptor(src_buf.va, src_bytes as u32),
            weights_desc: gpu::amd::shader::build_buffer_descriptor(wb_buf.va, wb_bytes as u32),
            dst_desc: gpu::amd::shader::build_buffer_descriptor(dst_buf.va, dst_bytes as u32),
            in_size: in_d as u32,
            out_size: out_d as u32,
            stride: stride as u32,
            activation: 0,
            batch_size: rows as u32,
        };
        let ib = gpu::amd::pm4::build_matmul_ib(&params);

        let buffers = [
            gpu::amd::cs::BufferRef { handle: shader_buf.handle, write: false, domain: shader_buf.domain },
            gpu::amd::cs::BufferRef { handle: src_buf.handle, write: false, domain: src_buf.domain },
            gpu::amd::cs::BufferRef { handle: wb_buf.handle, write: false, domain: wb_buf.domain },
            gpu::amd::cs::BufferRef { handle: dst_buf.handle, write: true, domain: dst_buf.domain },
        ];
        if !gpu::amd::cs::submit(fd, &ib.words[..ib.len], &buffers) {
            return None;
        }
        if !dst_buf.wait_idle() {
            return None;
        }

        let row_out_bytes = out_d * elem_size;
        let mut r = 0;
        while r < rows {
            let dst_off = r * stride * elem_size;
            let y_off = r * out_d * elem_size;
            unsafe { core::ptr::copy_nonoverlapping(dst_ptr.add(dst_off), y.as_mut_ptr().add(y_off), row_out_bytes); }
            r += 1;
        }
        Some(())
    })();

    let _ = syscall::sys_close(fd);
    ok.is_some()
}

#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn radeon_gemm(
    _elem_size: usize,
    _x: &[u8],
    _y: &mut [u8],
    _rows: usize,
    _in_d: usize,
    _out_d: usize,
    _weights: &[u8],
) -> bool {
    false
}
