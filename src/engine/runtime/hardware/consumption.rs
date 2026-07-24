use super::arch::detect_hardware_profile;
use super::types::HardwareProfile;
use crate::engine::ComputeBackend;

pub const COMPUTE_CAP_PPM: u32 = 800_000;
pub const RAM_CAP_PPM: u32 = 700_000;

#[derive(Clone, Copy, Debug, Default)]
pub struct ConsumptionGuard {
    pub cpu_cap_ppm: u32,
    pub gpu_cap_ppm: u32,
    pub tpu_cap_ppm: u32,
    pub lpu_cap_ppm: u32,
    pub ram_cap_ppm: u32,
    ram_base_bytes: usize,
    cores: usize,
    cpu_avg_mhz: u32,
    cpu_max_mhz: u32,
}

fn apply_ppm(value: usize, ppm: u32) -> usize {
    let capped = if ppm > 1_000_000 { 1_000_000u128 } else { ppm as u128 };
    ((value as u128).saturating_mul(capped) / 1_000_000u128) as usize
}

impl ConsumptionGuard {
    pub fn from_profile(profile: HardwareProfile) -> ConsumptionGuard {
        let ram_base = if profile.ram_available > 0 {
            profile.ram_available
        } else {
            profile.ram_total
        };
        ConsumptionGuard {
            cpu_cap_ppm: COMPUTE_CAP_PPM,
            gpu_cap_ppm: if profile.gpu { COMPUTE_CAP_PPM } else { 0 },
            tpu_cap_ppm: if profile.tpu { COMPUTE_CAP_PPM } else { 0 },
            lpu_cap_ppm: if profile.lpu { COMPUTE_CAP_PPM } else { 0 },
            ram_cap_ppm: RAM_CAP_PPM,
            ram_base_bytes: ram_base,
            cores: profile.cores.max(1),
            cpu_avg_mhz: profile.cpu_avg_mhz,
            cpu_max_mhz: profile.cpu_max_mhz,
        }
    }

    pub fn detect() -> ConsumptionGuard {
        ConsumptionGuard::from_profile(detect_hardware_profile())
    }

    pub fn ram_budget_bytes(&self) -> usize {
        apply_ppm(self.ram_base_bytes, self.ram_cap_ppm)
    }

    pub fn ram_fits(&self, requested_bytes: usize) -> bool {
        let budget = self.ram_budget_bytes();
        budget == 0 || requested_bytes <= budget
    }

    pub fn cpu_workers(&self) -> usize {
        apply_ppm(self.cores, self.cpu_cap_ppm).max(1)
    }

    pub fn logical_cores(&self) -> usize {
        self.cores
    }

    pub fn cpu_avg_mhz(&self) -> u32 {
        self.cpu_avg_mhz
    }

    pub fn cpu_max_mhz(&self) -> u32 {
        self.cpu_max_mhz
    }

    pub fn clamp_workers(&self, requested: usize) -> usize {
        requested.min(self.cpu_workers()).max(1)
    }

    pub fn compute_cap(&self) -> f64 {
        self.cpu_cap_ppm as f64 / 1_000_000.0
    }

    pub fn accelerator_cap_ppm(&self, backend: ComputeBackend) -> u32 {
        match backend {
            ComputeBackend::Gpu => self.gpu_cap_ppm,
            ComputeBackend::Tpu => self.tpu_cap_ppm,
            ComputeBackend::Lpu => self.lpu_cap_ppm,
            ComputeBackend::Cpu => self.cpu_cap_ppm,
        }
    }

    pub fn accelerator_headroom_ppm(&self, backend: ComputeBackend) -> u32 {
        let cap = self.accelerator_cap_ppm(backend);
        let cap = if cap == 0 { self.cpu_cap_ppm } else { cap };
        1_000_000u32.saturating_sub(cap)
    }
}
