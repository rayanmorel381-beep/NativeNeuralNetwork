#![no_std]

mod arch;
mod guardian;
mod types;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arm;

pub mod syscall;
pub mod consumption;
