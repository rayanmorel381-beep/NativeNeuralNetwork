#![no_std]

extern crate alloc;

pub use alloc::{string::String, vec::Vec};

pub mod bmk;
pub mod csv;
pub(crate) mod fs;
pub mod html;
pub mod ini;
pub mod json;
pub mod toml;
pub mod txt;
pub mod xml;
pub mod yaml;
