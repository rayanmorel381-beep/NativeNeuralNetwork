use core::ffi::c_void;

const MAX_WIDE: usize = 256;
const MAX_BRAND: usize = 128;
const MAX_REG_BUF: usize = 512;

type Hkey = *mut c_void;
const HKEY_LOCAL_MACHINE: Hkey = 0x80000002_usize as Hkey;
const KEY_READ: u32 = 0x20019;

#[repr(C)]
struct SystemInfo {
    processor_architecture: u16,
    _reserved: u16,
    _page_size: u32,
    _minimum_application_address: *mut c_void,
    _maximum_application_address: *mut c_void,
    _active_processor_mask: usize,
    number_of_processors: u32,
    _processor_type: u32,
    _allocation_granularity: u32,
    _processor_level: u16,
    _processor_revision: u16,
}

unsafe extern "system" {
    fn RegOpenKeyExW(key: Hkey, sub_key: *const u16, options: u32, desired: u32, result: *mut Hkey) -> i32;
    fn RegQueryValueExW(key: Hkey, value_name: *const u16, reserved: *mut u32, reg_type: *mut u32, data: *mut u8, data_len: *mut u32) -> i32;
    fn RegCloseKey(key: Hkey) -> i32;
    fn GetSystemInfo(lp_system_info: *mut SystemInfo);
}

fn to_wide(s: &str, buf: &mut [u16; MAX_WIDE]) -> usize {
    let mut len = 0;
    for c in s.encode_utf16() {
        if len + 1 >= MAX_WIDE { break; }
        buf[len] = c;
        len += 1;
    }
    buf[len] = 0;
    len + 1
}

fn wide_to_brand(wide: &[u16], out: &mut [u8; MAX_BRAND]) -> usize {
    let mut len = 0;
    for &c in wide {
        if c == 0 || len >= MAX_BRAND { break; }
        out[len] = if c < 128 { c as u8 } else { b'?' };
        len += 1;
    }
    len
}

fn registry_brand(key: &str, value: &str, out: &mut [u8; MAX_BRAND]) -> usize {
    let mut key_wide = [0u16; MAX_WIDE];
    let mut val_wide = [0u16; MAX_WIDE];
    to_wide(key, &mut key_wide);
    to_wide(value, &mut val_wide);
    let mut hkey: Hkey = core::ptr::null_mut();
    let open = unsafe { RegOpenKeyExW(HKEY_LOCAL_MACHINE, key_wide.as_ptr(), 0, KEY_READ, &mut hkey) };
    if open != 0 { return 0; }
    let mut buf = [0u8; MAX_REG_BUF];
    let mut len: u32 = MAX_REG_BUF as u32;
    let mut reg_type: u32 = 0;
    let res = unsafe {
        RegQueryValueExW(
            hkey,
            val_wide.as_ptr(),
            core::ptr::null_mut(),
            &mut reg_type,
            buf.as_mut_ptr(),
            &mut len,
        )
    };
    unsafe { RegCloseKey(hkey) };
    if res != 0 { return 0; }
    let u16_len = (len as usize) / 2;
    let wide = unsafe { core::slice::from_raw_parts(buf.as_ptr() as *const u16, u16_len) };
    wide_to_brand(wide, out)
}

fn get_system_info() -> SystemInfo {
    let mut info = SystemInfo {
        processor_architecture: 0,
        _reserved: 0,
        _page_size: 0,
        _minimum_application_address: core::ptr::null_mut(),
        _maximum_application_address: core::ptr::null_mut(),
        _active_processor_mask: 0,
        number_of_processors: 0,
        _processor_type: 0,
        _allocation_granularity: 0,
        _processor_level: 0,
        _processor_revision: 0,
    };
    unsafe { GetSystemInfo(&mut info) };
    info
}

const CPU_KEY: &str = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0";
const PROCESSOR_ARCHITECTURE_ARM64: u16 = 12;

pub(crate) struct ArmWinInfo {
    pub brand: [u8; MAX_BRAND],
    pub brand_len: usize,
    pub core_count: u32,
}

impl ArmWinInfo {
    pub fn brand_contains_ignore_case(&self, needle: &[u8]) -> bool {
        let hay = &self.brand[..self.brand_len];
        if needle.is_empty() { return true; }
        if hay.len() < needle.len() { return false; }
        let end = hay.len() - needle.len();
        'outer: for i in 0..=end {
            for (j, &n) in needle.iter().enumerate() {
                if !hay[i + j].eq_ignore_ascii_case(&n) { continue 'outer; }
            }
            return true;
        }
        false
    }
}

pub(crate) fn detect_arm() -> Option<ArmWinInfo> {
    let info = get_system_info();
    if info.processor_architecture != PROCESSOR_ARCHITECTURE_ARM64 {
        return None;
    }
    let mut brand = [0u8; MAX_BRAND];
    let brand_len = registry_brand(CPU_KEY, "ProcessorNameString", &mut brand);
    Some(ArmWinInfo {
        brand,
        brand_len,
        core_count: info.number_of_processors,
    })
}
