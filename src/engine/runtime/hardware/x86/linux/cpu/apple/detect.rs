const MAX_BRAND: usize = 128;

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
    None
}
