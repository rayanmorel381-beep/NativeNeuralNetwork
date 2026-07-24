use core::convert::TryInto;

pub struct Sha256Ctx {
    h: [u32; 8],
    len_bits: u64,
    buf: [u8; 64],
    buf_len: usize,
}

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn rotr32(x: u32, n: u32) -> u32 {
    x.rotate_right(n)
}

fn read_be_u32(block: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([block[off], block[off + 1], block[off + 2], block[off + 3]])
}

impl Sha256Ctx {
    pub fn new() -> Self {
        Sha256Ctx {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            len_bits: 0,
            buf: [0u8; 64],
            buf_len: 0,
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (t, wslot) in w.iter_mut().enumerate().take(16) {
            let off = t * 4;
            *wslot = read_be_u32(block, off);
        }
        for t in 16..64 {
            let s0 = rotr32(w[t - 15], 7) ^ rotr32(w[t - 15], 18) ^ (w[t - 15] >> 3);
            let s1 = rotr32(w[t - 2], 17) ^ rotr32(w[t - 2], 19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16]
                .wrapping_add(s0)
                .wrapping_add(w[t - 7])
                .wrapping_add(s1);
        }
        let mut a = self.h[0];
        let mut b = self.h[1];
        let mut c = self.h[2];
        let mut d = self.h[3];
        let mut e = self.h[4];
        let mut f = self.h[5];
        let mut g = self.h[6];
        let mut h = self.h[7];
        for t in 0..64 {
            let s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[t])
                .wrapping_add(w[t]);
            let s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        self.h[0] = self.h[0].wrapping_add(a);
        self.h[1] = self.h[1].wrapping_add(b);
        self.h[2] = self.h[2].wrapping_add(c);
        self.h[3] = self.h[3].wrapping_add(d);
        self.h[4] = self.h[4].wrapping_add(e);
        self.h[5] = self.h[5].wrapping_add(f);
        self.h[6] = self.h[6].wrapping_add(g);
        self.h[7] = self.h[7].wrapping_add(h);
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.len_bits = self.len_bits.wrapping_add((data.len() as u64) << 3);
        if self.buf_len > 0 {
            let need = 64 - self.buf_len;
            if data.len() >= need {
                self.buf[self.buf_len..].copy_from_slice(&data[..need]);
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
                data = &data[need..];
            } else {
                self.buf[self.buf_len..self.buf_len + data.len()].copy_from_slice(data);
                self.buf_len += data.len();
                return;
            }
        }
        while data.len() >= 64 {
            let block: &[u8; 64] = match data[..64].try_into() {
                Ok(b) => b,
                Err(_) => unreachable!(),
            };
            self.compress(block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    pub fn finalize(&mut self, out: &mut [u8; 32]) {
        let msg_len_bits = self.len_bits;
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;

        if self.buf_len > 56 {
            for b in &mut self.buf[self.buf_len..64] {
                *b = 0;
            }
            let block = self.buf;
            self.compress(&block);
            self.buf_len = 0;
        }

        for b in &mut self.buf[self.buf_len..56] {
            *b = 0;
        }
        self.buf[56..64].copy_from_slice(&msg_len_bits.to_be_bytes());
        let block = self.buf;
        self.compress(&block);

        self.buf_len = 0;
        for (i, v) in self.h.iter().enumerate() {
            let bytes = v.to_be_bytes();
            out[i * 4..i * 4 + 4].copy_from_slice(&bytes);
        }
    }
}
impl Default for Sha256Ctx {
    fn default() -> Self {
        Self::new()
    }
}

pub fn sha256_bytes(data: &[u8], out: &mut [u8; 32]) {
    let mut c = Sha256Ctx::new();
    c.update(data);
    c.finalize(out);
}

pub struct Sha512Ctx {
    h: [u64; 8],
    len_bits: u128,
    buf: [u8; 128],
    buf_len: usize,
}

const SHA512_K: [u64; 80] = [
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
];

fn rotr64(x: u64, n: u32) -> u64 {
    x.rotate_right(n)
}

fn read_be_u64(block: &[u8], off: usize) -> u64 {
    u64::from_be_bytes([
        block[off],
        block[off + 1],
        block[off + 2],
        block[off + 3],
        block[off + 4],
        block[off + 5],
        block[off + 6],
        block[off + 7],
    ])
}

impl Sha512Ctx {
    pub fn new() -> Self {
        Sha512Ctx {
            h: [
                0x6a09e667f3bcc908,
                0xbb67ae8584caa73b,
                0x3c6ef372fe94f82b,
                0xa54ff53a5f1d36f1,
                0x510e527fade682d1,
                0x9b05688c2b3e6c1f,
                0x1f83d9abfb41bd6b,
                0x5be0cd19137e2179,
            ],
            len_bits: 0,
            buf: [0u8; 128],
            buf_len: 0,
        }
    }

    fn compress(&mut self, block: &[u8; 128]) {
        let mut w = [0u64; 80];
        for (t, wslot) in w.iter_mut().enumerate().take(16) {
            let off = t * 8;
            *wslot = read_be_u64(block, off);
        }
        for t in 16..80 {
            let s0 = rotr64(w[t - 15], 1) ^ rotr64(w[t - 15], 8) ^ (w[t - 15] >> 7);
            let s1 = rotr64(w[t - 2], 19) ^ rotr64(w[t - 2], 61) ^ (w[t - 2] >> 6);
            w[t] = w[t - 16]
                .wrapping_add(s0)
                .wrapping_add(w[t - 7])
                .wrapping_add(s1);
        }
        let mut a = self.h[0];
        let mut b = self.h[1];
        let mut c = self.h[2];
        let mut d = self.h[3];
        let mut e = self.h[4];
        let mut f = self.h[5];
        let mut g = self.h[6];
        let mut h = self.h[7];
        for t in 0..80 {
            let s1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA512_K[t])
                .wrapping_add(w[t]);
            let s0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for i in 0..8 {
            self.h[i] = self.h[i].wrapping_add([a, b, c, d, e, f, g, h][i]);
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.len_bits = self.len_bits.wrapping_add((data.len() as u128) << 3);
        if self.buf_len > 0 {
            let need = 128 - self.buf_len;
            if data.len() >= need {
                self.buf[self.buf_len..].copy_from_slice(&data[..need]);
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
                data = &data[need..];
            } else {
                self.buf[self.buf_len..self.buf_len + data.len()].copy_from_slice(data);
                self.buf_len += data.len();
                return;
            }
        }
        while data.len() >= 128 {
            let block: &[u8; 128] = match data[..128].try_into() {
                Ok(b) => b,
                Err(_) => unreachable!(),
            };
            self.compress(block);
            data = &data[128..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    pub fn finalize(&mut self, out: &mut [u8; 64]) {
        let msg_len_bits = self.len_bits;
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;

        if self.buf_len > 112 {
            for b in &mut self.buf[self.buf_len..128] {
                *b = 0;
            }
            let block = self.buf;
            self.compress(&block);
            self.buf_len = 0;
        }

        for b in &mut self.buf[self.buf_len..112] {
            *b = 0;
        }
        self.buf[112..128].copy_from_slice(&msg_len_bits.to_be_bytes());
        let block = self.buf;
        self.compress(&block);

        self.buf_len = 0;
        for (i, v) in self.h.iter().enumerate() {
            let bytes = v.to_be_bytes();
            out[i * 8..i * 8 + 8].copy_from_slice(&bytes);
        }
    }
}

impl Default for Sha512Ctx {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256Ctx {
    fn zeroize(&mut self) {
        for v in &mut self.h {
            unsafe { core::ptr::write_volatile(v, 0u32) }
        }
        unsafe { core::ptr::write_volatile(&mut self.len_bits, 0u64) }
        for b in &mut self.buf {
            unsafe { core::ptr::write_volatile(b, 0u8) }
        }
        unsafe { core::ptr::write_volatile(&mut self.buf_len, 0usize) }
    }
}

impl Drop for Sha256Ctx {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl Sha512Ctx {
    fn zeroize(&mut self) {
        for v in &mut self.h {
            unsafe { core::ptr::write_volatile(v, 0u64) }
        }
        unsafe { core::ptr::write_volatile(&mut self.len_bits, 0u128) }
        for b in &mut self.buf {
            unsafe { core::ptr::write_volatile(b, 0u8) }
        }
        unsafe { core::ptr::write_volatile(&mut self.buf_len, 0usize) }
    }
}

impl Drop for Sha512Ctx {
    fn drop(&mut self) {
        self.zeroize();
    }
}

pub fn sha512_bytes(data: &[u8], out: &mut [u8; 64]) {
    let mut c = Sha512Ctx::new();
    c.update(data);
    c.finalize(out);
}
