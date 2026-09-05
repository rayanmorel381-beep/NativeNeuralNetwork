use super::shader;

const PKT3_TYPE: u32 = 3 << 30;
const PKT3_COMPUTE_BIT: u32 = 1 << 1;
const IT_SET_SH_REG: u32 = 0x76;
const IT_DISPATCH_DIRECT: u32 = 0x15;
const IT_EVENT_WRITE: u32 = 0x46;

const SH_REG_BASE: u32 = 0xB000;

const COMPUTE_NUM_THREAD_X: u32 = 0xB81C;
const COMPUTE_PGM_LO: u32 = 0xB830;
const COMPUTE_PGM_RSRC1: u32 = 0xB848;
const COMPUTE_RESOURCE_LIMITS: u32 = 0xB854;
const COMPUTE_STATIC_THREAD_MGMT_SE0: u32 = 0xB858;
const COMPUTE_USER_DATA_0: u32 = 0xB900;

const COMPUTE_DISPATCH_INITIATOR_SHADER_EN: u32 = 1;
const COMPUTE_STATIC_THREAD_MGMT_ALL_CU: u32 = 0xFFFF_FFFF;
const COMPUTE_RESOURCE_LIMITS_NONE: u32 = 0;

const RSRC1_FLOAT_MODE_ROUND_NEAREST: u32 = 0xC0;
const RSRC1_DX10_CLAMP: u32 = 1;
const RSRC1_VGPRS_FIELD: u32 = (shader::KERNEL_SCALE_VEC_VGPR_COUNT - 1) / 4;
const RSRC1_SGPRS_FIELD: u32 = (shader::KERNEL_SCALE_VEC_SGPR_COUNT - 1) / 8;
const COMPUTE_PGM_RSRC1_VALUE: u32 = RSRC1_VGPRS_FIELD
    | (RSRC1_SGPRS_FIELD << 6)
    | (RSRC1_FLOAT_MODE_ROUND_NEAREST << 12)
    | (RSRC1_DX10_CLAMP << 21);
const COMPUTE_PGM_RSRC2_VALUE: u32 = shader::KERNEL_SCALE_VEC_USER_SGPR_COUNT << 1;

const EVENT_TYPE_CS_PARTIAL_FLUSH: u32 = 7;
const EVENT_INDEX_CS_VS_PS: u32 = 4;

const DISPATCH_GROUPS_X: u32 = 1;
const DISPATCH_GROUPS_Y: u32 = 1;
const DISPATCH_GROUPS_Z: u32 = 1;

pub(crate) const DISPATCH_IB_MAX_DWORDS: usize = 64;

pub(crate) struct DispatchIb {
    pub words: [u32; DISPATCH_IB_MAX_DWORDS],
    pub len: usize,
}

pub(crate) struct DispatchParams {
    pub shader_va: u64,
    pub input_desc: [u32; 4],
    pub output_desc: [u32; 4],
    pub scale_bits: u32,
    pub threads_x: u32,
}

const fn pkt3_compute(op: u32, count: u32) -> u32 {
    PKT3_TYPE | ((count & 0x3FFF) << 16) | ((op & 0xFF) << 8) | PKT3_COMPUTE_BIT
}

fn push(ib: &mut DispatchIb, value: u32) {
    ib.words[ib.len] = value;
    ib.len += 1;
}

fn set_sh_reg(ib: &mut DispatchIb, reg: u32, values: &[u32]) {
    push(ib, pkt3_compute(IT_SET_SH_REG, values.len() as u32));
    push(ib, (reg - SH_REG_BASE) >> 2);
    let mut i = 0;
    while i < values.len() {
        push(ib, values[i]);
        i += 1;
    }
}

pub(crate) fn build_dispatch_ib(params: &DispatchParams) -> DispatchIb {
    let mut ib = DispatchIb {
        words: [0; DISPATCH_IB_MAX_DWORDS],
        len: 0,
    };

    let pgm = params.shader_va >> 8;
    set_sh_reg(
        &mut ib,
        COMPUTE_PGM_LO,
        &[(pgm & 0xFFFF_FFFF) as u32, ((pgm >> 32) & 0xFF) as u32],
    );
    set_sh_reg(
        &mut ib,
        COMPUTE_PGM_RSRC1,
        &[COMPUTE_PGM_RSRC1_VALUE, COMPUTE_PGM_RSRC2_VALUE],
    );
    set_sh_reg(&mut ib, COMPUTE_NUM_THREAD_X, &[params.threads_x, 1, 1]);

    let user_data = [
        params.input_desc[0],
        params.input_desc[1],
        params.input_desc[2],
        params.input_desc[3],
        params.output_desc[0],
        params.output_desc[1],
        params.output_desc[2],
        params.output_desc[3],
        params.scale_bits,
    ];
    set_sh_reg(&mut ib, COMPUTE_USER_DATA_0, &user_data);

    set_sh_reg(
        &mut ib,
        COMPUTE_RESOURCE_LIMITS,
        &[COMPUTE_RESOURCE_LIMITS_NONE],
    );
    set_sh_reg(
        &mut ib,
        COMPUTE_STATIC_THREAD_MGMT_SE0,
        &[
            COMPUTE_STATIC_THREAD_MGMT_ALL_CU,
            COMPUTE_STATIC_THREAD_MGMT_ALL_CU,
        ],
    );

    push(&mut ib, pkt3_compute(IT_DISPATCH_DIRECT, 3));
    push(&mut ib, DISPATCH_GROUPS_X);
    push(&mut ib, DISPATCH_GROUPS_Y);
    push(&mut ib, DISPATCH_GROUPS_Z);
    push(&mut ib, COMPUTE_DISPATCH_INITIATOR_SHADER_EN);

    push(&mut ib, pkt3_compute(IT_EVENT_WRITE, 0));
    push(
        &mut ib,
        EVENT_TYPE_CS_PARTIAL_FLUSH | (EVENT_INDEX_CS_VS_PS << 8),
    );

    ib
}

const RSRC1_MATMUL_F32_VGPRS_FIELD: u32 = (shader::KERNEL_MATMUL_F32_VGPR_COUNT - 1) / 4;
const RSRC1_MATMUL_F32_SGPRS_FIELD: u32 = (shader::KERNEL_MATMUL_F32_SGPR_COUNT - 1) / 8;
pub(crate) const COMPUTE_PGM_RSRC1_MATMUL_F32_VALUE: u32 = RSRC1_MATMUL_F32_VGPRS_FIELD
    | (RSRC1_MATMUL_F32_SGPRS_FIELD << 6)
    | (RSRC1_FLOAT_MODE_ROUND_NEAREST << 12)
    | (RSRC1_DX10_CLAMP << 21);

const RSRC1_MATMUL_F64_VGPRS_FIELD: u32 = (shader::KERNEL_MATMUL_F64_VGPR_COUNT - 1) / 4;
const RSRC1_MATMUL_F64_SGPRS_FIELD: u32 = (shader::KERNEL_MATMUL_F64_SGPR_COUNT - 1) / 8;
pub(crate) const COMPUTE_PGM_RSRC1_MATMUL_F64_VALUE: u32 = RSRC1_MATMUL_F64_VGPRS_FIELD
    | (RSRC1_MATMUL_F64_SGPRS_FIELD << 6)
    | (RSRC1_FLOAT_MODE_ROUND_NEAREST << 12)
    | (RSRC1_DX10_CLAMP << 21);

const RSRC2_TGID_X_EN: u32 = 1 << 7;
pub(crate) const COMPUTE_PGM_RSRC2_MATMUL_F32_VALUE: u32 =
    (shader::KERNEL_MATMUL_F32_USER_SGPR_COUNT << 1) | RSRC2_TGID_X_EN;
pub(crate) const COMPUTE_PGM_RSRC2_MATMUL_F64_VALUE: u32 =
    (shader::KERNEL_MATMUL_F64_USER_SGPR_COUNT << 1) | RSRC2_TGID_X_EN;

const MATMUL_THREADS_PER_GROUP: u32 = 64;

pub(crate) struct MatmulParams {
    pub shader_va: u64,
    pub rsrc1: u32,
    pub rsrc2: u32,
    pub src_desc: [u32; 4],
    pub weights_desc: [u32; 4],
    pub dst_desc: [u32; 4],
    pub in_size: u32,
    pub out_size: u32,
    pub stride: u32,
    pub activation: u32,
    pub batch_size: u32,
}

pub(crate) fn build_matmul_ib(params: &MatmulParams) -> DispatchIb {
    let mut ib = DispatchIb {
        words: [0; DISPATCH_IB_MAX_DWORDS],
        len: 0,
    };

    let pgm = params.shader_va >> 8;
    set_sh_reg(
        &mut ib,
        COMPUTE_PGM_LO,
        &[(pgm & 0xFFFF_FFFF) as u32, ((pgm >> 32) & 0xFF) as u32],
    );
    set_sh_reg(
        &mut ib,
        COMPUTE_PGM_RSRC1,
        &[
            params.rsrc1,
            params.rsrc2,
        ],
    );
    set_sh_reg(
        &mut ib,
        COMPUTE_NUM_THREAD_X,
        &[MATMUL_THREADS_PER_GROUP, 1, 1],
    );

    let user_data = [
        params.src_desc[0],
        params.src_desc[1],
        params.src_desc[2],
        params.src_desc[3],
        params.weights_desc[0],
        params.weights_desc[1],
        params.weights_desc[2],
        params.weights_desc[3],
        params.dst_desc[0],
        params.dst_desc[1],
        params.dst_desc[2],
        params.dst_desc[3],
        params.in_size | (params.out_size << 16),
        params.stride | (params.activation << 16),
        params.batch_size,
    ];
    set_sh_reg(&mut ib, COMPUTE_USER_DATA_0, &user_data);

    set_sh_reg(
        &mut ib,
        COMPUTE_RESOURCE_LIMITS,
        &[COMPUTE_RESOURCE_LIMITS_NONE],
    );
    set_sh_reg(
        &mut ib,
        COMPUTE_STATIC_THREAD_MGMT_SE0,
        &[
            COMPUTE_STATIC_THREAD_MGMT_ALL_CU,
            COMPUTE_STATIC_THREAD_MGMT_ALL_CU,
        ],
    );

    let groups_x = params.out_size.div_ceil(MATMUL_THREADS_PER_GROUP);

    push(&mut ib, pkt3_compute(IT_DISPATCH_DIRECT, 3));
    push(&mut ib, groups_x);
    push(&mut ib, DISPATCH_GROUPS_Y);
    push(&mut ib, DISPATCH_GROUPS_Z);
    push(&mut ib, COMPUTE_DISPATCH_INITIATOR_SHADER_EN);

    push(&mut ib, pkt3_compute(IT_EVENT_WRITE, 0));
    push(
        &mut ib,
        EVENT_TYPE_CS_PARTIAL_FLUSH | (EVENT_INDEX_CS_VS_PS << 8),
    );

    ib
}
