#[derive(Clone, Copy)]
pub(crate) struct Delivery {
    pub delay: u32,
    pub deliver: bool,
}
