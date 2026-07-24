use crate::observability::visualization::errors::VisualizeError;
use crate::observability::visualization::view::NetworkView;

pub fn get_network_view<'a>(bytes: &'a [u8]) -> Result<NetworkView<'a>, VisualizeError> {
    NetworkView::from_rnn_bytes(bytes)
}
