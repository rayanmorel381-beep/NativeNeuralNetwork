pub(super) mod backend;
pub(super) mod scheduler;

pub(crate) fn default_config() -> backend::VendorBackendConfig {
	backend::default_backend_config()
}

pub(crate) fn build_schedule(work_items: usize) -> scheduler::VendorSchedule {
	scheduler::build_schedule(work_items)
}

pub(crate) fn clamp_workers(requested: usize) -> usize {
	backend::clamp_workers(requested)
}
