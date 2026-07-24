use crate::engine::runtime::hardware;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsError {
    InvalidPath,
    OpenFailed,
    ReadFailed,
    WriteFailed,
    MkdirFailed,
}

pub const DEFAULT_FILE_MODE: u32 = 0o644;
pub const DEFAULT_DIR_MODE: u32 = 0o755;

fn validate_path(path: &[u8]) -> Result<(), FsError> {
    if path.is_empty() || path[path.len() - 1] != 0 {
        return Err(FsError::InvalidPath);
    }
    let mut i = 0usize;
    while i < path.len() - 1 {
        if path[i] == 0 {
            return Err(FsError::InvalidPath);
        }
        i += 1;
    }
    Ok(())
}

pub fn file_exists(path: &[u8]) -> bool {
    if validate_path(path).is_err() {
        return false;
    }
    let fd = hardware::sys_open(path, hardware::O_RDONLY, 0);
    if fd < 0 {
        return false;
    }
    let _ = hardware::sys_close(fd);
    true
}

pub fn read_file(path: &[u8], on_chunk: &mut dyn FnMut(&[u8]) -> bool) -> Result<usize, FsError> {
    validate_path(path)?;
    let fd = hardware::sys_open(path, hardware::O_RDONLY, 0);
    if fd < 0 {
        return Err(FsError::OpenFailed);
    }
    let mut buf = [0u8; 4096];
    let mut total = 0usize;
    loop {
        let n = hardware::sys_read_fd(fd, &mut buf);
        if n < 0 {
            let _ = hardware::sys_close(fd);
            return Err(FsError::ReadFailed);
        }
        if n == 0 {
            break;
        }
        let len = n as usize;
        total = total.saturating_add(len);
        if !on_chunk(&buf[..len]) {
            break;
        }
    }
    let _ = hardware::sys_close(fd);
    Ok(total)
}

pub fn write_file(path: &[u8], bytes: &[u8]) -> Result<(), FsError> {
    validate_path(path)?;
    let flags = hardware::O_WRONLY | hardware::o_creat() | hardware::o_trunc();
    let fd = hardware::sys_open(path, flags, DEFAULT_FILE_MODE);
    if fd < 0 {
        return Err(FsError::OpenFailed);
    }
    let mut written = 0usize;
    while written < bytes.len() {
        let n = hardware::sys_write_fd(fd, &bytes[written..]);
        if n <= 0 {
            let _ = hardware::sys_close(fd);
            return Err(FsError::WriteFailed);
        }
        written = written.saturating_add(n as usize);
    }
    let _ = hardware::sys_close(fd);
    Ok(())
}

pub fn append_file(path: &[u8], bytes: &[u8]) -> Result<(), FsError> {
    validate_path(path)?;
    let flags = hardware::O_WRONLY | hardware::o_creat() | hardware::O_APPEND;
    let fd = hardware::sys_open(path, flags, DEFAULT_FILE_MODE);
    if fd < 0 {
        return Err(FsError::OpenFailed);
    }
    let mut written = 0usize;
    while written < bytes.len() {
        let n = hardware::sys_write_fd(fd, &bytes[written..]);
        if n <= 0 {
            let _ = hardware::sys_close(fd);
            return Err(FsError::WriteFailed);
        }
        written = written.saturating_add(n as usize);
    }
    let _ = hardware::sys_close(fd);
    Ok(())
}

pub fn ensure_dir(path: &[u8]) -> Result<(), FsError> {
    validate_path(path)?;
    if file_exists(path) {
        return Ok(());
    }
    let rc = hardware::sys_mkdir(path, DEFAULT_DIR_MODE);
    if rc < 0 && !file_exists(path) {
        return Err(FsError::MkdirFailed);
    }
    Ok(())
}

pub fn monotonic_ns() -> u64 {
    hardware::monotonic_ns()
}

pub fn process_exit(code: i32) -> ! {
    hardware::exit(code)
}
