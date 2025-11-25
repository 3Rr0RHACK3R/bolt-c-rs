use std::time::Duration;

#[derive(Debug, Clone, Copy, Default)]
pub struct OsStats {
    pub user_cpu_time: Duration,
    pub sys_cpu_time: Duration,
    pub rss_bytes: u64,
}

#[cfg(unix)]
pub fn get_os_stats() -> OsStats {
    let mut rusage = std::mem::MaybeUninit::uninit();
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, rusage.as_mut_ptr()) };

    if ret != 0 {
        return OsStats::default();
    }

    let rusage = unsafe { rusage.assume_init() };

    let user_cpu_time = Duration::new(
        rusage.ru_utime.tv_sec as u64,
        (rusage.ru_utime.tv_usec as u32) * 1000,
    );
    let sys_cpu_time = Duration::new(
        rusage.ru_stime.tv_sec as u64,
        (rusage.ru_stime.tv_usec as u32) * 1000,
    );

    #[cfg(target_os = "macos")]
    let rss_bytes = rusage.ru_maxrss as u64;

    #[cfg(target_os = "linux")]
    let rss_bytes = (rusage.ru_maxrss as u64) * 1024;

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let rss_bytes = 0;

    OsStats {
        user_cpu_time,
        sys_cpu_time,
        rss_bytes,
    }
}

#[cfg(not(unix))]
pub fn get_os_stats() -> OsStats {
    OsStats::default()
}
