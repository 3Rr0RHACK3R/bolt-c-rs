use std::time::Duration;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn mach_port_deallocate(
        task: libc::mach_port_t,
        name: libc::mach_port_t,
    ) -> libc::kern_return_t;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OsStats {
    pub user_cpu_time: Duration,
    pub sys_cpu_time: Duration,
    pub thread_cpu_time: Duration,
    pub rss_bytes: u64,
}

#[cfg(unix)]
pub fn get_os_stats() -> OsStats {
    let mut rusage = std::mem::MaybeUninit::uninit();
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, rusage.as_mut_ptr()) };

    let (user_cpu_time, sys_cpu_time, rss_bytes) = if ret != 0 {
        (Duration::ZERO, Duration::ZERO, 0)
    } else {
        let rusage = unsafe { rusage.assume_init() };

        let user = Duration::new(
            rusage.ru_utime.tv_sec as u64,
            (rusage.ru_utime.tv_usec as u32) * 1000,
        );
        let sys = Duration::new(
            rusage.ru_stime.tv_sec as u64,
            (rusage.ru_stime.tv_usec as u32) * 1000,
        );

        #[cfg(target_os = "macos")]
        let rss = rusage.ru_maxrss as u64;

        #[cfg(target_os = "linux")]
        let rss = (rusage.ru_maxrss as u64) * 1024;

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        let rss = 0u64;

        (user, sys, rss)
    };

    let thread_cpu_time = get_thread_cpu_time();

    OsStats {
        user_cpu_time,
        sys_cpu_time,
        thread_cpu_time,
        rss_bytes,
    }
}

#[cfg(target_os = "linux")]
fn get_thread_cpu_time() -> Duration {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let ret = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    if ret != 0 {
        return Duration::ZERO;
    }
    Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
}

#[cfg(target_os = "macos")]
fn get_thread_cpu_time() -> Duration {
    use std::mem::MaybeUninit;

    let mut info = MaybeUninit::<libc::thread_basic_info>::uninit();
    let mut count = (std::mem::size_of::<libc::thread_basic_info>() / std::mem::size_of::<i32>())
        as libc::mach_msg_type_number_t;

    #[allow(deprecated)]
    let thread_self = unsafe { libc::mach_thread_self() };

    let flavor: libc::thread_flavor_t = libc::THREAD_BASIC_INFO as libc::thread_flavor_t;
    let ret = unsafe {
        libc::thread_info(
            thread_self,
            flavor,
            info.as_mut_ptr() as libc::thread_info_t,
            &mut count,
        )
    };

    #[allow(deprecated)]
    unsafe {
        // mach_task_self is deprecated in libc; acceptable here for cleanup in tests.
        let task = libc::mach_task_self();
        mach_port_deallocate(task, thread_self);
    }

    if ret != libc::KERN_SUCCESS {
        return Duration::ZERO;
    }

    let info = unsafe { info.assume_init() };
    let user = Duration::new(
        info.user_time.seconds as u64,
        (info.user_time.microseconds as u32) * 1000,
    );
    let sys = Duration::new(
        info.system_time.seconds as u64,
        (info.system_time.microseconds as u32) * 1000,
    );
    user + sys
}

#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
fn get_thread_cpu_time() -> Duration {
    Duration::ZERO
}

#[cfg(not(unix))]
pub fn get_os_stats() -> OsStats {
    OsStats::default()
}
