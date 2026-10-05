//! Finding a running shiftpaperd and telling it to reload its config.

use nix::sys::signal::{Signal, kill};
use nix::unistd::Pid;
use std::os::unix::fs::MetadataExt;

/// Tell every shiftpaperd this user is running to reload its config, the
/// same as `systemctl --user reload shiftpaperd` but without needing the
/// systemd unit. Returns how many were told.
pub fn reload() -> usize {
    running().filter(|&pid| hup(pid)).count()
}

fn hup(pid: Pid) -> bool {
    kill(pid, Signal::SIGHUP).is_ok()
}

/// The shiftpaperd processes owned by this user, found through /proc.
fn running() -> impl Iterator<Item = Pid> {
    let me = std::fs::metadata("/proc/self").map(|m| m.uid()).ok();
    std::fs::read_dir("/proc")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(move |entry| {
            let pid: i32 = entry.file_name().to_str()?.parse().ok()?;
            let name = std::fs::read_to_string(entry.path().join("comm")).ok()?;
            let owner = entry.metadata().ok()?.uid();
            (name.trim_end() == "shiftpaperd" && Some(owner) == me).then(|| Pid::from_raw(pid))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;
    use std::process::Command;

    /// Start a stand-in daemon: a copy of `sleep` named shiftpaperd.
    fn stand_in(dir: &std::path::Path) -> std::process::Child {
        let sleep = ["/usr/bin/sleep", "/bin/sleep"]
            .into_iter()
            .find(|p| std::path::Path::new(p).exists())
            .expect("sleep is installed");
        let exe = dir.join("shiftpaperd");
        std::fs::copy(sleep, &exe).unwrap();
        Command::new(&exe).arg("30").spawn().unwrap()
    }

    // Only the stand-in is signalled, so running the tests doesn't reload
    // a real daemon.
    #[test]
    fn finds_a_running_daemon_and_signals_it() {
        let dir = tempfile::tempdir().unwrap();
        let mut child = stand_in(dir.path());
        let pid = Pid::from_raw(child.id() as i32);
        // Give it a moment to exec and take its name.
        std::thread::sleep(std::time::Duration::from_millis(100));

        assert!(running().any(|p| p == pid));
        assert!(hup(pid));
        // sleep doesn't handle SIGHUP, so it ends from it.
        assert_eq!(child.wait().unwrap().signal(), Some(Signal::SIGHUP as i32));
    }
}
