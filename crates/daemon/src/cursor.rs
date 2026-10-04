use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

/// Reads the global cursor position over Hyprland's IPC socket.
pub struct HyprlandCursor {
    socket_path: PathBuf,
}

impl HyprlandCursor {
    /// None when not running under Hyprland.
    pub fn new() -> Option<Self> {
        Some(Self {
            socket_path: hyprland_socket_path()?,
        })
    }

    /// Cursor position in global logical coordinates, the same space as
    /// output positions.
    pub fn position(&self) -> Option<(f32, f32)> {
        let mut stream = UnixStream::connect(&self.socket_path).ok()?;
        stream
            .set_read_timeout(Some(Duration::from_millis(50)))
            .ok()?;
        stream.write_all(b"cursorpos").ok()?;

        let mut buf = [0u8; 64];
        let n = stream.read(&mut buf).ok()?;
        parse_cursor_pos(std::str::from_utf8(&buf[..n]).ok()?)
    }
}

/// Where `cursor` sits relative to the centre of an output at (x, y) with
/// size (w, h), as a fraction of that size in [-0.5, 0.5]. Outputs the
/// cursor isn't on clamp to their nearest edge, so they lean towards it.
pub fn offset_from_centre(cursor: (f32, f32), x: i32, y: i32, w: u32, h: u32) -> (f32, f32) {
    let ox = ((cursor.0 - x as f32) / w as f32 - 0.5).clamp(-0.5, 0.5);
    let oy = ((cursor.1 - y as f32) / h as f32 - 0.5).clamp(-0.5, 0.5);
    (ox, oy)
}

fn hyprland_socket_path() -> Option<PathBuf> {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").ok()?;
    let instance_sig = std::env::var("HYPRLAND_INSTANCE_SIGNATURE").ok()?;
    let path = PathBuf::from(runtime_dir)
        .join("hypr")
        .join(instance_sig)
        .join(".socket.sock");
    if path.exists() {
        Some(path)
    } else {
        tracing::warn!(?path, "Hyprland socket not found");
        None
    }
}

/// Parse Hyprland's `cursorpos` response, e.g. "1234, 567".
fn parse_cursor_pos(response: &str) -> Option<(f32, f32)> {
    let (x, y) = response.trim().split_once(',')?;
    Some((x.trim().parse().ok()?, y.trim().parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Layout from a real setup: laptop eDP-1 at (1600, 400) sized
    // 1600x1000 logical, external HDMI-A-1 at (3200, 0) sized 2560x1440.

    #[test]
    fn cursor_at_output_centre_is_zero() {
        assert_eq!(
            offset_from_centre((4480.0, 720.0), 3200, 0, 2560, 1440),
            (0.0, 0.0)
        );
        assert_eq!(
            offset_from_centre((2400.0, 900.0), 1600, 400, 1600, 1000),
            (0.0, 0.0)
        );
    }

    #[test]
    fn other_outputs_lean_towards_cursor() {
        // Cursor centred on the laptop, which is left of the external.
        assert_eq!(
            offset_from_centre((2400.0, 900.0), 3200, 0, 2560, 1440),
            (-0.5, 0.125)
        );
    }

    #[test]
    fn parses_cursor_position() {
        assert_eq!(parse_cursor_pos("1234, 567\n"), Some((1234.0, 567.0)));
    }

    #[test]
    fn rejects_malformed_response() {
        assert_eq!(parse_cursor_pos("unknown request"), None);
    }
}
