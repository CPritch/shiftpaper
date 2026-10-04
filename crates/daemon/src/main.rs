mod battery;
mod config;
mod cursor;
mod depth;
mod idle;
mod renderer;
mod wayland;

use anyhow::{Context, Result};
use calloop::timer::{TimeoutAction, Timer};
use shiftpaper_config::TrackingMode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;
use wayland_client::Connection;

const BATTERY_POLL_INTERVAL: Duration = Duration::from_secs(30);
const SIGNAL_CHECK_INTERVAL: Duration = Duration::from_millis(500);

static SHUTDOWN: AtomicBool = AtomicBool::new(false);
static RELOAD: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_shutdown(_: std::ffi::c_int) {
    SHUTDOWN.store(true, Ordering::Relaxed);
}
extern "C" fn handle_reload(_: std::ffi::c_int) {
    RELOAD.store(true, Ordering::Relaxed);
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("shiftpaper_daemon=info")),
        )
        .init();

    info!("starting shiftpaperd");

    install_signal_handlers();

    let cfg = config::load()?;
    info!(?cfg, "configuration loaded");

    let conn = Connection::connect_to_env().context("failed to connect to Wayland display")?;

    let (globals, mut event_queue) =
        wayland_client::globals::registry_queue_init::<wayland::App>(&conn)
            .context("failed to init Wayland registry")?;

    let qh = event_queue.handle();
    let mut app = wayland::App::new(cfg.clone(), &globals, &qh)?;

    // First roundtrip delivers output info, and new_output creates a layer
    // surface for each. Second delivers the compositor's configure for them.
    event_queue.roundtrip(&mut app)?;
    event_queue.roundtrip(&mut app)?;

    if app.render_targets.is_empty() {
        error!("no render targets initialized");
        for (i, o) in app.outputs.iter().enumerate() {
            error!(
                idx = i,
                name = o.name,
                configured = o.configured,
                "output state"
            );
        }
        anyhow::bail!("failed to initialize any outputs");
    }

    info!(
        outputs = app.outputs.len(),
        render_targets = app.render_targets.len(),
        "outputs ready"
    );

    app.idle = idle::try_bind(
        &globals,
        &qh,
        app.seat.as_ref(),
        cfg.daemon.idle_timeout_secs,
    );

    app.refresh_battery();

    match cfg.daemon.tracking_mode {
        TrackingMode::Hyprland => {
            info!("tracking mode: hyprland (IPC polling)");
            app.init_cursor();
        }
        TrackingMode::Pointer => {
            info!("tracking mode: pointer (Wayland-native, event-driven)");
        }
    }

    let mut event_loop: calloop::EventLoop<wayland::App> =
        calloop::EventLoop::try_new().context("failed to create calloop event loop")?;
    let loop_handle = event_loop.handle();

    calloop_wayland_source::WaylandSource::new(conn, event_queue)
        .insert(loop_handle.clone())
        .map_err(|e| anyhow::anyhow!("failed to insert Wayland source: {e}"))?;

    if matches!(cfg.daemon.tracking_mode, TrackingMode::Hyprland) {
        let poll_interval = Duration::from_secs_f64(1.0 / cfg.daemon.cursor_poll_hz.get() as f64);
        let tick_timer = Timer::immediate();
        let qh_tick = qh.clone();

        loop_handle
            .insert_source(
                tick_timer,
                move |_deadline, _metadata, app: &mut wayland::App| {
                    app.tick(&qh_tick);
                    TimeoutAction::ToDuration(poll_interval)
                },
            )
            .map_err(|e| anyhow::anyhow!("failed to insert timer source: {e}"))?;

        info!(
            hz = cfg.daemon.cursor_poll_hz.get(),
            "hyprland tick timer inserted"
        );
    }

    loop_handle
        .insert_source(
            Timer::from_duration(BATTERY_POLL_INTERVAL),
            move |_deadline, _metadata, app: &mut wayland::App| {
                app.refresh_battery();
                TimeoutAction::ToDuration(BATTERY_POLL_INTERVAL)
            },
        )
        .map_err(|e| anyhow::anyhow!("failed to insert battery timer: {e}"))?;

    info!("entering calloop event loop");

    while app.running {
        event_loop
            .dispatch(Some(SIGNAL_CHECK_INTERVAL), &mut app)
            .context("calloop dispatch error")?;

        if SHUTDOWN.swap(false, Ordering::Relaxed) {
            info!("received shutdown signal, exiting");
            app.running = false;
        }

        if RELOAD.swap(false, Ordering::Relaxed) {
            info!("received SIGHUP, reloading config");
            app.reload_config();
        }

        if app.needs_render {
            app.needs_render = false;
            app.redraw_all(&qh);
        }
    }

    info!("shiftpaperd exiting");
    Ok(())
}

fn install_signal_handlers() {
    use nix::sys::signal::{SaFlags, SigAction, SigHandler, SigSet, Signal, sigaction};

    let shutdown = SigAction::new(
        SigHandler::Handler(handle_shutdown),
        SaFlags::SA_RESTART,
        SigSet::empty(),
    );
    let reload = SigAction::new(
        SigHandler::Handler(handle_reload),
        SaFlags::SA_RESTART,
        SigSet::empty(),
    );

    unsafe {
        if let Err(e) = sigaction(Signal::SIGTERM, &shutdown) {
            warn!("failed to install SIGTERM handler: {e}");
        }
        if let Err(e) = sigaction(Signal::SIGINT, &shutdown) {
            warn!("failed to install SIGINT handler: {e}");
        }
        if let Err(e) = sigaction(Signal::SIGHUP, &reload) {
            warn!("failed to install SIGHUP handler: {e}");
        }
    }
}
