use anyhow::{Context, Result};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_layer, delegate_output, delegate_pointer, delegate_registry,
    delegate_seat, delegate_shm,
    output::{OutputHandler, OutputInfo as SctkOutputInfo, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    seat::{
        Capability, SeatHandler, SeatState,
        pointer::{PointerEvent, PointerEventKind, PointerHandler},
    },
    shell::{
        WaylandSurface,
        wlr_layer::{
            Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
            LayerSurfaceConfigure,
        },
    },
    shm::{Shm, ShmHandler},
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use tracing::{debug, info, warn};
use wayland_client::{
    Connection, Proxy, QueueHandle,
    globals::GlobalList,
    protocol::{wl_output, wl_pointer, wl_seat, wl_surface},
};

use crate::config::{Config, TrackingMode};
use crate::cursor::HyprlandCursor;
use crate::renderer::{OutputRenderState, Renderer};
use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle,
};

pub struct OutputInfo {
    pub name: String,
    pub wl_output: wl_output::WlOutput,
    /// Position in global logical coordinates.
    pub x: i32,
    pub y: i32,
    /// Surface size in logical pixels, from the layer surface configure.
    pub width: u32,
    pub height: u32,
    pub scale: i32,
    pub layer_surface: LayerSurface,
    pub configured: bool,
}

impl OutputInfo {
    fn refresh(&mut self, info: &SctkOutputInfo) {
        self.name = info.name.clone().unwrap_or_default();
        (self.x, self.y) = info.logical_position.unwrap_or(info.location);
        self.scale = info.scale_factor;
    }
}

pub struct App {
    pub config: Config,
    pub registry_state: RegistryState,
    pub compositor_state: CompositorState,
    pub output_state: OutputState,
    pub seat_state: SeatState,
    pub layer_shell: LayerShell,
    pub shm: Shm,
    pub outputs: Vec<OutputInfo>,
    pub renderer: Option<Renderer>,
    pub render_targets: HashMap<String, OutputRenderState>,
    pub cursor: Option<HyprlandCursor>,
    pub pointer: Option<wl_pointer::WlPointer>,
    pub seat: Option<wl_seat::WlSeat>,
    pub idle: Option<crate::idle::IdleState>,
    pub battery_ok: bool,
    /// Set by gate transitions (idle resume, battery resume) to request
    /// a one-shot render from the main loop after dispatch returns.
    pub needs_render: bool,
    pub running: bool,
}

impl App {
    pub fn new(config: Config, globals: &GlobalList, qh: &QueueHandle<Self>) -> Result<Self> {
        let registry_state = RegistryState::new(globals);
        let compositor_state =
            CompositorState::bind(globals, qh).context("wl_compositor not available")?;
        let output_state = OutputState::new(globals, qh);
        let seat_state = SeatState::new(globals, qh);
        let layer_shell = LayerShell::bind(globals, qh).map_err(|_| {
            anyhow::anyhow!(
                "wlr-layer-shell (zwlr_layer_shell_v1) is not available on this compositor.\n\
                 \n\
                 shiftpaperd requires a compositor that supports the wlr-layer-shell protocol.\n\
                 Supported: Hyprland, Sway, River, Wayfire, niri, labwc\n\
                 Not supported: GNOME Wayland, KDE Plasma (without third-party patches)"
            )
        })?;
        let shm = Shm::bind(globals, qh).context("wl_shm not available")?;

        Ok(Self {
            config,
            registry_state,
            compositor_state,
            output_state,
            seat_state,
            layer_shell,
            shm,
            outputs: Vec::new(),
            renderer: None,
            render_targets: HashMap::new(),
            cursor: None,
            pointer: None,
            seat: None,
            idle: None,
            battery_ok: true,
            needs_render: false,
            running: true,
        })
    }

    /// True if rendering should currently happen. Combines idle state
    /// (set by ext-idle-notify-v1 events) and battery state (refreshed
    /// by a slow calloop timer).
    pub fn render_allowed(&self) -> bool {
        let not_idle = self.idle.as_ref().map(|i| !i.idle).unwrap_or(true);
        not_idle && self.battery_ok
    }

    /// Refresh the battery_ok flag from sysfs. Called from a slow timer.
    pub fn refresh_battery(&mut self) {
        let state = crate::battery::read();
        let allowed = crate::battery::render_allowed(state, self.config.daemon.battery_threshold);
        if allowed != self.battery_ok {
            tracing::info!(allowed, ?state, "battery gate changed");
            self.battery_ok = allowed;
            if allowed {
                self.needs_render = true;
            }
        }
    }

    /// Hot-reload config.toml. Reloads wallpaper textures and rebuilds
    /// bind groups for every output. Tracking mode and idle timeout
    /// changes require a full restart.
    pub fn reload_config(&mut self) {
        let mut new_cfg = match crate::config::Config::load() {
            Ok(c) => c,
            Err(e) => {
                warn!("SIGHUP reload failed: {e:#}");
                return;
            }
        };

        if new_cfg.daemon.tracking_mode != self.config.daemon.tracking_mode {
            warn!("tracking_mode changed, restart required to take effect");
            // pointer_frame reads the mode at runtime, but the hyprland
            // timer is only set up at startup. Keep running in the old mode.
            new_cfg.daemon.tracking_mode = self.config.daemon.tracking_mode;
        }
        if new_cfg.daemon.idle_timeout_secs != self.config.daemon.idle_timeout_secs {
            warn!("idle_timeout_secs changed, restart required to take effect");
        }

        let renderer = match &self.renderer {
            Some(r) => r,
            None => {
                self.config = new_cfg;
                return;
            }
        };

        for output in &self.outputs {
            if !output.configured {
                continue;
            }
            if let Some(rt) = self.render_targets.get_mut(&output.name) {
                let color_path = new_cfg.color_for(&output.name).to_path_buf();
                let depth_path = new_cfg.depth_for(&output.name);

                let color_view = match renderer.load_wallpaper_texture(&color_path) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!(name = output.name, "reload: failed to load color: {e:#}");
                        continue;
                    }
                };

                let bind_group = match crate::depth::load_depth_map(&depth_path) {
                    Ok(depth) => {
                        let (_tex, view) = renderer.upload_depth_map(&depth);
                        renderer.create_bind_group(&color_view, &view, &rt.uniform_buffer)
                    }
                    Err(e) => {
                        warn!(name = output.name, "reload: failed to load depth: {e:#}");
                        renderer.create_bind_group(
                            &color_view,
                            &renderer.depth_view,
                            &rt.uniform_buffer,
                        )
                    }
                };

                rt.color_view = color_view;
                rt.bind_group = bind_group;
                info!(name = output.name, "reloaded wallpaper");
            }
        }

        self.config = new_cfg;
        self.needs_render = true;
        info!("config reloaded via SIGHUP");
    }

    pub fn init_cursor(&mut self) {
        self.cursor = HyprlandCursor::new();
        if self.cursor.is_none() {
            warn!("Hyprland IPC unavailable, parallax disabled");
        }
    }

    /// Hyprland-mode tick: read the global cursor position, step each
    /// output's offset towards it, and redraw only the outputs that moved.
    /// Not called in pointer mode (the timer source is not inserted).
    pub fn tick(&mut self, qh: &QueueHandle<Self>) {
        if !self.render_allowed() {
            return;
        }
        let (Some(cursor), Some(renderer)) = (&self.cursor, &self.renderer) else {
            return;
        };
        let Some(pos) = cursor.position() else {
            return;
        };

        for o in &self.outputs {
            if !o.configured {
                continue;
            }
            let Some(rt) = self.render_targets.get_mut(&o.name) else {
                continue;
            };
            rt.target_offset = crate::cursor::offset_from_centre(pos, o.x, o.y, o.width, o.height);
            if rt.step_and_write(&renderer.queue, self.config.intensity_for(&o.name)) {
                let surface = o.layer_surface.wl_surface();
                surface.frame(qh, surface.clone());
                renderer.render_frame(rt);
            }
        }
    }

    fn create_layer_surface(
        &self,
        qh: &QueueHandle<Self>,
        output: &wl_output::WlOutput,
    ) -> LayerSurface {
        let surface = self.compositor_state.create_surface(qh);
        let layer_surface = self.layer_shell.create_layer_surface(
            qh,
            surface,
            Layer::Background,
            Some("shiftpaper"),
            Some(output),
        );
        layer_surface.set_anchor(Anchor::all());
        layer_surface.set_exclusive_zone(-1);
        layer_surface.set_keyboard_interactivity(KeyboardInteractivity::None);
        layer_surface.set_size(0, 0);
        layer_surface.commit();
        layer_surface
    }

    pub fn render_all(&self, qh: &QueueHandle<Self>) {
        if !self.render_allowed() {
            return;
        }
        let renderer = match &self.renderer {
            Some(r) => r,
            None => return,
        };

        for output in &self.outputs {
            if !output.configured {
                continue;
            }

            if let Some(render_state) = self.render_targets.get(&output.name) {
                let surface = output.layer_surface.wl_surface();
                surface.frame(qh, surface.clone());
                renderer.render_frame(render_state);
            }
        }
    }

    /// Render a single named output. Used by pointer mode to avoid
    /// re-rendering monitors that didn't receive the cursor event.
    pub fn render_output(&self, qh: &QueueHandle<Self>, output_name: &str) {
        if !self.render_allowed() {
            return;
        }
        let renderer = match &self.renderer {
            Some(r) => r,
            None => return,
        };
        let output = match self.outputs.iter().find(|o| o.name == output_name) {
            Some(o) => o,
            None => return,
        };
        if !output.configured {
            return;
        }
        if let Some(rt) = self.render_targets.get(output_name) {
            let surface = output.layer_surface.wl_surface();
            surface.frame(qh, surface.clone());
            renderer.render_frame(rt);
        }
    }
}

impl CompositorHandler for App {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        new_factor: i32,
    ) {
        debug!(scale = new_factor, "surface scale factor changed");
    }

    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_transform: wl_output::Transform,
    ) {
    }

    fn frame(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _time: u32,
    ) {
    }

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }
}

impl OutputHandler for App {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    /// Called once the output's info is complete, both at startup and
    /// when a monitor is plugged in.
    fn new_output(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        let mut o = OutputInfo {
            name: String::new(),
            wl_output: output.clone(),
            x: 0,
            y: 0,
            width: 0,
            height: 0,
            scale: 1,
            layer_surface: self.create_layer_surface(qh, &output),
            configured: false,
        };
        if let Some(info) = self.output_state.info(&output) {
            o.refresh(&info);
        }
        info!(name = o.name, "created layer surface for output");
        self.outputs.push(o);
    }

    fn update_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        let Some(info) = self.output_state.info(&output) else {
            return;
        };
        if let Some(o) = self.outputs.iter_mut().find(|o| o.wl_output == output) {
            o.refresh(&info);
            debug!(name = o.name, scale = o.scale, "output updated");
        }
    }

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        // The wgpu surface holds a raw pointer to the wl_surface, so the
        // render target must be dropped before the layer surface.
        if let Some(o) = self.outputs.iter().find(|o| o.wl_output == output) {
            info!(name = o.name, "output removed");
            self.render_targets.remove(&o.name);
        }
        self.outputs.retain(|o| o.wl_output != output);
    }
}

impl LayerShellHandler for App {
    fn closed(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _layer: &LayerSurface) {
        warn!("layer surface closed by compositor");
    }

    fn configure(
        &mut self,
        conn: &Connection,
        qh: &QueueHandle<Self>,
        layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let (w, h) = (configure.new_size.0, configure.new_size.1);
        debug!(w, h, "layer surface configured");
        // We anchor to all four edges, so the protocol requires the
        // compositor to choose a size. wgpu panics on a zero-sized surface.
        if w == 0 || h == 0 {
            warn!("configure: compositor sent a zero size, ignoring");
            return;
        }

        if self.renderer.is_none() {
            info!("initializing wgpu renderer...");
            match pollster::block_on(Renderer::new()) {
                Ok(r) => self.renderer = Some(r),
                Err(e) => {
                    warn!("failed to init renderer: {e:#}");
                    return;
                }
            }
        }

        let output_idx = match self.outputs.iter().position(|o| &o.layer_surface == layer) {
            Some(i) => i,
            None => {
                warn!("configure: no matching output for layer surface");
                return;
            }
        };

        self.outputs[output_idx].width = w;
        self.outputs[output_idx].height = h;
        self.outputs[output_idx].configured = true;

        let output_name = self.outputs[output_idx].name.clone();
        let output_w = self.outputs[output_idx].width;
        let output_h = self.outputs[output_idx].height;

        let renderer = self.renderer.as_ref().unwrap();

        if !self.render_targets.contains_key(&output_name) {
            let display_ptr = conn.backend().display_ptr() as *mut c_void;
            let wl_surface = layer.wl_surface();
            let surface_ptr = wl_surface.id().as_ptr() as *mut c_void;

            let display_handle =
                WaylandDisplayHandle::new(NonNull::new(display_ptr).expect("null display ptr"));
            let window_handle =
                WaylandWindowHandle::new(NonNull::new(surface_ptr).expect("null surface ptr"));

            let target = wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: RawDisplayHandle::Wayland(display_handle),
                raw_window_handle: RawWindowHandle::Wayland(window_handle),
            };

            let surface = match unsafe { renderer.instance.create_surface_unsafe(target) } {
                Ok(s) => s,
                Err(e) => {
                    warn!(name = output_name, "failed to create wgpu surface: {e}");
                    return;
                }
            };

            let surface_caps = surface.get_capabilities(&renderer.adapter);
            let alpha_mode = if surface_caps
                .alpha_modes
                .contains(&wgpu::CompositeAlphaMode::PreMultiplied)
            {
                wgpu::CompositeAlphaMode::PreMultiplied
            } else {
                surface_caps.alpha_modes[0]
            };

            let surface_config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                width: output_w,
                height: output_h,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            };

            surface.configure(&renderer.device, &surface_config);

            let color_path = self.config.color_for(&output_name).to_path_buf();
            let depth_path = self.config.depth_for(&output_name);

            let color_view = match renderer.load_wallpaper_texture(&color_path) {
                Ok(v) => v,
                Err(e) => {
                    warn!("failed to load color texture: {e:#}");
                    return;
                }
            };

            let uniform_buffer = renderer.create_uniform_buffer();

            let bind_group = match crate::depth::load_depth_map(&depth_path) {
                Ok(depth) => {
                    let (_tex, view) = renderer.upload_depth_map(&depth);
                    info!(
                        name = output_name,
                        w = depth.width,
                        h = depth.height,
                        path = %depth_path.display(),
                        "depth map loaded"
                    );
                    renderer.create_bind_group(&color_view, &view, &uniform_buffer)
                }
                Err(e) => {
                    warn!(
                        path = %depth_path.display(),
                        "failed to load depth map, using flat placeholder: {e:#}"
                    );
                    renderer.create_bind_group(&color_view, &renderer.depth_view, &uniform_buffer)
                }
            };

            let render_state = OutputRenderState {
                surface,
                config: surface_config,
                bind_group,
                color_view,
                uniform_buffer,
                current_offset: (0.0, 0.0),
                target_offset: (0.0, 0.0),
            };

            self.render_targets
                .insert(output_name.clone(), render_state);
            info!(
                name = output_name,
                w = output_w,
                h = output_h,
                "output initialized"
            );

            layer.wl_surface().frame(qh, layer.wl_surface().clone());
            layer.wl_surface().commit();
        } else {
            if let Some(rt) = self.render_targets.get_mut(&output_name) {
                rt.config.width = output_w;
                rt.config.height = output_h;
                rt.surface.configure(&renderer.device, &rt.config);
                info!(
                    name = output_name,
                    w = output_w,
                    h = output_h,
                    "reconfigured swapchain"
                );
            }
        }

        // Draw now rather than waiting for the next cursor movement, so a
        // newly plugged in or resized output never shows a blank or stale frame.
        self.needs_render = true;
    }
}

impl SeatHandler for App {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat_state
    }

    fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, seat: wl_seat::WlSeat) {
        if self.seat.is_none() {
            self.seat = Some(seat);
        }
    }

    fn new_capability(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        seat: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Pointer && self.pointer.is_none() {
            match self.seat_state.get_pointer(qh, &seat) {
                Ok(pointer) => {
                    info!("acquired pointer from seat");
                    self.pointer = Some(pointer);
                }
                Err(e) => warn!("failed to get pointer: {e}"),
            }
        }
    }

    fn remove_capability(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Pointer
            && let Some(pointer) = self.pointer.take()
        {
            pointer.release();
        }
    }

    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
}

impl PointerHandler for App {
    fn pointer_frame(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        _pointer: &wl_pointer::WlPointer,
        events: &[PointerEvent],
    ) {
        // Pointer events arrive regardless of mode; only act on them in
        // pointer mode. Hyprland mode runs its own polling tick.
        if self.config.daemon.tracking_mode != TrackingMode::Pointer {
            return;
        }

        for event in events {
            let Some(o) = self
                .outputs
                .iter()
                .find(|o| o.layer_surface.wl_surface() == &event.surface)
            else {
                continue;
            };

            match event.kind {
                PointerEventKind::Enter { .. } => debug!(name = o.name, "pointer entered"),
                PointerEventKind::Leave { .. } => debug!(name = o.name, "pointer left"),
                PointerEventKind::Motion { .. } => {
                    let (x, y) = event.position;
                    let target_x = x as f32 / o.width as f32 - 0.5;
                    let target_y = y as f32 / o.height as f32 - 0.5;
                    let intensity = self.config.intensity_for(&o.name);
                    if let (Some(renderer), Some(rt)) =
                        (&self.renderer, self.render_targets.get_mut(&o.name))
                    {
                        rt.target_offset = (target_x, target_y);
                        rt.step_and_write(&renderer.queue, intensity);
                    }
                    self.render_output(qh, &o.name);
                }
                _ => {}
            }
        }
    }
}

impl ShmHandler for App {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

impl ProvidesRegistryState for App {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }

    registry_handlers!(OutputState, SeatState);
}

delegate_compositor!(App);
delegate_output!(App);
delegate_layer!(App);
delegate_seat!(App);
delegate_pointer!(App);
delegate_registry!(App);
delegate_shm!(App);
