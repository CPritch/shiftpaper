use crate::depth::{DepthGrid, DepthMap, GRID_CELLS, Ground, RANK_POINTS, load_depth_map};
use anyhow::{Context, Result};
use bytemuck::Zeroable;
use image::imageops::{self, FilterType};
use shiftpaper_config::Transition;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};
use wgpu::CurrentSurfaceTexture;
use wgpu::util::DeviceExt;

/// Mirrors `Uniforms` in shader.wgsl. `repr(C)` fixes the field order and
/// padding to match the shader, and `Pod` lets bytemuck view it as bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    cursor_offset: [f32; 2],
    intensity: f32,
    progress: f32,
    uv_scale: [f32; 2],
    next_uv_scale: [f32; 2],
    /// From `shader_style`.
    style: u32,
    /// The screen's width over its height.
    aspect: f32,
    /// The portal's trail: how many points are in use, how far the first
    /// sphere has to reach, the points themselves and how the spheres are
    /// paced, from `Trail`.
    trail_len: u32,
    portal_reach: f32,
    trail: [[f32; 4]; TRAIL_POINTS],
    portal_pace: [[f32; 4]; PACE_POINTS.div_ceil(4)],
    /// Where the ground is in each wallpaper, for the tide.
    ground: [f32; 4],
    next_ground: [f32; 4],
}

/// How many cursor positions a portal transition follows. Must match
/// TRAIL_POINTS in shader.wgsl.
const TRAIL_POINTS: usize = 32;
/// How far the cursor moves, in screen heights, before the portal grows
/// from a new point.
const TRAIL_SPACING: f32 = 0.05;
/// How close a depth of 0 is taken to be, which keeps the farthest pixels
/// at a finite distance, as with the tide's TIDE_NEAR. Must match
/// PORTAL_NEAR in shader.wgsl.
const PORTAL_NEAR: f32 = 0.1;
/// How many distances `Trail::pace` covers. Must match PACE_POINTS in
/// shader.wgsl.
const PACE_POINTS: usize = 33;

/// Where the cursor has been during a portal transition. The new wallpaper
/// grows from each point like a sphere, from when the cursor got there.
#[derive(Clone, Copy)]
struct Trail {
    /// x and y in screen uv, then the transition's progress when the cursor
    /// was there. The fourth number pads to the shader's vec4.
    points: [[f32; 4]; TRAIL_POINTS],
    len: usize,
    /// The furthest any pixel can be from where the first sphere starts,
    /// in the units of scene_point in shader.wgsl.
    reach: f32,
    /// How far through its growth a sphere is, from 0 to 1, once it has
    /// grown to each of PACE_POINTS evenly spaced distances out to `reach`,
    /// packed four to a vec4 for the shader. It's the fraction of the
    /// screen within that distance of the first sphere's centre, so the
    /// spheres change a similar amount of the picture at every moment, like
    /// the ranks the sweeps use.
    pace: [[f32; 4]; PACE_POINTS.div_ceil(4)],
}

impl Trail {
    /// Start a trail at `cursor`, over a wallpaper with depths `grid`,
    /// cropped by `uv_scale` to fit a screen `aspect` wide.
    fn new(cursor: [f32; 2], aspect: f32, grid: &DepthGrid, uv_scale: [f32; 2]) -> Self {
        // As crop in shader.wgsl, and back again.
        let zoom = |axis: usize| uv_scale[axis] * (1.0 - 2.0 * MARGIN);
        let to_image = |screen: f32, axis: usize| 0.5 + (screen - 0.5) * zoom(axis);
        let to_screen = |image: f32, axis: usize| 0.5 + (image - 0.5) / zoom(axis);
        let cell = |image: f32| ((image * GRID_CELLS as f32) as usize).min(GRID_CELLS - 1);
        // As scene_point in shader.wgsl.
        let point = |screen: [f32; 2], depth: f32| {
            [
                (screen[0] - 0.5) * aspect,
                screen[1] - 0.5,
                -(depth + PORTAL_NEAR).ln(),
            ]
        };
        let distance =
            |a: [f32; 3], b: [f32; 3]| (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f32>().sqrt();

        // The depth under the cursor is somewhere in the range of its cell
        // or the ones around it, in case it's just over the edge.
        let (col, row) = (cell(to_image(cursor[0], 0)), cell(to_image(cursor[1], 1)));
        let mut under = [f32::MAX, f32::MIN];
        for r in row.saturating_sub(1)..(row + 2).min(GRID_CELLS) {
            for c in col.saturating_sub(1)..(col + 2).min(GRID_CELLS) {
                let cell = grid.cells[r * GRID_CELLS + c];
                under = [under[0].min(cell.least), under[1].max(cell.greatest)];
            }
        }
        let center = point(cursor, grid.cells[row * GRID_CELLS + col].mean);

        // A cell's pixels are within its part of the screen and its range
        // of depths, so the furthest of them is no further than one of its
        // corners at one end of that range, from either end of the range
        // under the cursor. For pacing, each cell counts as its area at its
        // middle and mean depth.
        let mut reach: f32 = 1e-6;
        let mut cells = Vec::with_capacity(grid.cells.len());
        for row in 0..GRID_CELLS {
            let top = to_screen(row as f32 / GRID_CELLS as f32, 1).max(0.0);
            let bottom = to_screen((row + 1) as f32 / GRID_CELLS as f32, 1).min(1.0);
            for col in 0..GRID_CELLS {
                let left = to_screen(col as f32 / GRID_CELLS as f32, 0).max(0.0);
                let right = to_screen((col + 1) as f32 / GRID_CELLS as f32, 0).min(1.0);
                // Cropped off the screen.
                if top >= bottom || left >= right {
                    continue;
                }
                let depths = grid.cells[row * GRID_CELLS + col];
                for x in [left, right] {
                    for y in [top, bottom] {
                        for depth in [depths.least, depths.greatest] {
                            for from in under {
                                let d = distance(point([x, y], depth), point(cursor, from));
                                reach = reach.max(d);
                            }
                        }
                    }
                }
                let middle = [(left + right) / 2.0, (top + bottom) / 2.0];
                let area = (right - left) * (bottom - top);
                cells.push((distance(point(middle, depths.mean), center), area));
            }
        }

        // How much of the screen is within each distance.
        let mut within = [0.0; PACE_POINTS];
        for (d, area) in cells {
            let step = (d / reach * (PACE_POINTS - 1) as f32).ceil() as usize;
            within[step.min(PACE_POINTS - 1)] += area;
        }
        let mut total = 0.0;
        for w in &mut within {
            total += *w;
            *w = total;
        }
        let mut pace = [[0.0; 4]; PACE_POINTS.div_ceil(4)];
        for (i, w) in within.into_iter().enumerate() {
            pace[i / 4][i % 4] = w / total;
        }

        let mut points = [[0.0; 4]; TRAIL_POINTS];
        points[0] = [cursor[0], cursor[1], 0.0, 0.0];
        Self {
            points,
            len: 1,
            reach,
            pace,
        }
    }

    /// Follow the cursor, adding a point once it's moved far enough from
    /// the last. Once the trail is full it stops growing.
    fn follow(&mut self, cursor: [f32; 2], progress: f32, aspect: f32) {
        if self.len == TRAIL_POINTS {
            return;
        }
        let [x, y, ..] = self.points[self.len - 1];
        if ((cursor[0] - x) * aspect).hypot(cursor[1] - y) >= TRAIL_SPACING {
            self.points[self.len] = [cursor[0], cursor[1], progress, 0.0];
            self.len += 1;
        }
    }
}

/// How far the shader zooms in on each side. Must match MARGIN in
/// shader.wgsl.
const MARGIN: f32 = 0.025;

/// A wallpaper's images decoded into memory, ready to upload. Decoding is
/// the slow part, so it can happen off the main thread.
pub struct DecodedWallpaper {
    color: image::RgbaImage,
    /// None if the depth map couldn't be loaded, so the wallpaper still
    /// shows, just without parallax.
    depth: Option<DepthMap>,
    /// From `DepthMap::ranks` and `DepthMap::height_ranks`, for ordering
    /// transitions.
    ranks: Vec<[f32; 2]>,
    /// From `DepthMap::grid`, for the portal.
    grid: DepthGrid,
    /// From `DepthMap::ground`, for the tide.
    ground: Ground,
    files: WallpaperFiles,
}

/// The color and depth images a wallpaper is loaded from.
pub type WallpaperFiles = (PathBuf, PathBuf);

impl DecodedWallpaper {
    /// Load a baked wallpaper. If it's bigger than any of `screens` needs,
    /// it's scaled down, which makes every frame cheaper to draw and
    /// saves memory.
    pub fn load(color_path: &Path, depth_path: &Path, screens: &[(u32, u32)]) -> Result<Self> {
        let mut color = image::open(color_path)
            .with_context(|| format!("failed to load image: {}", color_path.display()))?
            .to_rgba8();
        let mut depth = match load_depth_map(depth_path) {
            Ok(map) => Some(map),
            Err(e) => {
                warn!("using a flat depth map: {e:#}");
                None
            }
        };
        if let Some(scale) = fit_scale(color.dimensions(), screens) {
            let (w, h) = scale_size(color.dimensions(), scale);
            debug!(w, h, "scaling wallpaper down to fit the screens");
            color = imageops::resize(&color, w, h, FilterType::Triangle);
            depth = depth.map(|map| {
                let (w, h) = scale_size((map.width, map.height), scale);
                map.resized(w, h)
            });
        }
        let ground = depth.as_ref().map_or(Ground::LEVEL, DepthMap::ground);
        // A flat wallpaper is all one depth, so it changes all at once,
        // halfway through a transition.
        let ranks = match &depth {
            Some(map) => (map.ranks().into_iter())
                .zip(map.height_ranks(&ground))
                .map(|(depth, height)| [depth, height])
                .collect(),
            None => vec![[0.5, 0.5]; RANK_POINTS],
        };
        let grid = depth.as_ref().map_or_else(DepthGrid::flat, DepthMap::grid);
        Ok(Self {
            color,
            depth,
            ranks,
            grid,
            ground,
            files: (color_path.to_path_buf(), depth_path.to_path_buf()),
        })
    }
}

/// How much to shrink an image so it still covers each of `screens` once
/// cropped to their shape, or None if it isn't bigger than they need.
fn fit_scale((width, height): (u32, u32), screens: &[(u32, u32)]) -> Option<f32> {
    let needed = screens
        .iter()
        .map(|&(w, h)| f32::max(w as f32 / width as f32, h as f32 / height as f32))
        .fold(0.0, f32::max)
        // The shader zooms in a little, which needs a little more detail.
        / (1.0 - 2.0 * MARGIN);
    (needed > 0.0 && needed < 1.0).then_some(needed)
}

/// `size` multiplied by `scale`, rounded up.
fn scale_size((width, height): (u32, u32), scale: f32) -> (u32, u32) {
    let scale = |n: u32| ((n as f32 * scale).ceil() as u32).max(1);
    (scale(width), scale(height))
}

/// A wallpaper's textures on the GPU. Cloning shares the textures, so
/// outputs showing the same wallpaper upload it once.
#[derive(Clone)]
pub struct Wallpaper {
    color: wgpu::TextureView,
    depth: wgpu::TextureView,
    ranks: wgpu::TextureView,
    /// Where things are in the scene, roughly, for the portal.
    grid: DepthGrid,
    /// Where the ground is, for the tide.
    ground: Ground,
    /// Image size in pixels, for cropping to the screen's aspect ratio.
    size: (u32, u32),
    pub files: WallpaperFiles,
}

/// A transition to another wallpaper that's under way.
struct ActiveTransition {
    next: Wallpaper,
    start: Instant,
    duration: Duration,
    style: Transition,
    trail: Trail,
    /// Frames drawn so far, logged at the end to show how smooth it was.
    frames: u32,
}

pub struct OutputRenderState {
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    bind_group: wgpu::BindGroup,
    current: Wallpaper,
    transition: Option<ActiveTransition>,
    uniform_buffer: wgpu::Buffer,
    current_offset: (f32, f32),
    pub target_offset: (f32, f32),
}

impl OutputRenderState {
    pub fn new(
        renderer: &Renderer,
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
        wallpaper: Wallpaper,
    ) -> Self {
        let uniform_buffer = renderer.create_uniform_buffer();
        let bind_group = renderer.create_bind_group(&wallpaper, &wallpaper, &uniform_buffer);
        Self {
            surface,
            config,
            bind_group,
            current: wallpaper,
            transition: None,
            uniform_buffer,
            current_offset: (0.0, 0.0),
            target_offset: (0.0, 0.0),
        }
    }

    /// The wallpaper this output is showing, or changing to if a transition
    /// is under way.
    pub fn showing(&self) -> &Wallpaper {
        match &self.transition {
            Some(t) => &t.next,
            None => &self.current,
        }
    }

    /// Show a wallpaper straight away, cancelling any transition.
    pub fn set_wallpaper(&mut self, renderer: &Renderer, wallpaper: Wallpaper) {
        self.transition = None;
        self.bind_group = renderer.create_bind_group(&wallpaper, &wallpaper, &self.uniform_buffer);
        self.current = wallpaper;
    }

    /// Start a transition to `next`. A transition that is already running
    /// skips to its end first.
    pub fn start_transition(
        &mut self,
        renderer: &Renderer,
        next: Wallpaper,
        duration: Duration,
        style: Transition,
    ) {
        if let Some(running) = self.transition.take() {
            self.current = running.next;
        }
        self.bind_group = renderer.create_bind_group(&self.current, &next, &self.uniform_buffer);
        let (x, y) = self.current_offset;
        self.transition = Some(ActiveTransition {
            next,
            start: Instant::now(),
            duration,
            style,
            trail: Trail::new(
                [x + 0.5, y + 0.5],
                self.aspect(),
                &self.current.grid,
                cover_uv_scale(self.current.size, (self.config.width, self.config.height)),
            ),
            frames: 0,
        });
    }

    /// Once a transition has run its course, make its wallpaper the current
    /// one. Call after drawing, so the final frame of the transition has
    /// been shown.
    pub fn finish_transition_if_done(&mut self, renderer: &Renderer) {
        let done = self
            .transition
            .as_ref()
            .is_some_and(|t| t.start.elapsed() >= t.duration);
        if done && let Some(finished) = self.transition.take() {
            debug!(frames = finished.frames, "transition finished");
            self.set_wallpaper(renderer, finished.next);
        }
    }

    fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height as f32
    }

    /// Step the lerp 30% of the way toward target_offset and write the new
    /// uniforms. Returns whether anything changed, so callers can skip
    /// redrawing once the offset has settled and no transition is running.
    pub fn step_and_write(&mut self, queue: &wgpu::Queue, intensity: f32) -> bool {
        let (tx, ty) = self.target_offset;
        let (cx, cy) = self.current_offset;
        let nx = cx + (tx - cx) * 0.3;
        let ny = cy + (ty - cy) * 0.3;
        self.current_offset = (nx, ny);

        let screen = (self.config.width, self.config.height);
        let aspect = self.aspect();
        let (progress, next, style, trail) = match &mut self.transition {
            Some(t) => {
                t.frames += 1;
                let progress = transition_progress(t.start.elapsed(), t.duration);
                t.trail.follow([nx + 0.5, ny + 0.5], progress, aspect);
                (progress, &t.next, t.style, Some(t.trail))
            }
            None => (0.0, &self.current, Transition::default(), None),
        };
        let uniforms = Uniforms {
            cursor_offset: [nx, ny],
            intensity,
            progress,
            uv_scale: cover_uv_scale(self.current.size, screen),
            next_uv_scale: cover_uv_scale(next.size, screen),
            style: shader_style(style),
            aspect,
            trail_len: trail.map_or(0, |t| t.len as u32),
            portal_reach: trail.map_or(0.0, |t| t.reach),
            trail: trail.map_or([[0.0; 4]; TRAIL_POINTS], |t| t.points),
            portal_pace: trail.map_or(Zeroable::zeroed(), |t| t.pace),
            ground: self.current.ground.0,
            next_ground: next.ground.0,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        let moved = (nx - cx).abs() > 1e-5 || (ny - cy).abs() > 1e-5;
        moved || self.transition.is_some()
    }
}

/// The shader's number for each transition. Must match the constants in
/// shader.wgsl.
fn shader_style(style: Transition) -> u32 {
    match style {
        Transition::SweepIn => 0,
        Transition::SweepOut => 1,
        Transition::Morph => 2,
        Transition::Dissolve => 3,
        Transition::Portal => 4,
        Transition::TideIn => 5,
        Transition::TideOut => 6,
    }
}

/// How far through a transition we are, eased in and out so it
/// starts and settles gently. Reaches exactly 1 once `duration` has passed.
fn transition_progress(elapsed: Duration, duration: Duration) -> f32 {
    if duration.is_zero() {
        return 1.0;
    }
    let t = (elapsed.as_secs_f32() / duration.as_secs_f32()).min(1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How much of the image to show on each axis so it fills a screen of a
/// different aspect ratio without stretching. The overflowing axis gets a
/// value below 1 and is cropped equally on both sides.
fn cover_uv_scale(image: (u32, u32), screen: (u32, u32)) -> [f32; 2] {
    let image_aspect = image.0 as f32 / image.1 as f32;
    let screen_aspect = screen.0 as f32 / screen.1 as f32;
    if image_aspect > screen_aspect {
        [screen_aspect / image_aspect, 1.0]
    } else {
        [1.0, image_aspect / screen_aspect]
    }
}

pub struct Renderer {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
    _depth_placeholder: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
}

impl Renderer {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                ..Default::default()
            })
            .await
            .context("no suitable GPU adapter found")?;

        info!(name = adapter.get_info().name, "using GPU");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("shiftpaper"),
                required_features: wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .context("failed to create device")?;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shiftpaper_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // The wallpaper being transitioned to: color, then depth.
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Depth ranks for ordering transitions: current, then next.
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D1,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D1,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shiftpaper_pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shiftpaper_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shiftpaper_rp"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shiftpaper_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let depth_placeholder = create_placeholder_depth(&device, &queue);
        let depth_view = depth_placeholder.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            pipeline,
            bind_group_layout,
            sampler,
            _depth_placeholder: depth_placeholder,
            depth_view,
        })
    }

    /// Create a uniform buffer for an output. Called once per output during
    /// layer surface configure, and filled in by `step_and_write` before
    /// the first draw.
    pub fn create_uniform_buffer(&self) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniforms"),
                contents: bytemuck::bytes_of(&Uniforms::zeroed()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Load a baked wallpaper from disk onto the GPU, scaled down to fit
    /// `screens` if it's bigger than they need.
    pub fn load_wallpaper(
        &self,
        color: &Path,
        depth: &Path,
        screens: &[(u32, u32)],
    ) -> Result<Wallpaper> {
        Ok(self.upload_wallpaper(&DecodedWallpaper::load(color, depth, screens)?))
    }

    pub fn upload_wallpaper(&self, decoded: &DecodedWallpaper) -> Wallpaper {
        Wallpaper {
            color: self.upload_color(&decoded.color),
            depth: match &decoded.depth {
                Some(map) => self.upload_depth_map(map),
                None => self.depth_view.clone(),
            },
            ranks: self.upload_ranks(&decoded.ranks),
            grid: decoded.grid.clone(),
            ground: decoded.ground,
            size: decoded.color.dimensions(),
            files: decoded.files.clone(),
        }
    }

    fn upload_ranks(&self, ranks: &[[f32; 2]]) -> wgpu::TextureView {
        let texture = self.device.create_texture_with_data(
            &self.queue,
            &wgpu::TextureDescriptor {
                label: Some("depth_ranks"),
                size: wgpu::Extent3d {
                    width: ranks.len() as u32,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D1,
                format: wgpu::TextureFormat::Rg32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(ranks),
        );
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn upload_depth_map(&self, depth: &DepthMap) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width: depth.width,
            height: depth.height,
            depth_or_array_layers: 1,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_map"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&depth.data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(2 * depth.width),
                rows_per_image: Some(depth.height),
            },
            size,
        );

        debug!(
            w = depth.width,
            h = depth.height,
            "uploaded depth map to GPU"
        );
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Bind the current wallpaper and the one being transitioned to, which
    /// is the same wallpaper when there's no transition.
    fn create_bind_group(
        &self,
        current: &Wallpaper,
        next: &Wallpaper,
        uniform_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shiftpaper_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&current.color),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&current.depth),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&next.color),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&next.depth),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&current.ranks),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&next.ranks),
                },
            ],
        })
    }

    fn upload_color(&self, img: &image::RgbaImage) -> wgpu::TextureView {
        let (w, h) = img.dimensions();

        let size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("wallpaper_color"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            img,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            size,
        );
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Draw and present one frame. Returns false if nothing was presented,
    /// so the caller can retry on the next frame callback.
    pub fn render_frame(&self, output: &OutputRenderState) -> bool {
        let frame = match output.surface.get_current_texture() {
            // A suboptimal frame is still usable, and our size only changes on
            // a compositor configure, which reconfigures the surface anyway.
            CurrentSurfaceTexture::Success(frame) | CurrentSurfaceTexture::Suboptimal(frame) => {
                frame
            }
            CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => return false,
            CurrentSurfaceTexture::Outdated | CurrentSurfaceTexture::Lost => {
                output.surface.configure(&self.device, &output.config);
                return false;
            }
            CurrentSurfaceTexture::Validation => {
                tracing::warn!("surface validation error while acquiring a frame");
                return false;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shiftpaper_enc"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shiftpaper_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &output.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(frame);
        true
    }
}

fn create_placeholder_depth(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_placeholder"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R16Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[0u8; 2],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(2),
            rows_per_image: Some(1),
        },
        size,
    );
    tex
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: [f32; 2], expected: [f32; 2]) {
        assert!(
            (actual[0] - expected[0]).abs() < 1e-6 && (actual[1] - expected[1]).abs() < 1e-6,
            "{actual:?} != {expected:?}"
        );
    }

    #[test]
    fn matching_aspect_is_not_cropped() {
        assert_close(cover_uv_scale((3840, 2160), (2560, 1440)), [1.0, 1.0]);
    }

    #[test]
    fn wider_image_is_cropped_horizontally() {
        // A 16:9 image on a 16:10 laptop panel shows 90% of its width.
        assert_close(cover_uv_scale((3840, 2160), (2560, 1600)), [0.9, 1.0]);
    }

    #[test]
    fn taller_image_is_cropped_vertically() {
        assert_close(cover_uv_scale((1000, 2000), (2000, 1000)), [1.0, 0.25]);
    }

    #[test]
    fn transition_numbers_match_the_shader() {
        let src = include_str!("shader.wgsl");
        for (name, style) in [
            ("SWEEP_IN", Transition::SweepIn),
            ("SWEEP_OUT", Transition::SweepOut),
            ("MORPH", Transition::Morph),
            ("DISSOLVE", Transition::Dissolve),
            ("PORTAL", Transition::Portal),
            ("TIDE_IN", Transition::TideIn),
            ("TIDE_OUT", Transition::TideOut),
        ] {
            let line = format!("const {name}: u32 = {}u;", shader_style(style));
            assert!(src.contains(&line), "{line}");
        }
    }

    #[test]
    fn portal_constants_match_the_shader() {
        let src = include_str!("shader.wgsl");
        for line in [
            format!("const PORTAL_NEAR: f32 = {PORTAL_NEAR:?};"),
            format!("const PACE_POINTS: u32 = {PACE_POINTS}u;"),
            format!("const TRAIL_POINTS: u32 = {TRAIL_POINTS}u;"),
        ] {
            assert!(src.contains(&line), "{line}");
        }
    }

    /// How far a pixel at `screen` is from where a trail from `cursor`
    /// starts, as portal_arrival in shader.wgsl works it out.
    fn portal_distance(
        map: &DepthMap,
        uv_scale: [f32; 2],
        aspect: f32,
        cursor: [f32; 2],
        screen: [f32; 2],
    ) -> f32 {
        // As crop, load_depth and scene_point in shader.wgsl.
        let point = |s: [f32; 2]| {
            let image = |axis: usize| 0.5 + (s[axis] - 0.5) * uv_scale[axis] * (1.0 - 2.0 * MARGIN);
            let x = ((image(0) * map.width as f32) as u32).min(map.width - 1);
            let y = ((image(1) * map.height as f32) as u32).min(map.height - 1);
            let depth = f32::from(map.data[(y * map.width + x) as usize]) / 65535.0;
            [
                (s[0] - 0.5) * aspect,
                s[1] - 0.5,
                -(depth + PORTAL_NEAR).ln(),
            ]
        };
        let (p, c) = (point(screen), point(cursor));
        (0..3).map(|i| (p[i] - c[i]).powi(2)).sum::<f32>().sqrt()
    }

    /// As pace in shader.wgsl.
    fn pace(trail: &Trail, distance: f32) -> f32 {
        let x = (distance / trail.reach).clamp(0.0, 1.0) * (PACE_POINTS - 1) as f32;
        let i = (x as usize).min(PACE_POINTS - 2);
        let (a, b) = (
            trail.pace[i / 4][i % 4],
            trail.pace[(i + 1) / 4][(i + 1) % 4],
        );
        a + (b - a) * (x - i as f32)
    }

    /// A depth map wider than the screen, with depths from `depth` at
    /// each pixel.
    fn depth_map(depth: impl Fn(u32, u32) -> f32) -> DepthMap {
        let (width, height) = (300, 160);
        let data = (0..width * height)
            .map(|i| (depth(i % width, i / width) * 65535.0) as u16)
            .collect();
        DepthMap {
            data,
            width,
            height,
        }
    }

    /// Pixel centres all over a 160 x 100 screen.
    fn screen_pixels() -> impl Iterator<Item = [f32; 2]> {
        (0..160 * 100).map(|i| {
            [
                ((i % 160) as f32 + 0.5) / 160.0,
                ((i / 160) as f32 + 0.5) / 100.0,
            ]
        })
    }

    const CURSORS: [[f32; 2]; 4] = [[0.5, 0.5], [0.0, 0.0], [0.9, 0.2], [0.3, 0.97]];

    #[test]
    fn the_portal_reaches_every_pixel() {
        // A slope with sharp bumps all over it.
        let map =
            depth_map(|x, y| y as f32 / 160.0 * 0.6 + ((x * 7 + y * 13) % 23) as f32 / 22.0 * 0.4);
        let uv_scale = cover_uv_scale((map.width, map.height), (1600, 1000));
        for cursor in CURSORS {
            let trail = Trail::new(cursor, 1.6, &map.grid(), uv_scale);
            for screen in screen_pixels() {
                let distance = portal_distance(&map, uv_scale, 1.6, cursor, screen);
                // To within a pixel.
                assert!(distance <= trail.reach + 1e-3, "{cursor:?} {screen:?}");
            }
        }
    }

    #[test]
    fn the_portal_changes_the_picture_steadily() {
        // A slope, and a slope with a few specks far away, like sky through
        // trees, which shouldn't hold the rest up.
        let slope = |_, y| y as f32 / 160.0;
        let specks = |x, y| {
            if (x + y * 3) % 97 == 0 {
                0.0
            } else {
                0.5 + y as f32 / 320.0
            }
        };
        for map in [depth_map(slope), depth_map(specks)] {
            let uv_scale = cover_uv_scale((map.width, map.height), (1600, 1000));
            for cursor in CURSORS {
                let trail = Trail::new(cursor, 1.6, &map.grid(), uv_scale);
                let paces: Vec<f32> = screen_pixels()
                    .map(|s| pace(&trail, portal_distance(&map, uv_scale, 1.6, cursor, s)))
                    .collect();
                for part in [0.25, 0.5, 0.75] {
                    let done =
                        paces.iter().filter(|&&p| p <= part).count() as f32 / paces.len() as f32;
                    assert!((done - part).abs() < 0.1, "{cursor:?} {part} {done}");
                }
            }
        }
    }

    #[test]
    fn the_trail_follows_the_cursor_in_steps() {
        let mut trail = Trail::new([0.5, 0.5], 1.0, &DepthGrid::flat(), [1.0, 1.0]);
        trail.follow([0.51, 0.5], 0.1, 1.0);
        assert_eq!(trail.len, 1, "a small move doesn't add a point");
        trail.follow([0.6, 0.5], 0.2, 1.0);
        assert_eq!(trail.len, 2);
        assert_eq!(trail.points[1], [0.6, 0.5, 0.2, 0.0]);
        for i in 0..100 {
            trail.follow([0.6 + 0.1 * i as f32, 0.5], 0.3, 1.0);
        }
        assert_eq!(trail.len, TRAIL_POINTS, "it stops when full");
    }

    #[test]
    fn tide_near_matches_the_shader() {
        let line = format!("const TIDE_NEAR: f32 = {:?};", crate::depth::TIDE_NEAR);
        assert!(include_str!("shader.wgsl").contains(&line), "{line}");
    }

    #[test]
    fn margin_matches_the_shader() {
        let line = format!("const MARGIN: f32 = {MARGIN};");
        assert!(include_str!("shader.wgsl").contains(&line), "{line}");
    }

    #[test]
    fn large_images_shrink_to_cover_the_largest_screen() {
        // A 16:10 screen is wider than a 3:2 image, so width sets the size.
        let scale = fit_scale((6000, 4000), &[(1920, 1080), (2560, 1600)]).unwrap();
        assert!((scale - 2560.0 / 6000.0 / 0.95).abs() < 1e-6, "{scale}");
    }

    #[test]
    fn rotated_screens_need_more_height() {
        let landscape = fit_scale((6000, 4000), &[(2560, 1440)]).unwrap();
        let portrait = fit_scale((6000, 4000), &[(1440, 2560)]).unwrap();
        assert!(portrait > landscape);
    }

    #[test]
    fn small_images_are_left_alone() {
        assert_eq!(fit_scale((1920, 1080), &[(2560, 1440)]), None);
        assert_eq!(fit_scale((2560, 1440), &[(2560, 1440)]), None);
        assert_eq!(fit_scale((6000, 4000), &[]), None);
    }

    #[test]
    fn transitions_ease_from_zero_to_one() {
        let second = Duration::from_secs(1);
        assert_eq!(transition_progress(Duration::ZERO, second), 0.0);
        assert_eq!(transition_progress(second / 2, second), 0.5);
        assert_eq!(transition_progress(second, second), 1.0);
        assert_eq!(transition_progress(second * 2, second), 1.0);
        // Eased, so it starts slower than linear.
        assert!(transition_progress(second / 10, second) < 0.1);
    }

    #[test]
    fn zero_length_transitions_finish_at_once() {
        assert_eq!(transition_progress(Duration::ZERO, Duration::ZERO), 1.0);
    }

    #[test]
    fn shader_is_valid_and_uniforms_match() {
        let src = include_str!("shader.wgsl");
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));

        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        let (uniforms, _) = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("Uniforms"))
            .expect("shader declares Uniforms");
        assert_eq!(
            layouter[uniforms].size as usize,
            std::mem::size_of::<Uniforms>()
        );
    }
}
