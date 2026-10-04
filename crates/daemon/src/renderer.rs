use anyhow::{Context, Result};
use bytemuck::Zeroable;
use std::path::Path;
use tracing::{debug, info};
use wgpu::CurrentSurfaceTexture;
use wgpu::util::DeviceExt;

/// Mirrors `Uniforms` in shader.wgsl. `repr(C)` fixes the field order and
/// padding to match the shader, and `Pod` lets bytemuck view it as bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    cursor_offset: [f32; 2],
    intensity: f32,
    _pad: f32,
    uv_scale: [f32; 2],
}

pub struct OutputRenderState {
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub bind_group: wgpu::BindGroup,
    pub color_view: wgpu::TextureView,
    /// Wallpaper size in pixels, for cropping to the screen's aspect ratio.
    pub image_size: (u32, u32),
    pub uniform_buffer: wgpu::Buffer,
    pub current_offset: (f32, f32),
    pub target_offset: (f32, f32),
}

impl OutputRenderState {
    /// Step the lerp 30% of the way toward target_offset and write the new
    /// uniform value. Returns whether the offset moved, so callers can skip
    /// redrawing once it has settled.
    pub fn step_and_write(&mut self, queue: &wgpu::Queue, intensity: f32) -> bool {
        let (tx, ty) = self.target_offset;
        let (cx, cy) = self.current_offset;
        let nx = cx + (tx - cx) * 0.3;
        let ny = cy + (ty - cy) * 0.3;
        self.current_offset = (nx, ny);
        let uniforms = Uniforms {
            cursor_offset: [nx, ny],
            intensity,
            _pad: 0.0,
            uv_scale: cover_uv_scale(self.image_size, (self.config.width, self.config.height)),
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        (nx - cx).abs() > 1e-5 || (ny - cy).abs() > 1e-5
    }
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

    pub fn upload_depth_map(
        &self,
        depth: &crate::depth::DepthMap,
    ) -> (wgpu::Texture, wgpu::TextureView) {
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

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        debug!(
            w = depth.width,
            h = depth.height,
            "uploaded depth map to GPU"
        );
        (texture, view)
    }

    pub fn create_bind_group(
        &self,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        uniform_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shiftpaper_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Upload the wallpaper image. Returns its view and size in pixels.
    pub fn load_wallpaper_texture(&self, path: &Path) -> Result<(wgpu::TextureView, (u32, u32))> {
        let img = image::open(path)
            .with_context(|| format!("failed to load image: {}", path.display()))?
            .to_rgba8();

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
            &img,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            size,
        );

        Ok((
            texture.create_view(&wgpu::TextureViewDescriptor::default()),
            (w, h),
        ))
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
