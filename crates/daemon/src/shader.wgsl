// Fullscreen depth-based parallax shader.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle from vertex index, so no vertex buffers are needed.
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(idx & 1u)) * 4.0 - 1.0;
    let y = f32(i32(idx >> 1u)) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

@group(0) @binding(0) var color_tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;

// Must match `Uniforms` in renderer.rs.
struct Uniforms {
    cursor_offset: vec2<f32>,
    intensity: f32,
    _pad: f32,
    // Fraction of the image shown on each axis after cropping to the
    // screen's aspect ratio.
    uv_scale: vec2<f32>,
};
@group(0) @binding(3) var<uniform> u: Uniforms;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Crop to the screen's aspect ratio, then zoom in a further 2.5% each
    // side so displacement pulls in real pixels from beyond the visible
    // area instead of stretching the edges.
    let margin = 0.025;
    let base_uv = 0.5 + (in.uv - 0.5) * u.uv_scale * (1.0 - 2.0 * margin);

    // Depth must come from the same image position as the colour.
    let depth_size = textureDimensions(depth_tex);
    let depth = textureLoad(depth_tex, vec2<u32>(base_uv * vec2<f32>(depth_size)), 0).r;

    // Cursor right (+x) shifts near pixels left. Scaled by uv_scale so a
    // cropped axis moves the same amount on screen as an uncropped one.
    let displaced_uv = base_uv - u.cursor_offset * depth * u.intensity * u.uv_scale;
    return textureSample(color_tex, tex_sampler, displaced_uv);
}
