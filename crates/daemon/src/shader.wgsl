// Fullscreen depth-based parallax shader, with depth-ordered transitions
// between two wallpapers.

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
// The wallpaper being transitioned to. The same as the current one when
// no transition is running.
@group(0) @binding(4) var next_color_tex: texture_2d<f32>;
@group(0) @binding(5) var next_depth_tex: texture_2d<f32>;
// Each wallpaper's depth ranks, from `DepthMap::ranks`.
@group(0) @binding(6) var ranks: texture_1d<f32>;
@group(0) @binding(7) var next_ranks: texture_1d<f32>;

// Must match `Uniforms` in renderer.rs.
struct Uniforms {
    cursor_offset: vec2<f32>,
    intensity: f32,
    // 0 shows the current wallpaper, 1 the next one.
    progress: f32,
    // Fraction of each image shown on each axis after cropping to the
    // screen's aspect ratio.
    uv_scale: vec2<f32>,
    next_uv_scale: vec2<f32>,
    // Which transition, one of the constants below.
    style: u32,
    // The screen's width over its height.
    aspect: f32,
    // Where the cursor has been in a portal transition: screen uv, then the
    // progress when it was there. The first `trail_len` are in use.
    trail_len: u32,
    // How fast each of the portal's bubbles grows, in screen heights over
    // the whole transition.
    portal_speed: f32,
    trail: array<vec4<f32>, TRAIL_POINTS>,
};
@group(0) @binding(3) var<uniform> u: Uniforms;

// Zoom in 2.5% each side so displacement pulls in real pixels from beyond
// the visible area instead of stretching the edges. Must match MARGIN in
// renderer.rs.
const MARGIN: f32 = 0.025;
// Transitions. Must match `shader_style` in renderer.rs.
const SWEEP_IN: u32 = 0u;
const SWEEP_OUT: u32 = 1u;
const MORPH: u32 = 2u;
const FLATTEN: u32 = 3u;
const DISSOLVE: u32 = 4u;
const PORTAL: u32 = 5u;

// The portal's settings, in screen heights. Must match renderer.rs.
const TRAIL_POINTS: u32 = 32u;
const PORTAL_DEPTH: f32 = 1.0;
const PORTAL_EDGE: f32 = 0.04;

// Half the width of the slice of depth ranks that is part way through
// switching at any moment. Narrower gives a crisper wavefront, wider a
// softer dissolve.
const BAND: f32 = 0.08;

// Crop to the screen's aspect ratio and apply the margin.
fn crop(uv: vec2<f32>, scale: vec2<f32>) -> vec2<f32> {
    return 0.5 + (uv - 0.5) * scale * (1.0 - 2.0 * MARGIN);
}

fn load_depth(tex: texture_2d<f32>, uv: vec2<f32>) -> f32 {
    let size = textureDimensions(tex);
    return textureLoad(tex, vec2<u32>(uv * vec2<f32>(size)), 0).r;
}

// The fraction of a wallpaper's pixels farther away than `depth`,
// interpolated from its 257 ranks.
fn depth_rank(table: texture_1d<f32>, depth: f32) -> f32 {
    let x = clamp(depth, 0.0, 1.0) * 256.0;
    let i = min(u32(x), 255u);
    let farther = textureLoad(table, i, 0).r;
    let next = textureLoad(table, i + 1u, 0).r;
    return mix(farther, next, x - f32(i));
}

// How far a pixel has switched to the next wallpaper in a sweep. Think of
// a plane sweeping through both scenes, taking away the current
// wallpaper's surfaces and putting the next one's in place as it passes
// them, with each pixel showing whichever surface is nearest.
//
// Sweeping in, from near to far, a pixel switches once the plane reaches
// the next wallpaper's surface there, which is then in front of anything
// left of the current one. Sweeping out, from far to near, it also has to
// wait for the current wallpaper's surface to go, as that is in front
// until it does.
//
// Depths are compared by rank, so the sweep spends similar time on each
// part of the picture.
fn sweep_weight(depth: f32, next_depth: f32, t: f32) -> f32 {
    let rank = depth_rank(ranks, depth);
    let next_rank = depth_rank(next_ranks, next_depth);
    // How far the plane has to travel before this pixel switches.
    let distance = select(max(rank, next_rank), 1.0 - next_rank, u.style == SWEEP_IN);
    return swept(distance, t);
}

// How far a pixel `distance` along has switched as a plane travels from 0
// to 1 over the transition. It actually goes from -BAND to 1 + BAND, so
// every pixel starts at 0 and ends at 1.
fn swept(distance: f32, t: f32) -> f32 {
    let plane = t * (1.0 + 2.0 * BAND) - BAND;
    return 1.0 - smoothstep(plane - BAND, plane + BAND, distance);
}

// A pseudo-random number in [0, 1) for each pair of whole numbers.
fn hash(p: vec2<u32>) -> f32 {
    var h = p.x * 1664525u + p.y * 1013904223u;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return f32(h) / 4294967296.0;
}

// Smooth noise: a random value at each whole number, blended in between.
fn value_noise(p: vec2<f32>) -> f32 {
    let cell = vec2<u32>(floor(p));
    let f = fract(p);
    let s = f * f * (3.0 - 2.0 * f);
    let below = mix(hash(cell), hash(cell + vec2<u32>(1u, 0u)), s.x);
    let above = mix(hash(cell + vec2<u32>(0u, 1u)), hash(cell + vec2<u32>(1u, 1u)), s.x);
    return mix(below, above, s.y);
}

// Soft blobs, about six to the screen's height, with finer detail at
// their edges.
fn blobs(screen_uv: vec2<f32>) -> f32 {
    let p = vec2<f32>(screen_uv.x * u.aspect, screen_uv.y) * 6.0;
    return 0.7 * value_noise(p) + 0.3 * value_noise(p * 3.1 + 17.0);
}

// How far a pixel has switched in a portal. The new wallpaper grows like
// a bubble in the scene from each point on the cursor's trail, so it
// spreads across the surface under the cursor first and has to wrap round
// things nearer or further away.
fn portal_weight(screen_uv: vec2<f32>, rank: f32, t: f32) -> f32 {
    var weight = 0.0;
    for (var i = 0u; i < u.trail_len && weight < 1.0; i += 1u) {
        let point = u.trail[i];
        let radius = (t - point.z) * u.portal_speed - PORTAL_EDGE;
        let across = (screen_uv - point.xy) * vec2<f32>(u.aspect, 1.0);
        // Depth only adds distance, so a bubble that can't reach this far
        // across the screen can be skipped before looking up its depth.
        if length(across) >= radius + PORTAL_EDGE {
            continue;
        }
        let point_rank = depth_rank(ranks, load_depth(depth_tex, crop(point.xy, u.uv_scale)));
        let offset = vec3<f32>(across, (rank - point_rank) * PORTAL_DEPTH);
        let inside = 1.0 - smoothstep(radius - PORTAL_EDGE, radius + PORTAL_EDGE, length(offset));
        weight = max(weight, inside);
    }
    return weight;
}

// How far a pixel has changed into the next wallpaper, for its colour and
// its depth separately, and how much depth the scene has right now.
struct Blend {
    color: f32,
    depth: f32,
    depth_scale: f32,
};

fn transition_blend(screen_uv: vec2<f32>, depth: f32, next_depth: f32, t: f32) -> Blend {
    if u.style == MORPH {
        // The shape of the scene leads and its colours follow, so it reads
        // as a morph rather than a crossfade.
        return Blend(smoothstep(0.25, 1.0, t), smoothstep(0.0, 0.75, t), 1.0);
    }
    if u.style == FLATTEN {
        // Flat through the middle fifth, while the pictures swap over, so
        // the change of shape can't be seen.
        let flatness = smoothstep(0.0, 0.4, t) - smoothstep(0.6, 1.0, t);
        return Blend(smoothstep(0.35, 0.65, t), step(0.5, t), 1.0 - flatness);
    }
    if u.style == DISSOLVE {
        // Half depth order, as in sweep-in, and half random blobs.
        let next_rank = depth_rank(next_ranks, next_depth);
        let weight = swept(mix(1.0 - next_rank, blobs(screen_uv), 0.5), t);
        return Blend(weight, weight, 1.0);
    }
    if u.style == PORTAL {
        let weight = portal_weight(screen_uv, depth_rank(ranks, depth), t);
        return Blend(weight, weight, 1.0);
    }
    let weight = sweep_weight(depth, next_depth, t);
    return Blend(weight, weight, 1.0);
}

// Linear sRGB to Oklab, from https://bottosson.github.io/posts/oklab/
fn to_oklab(c: vec3<f32>) -> vec3<f32> {
    let lms = vec3<f32>(
        dot(c, vec3<f32>(0.4122214708, 0.5363325363, 0.0514459929)),
        dot(c, vec3<f32>(0.2119034982, 0.6806995451, 0.1073969566)),
        dot(c, vec3<f32>(0.0883024619, 0.2817188376, 0.6299787005)),
    );
    let l = pow(max(lms, vec3<f32>(0.0)), vec3<f32>(1.0 / 3.0));
    return vec3<f32>(
        dot(l, vec3<f32>(0.2104542553, 0.7936177850, -0.0040720468)),
        dot(l, vec3<f32>(1.9779984951, -2.4285922050, 0.4505937099)),
        dot(l, vec3<f32>(0.0259040371, 0.7827717662, -0.8086757660)),
    );
}

fn from_oklab(lab: vec3<f32>) -> vec3<f32> {
    let l = vec3<f32>(
        dot(lab, vec3<f32>(1.0, 0.3963377774, 0.2158037573)),
        dot(lab, vec3<f32>(1.0, -0.1055613458, -0.0638541728)),
        dot(lab, vec3<f32>(1.0, -0.0894841775, -1.2914855480)),
    );
    let lms = l * l * l;
    return vec3<f32>(
        dot(lms, vec3<f32>(4.0767416621, -3.3077115913, 0.2309699292)),
        dot(lms, vec3<f32>(-1.2684380046, 2.6097574011, -0.3413193965)),
        dot(lms, vec3<f32>(-0.0041960771, -0.7034186147, 1.7076147010)),
    );
}

// Mix two colours in Oklab, where halfway between them looks halfway to
// the eye. Mixing the light itself lets a bright picture swamp a dark one,
// so a fade looks lopsided.
fn mix_oklab(a: vec4<f32>, b: vec4<f32>, t: f32) -> vec4<f32> {
    if t <= 0.0 {
        return a;
    }
    if t >= 1.0 {
        return b;
    }
    let lab = mix(to_oklab(a.rgb), to_oklab(b.rgb), t);
    return vec4<f32>(clamp(from_oklab(lab), vec3<f32>(0.0), vec3<f32>(1.0)), mix(a.a, b.a, t));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = crop(in.uv, u.uv_scale);
    // Depth comes from the same image position as the colour.
    var depth = load_depth(depth_tex, uv);
    if u.progress <= 0.0 {
        // Cursor right (+x) shifts near pixels left. Scaled by uv_scale so a
        // cropped axis moves the same amount on screen as an uncropped one.
        let shift = u.cursor_offset * depth * u.intensity;
        return textureSample(color_tex, tex_sampler, uv - shift * u.uv_scale);
    }

    let next_uv = crop(in.uv, u.next_uv_scale);
    let next_depth = load_depth(next_depth_tex, next_uv);
    let blend = transition_blend(in.uv, depth, next_depth, u.progress);
    // Blending depth as well as colour morphs the parallax geometry, and
    // both images shift by it so they move together.
    depth = mix(depth, next_depth, blend.depth) * blend.depth_scale;
    let shift = u.cursor_offset * depth * u.intensity;
    let color = textureSample(color_tex, tex_sampler, uv - shift * u.uv_scale);
    let next_color = textureSample(next_color_tex, tex_sampler, next_uv - shift * u.next_uv_scale);
    return mix_oklab(color, next_color, blend.color);
}
