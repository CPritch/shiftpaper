// Fullscreen depth-based parallax shader, with transitions between two
// wallpapers that work through the depth of the scene.

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
// Each wallpaper's depth and height ranks, from `DepthMap::ranks` and
// `DepthMap::height_ranks`.
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
    // Where the spheres of a portal or dissolve grow from: screen uv, then
    // the progress when each starts. The first `trail_len` are in use.
    trail_len: u32,
    // How far a sphere grows, in the units of `scene_point`. See `Trail`
    // in renderer.rs.
    portal_reach: f32,
    trail: array<vec4<f32>, TRAIL_POINTS>,
    // How the portal's spheres are paced, PACE_POINTS of them four to
    // a vec4. See `pace`.
    portal_pace: array<vec4<f32>, 9>,
    // When a pixel switches, from when the first sphere reaches it, packed
    // the same way. See `timing`.
    portal_timing: array<vec4<f32>, 9>,
    // Where the ground is in each wallpaper, for the tide. See
    // `tide_height`.
    ground: vec4<f32>,
    next_ground: vec4<f32>,
};
@group(0) @binding(3) var<uniform> u: Uniforms;

// Zoom in each side so displacement pulls in real pixels from beyond the
// visible area instead of stretching the edges: by at least 2.5%, and by
// half the intensity, as far as anything shifts. Must match MARGIN and
// `margin` in renderer.rs.
const MARGIN: f32 = 0.025;
// Transitions. Must match `shader_style` in renderer.rs.
const SWEEP_IN: u32 = 0u;
const SWEEP_OUT: u32 = 1u;
const MORPH: u32 = 2u;
const DISSOLVE: u32 = 3u;
const PORTAL: u32 = 4u;
const TIDE_IN: u32 = 5u;
const TIDE_OUT: u32 = 6u;

// The portal's settings, explained in renderer.rs. Must match it.
const TRAIL_POINTS: u32 = 32u;
const PORTAL_NEAR: f32 = 0.1;
const PACE_POINTS: u32 = 33u;
// How soft the portal's edge is, as a fraction of the transition.
const PORTAL_EDGE: f32 = 0.015;
// How far noise pushes the portal's edge in and out, in the units of
// `scene_point`, and how many bumps it has to a unit.
const PORTAL_NOISE: f32 = 0.12;
const PORTAL_NOISE_SCALE: f32 = 6.0;

// How close a depth of 0 is for the tide. Must match depth.rs.
const TIDE_NEAR: f32 = 0.1;
// Just under the tide's surface the water ripples and darkens a little.
// How far down that goes, in height rank, how far the ripple moves the
// picture, as a fraction of it, and how much darker it gets at the surface.
const WATER_DEPTH: f32 = 0.15;
const RIPPLE: f32 = 0.0025;
const WATER_SHADE: f32 = 0.18;

// Half the width of the band of ranks part way through switching at any
// moment, in the sweeps, the dissolve and the tide. Narrower gives a
// crisper edge, wider a softer one.
const BAND: f32 = 0.08;

// Crop to the screen's aspect ratio and apply the margin.
fn crop(uv: vec2<f32>, scale: vec2<f32>) -> vec2<f32> {
    let margin = max(MARGIN, 0.5 * u.intensity);
    return 0.5 + (uv - 0.5) * scale * (1.0 - 2.0 * margin);
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

// Roughly how high a point is in the scene above `ground`, squashed into
// (-1, 1). Must match tide_height in depth.rs, which explains it.
fn tide_height(uv: vec2<f32>, depth: f32, ground: vec4<f32>) -> f32 {
    let distance = 1.0 / (depth + TIDE_NEAR);
    let height = distance * dot(ground.xyz, vec3<f32>(uv, 1.0)) + ground.w;
    return height / (1.0 + abs(height));
}

// The fraction of a wallpaper's pixels lower in the scene than `height`,
// from the second column of its ranks.
fn height_rank(table: texture_1d<f32>, height: f32) -> f32 {
    let x = clamp((height + 1.0) * 0.5, 0.0, 1.0) * 256.0;
    let i = min(u32(x), 255u);
    let lower = textureLoad(table, i, 0).g;
    let next = textureLoad(table, i + 1u, 0).g;
    return mix(lower, next, x - f32(i));
}

// How far under the tide's water a pixel is, in height rank, so negative
// above it. Coming in, the new wallpaper rises like water over the old
// one's ground, filling the lowest places first, lapping round whatever
// stands out of it and reaching the sky last. Going out, the old wallpaper
// drains off the new one's ground, uncovering its highest places first.
// The water moves through height rank, like the sweeps through depth rank.
fn under_water(uv: vec2<f32>, depth: f32, next_uv: vec2<f32>, next_depth: f32, t: f32) -> f32 {
    let level = t * (1.0 + 2.0 * BAND) - BAND;
    if u.style == TIDE_IN {
        return level - height_rank(ranks, tide_height(uv, depth, u.ground));
    }
    return (1.0 - level) - height_rank(next_ranks, tide_height(next_uv, next_depth, u.next_ground));
}

// Small waves across the tide's surface, which move as it does.
fn wave(screen_uv: vec2<f32>, t: f32) -> f32 {
    return sin(screen_uv.x * 90.0 + t * 50.0)
        + 0.5 * sin(screen_uv.x * 210.0 - t * 70.0 + screen_uv.y * 40.0);
}

// Where a pixel is in the scene: across the screen in screen heights, then
// the log of its distance, from depth as in `tide_height`. Near any one
// point this is the scene shrunk by that point's distance, so a sphere in
// it looks like a sphere in the scene, but far things aren't spread out
// any more than near ones. Trail::new in renderer.rs relies on how far
// apart this puts any two pixels.
fn scene_point(screen_uv: vec2<f32>, depth: f32) -> vec3<f32> {
    return vec3<f32>((screen_uv - 0.5) * vec2<f32>(u.aspect, 1.0), -log(depth + PORTAL_NEAR));
}

// How far through its growth a portal sphere is once it has grown to
// `distance`, from `Trail::pace` in renderer.rs.
fn pace(distance: f32) -> f32 {
    let x = clamp(distance / u.portal_reach, 0.0, 1.0) * f32(PACE_POINTS - 1u);
    let i = min(u32(x), PACE_POINTS - 2u);
    let j = i + 1u;
    return mix(u.portal_pace[i / 4u][i % 4u], u.portal_pace[j / 4u][j % 4u], x - f32(i));
}

// When a pixel switches, as transition progress, once the first sphere
// reaches it at `arrival`, from `Trail::timing` in renderer.rs.
fn timing(arrival: f32) -> f32 {
    let x = clamp(arrival, 0.0, 1.0) * f32(PACE_POINTS - 1u);
    let i = min(u32(x), PACE_POINTS - 2u);
    let j = i + 1u;
    return mix(u.portal_timing[i / 4u][i % 4u], u.portal_timing[j / 4u][j % 4u], x - f32(i));
}

// Smooth noise in the scene, from 0 to 1, made of two layers of
// value_noise blended through depth.
fn scene_noise(p: vec3<f32>) -> f32 {
    let q = p * PORTAL_NOISE_SCALE + 64.0;
    let layer = floor(q.z);
    let s = fract(q.z);
    let below = value_noise(q.xy + layer * 17.0);
    let above = value_noise(q.xy + (layer + 1.0) * 17.0);
    return mix(below, above, s * s * (3.0 - 2.0 * s));
}

// The new wallpaper grows like a sphere in the scene from each point on
// the trail, starting at the surface under it: the cursor's trail for the
// portal, and random points for the dissolve. It spreads over that
// surface first and reaches things nearer or further away later, so its
// front wraps round the shape of the scene. Noise pushes its edge in and
// out, across the screen and in depth. Returns when this pixel switches,
// as transition progress, from when the first sphere reaches it.
fn portal_arrival(screen_uv: vec2<f32>, depth: f32) -> f32 {
    let here = scene_point(screen_uv, depth);
    let bump = (scene_noise(here) - 0.5) * PORTAL_NOISE;
    // Each sphere grows over this much of the transition, so the first
    // has reached everywhere by the end, at the latest.
    let growth = 1.0 - 2.0 * PORTAL_EDGE;
    var first = 2.0;
    for (var i = 0u; i < u.trail_len; i += 1u) {
        let point = u.trail[i];
        // The trail is in the order the spheres start, and no sphere
        // arrives before it starts, so none of the rest can arrive any
        // sooner.
        let start = point.z + PORTAL_EDGE;
        if start >= first {
            break;
        }
        // Depth only adds distance, so a sphere that can't get here sooner
        // going straight across the screen can be skipped before looking
        // up its depth.
        let across = length((screen_uv - point.xy) * vec2<f32>(u.aspect, 1.0));
        if start + pace(across + bump) * growth >= first {
            continue;
        }
        let center = scene_point(point.xy, load_depth(depth_tex, crop(point.xy, u.uv_scale)));
        first = min(first, start + pace(distance(here, center) + bump) * growth);
    }
    return timing(first);
}

// How far a pixel has changed into the next wallpaper, for its colour and
// its depth separately. For the tide, also how close it is under the
// water's surface, from 0 to 1.
struct Blend {
    color: f32,
    depth: f32,
    surface: f32,
};

fn transition_blend(screen_uv: vec2<f32>, depth: f32, next_depth: f32, t: f32) -> Blend {
    if u.style == MORPH {
        // The shape of the scene leads and its colours follow, so it reads
        // as a morph rather than a crossfade.
        return Blend(smoothstep(0.25, 1.0, t), smoothstep(0.0, 0.75, t), 0.0);
    }
    if u.style == PORTAL || u.style == DISSOLVE {
        let arrival = portal_arrival(screen_uv, depth);
        let weight = smoothstep(arrival - PORTAL_EDGE, arrival + PORTAL_EDGE, t);
        return Blend(weight, weight, 0.0);
    }
    if u.style == TIDE_IN || u.style == TIDE_OUT {
        let uv = crop(screen_uv, u.uv_scale);
        let next_uv = crop(screen_uv, u.next_uv_scale);
        let under = under_water(uv, depth, next_uv, next_depth, t);
        // The new wallpaper shows under the water coming in, and above it
        // going out.
        let weight = smoothstep(-BAND, BAND, select(-under, under, u.style == TIDE_IN));
        let surface = smoothstep(-0.02, 0.02, under) * (1.0 - smoothstep(0.0, WATER_DEPTH, under));
        return Blend(weight, weight, surface);
    }
    let weight = sweep_weight(depth, next_depth, t);
    return Blend(weight, weight, 0.0);
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
    depth = mix(depth, next_depth, blend.depth);
    let shift = u.cursor_offset * depth * u.intensity;

    // Under the tide's surface the water ripples and darkens a little. The
    // water is the new wallpaper coming in and the old one going out.
    let ripple = vec2<f32>(0.0, wave(in.uv, u.progress) * RIPPLE * blend.surface);
    let shade = 1.0 - WATER_SHADE * blend.surface;
    let old_is_water = u.style == TIDE_OUT;
    let color = textureSample(
        color_tex,
        tex_sampler,
        uv - shift * u.uv_scale + select(vec2<f32>(0.0), ripple, old_is_water),
    );
    let next_color = textureSample(
        next_color_tex,
        tex_sampler,
        next_uv - shift * u.next_uv_scale + select(ripple, vec2<f32>(0.0), old_is_water),
    );
    return mix_oklab(
        vec4<f32>(color.rgb * select(1.0, shade, old_is_water), color.a),
        vec4<f32>(next_color.rgb * select(shade, 1.0, old_is_water), next_color.a),
        blend.color,
    );
}
