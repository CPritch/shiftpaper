//! Shiftpaper's depth parallax renderer. It draws baked wallpapers to wgpu
//! surfaces and knows nothing about Wayland, so any front end can use it:
//! shiftpaperd creates the surfaces from its layer shell windows.

mod depth;
mod renderer;

pub use renderer::{DecodedWallpaper, OutputRenderState, Renderer, Wallpaper, WallpaperFiles};
pub use shiftpaper_config::Transition;
