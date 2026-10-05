//! Steps through [slideshow]'s images on a timer.

use crate::renderer::{DecodedWallpaper, Renderer, Wallpaper};
use anyhow::{Result, anyhow};
use shiftpaper_config::SlideshowConfig;
use std::thread::JoinHandle;
use std::time::SystemTime;
use tracing::{info, warn};

pub struct Slideshow {
    config: SlideshowConfig,
    /// The slide on screen, or the last one tried if it failed to load.
    index: Option<usize>,
    /// The slide's textures, shared by every output showing it.
    wallpaper: Option<Wallpaper>,
    /// A slide being decoded on a background thread, as decoding a large
    /// image takes long enough to stall the parallax.
    loading: Option<(usize, JoinHandle<Result<DecodedWallpaper>>)>,
}

impl Slideshow {
    pub fn new(config: SlideshowConfig) -> Self {
        Self {
            config,
            index: None,
            wallpaper: None,
            loading: None,
        }
    }

    /// The slide to show now, loaded straight away if it isn't already, for
    /// an output that needs something to show. Skips slides that fail to
    /// load. `screens` are the screen sizes to scale slides down to.
    pub fn current(&mut self, renderer: &Renderer, screens: &[(u32, u32)]) -> Option<Wallpaper> {
        if self.wallpaper.is_none() {
            let len = self.config.images.len();
            let due = self.config.due(SystemTime::now());
            for index in (due..len).chain(0..due) {
                let (color, depth) = self.config.slide(index);
                match renderer.load_wallpaper(color, &depth, screens) {
                    Ok(wallpaper) => {
                        info!(path = %color.display(), "slideshow started");
                        self.index = Some(index);
                        self.wallpaper = Some(wallpaper);
                        break;
                    }
                    Err(e) => warn!("slideshow: {e:#}"),
                }
            }
        }
        self.wallpaper.clone()
    }

    /// Check whether it's time for the next slide. Returns it once it's
    /// due and has finished loading in the background.
    pub fn poll(&mut self, renderer: &Renderer, screens: &[(u32, u32)]) -> Option<Wallpaper> {
        let due = self.config.due(SystemTime::now());
        if self.index == Some(due) {
            return None;
        }

        let handle = match self.loading.take() {
            Some((index, handle)) if index == due => handle,
            // Nothing is loading, or a slide that's no longer due is, which
            // happens after a long sleep. Dropping its handle leaves the
            // thread to finish on its own.
            _ => {
                let (color, depth) = self.config.slide(due);
                let color = color.to_path_buf();
                let screens = screens.to_vec();
                let handle =
                    std::thread::spawn(move || DecodedWallpaper::load(&color, &depth, &screens));
                self.loading = Some((due, handle));
                return None;
            }
        };
        if !handle.is_finished() {
            self.loading = Some((due, handle));
            return None;
        }

        // Either way this slide has had its turn, so a broken one isn't
        // retried until it comes round again.
        self.index = Some(due);
        let decoded = handle
            .join()
            .unwrap_or_else(|_| Err(anyhow!("the loading thread panicked")));
        match decoded {
            Ok(decoded) => {
                info!(path = %self.config.slide(due).0.display(), "next slide");
                let wallpaper = renderer.upload_wallpaper(&decoded);
                self.wallpaper = Some(wallpaper.clone());
                Some(wallpaper)
            }
            Err(e) => {
                warn!("slideshow: {e:#}");
                None
            }
        }
    }
}
