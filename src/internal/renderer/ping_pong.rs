//! Ping-pong render targets.
//!
//! Compositing a stack of N layers means repeatedly reading "everything blended
//! so far" while writing the next blend on top. A texture can't be sampled and
//! used as a render attachment in the same pass, so the naive approach snapshots
//! the accumulator with a full-screen `copy_texture_to_texture` before each
//! blend — N-1 redundant GPU copies per stack, per frame.
//!
//! `PingPong` owns two same-format textures and alternates them: each blend reads
//! from `background_view()` and writes to `target_view()`, then `advance()` makes
//! that fresh result the new background. `result_view()` is *invariantly* the
//! latest content, so downstream stages never see the scratch texture and there
//! is no "swap it back if it landed in the wrong texture" footgun to forget.
//!
//! This is the one place the parity/swap invariant lives. Every compositing loop
//! (channel deck-blend, channel effect-chain, mixer channel-blend, sub-mix,
//! master-effects) drives it instead of open-coding the snapshot copy.

use super::GpuContext;

pub struct PingPong {
    /// Holds the latest composited content (read-from for the next blend).
    front_tex: wgpu::Texture,
    front_view: wgpu::TextureView,
    /// Non-sRGB reinterpret view of `front_tex` for egui previews (see
    /// `result_view_linear`). Swapped in lockstep with `front_view`.
    front_view_linear: wgpu::TextureView,
    /// Scratch attachment the next blend writes into.
    back_tex: wgpu::Texture,
    back_view: wgpu::TextureView,
    back_view_linear: wgpu::TextureView,
}

/// A view reinterpreting an sRGB texture as its linear counterpart (identity for
/// already-linear formats). The texture must allow it via `view_formats`.
fn linear_view(tex: &wgpu::Texture) -> wgpu::TextureView {
    tex.create_view(&wgpu::TextureViewDescriptor {
        format: Some(tex.format().remove_srgb_suffix()),
        ..Default::default()
    })
}

impl PingPong {
    pub fn new(context: &GpuContext, width: u32, height: u32) -> Self {
        let front_tex = context.create_render_texture(width, height);
        let front_view = front_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let front_view_linear = linear_view(&front_tex);
        let back_tex = context.create_render_texture(width, height);
        let back_view = back_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let back_view_linear = linear_view(&back_tex);
        Self { front_tex, front_view, front_view_linear, back_tex, back_view, back_view_linear }
    }

    /// Reallocate both targets at a new size.
    pub fn resize(&mut self, context: &GpuContext, width: u32, height: u32) {
        self.front_tex = context.create_render_texture(width, height);
        self.front_view = self.front_tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.front_view_linear = linear_view(&self.front_tex);
        self.back_tex = context.create_render_texture(width, height);
        self.back_view = self.back_tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.back_view_linear = linear_view(&self.back_tex);
    }

    /// The latest composited content — the canonical output, always current.
    pub fn result_view(&self) -> &wgpu::TextureView {
        &self.front_view
    }

    /// The latest composited content as a non-sRGB reinterpret view. egui samples
    /// preview textures expecting raw bytes (it does gamma in-shader); handing it
    /// the sRGB view instead lets the hardware decode on sample and double-darkens
    /// the preview versus the real output. Register this with egui, not `result_view`.
    pub fn result_view_linear(&self) -> &wgpu::TextureView {
        &self.front_view_linear
    }

    /// The latest composited content as a texture (for size queries / final copies).
    pub fn result_texture(&self) -> &wgpu::Texture {
        &self.front_tex
    }

    /// Attachment to render the next blend into.
    pub fn target_view(&self) -> &wgpu::TextureView {
        &self.back_view
    }

    /// Content to read as the "blended so far" background for the next blend.
    pub fn background_view(&self) -> &wgpu::TextureView {
        &self.front_view
    }

    /// Call after a blend has written into `target_view()`: the fresh result now
    /// lives in `back`, so swap it to `front` to become the next background.
    pub fn advance(&mut self) {
        std::mem::swap(&mut self.front_tex, &mut self.back_tex);
        std::mem::swap(&mut self.front_view, &mut self.back_view);
        std::mem::swap(&mut self.front_view_linear, &mut self.back_view_linear);
    }

    pub fn width(&self) -> u32 {
        self.front_tex.width()
    }

    pub fn height(&self) -> u32 {
        self.front_tex.height()
    }
}
