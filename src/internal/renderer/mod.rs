pub mod context;
pub mod dome;
pub mod dome_preview;
pub mod slicer;
pub mod pipeline;
pub mod blit;
pub mod edge_blend;
pub mod hap_convert;
pub mod readback;
pub mod subprocess;
pub mod transition;
pub mod warp;
pub mod ping_pong;

/// Maximum number of parameter slots available for batched rendering.
/// Covers all decks in a channel or all channels in the mixer.
pub const MAX_RENDER_SLOTS: usize = 128;

pub use context::*;
pub use dome::*;
pub use dome_preview::*;
pub use slicer::*;
pub use pipeline::*;
pub use blit::*;
pub use edge_blend::*;
pub use hap_convert::*;
pub use readback::*;
pub use subprocess::*;
pub use transition::*;
pub use warp::*;
pub use ping_pong::*;

