## 2026-05-29 - [Ping-Pong Compositing Optimization]
**Learning:** Using a parity-based target selection strategy for multi-layer compositing eliminates the need for intermediate GPU texture copies (`copy_texture_to_texture`). By pre-calculating whether a layer should target the final destination or a ping-pong buffer based on the remaining layer count, the final blend operation is guaranteed to land in the correct texture.
**Action:** Apply this pattern to all multi-pass or multi-layer rendering loops to reduce GPU overhead.
