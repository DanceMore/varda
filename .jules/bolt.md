## 2025-05-14 - Redundant Texture Copies in Compositing
**Learning:** The compositing loop was using  to snapshot the "composite-so-far" result before each new layer. This creates N-1 redundant copies for N layers, wasting GPU bandwidth.
**Action:** Use a ping-pong strategy alternating between two existing textures ( and ) and perform a final pointer/reference swap if needed. This eliminates all intermediate copies.
## 2025-05-14 - Redundant Texture Copies in Compositing
**Learning:** The compositing loop was using `copy_texture_to_texture` to snapshot the "composite-so-far" result before each new layer. This creates N-1 redundant copies for N layers, wasting GPU bandwidth.
**Action:** Use a ping-pong strategy alternating between two existing textures (`composite_texture` and `effect_ping_texture`) and perform a final pointer/reference swap if needed. This eliminates all intermediate copies.
