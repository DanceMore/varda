## 2025-05-14 - [Parity-based Ping-Pong Optimization]
**Learning:** Sub-mix compositing was using a `copy_texture_to_texture` snapshot pattern, which is $O(N)$ in GPU copies. By adopting the parity-based target selection used in the main Mixer and Channel render paths, we can eliminate these copies entirely.
**Action:** Always check for `copy_texture_to_texture` in compositing loops and replace with parity-based selection between `sub_view` and `scratch_view` to ensure the final result lands in the stable output view.

## 2025-05-14 - [Shader Parameter Hot Path Optimization]
**Learning:** `ShaderParams` was using `HashMap<String, ParamValue>` for storage, leading to $O(N)$ hash lookups every frame during uniform buffer construction. Furthermore, it performed a `ModulationEngine` lookup for every parameter, even when most were unmodulated.
**Action:** Switch to `Vec<ParamValue>` with index-based access and cache the "is_modulated" status per parameter, synchronized by a version counter in the engine. Use map-based conversion only at persistence/UI boundaries.
