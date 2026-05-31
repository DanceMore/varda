## 2025-05-14 - [Parity-based Ping-Pong Optimization]
**Learning:** Sub-mix compositing was using a `copy_texture_to_texture` snapshot pattern, which is $O(N)$ in GPU copies. By adopting the parity-based target selection used in the main Mixer and Channel render paths, we can eliminate these copies entirely.
**Action:** Always check for `copy_texture_to_texture` in compositing loops and replace with parity-based selection between `sub_view` and `scratch_view` to ensure the final result lands in the stable output view.

## 2025-05-14 - [Shader Parameter Hot Path Optimization]
**Learning:** `ShaderParams` was using `HashMap<String, ParamValue>` for storage, leading to $O(N)$ hash lookups every frame during uniform buffer construction. Furthermore, it performed a `ModulationEngine` lookup for every parameter, even when most were unmodulated.
**Action:** Switch to `Vec<ParamValue>` with index-based access and cache the "is_modulated" status per parameter, synchronized by a version counter in the engine. Use map-based conversion only at persistence/UI boundaries.

## 2025-05-14 - [Eliminating Per-Frame Allocations in Channel Hot Path]
**Learning:**  and  were allocating multiple s every frame for deck sorting and compositing metadata. In a performance-critical rendering loop, these heap allocations add significant pressure and latency.
**Action:** Move scratchpad vectors (, , ) into the  struct using . Use a two-pass iteration strategy in  to avoid intermediate "ordered" vectors. Ensure metadata structs implement  to avoid borrow checker conflicts when calling mutable struct methods while iterating.

## 2025-05-14 - [Eliminating Per-Frame Allocations in Channel Hot Path]
**Learning:** `Channel::render` and `Channel::tick_auto_transitions` were allocating multiple `Vec`s every frame for deck sorting and compositing metadata. In a performance-critical rendering loop, these heap allocations add significant pressure and latency.
**Action:** Move scratchpad vectors (`deck_indices`, `composite_info`, `just_started_transitioning`) into the `Channel` struct using `#[serde(skip)]`. Use a two-pass iteration strategy in `render` to avoid intermediate "ordered" vectors. Ensure metadata structs implement `Copy` to avoid borrow checker conflicts when calling mutable struct methods while iterating.

## 2025-05-14 - [Audio Analysis Arc Optimization]
**Learning:** Audio analysis data (`waveform` and `fft` arrays) is updated every frame and cloned multiple times (from audio thread to main loop, then into modulation engine). These were `Vec<f32>`, causing $O(N)$ heap allocations and data copies every frame.
**Action:** Use `Arc<[f32]>` for large audio data arrays. This makes clones $O(1)$ via reference counting. Since the data is read-only once published, `Arc` provides the correct semantics with significantly lower overhead.
