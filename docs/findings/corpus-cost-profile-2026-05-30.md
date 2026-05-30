# Corpus cost profile — generator fill at 1080p

Snapshot for cost-aware conducting. Regenerate with:

```
VARDA_PROFILE_CORPUS=1 VARDA_BENCH_SKIP_SLO=1 cargo bench --bench compositing -- profile_corpus
```

- Hardware: NVIDIA GTX 1660 Super, Vulkan, driver 550.163.01
- Resolution: 1920x1080. **All numbers scale ~linearly with pixel count** — at 720p
  multiply by ~0.44.
- `fill_us` = marginal GPU cost of adding the shader as a layer (total minus the
  250µs solid-deck floor of composite+submit). This is the number to budget a stack
  against: stack cost ≈ floor + Σ fill.
- Tiers: T1 < 1000µs (stack freely), T2 1000–4000µs (a couple at once),
  T3 ≥ 4000µs (raymarch tier — one held centerpiece).
- 76 effects/transitions are excluded: they render over an input, single-pass, and
  are an inherently cheap class (this is where color_balance, tint, blur, kaleidoscope,
  feedback_trails live). Profiled only generators that render standalone as a deck.

## Generators

| tier | total_us | fill_us | shader |
|------|---------:|--------:|--------|
| T3 | 67582 | 67332 | crystal_cave |
| T3 | 48052 | 47802 | fractal_mandelbulb |
| T3 | 37420 | 37170 | fractal_mandelbox |
| T3 | 24426 | 24176 | big_bang |
| T3 | 23712 | 23462 | fractal_menger |
| T3 | 21833 | 21583 | particle_collider |
| T3 | 13010 | 12760 | graph_network |
| T3 | 12546 | 12296 | black_hole |
| T3 |  9392 |  9142 | tas_psychedelic |
| T3 |  6763 |  6513 | hilbert_curve |
| T3 |  4640 |  4390 | oscilloscope |
| T3 |  4522 |  4272 | quantum_membrane |
| T2 |  2101 |  1851 | particle |
| T2 |  2031 |  1781 | turing_3d |
| T2 |  1875 |  1625 | liquid_light |
| T2 |  1710 |  1460 | turing_patterns |
| T2 |  1379 |  1129 | starfield |
| T1 |  1119 |   869 | abstract_field |
| T1 |   967 |   717 | noise |
| T1 |   914 |   664 | game_of_life |
| T1 |   795 |   545 | sacred_geometry |
| T1 |   768 |   518 | generative_feedback |
| T1 |   697 |   447 | voronoi |
| T1 |   612 |   362 | dark_matter |
| T1 |   598 |   348 | fire |
| T1 |   518 |   268 | fractal |
| T1 |   396 |   146 | cymatics |
| T1 |   377 |   127 | lines |
| T1 |   346 |    96 | gemma4 |
| T1 |   335 |    85 | radar |
| T1 |   331 |    81 | tunnelines |
| T1 |   307 |    57 | grid |
| T1 |   302 |    52 | rings |
| T1 |   299 |    49 | plasma |
| T1 |   297 |    47 | shaper |
| T1 |   293 |    43 | checkerboard |
| T1 |   289 |    39 | bars |
| T1 |   288 |    38 | gradient |
| T1 |   287 |    37 | solid_color |

## Note

Line count is not cost. `sacred_geometry` (434 lines) is T1; `black_hole` (541 lines)
is T3 at 12ms. The discriminator is per-pixel loop depth (raymarchers) vs bounded ALU
(2D procedural), not source size.
