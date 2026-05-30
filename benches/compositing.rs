/// GPU compositing benchmarks at 1080p.
///
///   channel_composite_solid  — solid-color decks (LoadOp::Clear, no fragment shader).
///                              The slope across deck counts isolates per-deck
///                              copy-on-composite cost. The level at decks/1
///                              includes fixed render-pass setup, not just
///                              compositing.
///
///   channel_composite_shader — same shape but with bars.fs running on every
///                              pixel. Difference vs solid at N decks ≈
///                              N × per-deck shader execution cost.
///
///   mixer_crossfade          — two channels through the crossfader at 50%.
///
/// After the criterion groups complete, the per-deck slope (decks/8 minus
/// decks/1, divided by 7) is computed from fresh samples and printed.

use std::time::Instant;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use varda::{
    audio::AudioData,
    deck::Deck,
    isf::ISFShader,
    mixer::Mixer,
    modulation::AudioValues,
    renderer::context::GpuContext,
};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

/// 60fps frame budget. Preflight panics if 8-deck solid composite at 1080p
/// exceeds this. Disable with `VARDA_BENCH_SKIP_SLO=1`.
const FRAME_BUDGET_US: u128 = 16_670;

const BARS_SHADER: &str = include_str!("../shaders/bars.fs");
// A real, heavy gig generator (raymarched), not a synthetic worst case — so the
// crossfade split below measures fill the way an actual set would produce it.
const HEAVY_SHADERS: &[(&str, &str)] = &[
    ("sacred_geometry", include_str!("../shaders/sacred_geometry.fs")),
    ("black_hole", include_str!("../shaders/black_hole.fs")),
];

fn make_context() -> Option<GpuContext> {
    GpuContext::new_headless().ok()
}

/// Name the adapter that produced every number in this run. With the Vega bound
/// to a (usually-off) VM, the host exposes the GTX 1660 Super and a software
/// llvmpipe fallback; LowPower headless selection must not silently pick the CPU.
fn print_adapter(ctx: &GpuContext) {
    let info = ctx.adapter.get_info();
    eprintln!(
        "bench adapter: {} | backend={:?} | device_type={:?} | driver={} {}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info,
    );
}

fn poll(ctx: &GpuContext) {
    ctx.device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
        .ok();
}

fn setup_mixer_solid(context: &GpuContext, n_decks: usize) -> Mixer {
    let mut mixer = Mixer::new(context, WIDTH, HEIGHT).expect("mixer");
    let ch = mixer.channel_mut(0).expect("channel 0");
    for i in 0..n_decks {
        let t = i as f32 / n_decks.max(1) as f32;
        let deck = Deck::new_solid_color(context, [t, 0.5, 1.0 - t, 1.0], WIDTH, HEIGHT)
            .expect("solid color deck");
        ch.add_deck(deck);
    }
    mixer
}

fn setup_mixer_shader(context: &GpuContext, n_decks: usize) -> Mixer {
    let mut mixer = Mixer::new(context, WIDTH, HEIGHT).expect("mixer");
    let ch = mixer.channel_mut(0).expect("channel 0");
    for _ in 0..n_decks {
        let shader = ISFShader::from_string(BARS_SHADER).expect("bars shader");
        let deck = Deck::new(context, shader, WIDTH, HEIGHT).expect("shader deck");
        ch.add_deck(deck);
    }
    mixer
}

fn time_render_us(ctx: &GpuContext, mixer: &mut Mixer, samples: usize) -> u128 {
    let audio = AudioData::default();
    let audio_values = AudioValues { sources: Default::default() };
    for _ in 0..3 {
        mixer.render(ctx, &audio, &audio_values).expect("warmup");
        poll(ctx);
    }
    let mut times = Vec::with_capacity(samples);
    for _ in 0..samples {
        let t0 = Instant::now();
        mixer.render(ctx, &audio, &audio_values).expect("render");
        poll(ctx);
        times.push(t0.elapsed().as_micros());
    }
    times.sort_unstable();
    times[samples / 2]
}

/// Preflight: assert the 8-deck render fits the 60fps budget. Runs once
/// before any criterion sampling. Skipped when VARDA_BENCH_SKIP_SLO is set.
fn preflight_slo(ctx: &GpuContext) {
    if std::env::var_os("VARDA_BENCH_SKIP_SLO").is_some() {
        return;
    }
    let mut mixer = setup_mixer_solid(ctx, 8);
    let median = time_render_us(ctx, &mut mixer, 11);
    if median > FRAME_BUDGET_US {
        panic!(
            "SLO violation: 8-deck solid composite at {WIDTH}x{HEIGHT} \
             median = {median}µs, exceeds 60fps budget {FRAME_BUDGET_US}µs"
        );
    }
    eprintln!("preflight: 8-deck solid composite median = {median}µs (budget {FRAME_BUDGET_US}µs)");
}

fn bench_channel_composite_solid(c: &mut Criterion) {
    let Some(ctx) = make_context() else {
        eprintln!("no GPU adapter — skipping");
        return;
    };
    preflight_slo(&ctx);

    let audio = AudioData::default();
    let audio_values = AudioValues { sources: Default::default() };

    let mut group = c.benchmark_group("channel_composite_solid");
    group.sample_size(50);

    for n_decks in [1, 2, 4, 8] {
        let mut mixer = setup_mixer_solid(&ctx, n_decks);
        group.bench_with_input(BenchmarkId::new("decks", n_decks), &n_decks, |b, _| {
            b.iter(|| {
                mixer.render(&ctx, &audio, &audio_values).expect("render");
                poll(&ctx);
            });
        });
    }

    group.finish();
}

fn bench_channel_composite_shader(c: &mut Criterion) {
    let Some(ctx) = make_context() else {
        eprintln!("no GPU adapter — skipping");
        return;
    };

    let audio = AudioData::default();
    let audio_values = AudioValues { sources: Default::default() };

    let mut group = c.benchmark_group("channel_composite_shader");
    group.sample_size(50);

    for n_decks in [1, 2, 4, 8] {
        let mut mixer = setup_mixer_shader(&ctx, n_decks);
        group.bench_with_input(BenchmarkId::new("decks", n_decks), &n_decks, |b, _| {
            b.iter(|| {
                mixer.render(&ctx, &audio, &audio_values).expect("render");
                poll(&ctx);
            });
        });
    }

    group.finish();
}

fn bench_mixer_crossfade(c: &mut Criterion) {
    let Some(ctx) = make_context() else {
        eprintln!("no GPU adapter — skipping");
        return;
    };

    let audio = AudioData::default();
    let audio_values = AudioValues { sources: Default::default() };

    let mut mixer = setup_mixer_solid(&ctx, 1);
    let deck = Deck::new_solid_color(&ctx, [0.0, 1.0, 0.5, 1.0], WIDTH, HEIGHT)
        .expect("solid color deck");
    mixer.channel_mut(1).unwrap().add_deck(deck);
    mixer.set_crossfader(0.5);

    let mut group = c.benchmark_group("mixer_crossfade");
    group.sample_size(50);
    group.bench_function("2ch_50pct", |b| {
        b.iter(|| {
            mixer.render(&ctx, &audio, &audio_values).expect("render");
            poll(&ctx);
        });
    });
    group.finish();
}

/// Re-times solid composites at 1 and 8 decks (warmed, median of 11) and
/// prints the per-deck slope to stderr.
fn report_per_deck_slope(_c: &mut Criterion) {
    let Some(ctx) = make_context() else { return };
    let mut m1 = setup_mixer_solid(&ctx, 1);
    let mut m8 = setup_mixer_solid(&ctx, 8);
    let t1 = time_render_us(&ctx, &mut m1, 11);
    let t8 = time_render_us(&ctx, &mut m8, 11);
    let slope = (t8 as i128 - t1 as i128) / 7;
    eprintln!(
        "per-deck copy-on-composite slope (solid, {WIDTH}x{HEIGHT}): \
         {slope}µs/deck   [decks/1={t1}µs, decks/8={t8}µs]"
    );
}

/// Two channels, one heavy gig shader deck each.
fn setup_two_channel_heavy(context: &GpuContext, src: &str) -> Mixer {
    let mut mixer = Mixer::new(context, WIDTH, HEIGHT).expect("mixer");
    for ch_idx in 0..2 {
        let shader = ISFShader::from_string(src).expect("heavy shader");
        let deck = Deck::new(context, shader, WIDTH, HEIGHT).expect("shader deck");
        mixer.channel_mut(ch_idx).expect("channel").add_deck(deck);
    }
    mixer
}

/// Median (CPU-submit µs, total-polled µs) over `samples` warmed renders.
///
/// `render()` returns after the last `queue.submit()` without blocking, so its
/// duration is pure CPU encode+submit. The subsequent poll-to-idle is the GPU
/// work still outstanding after that last submit. Caveat: GPU execution overlaps
/// CPU submission, so the tail *under*-counts true GPU time while the CPU figure
/// is clean — which is the conservative direction for "is this CPU-bound?".
fn time_split_us(ctx: &GpuContext, mixer: &mut Mixer, samples: usize) -> (u128, u128) {
    let audio = AudioData::default();
    let audio_values = AudioValues { sources: Default::default() };
    for _ in 0..5 {
        mixer.render(ctx, &audio, &audio_values).expect("warmup");
        poll(ctx);
    }
    let mut cpu = Vec::with_capacity(samples);
    let mut total = Vec::with_capacity(samples);
    for _ in 0..samples {
        let t0 = Instant::now();
        mixer.render(ctx, &audio, &audio_values).expect("render");
        let c = t0.elapsed().as_micros();
        poll(ctx);
        let t = t0.elapsed().as_micros();
        cpu.push(c);
        total.push(t);
    }
    cpu.sort_unstable();
    total.sort_unstable();
    (cpu[samples / 2], total[samples / 2])
}

/// CPU-submit vs GPU-tail during a real-shader crossfade, on named hardware.
///
/// crossfader=0.0 culls channel B (single-channel cost); crossfader=0.5 keeps
/// both live (the cull at mixer/render.rs:98 does not fire). The 0.0→0.5 delta
/// is the crossfade tax in isolation. The CPU vs GPU-tail split says whether the
/// wall is Varda's submission overhead or the GPU's fill rate.
fn report_crossfade_split(_c: &mut Criterion) {
    let Some(ctx) = make_context() else { return };
    print_adapter(&ctx);

    for (name, src) in HEAVY_SHADERS {
        let mut mixer = setup_two_channel_heavy(&ctx, src);

        mixer.set_crossfader(0.0);
        let (cpu0, tot0) = time_split_us(&ctx, &mut mixer, 21);

        mixer.set_crossfader(0.5);
        let (cpu5, tot5) = time_split_us(&ctx, &mut mixer, 21);

        eprintln!(
            "crossfade split ({name}, {WIDTH}x{HEIGHT}):\n  \
             cf=0.0 (1 ch live): total={tot0}µs  cpu_submit={cpu0}µs  gpu_tail={}µs\n  \
             cf=0.5 (2 ch live): total={tot5}µs  cpu_submit={cpu5}µs  gpu_tail={}µs\n  \
             crossfade tax: total +{}µs  cpu +{}µs  gpu_tail +{}µs",
            tot0.saturating_sub(cpu0),
            tot5.saturating_sub(cpu5),
            tot5.saturating_sub(tot0),
            cpu5.saturating_sub(cpu0),
            tot5.saturating_sub(tot0).saturating_sub(cpu5.saturating_sub(cpu0)),
        );
    }
}

criterion_group!(
    benches,
    bench_channel_composite_solid,
    bench_channel_composite_shader,
    bench_mixer_crossfade,
    report_per_deck_slope,
    report_crossfade_split,
);
criterion_main!(benches);
