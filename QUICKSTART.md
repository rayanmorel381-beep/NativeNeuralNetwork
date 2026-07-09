# native_neural_network - Quickstart

## Requirements

- Rust stable (recommended: latest stable)
- Project root: this repository (`Cargo.toml` at root)

## Build Check

```bash
cargo check
```

## Generate Sample Models

```bash
cargo run --example generate_sample_model
```

Generated files:

- `trained/f32/*_model.rnn`
- `trained/f32/*_model.rnn`

## Train Commands

The repository exposes cargo aliases for training bins, build them with this commands:

```bash
cargo build --bin small_model
cargo build --bin medium_model
cargo build --bin large_model
cargo build --bin enormous_model
```

Run the bin's training model with this commands: 

```bash
cargo run --bin small_model
cargo run --bin medium_model
cargo run --bin large_model
cargo run --bin enormous_model
```

## Training Duration

Change the 'train_seconds' if you want to increase or decrease the training time in all the 'training/*_model.rs, line 12.

## Training Storage Behavior

Training writes to existing files and appends history instead of creating new file names.

For each precision (`trained/f32` and `trained/f64`):

- model: `trained/<precision>/<model>.rnn`
- benchmark blob: `trained/<precision>/benchmark/<model>.bmk`
- benchmark timeline: `trained/<precision>/benchmark/<model>.csv`
- benchmark text exports: `trained/<precision>/benchmark/<model>.json` and `.yaml`

Important behavior:

- `.rnn`, `.bmk`, `.csv`, `.json`, `.yaml` are appended in place
- training resumes from the latest stored model snapshot
- global counters (`elapsed_ms`, `iterations`) resume from previous CSV state

## Quick Verification

After running a training command twice, verify that history grows:

```bash
for f in trained/f32/benchmark/*.csv trained/f64/benchmark/*.csv; do
  [ -f "$f" ] || continue
  lines=$(wc -l < "$f")
  headers=$(grep -c '^model,precision,elapsed_ms,iterations,train_samples,avg_loss,last_loss,iterations_per_sec,samples_per_sec$' "$f")
  runs=$(awk -F, 'NR==1{prev=-1; r=0; next} {e=$3+0; if(prev>=0 && e<prev) r++; prev=e} END{print (NR>1)?(r+1):0}' "$f")
  echo "$f lines=$lines headers=$headers estimated_runs=$runs"
done
```

Expected:

- `lines` increases over time
- `headers` stays `1`
- `estimated_runs` increases after each new training launch

## FFI and Wrappers

- C header: `include/rnn_api.h`
- Rust FFI implementation: `src/ffi_api/ffi_api.rs`
- Wrappers: `wrappers/cpp`, `wrappers/java`, `wrappers/javascript`, `wrappers/python`

The benchmark FFI structs are aligned across Rust/C/Python/Java/C++.

## Notes

- No UI is bundled.
- This quickstart focuses on local build, sample generation, and training lifecycle.
