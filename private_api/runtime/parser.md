# runtime/parser

Zero-dependency `no_std` parsers for JSON, YAML and CSV, plus a minimal
filesystem layer. Every parser is bounded by explicit limits to stay safe on
untrusted input.

Façade: `private_api::modules::parser`.

## Key types

- JSON: `JsonParser`, `JsonValue`, `JsonLimits`, `JsonError*`.
- YAML: `YamlParser`, `YamlValue`, `YamlLimits`, `YamlLine`, `YamlError*`.
- CSV: `CsvParser`, `CsvValue`, `CsvLimits`, `CsvError*`.
- Cursors: `Cursor`, `LineCursor`.
- FS: `FsError`, `DuplicateKeyPolicy`.

## Key constants

- `DEFAULT_LIMITS`, `DEFAULT_JSON_LIMITS`/`DEFAULT_CSV_LIMITS`/
  `DEFAULT_YAML_LIMITS`, `DEFAULT_MAX_DEPTH`, `DEFAULT_MAX_YAML_DEPTH`,
  `DEFAULT_FILE_MODE`, `DEFAULT_DIR_MODE`.

## Representative functions

- Parse: `parse_json`, `parse_json_with_limits`, `parse_json_with_max_depth`,
  `parse_csv`, `parse_csv_with_limits`, `parse` (YAML).
- FS: `file_exists`, `ensure_dir`, `append_file`.
- Time: `monotonic_ns`.

## Concrete usage

Parse a config file with an explicit depth limit so a hostile input can't blow
the stack:

```rust
use native_neural_network::private_api::modules::parser;

fn load_config(text: &str) -> bool {
    match parser::parse_json_with_max_depth(text.as_bytes(), 16) {
        Ok(value) => { let _ = value; true }
        Err(_) => false,      // malformed or too deep -> rejected
    }
}
```

Bounded CSV over a dataset shard:

```rust
let rows = parser::parse_csv_with_limits(bytes, &parser::DEFAULT_CSV_LIMITS);
```

## Performance levers

- Always parse untrusted input with the `*_with_limits` / `*_with_max_depth`
  variants — the defaults bound depth, length and key counts.
- `monotonic_ns` gives a cheap timing source without pulling in `std`.

## Integration

Reads dataset/config files that feed `build`/`train`; the FS helpers back the
dataset directory scanning used by the training pipeline.
