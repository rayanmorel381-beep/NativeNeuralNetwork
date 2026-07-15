# geometry/sphere5d

A 5D spherical representation of a network's neurons — places neurons as points
on a 5D sphere and aggregates them into kernels, enabling spatial reasoning and
visualization over the parameter field.

Façade: `private_api::modules::sphere5d`.

## Key types

- `Sphere5D` — the point cloud container (f32) with an f64 twin path.
- `NeuronPoint` / `NeuronPointF64` — a neuron's 5D position.
- `NeuronKernel` / `NeuronKernelF64` — an aggregated kernel.
- `KernelLayoutStats` — layout statistics.
- `SphereError` / `KernelizeError` — failure reasons.

## Representative functions

- Build: `from_network`, `fill_from_network`, `add_neuron`, `new`.
- Geometry: `centroid`, `mean_radius`, `radius`, `nearest`, `neighbors_within`,
  `layer_bounds`, `sphere_pos_from_seed_f32`, `sphere_pos_from_seed_f64`.
- Kernelize: `aggregate_constellation_to_kernels`, `..._f64`.
- Queries: `len`, `capacity`, `is_empty`, `neuron_exists`, `as_slice`,
  `as_mut_slice`.

## Concrete usage

Project a trained network into 5D space and query its structure:

```rust
use native_neural_network::private_api::modules::sphere5d::Sphere5D;
use native_neural_network::private_api::modules::network::NeuralNetwork;

fn analyze(net: &NeuralNetwork) {
    let sphere = Sphere5D::from_network(net);
    let center = sphere.centroid();
    let spread = sphere.mean_radius();
    // find the 8 neurons closest to the centroid:
    let neighbors = sphere.neighbors_within(&center, spread);
    let _ = (center, spread, neighbors);
}
```

## Performance levers

- `from_network` is a one-shot projection; cache the `Sphere5D` and re-query
  cheaply with `nearest` / `neighbors_within`.
- Use the f32 path unless you need f64 geometric precision.

## Integration

Built from [network](../core/network.md) (`conceptualize_5d`); powers
[visualization](../runtime/visualization.md) meshes.
