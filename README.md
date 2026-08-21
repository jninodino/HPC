# Parallel Lattice Boltzmann Solver

A two-dimensional **D2Q9 Lattice Boltzmann Method (LBM)** solver written in C++17. The project uses **Kokkos** for on-node parallel kernels and **MPI** for domain decomposition and halo exchange.

The implementation covers the main milestones of the HPC project:

- D2Q9 streaming
- BGK collision and local equilibrium
- density and velocity reconstruction
- bounce-back boundary conditions
- moving-wall boundary condition for the lid-driven cavity
- shear-wave decay for viscosity validation
- one-dimensional MPI domain decomposition with ghost cells
- strong-scaling measurements in runtime, MLUPS, mass and kinetic energy
- MPI-vs-serial validation

## Repository structure

```text
.
├── CMakeLists.txt
├── src/
│   ├── latticeBoltzmann.cpp      # LBM kernels, boundaries, MPI halo exchange
│   ├── latticeBoltzmann.h
│   ├── saveData.cpp              # binary output helpers
|   └── saveData.h
├── executables/
│   ├── main.cpp                  # small general demo
│   ├── streaming.cpp             # streaming validation
│   ├── relaxation.cpp            # density perturbation / collision validation
│   ├── shareWaveDecay.cpp        # shear-wave decay experiment
│   ├── lidDrivenCavity.cpp       # lid-driven cavity benchmark
│   └── distributedRun.cpp        # MPI scaling + serial-reference validation
├── tests/
│   └── test_latticeBoltzmann.cpp # GoogleTest unit/regression tests
├── data/                         # generated binary simulation data
└── visualize_*.py               # plotting and animation scripts
```

## Numerical model

The solver uses the D2Q9 lattice with nine discrete velocity directions. For every lattice cell, the macroscopic density and velocity are reconstructed from the particle distribution functions:

\[
\rho = \sum_i f_i,
\qquad
\mathbf{u} = \frac{1}{\rho}\sum_i f_i\mathbf{c}_i.
\]

The BGK collision step relaxes the populations towards local equilibrium:

\[
f_i^{\mathrm{post}}
= f_i - \omega\left(f_i-f_i^{\mathrm{eq}}\right).
\]

The D2Q9 equilibrium distribution implemented in the code is

\[
f_i^{\mathrm{eq}}
= w_i\rho\left[
1 + 3(\mathbf{c}_i\cdot\mathbf{u})
+ \frac{9}{2}(\mathbf{c}_i\cdot\mathbf{u})^2
- \frac{3}{2}|\mathbf{u}|^2
\right].
\]

For the BGK model in lattice units, the kinematic viscosity is related to the relaxation parameter by

\[
\nu = \frac{1}{3}\left(\frac{1}{\omega}-\frac{1}{2}\right).
\]

## Requirements

### C++ build

- CMake >= 3.14
- C++17 compiler
- MPI implementation, e.g. OpenMPI
- Kokkos 4.7.03
- Eigen 5.0.1
- GoogleTest for the test target

The top-level `CMakeLists.txt` tries to find Kokkos and Eigen first. If the requested versions are not available, CMake downloads them with `FetchContent`. GoogleTest is handled in the same way for the tests.

> If you build on a machine without internet access, install the required dependencies beforehand or provide them through the cluster environment.

### Python plots

The plotting scripts require Python 3 with:

```bash
python3 -m pip install numpy matplotlib pillow
```

`pillow` is required for GIF output from the animation scripts.

## Build

### Release build

Use a Release build for simulations and benchmarks:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

For development and debugging:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j
```

### Example cluster setup

On a system using environment modules, load a compiler and MPI implementation first. The exact module names depend on the cluster.

For example:

```bash
module load compiler/gnu
module load mpi/openmpi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run the tests

After compiling:

```bash
ctest --test-dir build --output-on-failure
```

or directly:

```bash
./build/tests/tests
```

The test suite contains checks for, among other things:

- propagation in all D2Q9 directions
- density calculation
- mass conservation
- momentum conservation during collision
- equilibrium as a collision fixed point
- bounce-back boundary conditions

## Executables

All executables are generated in `build/executables/`.

| Executable | Purpose | Current built-in setup |
|---|---|---|
| `streaming` | Validate pure propagation/streaming | 15 x 10, 30 steps |
| `relaxation` | Density perturbation and BGK relaxation | 100 x 100, `omega = 1.9`, 61 steps |
| `shareWaveDecay` | Shear-wave viscosity experiment | 50 x 50, `omega = 0.2`, 1000 steps |
| `lidDrivenCavity` | Lid-driven cavity and MLUPS timing | 128 x 128, `omega = 1.7`, 10000 steps |
| `distributedRun` | MPI strong scaling and serial validation | Default: 512 x 512, 500 steps, `omega = 1.7` |
| `main` | Small general MPI/Kokkos demo | 9 x 6, 1000 steps |

The parameters are currently set directly in the corresponding `.cpp` files rather than through command-line arguments.
Most milestone executables use parameters defined in their source files.
`distributedRun` additionally accepts simulation parameters through the command line:

```bash
distributedRun [width] [height] [steps] [omega]

## Reproducing the project experiments

Run the following commands from the **repository root**, because the output and plotting scripts use relative paths such as `data/...` and `plots/...`.

Create the output directories once:

```bash
mkdir -p data \
    plots/milestone2 \
    plots/milestone3 \
    plots/milestone4 \
    plots/milestone5
```

### 1. Streaming validation

```bash
mpirun -np 1 ./build/executables/streaming
python3 visualize_streaming.py
```

The executable writes:

```text
data/streaming_density.bin
```

The visualization script generates an animation of the propagating density field.

### 2. Collision / density relaxation

```bash
mpirun -np 1 ./build/executables/relaxation
python3 visualize_relaxation.py
```

The executable writes:

```text
data/relaxation_density.bin
```

The initial state consists of a uniform background with a stronger density perturbation in the center. The perturbation is then relaxed by the BGK collision and streaming steps.

### 3. Shear-wave decay

```bash
mpirun -np 1 ./build/executables/shearWaveDecay
```

With the current source settings, this produces data for `omega = 0.2`:

```text
data/shearWaveDecay_density.bin
data/shearWaveDecay_velocity_02.bin
```

`visualize_shearWaveDecay.py` compares simulated and theoretical viscosity for several values of `omega`. It currently expects the following velocity files:

```text
shearWaveDecay_velocity_02.bin
shearWaveDecay_velocity_04.bin
shearWaveDecay_velocity_06.bin
shearWaveDecay_velocity_08.bin
shearWaveDecay_velocity_10.bin
shearWaveDecay_velocity_12.bin
shearWaveDecay_velocity_14.bin
shearWaveDecay_velocity_16.bin
shearWaveDecay_velocity_18.bin
```

To regenerate the complete viscosity curve, change `omega` and the corresponding output suffix in `executables/shearWaveDecay.cpp`, rebuild, and rerun the executable for each value before running:

```bash
python3 visualize_shearWaveDecay.py
```

The theoretical reference used by the analysis is

```text
nu = (1/3) * (1/omega - 1/2)
```

### 4. Lid-driven cavity

```bash
mpirun -np 1 ./build/executables/lidDrivenCavity
```

The program reports runtime and MLUPS for the built-in 128 x 128 problem.

The moving top wall uses a lattice velocity of `u_lid = 0.1`; the remaining physical walls use bounce-back/no-slip boundary conditions.

The calls that save the cavity velocity and density fields are currently commented out in `executables/lidDrivenCavity.cpp`. To regenerate the input for `visualize_lidDrivenCavity.py`, enable at least

```cpp
save_velocity(
    velocity,
    local_width,
    height,
    step,
    steps,
    "data/lidDrivenCavity_velocity.bin"
);
```

then rebuild and rerun the simulation:

```bash
cmake --build build -j
mpirun -np 1 ./build/executables/lidDrivenCavity
python3 visualize_lidDrivenCavity.py
```

Saving every time step substantially increases I/O and is therefore intentionally disabled for the performance benchmark.

### 5. MPI strong scaling

`distributedRun` is the main MPI benchmark. Its default parameters are:

- grid: `512 x 512`
- time steps: `500`
- relaxation parameter: `omega = 1.7`

The executable accepts optional command-line arguments:

```bash
./distributedRun [width] [height] [steps] [omega]

Example runs:

```bash
mpirun -np 1  ./build/executables/distributedRun 512 512 500 1.7
mpirun -np 2  ./build/executables/distributedRun 512 512 500 1.7
mpirun -np 4  ./build/executables/distributedRun 512 512 500 1.7
mpirun -np 8  ./build/executables/distributedRun 512 512 500 1.7
mpirun -np 16 ./build/executables/distributedRun 512 512 500 1.7
mpirun -np 32 ./build/executables/distributedRun 512 512 500 1.7
mpirun -np 64 ./build/executables/distributedRun 512 512 500 1.7

If no arguments are given, the default values are used.

[width], [height] and [steps] must be positive and the relaxation parameter must satisfy
0 < omega < 2

The global x-dimension must be divisible by the number of MPI ranks.

For every run, rank 0 prints values of the form:

```text
Validation: OK
Runtime: ... s
MLUPS: ...
Total mass: ...
Total kinetic energy: ...
```

### Serial-reference validation

After the timed MPI section, `distributedRun` gathers the distributed
population field on rank 0 and runs the same solver for the complete
domain using a single MPI rank as a reference.

The distributed and single-rank population fields are compared with
a tolerance of `1e-12`.

The validation is performed outside the timed benchmark region and
therefore does not affect the reported runtime or MLUPS.
## MPI domain decomposition

The global domain is decomposed along the x-direction. Every MPI rank owns a contiguous block plus one ghost column on either side:

```text
rank 0        rank 1        rank 2        rank 3
+----------+ +----------+ +----------+ +----------+
| local    | | local    | | local    | | local    |
| domain   | | domain   | | domain   | | domain   |
+----------+ +----------+ +----------+ +----------+
     <-------- halo / ghost exchange -------->
```

Only populations that cross an MPI interface are packed into the communication buffers. The code currently stages these buffers through host mirrors before calling `MPI_Sendrecv`, which avoids requiring GPU-aware MPI.

Global mass and kinetic energy are combined across ranks with MPI reductions.

## Performance metrics

The code reports performance in **MLUPS** (million lattice updates per second):

\[
\mathrm{MLUPS}
= \frac{N_x N_y N_{steps}}
{t\cdot 10^6}.
\]

For strong scaling, speedup and parallel efficiency can be calculated from the measured runtimes:

\[
S(p)=\frac{T_1}{T_p},
\qquad
E(p)=\frac{S(p)}{p}.
\]

The benchmark uses the maximum runtime over all MPI ranks so that the reported time reflects the slowest rank.

## Optional CUDA / GPU build

The repository currently configures the bundled Kokkos dependency with the CUDA backend disabled:

```cmake
set(Kokkos_ENABLE_CUDA OFF CACHE BOOL "Disable CUDA backend" FORCE)
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Serial backend" FORCE)
set(Kokkos_ENABLE_THREADS ON CACHE BOOL "Enable Threads backend" FORCE)
```

For a CUDA build, change these settings to, for example:

```cmake
set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable CUDA backend" FORCE)
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Serial backend" FORCE)
set(Kokkos_ENABLE_THREADS OFF CACHE BOOL "Disable Threads backend" FORCE)
```

## Notes on reproducibility

- Use a Release build for benchmark numbers.
- Run benchmark commands from the repository root.
- Do not include file output or plotting inside the timed strong-scaling region.
- Record the compiler, MPI version, Kokkos backend and hardware when reporting performance.
- Use the same problem size and number of time steps when comparing MPI process counts.

## Implementation notes

- `distributedRun` supports command-line configuration of the grid size,
  number of time steps, and relaxation parameter.
- The milestone executables are intended primarily for single-rank
  validation experiments, while `distributedRun` is used for MPI
  strong-scaling measurements.
- The shear-wave analysis compares the measured viscosity with the
  theoretical BGK relation for several relaxation parameters.
- Performance measurements should be performed with a Release build
  and without file output inside the timed region.

## License

This project is distributed under the MIT License. See `LICENSE` for details.
