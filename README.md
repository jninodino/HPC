# Parallel Lattice Boltzmann Solver

A two-dimensional **D2Q9 Lattice Boltzmann Method (LBM)** solver written in C++17. The project uses **Kokkos** for on-node parallel kernels and **MPI** for domain decomposition and halo exchange.

The implementation covers the main milestones of the HPC project:

- D2Q9 streaming
- BGK collision and local equilibrium
- density and velocity reconstruction
- bounce-back boundary conditions
- moving-wall boundary condition for the lid-driven cavity
- shear-wave decay for viscosity validation
- one-dimensional MPI decomposition in x-direction
- two-dimensional Cartesian MPI decompositions
- x/y halo exchange and diagonal corner communication
- MPI-vs-serial regression validation
- strong-scaling measurements in runtime, MLUPS, mass and kinetic energy

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
│   ├── distributedRun.cpp        # MPI scaling + serial-reference validation
│   └── distributedRun2D.cpp      # 2D cartesian MPI decomposition
├── tests/
│   ├── test_latticeBoltzmann.cpp # regular unit/regression tests
│   ├── test_mpi_2d.cpp           # explicit 2D MPI communication tests
├── data/                         # generated binary simulation data
├── plots/                        # generated milestone plots
└── visualize_*.py               # plotting and animation scripts
```

## Numerical model

The solver uses the D2Q9 lattice with nine discrete velocity directions. For every lattice cell, the macroscopic density and velocity are reconstructed from the particle distribution functions:

$$
\rho = \sum_i f_i,
\qquad
\mathbf{u} = \frac{1}{\rho}\sum_i f_i\mathbf{c}_i.
$$

The BGK collision step relaxes the populations towards local equilibrium:

$$
f_i^{\mathrm{post}}
= f_i - \omega\left(f_i-f_i^{\mathrm{eq}}\right).
$$

The D2Q9 equilibrium distribution implemented in the code is

```math
f_i^{\mathrm{eq}}
= w_i\rho\left[
1 + 3(\mathbf{c}_i\cdot\mathbf{u})
+ \frac{9}{2}(\mathbf{c}_i\cdot\mathbf{u})^2
- \frac{3}{2}|\mathbf{u}|^2
\right].
```

For the BGK model in lattice units, the kinematic viscosity is related to the relaxation parameter by

$$
\nu = \frac{1}{3}\left(\frac{1}{\omega}-\frac{1}{2}\right).
$$

## Requirements

### C++ build

- CMake >= 3.14
- C++17 compiler
- MPI implementation, e.g. OpenMPI
- Kokkos 4.7.03
- Eigen 5.0.1
- GoogleTest for the test target

The top-level `CMakeLists.txt` tries to find Kokkos and Eigen first. If the requested versions are not available, CMake downloads them with `FetchContent`. GoogleTest is handled in the same way for the tests.

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

### 2D MPI tests

Milestone 06 additionally contains explicit tests for the 2D communication path:

- x-direction halo exchange
- y-direction halo exchange
- diagonal corner communication
- a 2 x 2 MPI-vs-serial regression test

Run them with four MPI processes:

```bash
mpirun -np 4 ./build/tests/mpi_2d_tests
```

The complete 2D regression test reconstructs the global distributed field and compares it against a single-rank reference solution with a tolerance of `1e-12`.

## Executables

All executables are generated in `build/executables/`.

| Executable | Purpose | Current built-in setup |
|---|---|---|
| `streaming` | Validate pure propagation/streaming | 15 x 10, 30 steps |
| `relaxation` | Density perturbation and BGK relaxation | 100 x 100, `omega = 1.9`, 61 steps |
| `shareWaveDecay` | Shear-wave viscosity experiment | 50 x 50, `omega = 0.2`, 1000 steps |
| `lidDrivenCavity` | Lid-driven cavity and MLUPS timing | 128 x 128, `omega = 1.7`, 10000 steps |
| `distributedRun` | MPI strong scaling and serial validation | Default: 512 x 512, 500 steps, `omega = 1.7` |
| `distributedRun2D` | 2D Cartesian MPI decomposition, benchmark and serial validation |
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

The shear-wave executable accepts the relaxation parameter from the command line:

```text
shearWaveDecay [omega] [steps] [width] [height] [epsilon] [output.csv]
```

Example:

```bash
./build/executables/shearWaveDecay \
    1.4 1500 50 100 1e-3 \
    data/shearWaveDecay/omega_1p4.csv
```

The shear-wave experiment uses periodic boundary conditions. The relevant shear-mode amplitude is stored as a function of time and is fitted using

$$
A(t)=A_0\exp(-\nu k^2t).
$$

The numerical viscosity is then compared with

$$
\nu_{\mathrm{analytic}}
= \frac{1}{3}\left(\frac{1}{\omega}-\frac{1}{2}\right).
$$

The complete sweep over multiple relaxation parameters can be reproduced automatically with

```bash
python3 visualize_shearWaveDecay.py
```

The script runs several values of `omega`, determines the numerical viscosity, writes a summary CSV and creates the comparison plot in `plots/milestone4/`.


### 4. Lid-driven cavity
e cavity executable accepts

```text
lidDrivenCavity [width] [height] [max_steps] [omega] [tolerance]
```

Example:

```bash
mpirun -np 1 ./build/executables/lidDrivenCavity \
    128 128 300000 1.7 1e-6
```
The simulation checks the maximum change of the velocity field between consecutive time steps:

$$
\max_{x,y}\left|\mathbf{u}^{n+1}(x,y)-\mathbf{u}^{n}(x,y)\right| < 10^{-6}.
$$

`max_steps` is only a safety limit. At the end of the run, the executable reports whether the steady-state criterion was reached and stores the final velocity field in

```text
data/lidDrivenCavity_velocity.bin
```

Generate the final streamline/velocity plot with

```bash
python3 visualize_lidDrivenCavity.py
```

The lid-driven cavity executable is intended to be run with one MPI rank.
 MPI domain decomposition

Two MPI decomposition strategies are kept in the repository. The 1D version represents the initial distributed implementation, while the 2D version extends it to a Cartesian process topology.

## 1D decomposition: `distributedRun`

The 1D implementation decomposes the global domain only along the x-direction. Every rank owns a contiguous x-slab of physical cells and one ghost column on each side.

```text
             x direction

 rank 0          rank 1          rank 2          rank 3
+---------+     +---------+     +---------+     +---------+
|  local  |<--->|  local  |<--->|  local  |<--->|  local  |
| domain  |     | domain  |     | domain  |     | domain  |
+---------+     +---------+     +---------+     +---------+
   ghost           ghost           ghost           ghost
   columns          columns         columns         columns
```

Only populations crossing a left/right process interface need MPI communication. The local x-extent therefore contains two additional ghost columns.

For equal-size subdomains, the global width must be divisible by the number of MPI ranks:

```text
width % number_of_ranks == 0
```

Run syntax:

```text
distributedRun [width] [height] [steps] [omega]
```

Examples:

```bash
mpirun -np 1 ./build/executables/distributedRun 512 512 500 1.7
mpirun -np 2 ./build/executables/distributedRun 512 512 500 1.7
mpirun -np 4 ./build/executables/distributedRun 512 512 500 1.7
```

## 2D Cartesian decomposition: `distributedRun2D`

The 2D implementation decomposes the global domain in both x and y. The process-grid dimensions are selected automatically with

```cpp
int dims[2] = {0, 0};
MPI_Dims_create(world_size, 2, dims);
```

`MPI_Dims_create` chooses a balanced two-dimensional factorization of the available number of ranks. For example, four ranks form a `2 x 2` process grid and six ranks typically form a `3 x 2` grid.

A Cartesian communicator is then created with

```cpp
int periods[2] = {0, 0};
MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
```

The process topology is non-periodic because the global simulation domain has physical boundaries.

The Cartesian coordinates of each rank are obtained with `MPI_Cart_coords`. Direct left/right and bottom/top neighbors are obtained with `MPI_Cart_shift`.

```text
                 y
                 ^
                 |
        +----------------+----------------+
        |                |                |
        |    rank 0      |    rank 1      |
        |                |                |
        +----------------+----------------+
        |                |                |
        |    rank 2      |    rank 3      |
        |                |                |
        +----------------+----------------+ --> x
```

### Ghost cells

Every local 2D tile contains one ghost-cell layer on all four sides:

```text
+----------------------------------+
|            top ghost             |
+-----+----------------------+-----+
|left |                      |right|
|ghost|    owned cells       |ghost|
|     |                      |     |
+-----+----------------------+-----+
|           bottom ghost            |
+----------------------------------+
```

Before populations stream across a process boundary, the required boundary populations are exchanged with the corresponding neighboring rank. This lets each rank perform the local streaming/collision update using data from adjacent subdomains.

The communication buffers are staged through host mirrors before MPI communication, so the current implementation does not require GPU-aware MPI.

## MPI-vs-serial validation

Both distributed executables contain a serial-reference regression check.

After the timed MPI section, the distributed population field is gathered on rank 0. Rank 0 independently runs the same global problem using a single-rank reference configuration. The complete population fields are then compared with a tolerance of

```text
1e-12
```

Successful runs report, for example,

```text
MPI 1D validation PASSED
Max error: 0
Validation: OK
```

or

```text
MPI 2D validation PASSED
Max error: 0
Validation: OK
```
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

For a CUDA build, change these settings to:

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

## License

This project is distributed under the MIT License. See `LICENSE` for details.
