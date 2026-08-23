#include "saveData.h"
#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

//____________________________________________________________________________
// Initialize the complete local array
void initialize_equilibrium(field2_t density, field3_t velocity, field3_t f,
                            int width, int height) {
    Kokkos::parallel_for(
        "initialize_equilibrium",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {width, height}),
        KOKKOS_LAMBDA(const int x, const int y) {
            density(x, y) = 1.0;

            velocity(x, y, 0) = 0.0;
            velocity(x, y, 1) = 0.0;

            for (int i = 0; i < v_dim; ++i) {
                // rho=1 and u=0 -> f_eq = weight(i)
                f(x, y, i) = weight(i);
            }
        });
}

//____________________________________________________________________________
// Return the rank of a diagonal neighbour in the Cart process grid
int diagonal_rank(MPI_Comm cart_comm, const int coords[2], const int dims[2],
                  int dx, int dy) {
    const int x = coords[0] + dx;
    const int y = coords[1] + dy;
    if (x < 0 || x >= dims[0] || y < 0 || y >= dims[1]) {
        return MPI_PROC_NULL; // Out of bounds, no diagonal neighbour
    }
    int diagonal_coords[2] = {x, y};
    int diagonal = MPI_PROC_NULL;
    MPI_Cart_rank(cart_comm, diagonal_coords, &diagonal);
    return diagonal;
}

//____________________________________________________________________________
// Build a non-periodic 2D Cartesian MPI topology and cache all direct and
// diagonal neighbours required by the D2Q9 halo exchange
MpiGrid2D create_cartesian_grid_2d(int world_size) {
    MpiGrid2D grid;
    grid.size = world_size;
    grid.dims[0] = 0;
    grid.dims[1] = 0;
    MPI_Dims_create(world_size, 2, grid.dims);

    // The global cavity has physical walls, so the process topology is not
    // periodic
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, grid.dims, periods, 0, &grid.comm);

    MPI_Comm_rank(grid.comm, &grid.rank);
    MPI_Comm_size(grid.comm, &grid.size);
    MPI_Cart_coords(grid.comm, grid.rank, 2, grid.coords);

    // Dimension 0 = x, Dimension 1 = y
    MPI_Cart_shift(grid.comm, 0, 1, &grid.left, &grid.right);
    MPI_Cart_shift(grid.comm, 1, 1, &grid.bottom, &grid.top);

    grid.bottom_left = diagonal_rank(grid.comm, grid.coords, grid.dims, -1, -1);
    grid.bottom_right =
        diagonal_rank(grid.comm, grid.coords, grid.dims, +1, -1);
    grid.top_left = diagonal_rank(grid.comm, grid.coords, grid.dims, -1, +1);
    grid.top_right = diagonal_rank(grid.comm, grid.coords, grid.dims, +1, +1);

    return grid;
}

//____________________________________________________________________________
// Validate the distributed solution against a single-rank reference 
bool validate_against_serial_2d(field3_t distributed_f, int global_width,
                                int global_height, int local_width,
                                int local_height, int steps, scalar_t omega,
                                const MpiGrid2D &grid) {
    // local field without ghost cells
    const int local_nx = local_width - 2;
    const int local_ny = local_height - 2;
    const int local_count = local_nx * local_ny * v_dim;
    
    // pack the owned subdomain into contiugous storage before moving it to
    // host memory for MPI_Gather()
    field_t packed_device("packed_mpi_field_2d", local_count);

    Kokkos::parallel_for(
        "pack_mpi_field_2d", Kokkos::RangePolicy<>(0, local_count),
        KOKKOS_LAMBDA(const int idx) {
            const int i = idx % v_dim;
            const int tmp = idx / v_dim;
            const int local_y = tmp % local_ny;
            const int local_x = tmp / local_ny;

            packed_device(idx) = distributed_f(local_x + 1, local_y + 1, i);
        });

    // MPI operates on host-accessible contiguous memory in this driver
    auto packed_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, packed_device);

    std::vector<scalar_t> gathered_tiles;
    if (grid.rank == 0) {
        gathered_tiles.resize(static_cast<std::size_t>(local_count) *
                              grid.size);
    }

    MPI_Gather(packed_host.data(), local_count, MPI_DOUBLE,
               grid.rank == 0 ? gathered_tiles.data() : nullptr, local_count,
               MPI_DOUBLE, 0, grid.comm);

    int validation_ok = 1;

    if (grid.rank == 0) {
        // Reconstruct the global field according to Cartesian process
        // coordinates
        // MPI rank order alone is not enough for a 2D tiling
        std::vector<scalar_t> global_mpi_field(
            static_cast<std::size_t>(global_width) * global_height * v_dim);

        for (int process = 0; process < grid.size; ++process) {
            int process_coords[2] = {0, 0};
            MPI_Cart_coords(grid.comm, process, 2, process_coords);

            const std::size_t tile_offset =
                static_cast<std::size_t>(process) * local_count;

            for (int local_x = 0; local_x < local_nx; ++local_x) {
                for (int local_y = 0; local_y < local_ny; ++local_y) {
                    const int global_x = process_coords[0] * local_nx + local_x;
                    const int global_y = process_coords[1] * local_ny + local_y;

                    for (int i = 0; i < v_dim; ++i) {
                        const std::size_t local_idx =
                            (static_cast<std::size_t>(local_x) * local_ny +
                             local_y) *
                                v_dim +
                            i;
                        const std::size_t global_idx =
                            (static_cast<std::size_t>(global_x) *
                                 global_height +
                             global_y) *
                                v_dim +
                            i;

                        global_mpi_field[global_idx] =
                            gathered_tiles[tile_offset + local_idx];
                    }
                }
            }
        }

        // Independent serial reference. It uses the same 2D implementation,
        // but with a 1 x 1 process grid and therefore only physical walls.
        const int serial_width = global_width + 2;
        const int serial_height = global_height + 2;

        field2_t serial_density("serial_density", serial_width, serial_height);
        field3_t serial_velocity("serial_velocity", serial_width, serial_height,
                                 2);
        field3_t serial_f("serial_f", serial_width, serial_height, v_dim);
        field3_t serial_post_f("serial_post_f", serial_width, serial_height,
                               v_dim);
        GhostBuffers2D serial_buffers(serial_width, serial_height);

        MpiGrid2D serial_grid{};
        serial_grid.comm = MPI_COMM_SELF;
        serial_grid.rank = 0;
        serial_grid.size = 1;

        serial_grid.dims[0] = 1;
        serial_grid.dims[1] = 1;

        serial_grid.coords[0] = 0;
        serial_grid.coords[1] = 0;

        serial_grid.left = MPI_PROC_NULL;
        serial_grid.right = MPI_PROC_NULL;
        serial_grid.bottom = MPI_PROC_NULL;
        serial_grid.top = MPI_PROC_NULL;

        serial_grid.bottom_left = MPI_PROC_NULL;
        serial_grid.bottom_right = MPI_PROC_NULL;
        serial_grid.top_left = MPI_PROC_NULL;
        serial_grid.top_right = MPI_PROC_NULL;

        initialize_equilibrium(serial_density, serial_velocity, serial_f,
                               serial_width, serial_height);

        for (int step = 0; step < steps; ++step) {
            execute_time_step_2d(serial_f, serial_post_f, serial_density,
                                 serial_velocity, serial_width, serial_height,
                                 omega, serial_grid, serial_buffers);
        }

        Kokkos::fence();
        auto serial_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, serial_f);

        scalar_t max_error = 0.0;
        int error_x = -1;
        int error_y = -1;
        int error_i = -1;

        for (int x = 0; x < global_width; ++x) {
            for (int y = 0; y < global_height; ++y) {
                for (int i = 0; i < v_dim; ++i) {
                    const std::size_t idx =
                        (static_cast<std::size_t>(x) * global_height + y) *
                            v_dim +
                        i;
                    const scalar_t reference = serial_host(x + 1, y + 1, i);
                    const scalar_t mpi_value = global_mpi_field[idx];
                    const scalar_t error = std::abs(mpi_value - reference);

                    if (error > max_error) {
                        max_error = error;
                        error_x = x;
                        error_y = y;
                        error_i = i;
                    }
                }
            }
        }

        // The comparison is intentionally performed on the distribution
        // functions themselves rather than only on global diagnostics
        constexpr scalar_t tolerance = 1e-12;
        if (max_error > tolerance) {
            validation_ok = 0;
            std::cout << "MPI 2D validation FAILED\n"
                      << "Max error: " << max_error << "\nAt x=" << error_x
                      << ", y=" << error_y << ", i=" << error_i << "\n";
        } else {
            std::cout << "MPI 2D validation PASSED\n"
                      << "Max error: " << max_error << "\n";
        }
    }

    MPI_Bcast(&validation_ok, 1, MPI_INT, 0, grid.comm);
    return validation_ok == 1;
}

//____________________________________________________________________________
int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get process rank and total number of processes
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int width = 512;
    int height = 512;
    int steps = 500;
    scalar_t omega = 1.7;

    // optional command-line parameters
    try {
        if (argc > 1) {
            width = std::stoi(argv[1]);
        }
        if (argc > 2) {
            height = std::stoi(argv[2]);
        }
        if (argc > 3) {
            steps = std::stoi(argv[3]);
        }
        if (argc > 4) {
            omega = std::stod(argv[4]);
        }
    } catch (const std::exception &exception) {
        if (world_rank == 0) {
            std::cerr << "Invalid argument: " << exception.what() << '\n';
        }

        MPI_Finalize();
        return 1;
    }

    if (width <= 0 || height <= 0 || steps <= 0 || omega <= 0.0 ||
        omega >= 2.0) {
        if (world_rank == 0) {
            std::cerr
                << "Usage: distributedRun2D [width] [height] [steps] [omega]\n"
                << "All dimensions and steps must be positive.\n"
                << "Omega must satisfy 0 < omega < 2.\n";
        }
        MPI_Finalize();
        return 1;
    }
    // Construct the 2D process topology before allocating local domains
    MpiGrid2D grid = create_cartesian_grid_2d(world_size);

    if (width % grid.dims[0] != 0 || height % grid.dims[1] != 0) {
        if (grid.rank == 0) {
            std::cerr << "For the current equal-tile 2D decomposition, width "
                         "must be divisible by "
                      << grid.dims[0] << " and height by " << grid.dims[1]
                      << ". Got " << width << " x " << height << ".\n";
        }
        MPI_Comm_free(&grid.comm); // free the Cartesian communicator before finalizing MPI
        MPI_Finalize();
        return 1;
    }

    // Initialize Kokkos
    Kokkos::initialize(argc, argv); // Changed bc: Cuda can't access device -1
    {
        // Constant parameters
        double total_mass = 0.0L;
        double total_kin_energy = 0.0L;

        const int local_nx = width / grid.dims[0];
        const int local_ny = height / grid.dims[1];

        // One ghost layer on every side
        const int local_width = local_nx + 2;
        const int local_height = local_ny + 2;

        // Data structures
        field2_t density("density", local_width, local_height);
        field3_t velocity("velocity", local_width, local_height, 2);
        field3_t f("f", local_width, local_height, v_dim);
        field3_t post_f("post_f", local_width, local_height, v_dim);
        GhostBuffers2D ghost_buffers(local_width, local_height);

        // Uniform equilibrium background: rho = 1, u = 0 everywhere.
        // Includes the ghost columns (x=0, local_width-1)
        initialize_equilibrium(density, velocity, f, local_width, local_height);

        Kokkos::fence();


        MPI_Barrier(grid.comm); // Ensure all ranks start timing together
        
        // start timing
        const double start = MPI_Wtime();

        for (int step = 0; step < steps; ++step) {
            execute_time_step_2d(f, post_f, density, velocity, local_width,
                                 local_height, omega, grid, ghost_buffers);
        }

        Kokkos::fence();

        const double local_runtime = MPI_Wtime() - start;

        double runtime = 0.0;

        MPI_Reduce(&local_runtime, &runtime, 1, MPI_DOUBLE, MPI_MAX, 0,
                   grid.comm);

        // Synchronize halo values once more for post-processing
        if (grid.dims[0] * grid.dims[1] > 1) {
            share_ghost_cells_2d(f, local_width, local_height, grid,
                                 ghost_buffers);
        }

        // Compute global diagnostics on the final state
        calc_macroscopic(f, density, velocity, local_width, local_height);

        Kokkos::fence();
        if (grid.rank == 0) {
            std::cout << "Grid: " << width << " x " << height << '\n';
            std::cout << "Steps: " << steps << '\n';
            std::cout << "MPI ranks: " << grid.size << '\n';
            std::cout << "MPI process grid: " << grid.dims[0] << " x "
                      << grid.dims[1] << '\n';
            std::cout << "Omega: " << omega << '\n';
        }

        calc_total_mass_2d(total_mass, density, local_width, local_height,
                           grid.comm);
        calc_total_kin_energy_2d(total_kin_energy, velocity, density,
                                 local_width, local_height, grid.comm);

        const bool valid = validate_against_serial_2d(
            f, width, height, local_width, local_height, steps, omega, grid);
        if (grid.rank == 0) {
            const double mlups =
                (static_cast<double>(width) * height * steps) / (runtime * 1e6);

            std::cout << "Validation: " << (valid ? "OK" : "FAILED") << "\n";
            std::cout << "Runtime: " << runtime << " s\n";
            std::cout << "MLUPS: " << mlups << "\n";
            std::cout << "Total mass: " << total_mass << "\n";
            std::cout << "Total kinetic energy: " << total_kin_energy << "\n";
        }
    }
    // Finalize MPI and Kokkos
    Kokkos::finalize();
    MPI_Comm_free(&grid.comm);
    MPI_Finalize();
    return 0;
}