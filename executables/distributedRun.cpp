#include "saveData.cpp"
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

bool validate_against_serial(field3_t distributed_f, int global_width,
                             int height, int steps, scalar_t omega, int rank,
                             int size) {
    const int local_nx = global_width / size;

    const int local_count = local_nx * height * v_dim;

    // Pack owned cells of each MPI rank
    field_t packed_device("packed_mpi_field", local_count);

    Kokkos::parallel_for(
        "pack_mpi_field", Kokkos::RangePolicy<>(0, local_count),
        KOKKOS_LAMBDA(const int idx) {
            const int i = idx % v_dim;

            const int tmp = idx / v_dim;

            const int y = tmp % height;

            // +1 because x=0 is ghost column
            const int x = tmp / height + 1;

            packed_device(idx) = distributed_f(x, y, i);
        });

    auto packed_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, packed_device);

    // Gather all local domains on rank 0
    std::vector<scalar_t> global_mpi_field;

    if (rank == 0) {
        global_mpi_field.resize(global_width * height * v_dim);
    }

    MPI_Gather(packed_host.data(), local_count, MPI_DOUBLE,

               rank == 0 ? global_mpi_field.data() : nullptr,

               local_count, MPI_DOUBLE,

               0, MPI_COMM_WORLD);

    int validation_ok = 1;

    // Rank 0 computes independent serial reference
    if (rank == 0) {
        const int serial_width = global_width + 2;
        field2_t serial_density("serial_density", serial_width, height);
        field3_t serial_velocity("serial_velocity", serial_width, height, 2);
        field3_t serial_f("serial_f", serial_width, height, v_dim);
        field3_t serial_post_f("serial_post_f", serial_width, height, v_dim);
        GhostBuffers serial_buffers(height);

        initialize_equilibrium(serial_density, serial_velocity, serial_f,
                               serial_width, height);
        for (int step = 0; step < steps; ++step) {
            execute_time_step(serial_f, serial_post_f, serial_density,
                              serial_velocity, serial_width, height, omega,
                              0, // rank
                              1, // size -> serial
                              serial_buffers);
        }
        Kokkos::fence();
        auto serial_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, serial_f);
        scalar_t max_error = 0.0;

        int error_x = -1;
        int error_y = -1;
        int error_i = -1;

        for (int x = 0; x < global_width; ++x) {
            for (int y = 0; y < height; ++y) {
                for (int i = 0; i < v_dim; ++i) {
                    const int idx = (x * height + y) * v_dim + i;
                    // +1 because serial x=0 is ghost
                    const scalar_t reference = serial_host(x + 1, y, i);
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

        constexpr scalar_t tolerance = 1e-12;
        if (max_error > tolerance) {
            validation_ok = 0;
            std::cout << "MPI validation FAILED\n"
                      << "Max error: " << max_error << "\nAt x=" << error_x
                      << ", y=" << error_y << ", i=" << error_i << "\n";
        } else {
            std::cout << "MPI validation PASSED\n"
                      << "Max error: " << max_error << "\n";
        }
    }
    MPI_Bcast(&validation_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return validation_ok == 1;
}

int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get process rank and total number of processes
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int width = 512;
    int height = 512;
    int steps = 500;
    scalar_t omega = 1.7;

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
        if (rank == 0) {
            std::cerr << "Invalid argument: " << exception.what() << '\n';
        }
        

        MPI_Finalize();
        return 1;
    }

    if (width <= 0 || height <= 0 || steps <= 0 || omega <= 0.0 ||
        omega >= 2.0) {
        if (rank == 0) {
            std::cerr
                << "Usage: distributedRun [width] [height] [steps] [omega]\n"
                << "All dimensions and steps must be positive.\n"
                << "Omega must satisfy 0 < omega < 2.\n";
        }
    if (width % size != 0) {
            if (rank == 0) {
                std::cerr << "Width must be a multiple of size got: " << width
                          << " and " << size << std::endl;
            }
        }

        MPI_Finalize();
        return 1;
    }

    // Initialize Kokkos
    Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(-1));
    {
        // Constant parameters
        double total_mass = 0.0L;
        double total_kin_energy = 0.0L;

        // Calculate local dimensions, add 2 for ghost cells at 0 and n_local
        const int local_width = width / size + 2;

        // Data structures
        field2_t density("density", local_width, height);
        field3_t velocity("velocity", local_width, height, 2);
        field3_t f("f", local_width, height, v_dim);
        field3_t post_f("post_f", local_width, height, v_dim);
        GhostBuffers ghost_buffers(height);

        // Uniform equilibrium background: rho = 1, u = 0 everywhere.
        // Includes the ghost columns (x=0, local_width-1) so the very first
        // ghost exchange (before any streaming has run) ships real
        // equilibrium data instead of zero-initialized garbage.
        initialize_equilibrium(density, velocity, f, local_width, height);

        Kokkos::fence();

        // start timing

        MPI_Barrier(MPI_COMM_WORLD);

        const double start = MPI_Wtime();

        for (int step = 0; step < steps; ++step) {
            execute_time_step(f, post_f, density, velocity, local_width, height,
                              omega, rank, size, ghost_buffers);
        }

        Kokkos::fence();

        const double local_runtime = MPI_Wtime() - start;

        double runtime = 0.0;

        MPI_Reduce(&local_runtime, &runtime, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);

        // Complete the final distributed state.
        // This is intentionally outside the timed region.
        if (size > 1) {
            share_ghost_cells(f, local_width, height, rank, size,
                              ghost_buffers);
        }

        // density/velocity currently belong to the state before the final
        // streaming. Recompute them from the final f.
        calc_macroscopic(f, density, velocity, local_width, height);

        Kokkos::fence();
        if (rank == 0) {
            std::cout << "Grid: " << width << " x " << height << '\n';
            std::cout << "Steps: " << steps << '\n';
            std::cout << "MPI ranks: " << size << '\n';
            std::cout << "Omega: " << omega << '\n';
        }

        calc_total_mass(total_mass, density, local_width, height);

        calc_total_kin_energy(total_kin_energy, velocity, density, local_width,
                              height);

        const bool valid =
            validate_against_serial(f, width, height, steps, omega, rank, size);
        if (rank == 0) {
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
    MPI_Finalize();
    return 0;
}