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

// This file works currently only on one thread!

namespace {

//____________________________________________________________________________
scalar_t max_velocity_change(field3_t velocity, field3_t previous_velocity,
                             int width, int height) {
    scalar_t max_squared_change = 0.0;

    // x = 0 and x = width - 1 are ghost columns
    Kokkos::parallel_reduce(
        "max_velocity_change",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {width - 1, height}),
        KOKKOS_LAMBDA(const int x, const int y, scalar_t &local_max) {
            const scalar_t du_x =
                velocity(x, y, 0) - previous_velocity(x, y, 0);
            const scalar_t du_y =
                velocity(x, y, 1) - previous_velocity(x, y, 1);
            const scalar_t squared_change = du_x * du_x + du_y * du_y;

            if (squared_change > local_max) {
                local_max = squared_change;
            }
        },
        Kokkos::Max<scalar_t>(max_squared_change)); // Kokkos::Max computes the maximum value across all threads

    return std::sqrt(max_squared_change);
} 
} // namespace

//____________________________________________________________________________
int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get process rank and total number of processes
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // cavity case is run serially
    if (size != 1) {
        if (rank == 0) {
            std::cerr << "lidDrivenCavity currently supports exactly one MPI "
                         "rank. Run it with -np 1.\n";
        }
        MPI_Finalize();
        return 1;
    }

    int width = 128;
    int height = 128;
    int max_steps = 100000;
    scalar_t omega = 1.7;
    scalar_t tolerance = 1e-6;
    
    // optional command-line parameters
    try {
        if (argc > 1) {
            width = std::stoi(argv[1]);
        }
        if (argc > 2) {
            height = std::stoi(argv[2]);
        }
        if (argc > 3) {
            max_steps = std::stoi(argv[3]);
        }
        if (argc > 4) {
            omega = std::stod(argv[4]);
        }
        if (argc > 5) {
            tolerance = std::stod(argv[5]);
        }
    } catch (const std::exception &exception) {
        if (rank == 0) {
            std::cerr << "Invalid argument: " << exception.what() << '\n';
        }
        MPI_Finalize();
        return 1;
    }

    if (width <= 0 || height <= 0 || max_steps <= 0 || omega <= 0.0 ||
        omega >= 2.0 || tolerance <= 0.0) {
        if (rank == 0) {
            std::cerr << "Usage: lidDrivenCavity [width] [height] [max_steps] "
                         "[omega] [tolerance]\n"
                      << "width, height, max_steps and tolerance must be "
                         "positive; omega must satisfy 0 < omega < 2.\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        // add one ghost column on each x side
        const int local_width = width + 2;

        // Data structures
        field2_t density("density", local_width, height);
        field3_t velocity("velocity", local_width, height, 2);
		field3_t previous_velocity("previous_velocity", local_width, height, 2);
        field3_t f("f", local_width, height, v_dim);
        field3_t post_f("post_f", local_width, height, v_dim);
        GhostBuffers ghost_buffers(height);

        // Initialze rho = 1 and u = 0, including ghost cells
        Kokkos::parallel_for(
            "initialize_liddrivencavity",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0},
                                                   {local_width, height}),
            KOKKOS_LAMBDA(const int x, const int y) {
                density(x, y) = 1.0;

                velocity(x, y, 0) = 0.0;
                velocity(x, y, 1) = 0.0;
                previous_velocity(x, y, 0) = 0.0;
                previous_velocity(x, y, 1) = 0.0;

                for (int i = 0; i < v_dim; i++) {
                    f(x, y, i) = calc_f_eq(density, velocity, x, y, i);
                }
            });

        // start timing
        Kokkos::fence();
        Kokkos::Timer timer;

        int executed_steps = 0;
        scalar_t residual = 0.0;
        bool converged = false;

        for (int step = 0; step < max_steps; step++) {

            // collision, streaming, then update macroscopic fields
            calc_collision(f, post_f, density, velocity, local_width, height,
                           omega);
            streaming(f, post_f, density, local_width, height, rank, size);
            // TODO: brauchen wir das noch? 
            // save_velocity(velocity, local_width, height, step, steps,
            //     "data/lidDrivenCavity_velocity.bin");
            // save_density(density, local_width, height, step, steps,
            //     "data/lidDrivenCavity_density.bin");
            calc_macroscopic(f, density, velocity, local_width, height);

            // stop once the velocity no longer changes significantly
            residual = max_velocity_change(velocity, previous_velocity,
                                           local_width, height);
            Kokkos::deep_copy(previous_velocity, velocity);

            executed_steps = step + 1;

            if (rank == 0 &&
                (executed_steps == 1 || executed_steps % 1000 == 0)) {
                std::cout << "Step " << executed_steps
                          << ", max |u_new - u_old| = " << residual << '\n';
            }

            if (residual < tolerance) {
                converged = true;
                break;
            }
        }

        Kokkos::fence();
        // stop timer
        const double runtime = timer.seconds();

        const double mlups =
            (static_cast<double>(width) * height * executed_steps) /
            (runtime * 1e6);

        // save only the final vel field
        save_velocity(velocity, local_width, height, 0, 1,
                      "data/lidDrivenCavity_velocity.bin");

        if (rank == 0) {
            std::cout << "\nLid-driven cavity finished\n"
                      << "Grid: " << width << " x " << height << '\n'
                      << "Omega: " << omega << '\n'
                      << "Tolerance: " << tolerance << '\n'
                      << "Steps executed: " << executed_steps << '\n'
                      << "Final max |u_new - u_old|: " << residual << '\n'
                      << "Steady state: "
                      << (converged ? "CONVERGED" : "NOT CONVERGED") << '\n'
                      << "Runtime: " << runtime << " s\n"
                      << "MLUPS: " << mlups << '\n'
                      << "Saved final velocity field to "
                         "data/lidDrivenCavity_velocity.bin\n";
        }
    }
    // Finalize MPI and Kokkos
    Kokkos::finalize();
    MPI_Finalize();
    std::cout << "Made it to the end\n";
    return 0;
}