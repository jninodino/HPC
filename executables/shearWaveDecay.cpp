#include "latticeBoltzmann.h"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>

// This file works currently only on one thread!

namespace {

constexpr scalar_t pi = 3.14159265358979323846;

//____________________________________________________________________________
void initialize_shear_wave(field2_t density, field3_t velocity, field3_t f,
                           int width, int height, scalar_t epsilon) {
    // reset all fields before init
    Kokkos::deep_copy(density, 0.0);
    Kokkos::deep_copy(velocity, 0.0);
    Kokkos::deep_copy(f, 0.0);

    Kokkos::parallel_for(
        "initialize_shear_wave",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {width - 1, height}),
        KOKKOS_LAMBDA(const int x, const int y) {
            // sinusoidal x-velocity perturbation in y
            const scalar_t rho = 1.0;
            const scalar_t ux = epsilon *
                                sin(2.0 * pi * static_cast<scalar_t>(y) /
                                    static_cast<scalar_t>(height));
            const scalar_t uy = 0.0;

            density(x, y) = rho;
            velocity(x, y, 0) = ux;
            velocity(x, y, 1) = uy;

            const scalar_t u_squared = ux * ux + uy * uy;
            for (int i = 0; i < v_dim; ++i) {
                const scalar_t cu = cx(i) * ux + cy(i) * uy;
                f(x, y, i) = weight(i) * rho *
                             (1.0 + 3.0 * cu + 4.5 * cu * cu -
                              1.5 * u_squared);
            }
        });
}

//____________________________________________________________________________
// periodic streaming used only for shear-wave decay validation experiment
void streaming_periodic(field3_t f, field3_t post_f, int width, int height) {
    Kokkos::parallel_for(
        "streaming_periodic",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 0, 0},
                                               {width - 1, height, v_dim}),
        KOKKOS_LAMBDA(const int x, const int y, const int i) {
            // wrap both coordinates periodically
            int x_next = x + cx(i);
            if (x_next == 0) {
                x_next = width - 2;
            } else if (x_next == width - 1) {
                x_next = 1;
            }

            int y_next = y + cy(i);
            if (y_next < 0) {
                y_next += height;
            } else if (y_next >= height) {
                y_next -= height;
            }

            f(x_next, y_next, i) = post_f(x, y, i);
        });
}

//____________________________________________________________________________
scalar_t shear_amplitude(field3_t velocity, int width, int height) {
    scalar_t projection = 0.0;

    // project u_x onto the initial sine mode
    Kokkos::parallel_reduce(
        "shear_amplitude",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {width - 1, height}),
        KOKKOS_LAMBDA(const int x, const int y, scalar_t &sum) {
            const scalar_t basis =
                sin(2.0 * pi * static_cast<scalar_t>(y) /
                    static_cast<scalar_t>(height));
            sum += velocity(x, y, 0) * basis;
        },
        projection);

    Kokkos::fence();
    const int physical_width = width - 2;
    return 2.0 * projection /
           static_cast<scalar_t>(physical_width * height);
}

//____________________________________________________________________________
void print_usage() {
    std::cerr
        << "Usage: shearWaveDecay [omega] [steps] [width] [height] [epsilon] "
           "[output.csv]\n"
        << "Defaults: omega=1.8, steps=1500, width=50, height=100, "
           "epsilon=1e-3, output=data/shearWaveDecay.csv\n";
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

	if (size != 1) {
			if (rank == 0) {
				std::cerr << "shearWaveDecay is a single-rank validation experiment. "
							"Run it with one MPI rank.\n";
			}
			MPI_Finalize();
			return 1;
		}
	
		scalar_t omega = 1.8;
    int steps = 1500;
    int physical_width = 50;
    int height = 100;
    scalar_t epsilon = 1e-3;
    std::string output_file = "data/shearWaveDecay.csv";
    
    // optional command-line parameters
    try {
        if (argc > 1) {
            omega = std::stod(argv[1]);
        }
        if (argc > 2) {
            steps = std::stoi(argv[2]);
        }
        if (argc > 3) {
            physical_width = std::stoi(argv[3]);
        }
        if (argc > 4) {
            height = std::stoi(argv[4]);
        }
        if (argc > 5) {
            epsilon = std::stod(argv[5]);
        }
        if (argc > 6) {
            output_file = argv[6];
        }
    } catch (const std::exception &exception) {
        if (rank == 0) {
            std::cerr << "Invalid argument: " << exception.what() << '\n';
            print_usage();
        }
        MPI_Finalize();
        return 1;
    }

    if (omega <= 0.0 || omega >= 2.0 || steps <= 0 || physical_width <= 0 ||
        height <= 2 || epsilon <= 0.0) {
        if (rank == 0) {
            print_usage();
            std::cerr << "Require 0 < omega < 2, positive steps/width/epsilon, "
                         "and height > 2.\n";
        }
        MPI_Finalize();
        return 1;
    }
	  Kokkos::initialize(argc, argv);
    {
	const std::filesystem::path output_path(output_file);
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream output(output_file, std::ios::trunc);
    if (!output) {
        std::cerr << "Failed to open output file: " << output_file << '\n';
        MPI_Finalize();
        return 1;
    }
    output << std::setprecision(17);
    output << "step,amplitude\n";
	// Keep two x ghost columns for the shared collision kernel
        const int width = physical_width + 2;

        field2_t density("density", width, height);
        field3_t velocity("velocity", width, height, 2);
        field3_t f("f", width, height, v_dim);
        field3_t post_f("post_f", width, height, v_dim);

        // Initialize the equilibrium distribution with the shear perturbation
        initialize_shear_wave(density, velocity, f, width, height, epsilon);
        Kokkos::deep_copy(post_f, 0.0);
        Kokkos::fence();

        output << 0 << ',' << shear_amplitude(velocity, width, height) << '\n';

        for (int step = 1; step <= steps; ++step) {
            // Advance one LBM step with periodic boundaries
            calc_collision(f, post_f, density, velocity, width, height, omega);
            streaming_periodic(f, post_f, width, height);

            // Recompute macros after streaming for the current amplitude
            calc_macroscopic(f, density, velocity, width, height);
            Kokkos::fence();

            output << step << ','
                   << shear_amplitude(velocity, width, height) << '\n';
        }
    }
    Kokkos::finalize();

    const scalar_t analytic_viscosity =
        (1.0 / 3.0) * ((1.0 / omega) - 0.5);

    std::cout << "Shear-wave decay finished\n"
              << "Omega: " << omega << '\n'
              << "Analytical viscosity: " << analytic_viscosity << '\n'
              << "Steps: " << steps << '\n'
              << "Grid: " << physical_width << " x " << height << '\n'
              << "Initial amplitude: " << epsilon << '\n'
              << "Saved amplitude history to: " << output_file << '\n';

    MPI_Finalize();
    return 0;
}