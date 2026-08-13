#include <iostream>
#include <filesystem>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <string>
#include <fstream>
#include <stdexcept>
#include "saveData.cpp"
#include <vector>
#include <cmath>
#include <numbers>
#include <chrono>

// This file works currently only on one thread!

int main(int argc, char *argv[]) {
	// Initialize MPI
	MPI_Init(&argc, &argv);

	// Get process rank and total number of processes
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Initialize Kokkos
	Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(-1));
	{
    	// Constant parameters
    	int width = 512;
    	int height = 512;
		scalar_t omega = 1.7; 

		int steps = 500;
        double total_mass = 0.0L;
        double total_kin_energy = 0.0L;

		if (width % size != 0) {
			std::cout << "Width must be a multiple of size got: " << width <<
			 " and " << size << std::endl;
			return 1;
		}
        
		// Calculate local dimensions, add 2 for ghost cells at 0 and n_local
		int local_width = (size > 0) ? width / size + 2 : width;

		// Data structures
		field2_t density("density", local_width, height);
		field3_t velocity("velocity", local_width, height, 2);
		field3_t f("f", local_width, height, v_dim);
		field3_t post_f("post_f", local_width, height, v_dim);

		// Uniform equilibrium background: rho = 1, u = 0 everywhere.
		// Includes the ghost columns (x=0, local_width-1) so the very first
		// ghost exchange (before any streaming has run) ships real
		// equilibrium data instead of zero-initialized garbage.
		for (int x = 0; x < local_width; x++) {
			for (int y = 0; y < height; y++) {
				density(x, y) = 1.0;
				for (int i = 0; i < v_dim; i++) {
					f(x, y, i) = calc_f_eq(density, velocity, x, y, i);
				}
			}
		}

        // start timing
        Kokkos::fence();
        Kokkos::Timer timer;
		for (int step=0; step<steps; step++) {
			execute_time_step(f, post_f, density, velocity, local_width, height,
            omega, rank, size);
		}

    calc_total_mass(total_mass, density, local_width, height);
    calc_total_kin_energy(total_kin_energy, velocity, density, local_width, 
                height);

    Kokkos::fence();
    // stop timer
    double runtime = timer.seconds();
    
    double mlups = (static_cast<double>(width) * height * steps) / (runtime *1e6);

    if (rank == 0) {
        std::cout << "Runtime: " << runtime << " s\n";
        std::cout << "MLUPS: " << mlups << "\n";
        std::cout << "Total mass: " << total_mass << "\n";
        std::cout << "Total kinetic energy: " << total_kin_energy << "\n";
    }

	}
	// Finalize MPI and Kokkos
	Kokkos::finalize();
	MPI_Finalize();
	std::cout << "Made it to the end\n";
	return 0;
}