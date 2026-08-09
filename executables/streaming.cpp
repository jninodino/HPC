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
    	int width = 15;
    	int height = 10;
		scalar_t omega = 0.0;

		int steps = 30;

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

		// Initial single non-zero blob in the center of the map
		f(local_width / 2, height / 2, 2) = 1;



		// Mass check
		double mass_t_new = 0.0;
		double mass_t = 0.0;

		for (int step=0; step<steps; step++) {
			execute_time_step(f, post_f, density, velocity, local_width, height, 
				omega, rank, size);

			// Check if total mass is constant
			calc_total_mass(mass_t_new, density, local_width, height);
			if (mass_t != mass_t_new && step != 0) {
				std::cout << "Mass conversation is not given in step: " <<
					step <<  " , mass changed from value " << total_mass <<
					" to " << mass_t_new << std::endl;
			}
			mass_t = mass_t_new;
			
			// Store density data
			save_density(density, local_width, height, step, steps, 
				"data/streaming_density.bin");
		}

	}
	// Finalize MPI and Kokkos
	Kokkos::finalize();
	MPI_Finalize();
	std::cout << "Made it to the end\n";
	return 0;
}