#include <iostream>
#include <filesystem>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <string>
#include <fstream>
#include <stdexcept>
#include "simulation.cpp"
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
    	int width = 100;
    	int height = 100;
		scalar_t omega = 0.0;

		int steps = 200;

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

		for (int i = 0; i < v_dim; i++) {
			f(local_width / 2, height / 2, i) = 0.1;
		}


		for (int step=0; step<steps; step++) {
			if (size > 1) {
				share_ghost_cells(f, local_width, height, rank, size);
			}
			calc_collision(f, post_f, density, velocity, local_width, 
				height, omega);
			streaming(f, post_f, local_width, height, rank, size);
			save_density(density, local_width, height, step, steps, 
				"data/relaxation_density.bin");
		}

	}
	// Finalize MPI and Kokkos
	Kokkos::finalize();
	MPI_Finalize();
	std::cout << "Made it to the end\n";
	return 0;
}

