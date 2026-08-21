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
    	int width = 9;
    	int height =  6;
		scalar_t omega = 1.7L;

		int steps = 10;

		if (width % size != 0) {
			std::cout << "Width must be a multiple of size got: " << width << " and " << size << std::endl;
			return 1;
		}

		// Calculate local dimensions, add 2 for ghost cells at 0 and n_local
		int local_width = (size > 0) ? width / size + 2 : width;

		// Data structures
		field2_t density("density", local_width, height);
		field3_t velocity("velocity", local_width, height, 2);
		field3_t f("f", local_width, height, v_dim);
		field3_t post_f("post_f", local_width, height, v_dim);

		double total_mass = 0.0L;
		double total_kin_energy = 0.0L;

		for (int step=0; step<steps; step++) {
			share_ghost_cells(f, local_width, height, rank, size);
			calc_collision(f, post_f, density, velocity, local_width, 
				height, omega);
			streaming(f, post_f, local_width, height, rank, size);
			calc_total_mass(total_mass, density, local_width, height);
			calc_total_kin_energy(total_kin_energy, velocity, density, width,
				height);
		}

	}
	// Finalize MPI and Kokkos
	Kokkos::finalize();
	MPI_Finalize();
	std::cout << "Made it to the end\n";
	return 0;
}

