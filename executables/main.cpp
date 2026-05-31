#include <iostream>
#include <filesystem>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <string>
#include <fstream>
#include <stdexcept>
#include "streaming.cpp"
#include <vector>




void save_density_to_file(Kokkos::View<int***> pdf, Kokkos::View<int**> density,
                          int width, int height, int VELO_DIM, int step, int steps, const std::string& filename) {
	(void)VELO_DIM;

	if (step < 0 || step >= steps) {
		throw std::out_of_range("step must be in [0, steps)");
	}

	if (step == 0) {
		// Initialize output file once and store metadata header.
		std::ofstream init_file(filename, std::ios::binary | std::ios::trunc);
		if (!init_file) {
			throw std::runtime_error("Failed to open output file for initialization: " + filename);
		}
		init_file.write(reinterpret_cast<const char*>(&steps), sizeof(int));
		init_file.write(reinterpret_cast<const char*>(&width), sizeof(int));
		init_file.write(reinterpret_cast<const char*>(&height), sizeof(int));
	}

	std::fstream file(filename, std::ios::binary | std::ios::in | std::ios::out);
	if (!file) {
		throw std::runtime_error("Failed to open output file: " + filename);
	}

	const std::streamoff header_bytes = static_cast<std::streamoff>(3 * sizeof(int));
	const std::streamoff slice_bytes = static_cast<std::streamoff>(width) *
																		 static_cast<std::streamoff>(height) *
																		 static_cast<std::streamoff>(sizeof(int));
	const std::streamoff step_offset = header_bytes + static_cast<std::streamoff>(step) * slice_bytes;
	file.seekp(step_offset, std::ios::beg);

  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < height; ++j) {
      calc_density(pdf, density, i, j);
      int d = density(i, j);
      file.write(reinterpret_cast<const char*>(&d), sizeof(int));
    }
  }
}

int main(int argc, char *argv[]) {
	Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(-1));
	{
    	// Constant parameters
    	int width = 60;
    	int height = 40;
    	int delta_t = 1;

		int steps = 100;

		// Data structures
		Kokkos::View<int**> density("density", width, height);
		Kokkos::View<int***> velocity("velocity", width, height, 2);
		Kokkos::View<int***> pdf("distribution", width, height, VELO_DIM);


		// Initialize probabilty density function
		Kokkos::deep_copy(pdf, 0);
		pdf(12, 1, 1) = 1;
		pdf(1, 6, 5) = 1;

		for (int step=0; step<steps; step++) {
			save_density_to_file(pdf, density, width, height, VELO_DIM, step, 100, "output.bin");
			streaming(pdf, density, velocity, width, height);			
		}

		
	}
	Kokkos::finalize();
	std::cout << "Made it to the end\n";
}
