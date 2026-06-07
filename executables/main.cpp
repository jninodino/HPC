#include <iostream>
#include <filesystem>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <string>
#include <fstream>
#include <stdexcept>
#include "streaming.cpp"
#include <vector>




void save_density_to_file(Kokkos::View<float***> pdf, Kokkos::View<float**> density,
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
		int dim = 99;
    	int width = dim;
    	int height = dim;
    	int delta_t = 1;
		float tau = 1.0f;

		int steps = 100;

		// Data structures
		Kokkos::View<float**> density("density", width, height);
		Kokkos::View<float***> velocity("velocity", width, height, 2);
		Kokkos::View<float***> pdf("distribution", width, height, VELO_DIM);


		// Initialize probabilty density function
		Kokkos::deep_copy(pdf, 0);
		for (int x=0; x < width; x++) {
			for (int y=0; y < height; y++) {
				pdf(x, y, 0) = 1.0;
			}
		}
		for (int i=1; i<9; i++) {
			pdf(width / 2, height / 2, i) = 100.0f;
		}


		for (int step=0; step<steps; step++) {
			save_density_to_file(pdf, density, width, height, VELO_DIM, step, 100, "output.bin");
			streaming(pdf, density, velocity, width, height, tau);			
		}

		
	}
	Kokkos::finalize();
	std::cout << "Made it to the end\n";
}
