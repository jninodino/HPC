#include "saveData.h"
#include <fstream>
#include <stdexcept>


//____________________________________________________________________________
void save_density(field2_t density, int width, int height, int step, int steps, const std::string& filename) {
	// Create a host mirror of the density view to ensure data is accessible on the host
	auto density_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, density);
	
	if (step == 0) {
		// Initialize output file once and store metadata header
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
									  static_cast<std::streamoff>(sizeof(scalar_t));
	const std::streamoff step_offset = header_bytes + static_cast<std::streamoff>(step) * slice_bytes;
	file.seekp(step_offset, std::ios::beg);

  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < height; ++j) {
      scalar_t d = density_host(i, j);
      file.write(reinterpret_cast<const char*>(&d), sizeof(scalar_t));
    }
  }
}


//____________________________________________________________________________
void save_velocity(field3_t velocity, int width, int height, 
	int step, int steps, const std::string& filename) {
	auto velocity_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, velocity);

	int dim = 2;
	if (step == 0) {
		// Initialize output file once and store metadata header.
		std::ofstream init_file(filename, std::ios::binary | std::ios::trunc);
		if (!init_file) {
			throw std::runtime_error("Failed to open output file for initialization: " + filename);
		}
		init_file.write(reinterpret_cast<const char*>(&steps), sizeof(int));
		init_file.write(reinterpret_cast<const char*>(&width), sizeof(int));
		init_file.write(reinterpret_cast<const char*>(&height), sizeof(int));
		init_file.write(reinterpret_cast<const char*>(&dim), sizeof(int));
	}

	std::fstream file(filename, std::ios::binary | std::ios::in | std::ios::out);
	if (!file) {
		throw std::runtime_error("Failed to open output file: " + filename);
	}

	const std::streamoff header_bytes = static_cast<std::streamoff>(4 * sizeof(int));
	const std::streamoff slice_bytes = static_cast<std::streamoff>(width) *
									  static_cast<std::streamoff>(height) *
									  static_cast<std::streamoff>(dim) *
									  static_cast<std::streamoff>(sizeof(scalar_t));
	const std::streamoff step_offset = header_bytes + static_cast<std::streamoff>(step) * slice_bytes;
	file.seekp(step_offset, std::ios::beg);

  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < height; ++j) {
      scalar_t v_x = velocity_host(i, j, 0);
	  scalar_t v_y = velocity_host(i, j, 1);
      file.write(reinterpret_cast<const char*>(&v_x), sizeof(scalar_t));
	  file.write(reinterpret_cast<const char*>(&v_y), sizeof(scalar_t));
    }
  }
}
