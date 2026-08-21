#include <iostream>
#include <filesystem>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <fstream>
#include "latticeBoltzmann.h"
using scalar_t = double;


constexpr double pi = 3.14159265358979323846;



void save_density(Kokkos::View<scalar_t**> density, int width, int height, int step, int steps, const std::string& filename) {

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
									  static_cast<std::streamoff>(sizeof(scalar_t));
	const std::streamoff step_offset = header_bytes + static_cast<std::streamoff>(step) * slice_bytes;
	file.seekp(step_offset, std::ios::beg);

  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < height; ++j) {
      scalar_t d = density(i, j);
      file.write(reinterpret_cast<const char*>(&d), sizeof(scalar_t));
    }
  }
}

void save_velocity(Kokkos::View<scalar_t***> velocity, int width, int height, 
	int step, int steps, const std::string& filename) {

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
      scalar_t v_x = velocity(i, j, 0);
	  scalar_t v_y = velocity(i, j, 1);
      file.write(reinterpret_cast<const char*>(&v_x), sizeof(scalar_t));
	  file.write(reinterpret_cast<const char*>(&v_y), sizeof(scalar_t));
    }
  }
}
