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

void save_velocity(Kokkos::View<scalar_t***> velocity, int width, int height, int step, int steps, const std::string& filename) {

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

void save_one_position(Kokkos::View<scalar_t***> f, int x, int y, int step, int steps, const std::string& filename) {
	// init/overide file
	if (step == 0) {
		// Initialiye output file once and store metadata header.
		int dim = 9;
		std::ofstream init_file(filename, std::ios::binary | std::ios::trunc);
		if (!init_file) {
			throw std::runtime_error("Failed to open output file for initialization: " + filename);
		}
		init_file.write(reinterpret_cast<const char*>(&steps), sizeof(int));
		init_file.write(reinterpret_cast<const char*>(&dim), sizeof(int));
	}

	// open file
	std::fstream file(filename, std::ios::binary | std::ios::in | std::ios::out);
	if (!file) {
		throw std::runtime_error("Failed to open output file: " + filename);
	}

	std::streamoff header_bytes = static_cast<std::streamoff>(2 * sizeof(int));
	std::streamoff slice_bytes = static_cast<std::streamoff>(sizeof(scalar_t) * 9);
	std::streamoff step_offset = header_bytes + static_cast<std::streamoff>(step) * slice_bytes;
	file.seekp(step_offset, std::ios::beg);

	for (int i=0; i<9; i++) {
		scalar_t d = f(x, y, i);
		file.write(reinterpret_cast<const char*>(&d), sizeof(scalar_t));
	}
}


void save_x_velocity(Kokkos::View<scalar_t***> velocity, int x, int y, int step, int steps, const::std::string& filename) {
	// init/overide file
	if (step == 0) {
		// Initialiye output file once and store metadata header.
		std::ofstream init_file(filename, std::ios::binary | std::ios::trunc);
		if (!init_file) {
			throw std::runtime_error("Failed to open output file for initialization: " + filename);
		}
		init_file.write(reinterpret_cast<const char*>(&steps), sizeof(int));
	}
	// open file
	std::fstream file(filename, std::ios::binary | std::ios::in | std::ios::out);
	if (!file) {
		throw std::runtime_error("Failed to open output file: " + filename);
	}

	std::streamoff header_bytes = static_cast<std::streamoff>(1 * sizeof(int));
	std::streamoff slice_bytes = static_cast<std::streamoff>(sizeof(scalar_t));
	std::streamoff step_offset = header_bytes + static_cast<std::streamoff>(step) * slice_bytes;
	file.seekp(step_offset, std::ios::beg);

	scalar_t d = velocity(x, y, 0);
	file.write(reinterpret_cast<const char*>(&d), sizeof(scalar_t));
}

void save_viscosity_meas(field_t a_sim, int steps, const::std::string& filename) {
	// Initialiye output file once and store metadata header.
	std::ofstream file(filename, std::ios::binary | std::ios::trunc);
	if (!file) {
		throw std::runtime_error("Failed to open output file for initialization: " + filename);
	}
	file.write(reinterpret_cast<const char*>(&steps), sizeof(int));

	for (int step=0; step<steps; step++) {
		scalar_t d = a_sim(step);
		file.write(reinterpret_cast<const char*>(&d), sizeof(scalar_t));
	}

}


void init_sinusiodal_pertubation(field2_t density, 
						  field3_t velocity,
						  field3_t f,
						  field3_t post_f,
						  int dim,
						  int width,
						  int height,
						  scalar_t omega,
						  scalar_t epsilon,
						  int steps
						) {
		// Initialize probabilty density function
		Kokkos::deep_copy(f, 0);

		// Set initual perturpation
		for (int x=0; x < width; x++) {
			for (int y=0; y < height; y++) {
				density(x, y) = 1.0;
				velocity(x, y, 0) = epsilon * std::sin(2 * pi * y / height);
				velocity(x, y, 1) = 0.0;
				for (int i=0; i<9; i++) {
					f(x, y, i) = calc_f_eq(density, velocity, x, y, i);
				}
			}
}
						}

void overserve_x_velocity(field2_t density, 
						  field3_t velocity,
						  field3_t f,
						  field3_t post_f,
						  int dim,
						  int width,
						  int height,
						  scalar_t omega,
						  scalar_t epsilon,
						  int steps,
						  int rank,
						  int size
						) {
		// Oberserved point (Milestone 4)
		int o_x = width / 4;
		int o_y = height / 4;

		init_sinusiodal_pertubation(density, velocity, f, post_f, dim, width,
			 height, omega, epsilon, steps);
	
		// Simulation
		for (int step=0; step<steps; step++) {
			calc_collision(f, post_f, density, velocity, width, height, omega);
			streaming(f, post_f, width, height, rank, size);			
			save_density(density, width, height, step, steps, "data/density.bin");
			save_one_position(f, o_x, o_y, step, steps, "data/observed_point.bin");
			save_x_velocity(velocity, o_x, o_y, step, steps, "data/observed_velocity.bin");
		}
}


void measure_viscosity(field2_t density, 
						  field3_t velocity,
						  field3_t f,
						  field3_t post_f,
						  int dim,
						  int width,
						  int height,
						  scalar_t omega,
						  scalar_t epsilon,
						  int steps,
						  int rank, 
						  int size
						) {
				
		// fourier amplitude
		field_t a_sim("a_sim", steps);

		init_sinusiodal_pertubation(density, velocity, f, post_f, dim, width,
			 height, omega, epsilon, steps);

		// Simulation
		for (int step=0; step<steps; step++) {
			calc_collision(f, post_f, density, velocity, width, height, omega);
			streaming(f, post_f, width, height, rank, size);
			
			// calculate a_sim(step)
			double sum = 0.0L;
			for (int j=0; j<(height-1); j++) {
				sum += velocity(0, j, 0) * std::sin(2 * pi * j / height);
			}
			a_sim(step) = 2 / static_cast<double>(height) * sum;
		}
		save_viscosity_meas(a_sim, steps, "data/viscosity_meas.bin");
}

void test_bounce_back(field2_t density, 
						  field3_t velocity,
						  field3_t f,
						  field3_t post_f,
						  int dim,
						  int width,
						  int height,
						  scalar_t omega,
						  scalar_t epsilon,
						  int steps,
						  int rank,
						  int size
						) {
	Kokkos::deep_copy(f, 1);

	// Simulation
	for (int step=0; step<steps; step++) {
		calc_collision(f, post_f, density, velocity, width, height, omega);
		streaming(f, post_f, width, height, rank, size);
		if (step == steps - 1) {
			save_velocity(velocity, width, height, 0, 1, "data/velocity.bin");
			}
	}
}