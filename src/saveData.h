#pragma once

#include <string>
#include <Kokkos_Core.hpp>
#include "latticeBoltzmann.h"

// Save the density field to a binary file with metadata header
void save_density(field2_t density, int width, int height, int step, int steps, const std::string& filename);

// Save the velocity field to a binary file with metadata header
void save_velocity(field3_t velocity, int width, int height, int step, int steps, const std::string& filename);