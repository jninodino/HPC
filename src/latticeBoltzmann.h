#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

// Type aliases
using scalar_t = double;
using field_t = Kokkos::View<scalar_t*>;
using field2_t = Kokkos::View<scalar_t**>;
using field3_t = Kokkos::View<scalar_t***>;


// Number of velocity directions in lattice boltzmann
inline constexpr int v_dim = 9;
// Inverted velocity indices for boundary
inline constexpr int opposite_i[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
// Velocity indices in x (right) direction
inline constexpr int right_dirs[3] = {1, 5, 8};
// Veloxity indices in minus x (left) directions
inline constexpr int left_dirs[3] = {3, 6, 7};


KOKKOS_INLINE_FUNCTION constexpr int cx(int i) {
	constexpr int tab[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
	return tab[i];
}

KOKKOS_INLINE_FUNCTION constexpr int cy(int i) {
	constexpr int tab[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
	return tab[i];
	
}

extern scalar_t W[9];
extern scalar_t cs_s;


// Functions
// TODO add _ for inline functions

// Calculate square of scalar_t input
scalar_t square(scalar_t input);

// Calculate density at point x, y
void calc_density(field3_t f, field2_t density, int x, int y);

// Calculate velocity at point x, y depended of density
void calc_velocity(field3_t f, field2_t density, field3_t velocity, int x, 
	int y);

// Help function for calculate the equilibrium distribution function of 
// probability density function
scalar_t calc_f_eq(field2_t density, field3_t velocity, int x, int y, int i);

// Calculate collision on borders implemented as in Milestone 5 explained
void calc_collision(field3_t f, field3_t post_f, field2_t density, 
	field3_t velocity, int width, int height, scalar_t omega, 
	bool relaxation = true);

// Check if 'particle moves into a bounce back wall
bool is_bounce_back(int x, int y, int i, int width, int height, int rank, 
	int size);

// Check if 'particle' moves into a moving wall
bool is_moving_wall(int x, int y, int i, int width, int height);

// For domain decomposition: Add input from neighbored field
void add_neighbor_input(field3_t f, int width, int height, int rank, 
	int size);

// Execute streaming step
void streaming(field3_t f, field3_t post_f, int width, int height, 
 int rank, int size);

// For domain decomposition: Exchange ghost cells with neighbored field
void share_ghost_cells(field3_t f, int width, int height, int rank, int size);

// Calculate total mass over complete domain
void calc_total_mass(double total_mass, field2_t density, int width, int height);

// Calculate total kinetic energy over complete domain
void calc_total_kin_energy(double total_kin_energy, field3_t velocity,
	field2_t density, int width, int height);

// Run one timestep of lattice boltzman simulation
void execute_time_step(field3_t f, field3_t post_f, field2_t density, 
	field3_t velocity, int width, int height, scalar_t omega, int rank, int size);

scalar_t total_mass(field3_t f, int width, int height);