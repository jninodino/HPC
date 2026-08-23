#pragma once

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <vector>

// Type aliases
using scalar_t = double;
using field_t = Kokkos::View<scalar_t *>;
using field2_t = Kokkos::View<scalar_t **>;
using field3_t = Kokkos::View<scalar_t ***>;

// Host pinned memory for MPI transfers
using pinned_field_t = Kokkos::View<scalar_t *, Kokkos::SharedHostPinnedSpace>; 


// Buffers for exchanging ghost cells with neighbouring MPI ranks
struct GhostBuffers {
    field_t send_device;
    field_t recv_device;

    pinned_field_t send_host;
    pinned_field_t recv_host;

    explicit GhostBuffers(int height)
        : send_device("ghost_send_device", height * 3),
          recv_device("ghost_recv_device", height * 3),
          send_host("ghost_send_host", height * 3),
          recv_host("ghost_recv_host", height * 3) {
    }
};

// Number of velocity directions
inline constexpr int v_dim = 9;

// Inverted velocity indices for boundary
KOKKOS_INLINE_FUNCTION
constexpr int opposite(int i) {
    constexpr int tab[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

    return tab[i];
}
// Velocity indices in x (right) direction
KOKKOS_INLINE_FUNCTION
constexpr int right_dir(int d) {
    switch (d) {
    case 0:
        return 1;
    case 1:
        return 5;
    case 2:
        return 8;
    default:
        return 0;
    }
}

// Veloxity indices in minus x (left) directions
KOKKOS_INLINE_FUNCTION
constexpr int left_dir(int d) {
    switch (d) {
    case 0:
        return 3;
    case 1:
        return 6;
    case 2:
        return 7;
    default:
        return 0;
    }
}

// D2Q9 velocity directions
KOKKOS_INLINE_FUNCTION constexpr int cx(int i) {
    constexpr int tab[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    return tab[i];
}

KOKKOS_INLINE_FUNCTION constexpr int cy(int i) {
    constexpr int tab[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    return tab[i];
}

// -----------------------------------------------------------------------------
// Functions
// -----------------------------------------------------------------------------

// Calculate square of scalar_t input
KOKKOS_INLINE_FUNCTION
constexpr scalar_t square(scalar_t value) {
    return value * value;
}

// Calculate density at point x, y
void calc_density(field3_t f, field2_t density, int x, int y);

// Calculate velocity at point x, y depended of density
void calc_velocity(field3_t f, field2_t density, field3_t velocity, int x,
                   int y);

KOKKOS_INLINE_FUNCTION
constexpr scalar_t weight(int i) {
    switch (i) {
    case 0:
        return 4.0 / 9.0;

    case 1:
    case 2:
    case 3:
    case 4:
        return 1.0 / 9.0;

    case 5:
    case 6:
    case 7:
    case 8:
        return 1.0 / 36.0;

    default:
        return 0.0;
    }
}
// Help function for calculate the equilibrium distribution function of
// probability density function
KOKKOS_INLINE_FUNCTION
scalar_t calc_f_eq(field2_t density, field3_t velocity, int x, int y, int i) {
    const scalar_t ux = velocity(x, y, 0);
    const scalar_t uy = velocity(x, y, 1);

    const scalar_t cu = cx(i) * ux + cy(i) * uy;

    const scalar_t u_squared = ux * ux + uy * uy;

    return weight(i) * density(x, y) *
           (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_squared);
}

// Calculate collision on borders implemented as in Milestone 5 explained
void calc_collision(field3_t f, field3_t post_f, field2_t density,
                    field3_t velocity, int width, int height, scalar_t omega,
                    bool relaxation = true);

// Add input from neighbored field
void add_neighbor_input(field3_t f, int width, int height, int rank, int size);

// Execute streaming step
void streaming(field3_t f, field3_t post_f, field2_t density, int width,
               int height, int rank, int size);

// Exchange ghost cells with neighbored field
void share_ghost_cells(field3_t f, int width, int height, int rank, int size,
                       GhostBuffers &buffers);

// Calculate total mass over complete domain
void calc_total_mass(double &total_mass, field2_t density, int width,
                     int height);

// Calculate total kinetic energy over complete domain
void calc_total_kin_energy(double &total_kin_energy, field3_t velocity,
                           field2_t density, int width, int height);

// Run one timestep of lattice boltzman simulation
void execute_time_step(field3_t f, field3_t post_f, field2_t density,
                       field3_t velocity, int width, int height, scalar_t omega,
                       int rank, int size, GhostBuffers &buffers);

scalar_t total_mass(field3_t f, int width, int height);

// Calculate density and velocity field in one kernel
void calc_macroscopic(field3_t f, field2_t density, field3_t velocity,
                      int width, int height);

// -----------------------------------------------------------------------------
// 2D MPI domain decomposition
// -----------------------------------------------------------------------------

// 2D MPI grid information for domain decomposition
struct MpiGrid2D {
    MPI_Comm comm = MPI_COMM_NULL; // Cartesian communicator for the 2D grid

    int rank = 0;
    int size = 1;

    // Number of processes in x and y direction
    int dims[2] = {1, 1};

    // Coordinates of this rank in the Cartesian process grid
    int coords[2] = {0, 0};

    // Direct neighbours
    int left = MPI_PROC_NULL;
    int right = MPI_PROC_NULL;
    int bottom = MPI_PROC_NULL;
    int top = MPI_PROC_NULL;

    // Diagonal neighbours for D2Q9 populations
    int bottom_left = MPI_PROC_NULL;
    int bottom_right = MPI_PROC_NULL;
    int top_left = MPI_PROC_NULL;
    int top_right = MPI_PROC_NULL;
};

struct GhostBuffers2D {
    // Buffers for left/right boundaries (3 pops per cell)
    field_t send_x_device;
    field_t recv_x_device;

    pinned_field_t send_x_host;
    pinned_field_t recv_x_host;

    // Buffers for top/bottom boundaries (3 pops per cell)
    field_t send_y_device;
    field_t recv_y_device;

    pinned_field_t send_y_host;
    pinned_field_t recv_y_host;

    // One buffer entry for each diag corner
    field_t send_corner_device;
    field_t recv_corner_device;

    pinned_field_t send_corner_host;
    pinned_field_t recv_corner_host;

    GhostBuffers2D(int width, int height)
        : send_x_device("ghost2d_send_x_device", (height - 2) * 3),
          recv_x_device("ghost2d_recv_x_device", (height - 2) * 3),
          send_x_host("ghost2d_send_x_host", (height - 2) * 3),
          recv_x_host("ghost2d_recv_x_host", (height - 2) * 3),

          send_y_device("ghost2d_send_y_device", (width - 2) * 3),
          recv_y_device("ghost2d_recv_y_device", (width - 2) * 3),
          send_y_host("ghost2d_send_y_host", (width - 2) * 3),
          recv_y_host("ghost2d_recv_y_host", (width - 2) * 3),

          send_corner_device("ghost2d_send_corner_device", 4),
          recv_corner_device("ghost2d_recv_corner_device", 4),
          send_corner_host("ghost2d_send_corner_host", 4),
          recv_corner_host("ghost2d_recv_corner_host", 4) {
    }
};

// Exchange D2Q9 populations across all four boundaries and corners.
void share_ghost_cells_2d(field3_t f, int width, int height,
                          const MpiGrid2D &grid, GhostBuffers2D &buffers);

// Run one timestep on a 2D MPI subdomain.
void execute_time_step_2d(field3_t f, field3_t post_f, field2_t density,
                          field3_t velocity, int width, int height,
                          scalar_t omega, const MpiGrid2D &grid,
                          GhostBuffers2D &buffers);

// Global mass over all owned cells of all MPI ranks.
void calc_total_mass_2d(double &total_mass, field2_t density, int width,
                        int height, MPI_Comm comm);

// Global kinetic energy over all owned cells of all MPI ranks.
void calc_total_kin_energy_2d(double &total_kin_energy, field3_t velocity,
                              field2_t density, int width, int height,
                              MPI_Comm comm);