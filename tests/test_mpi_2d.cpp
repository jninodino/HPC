#include "latticeBoltzmann.h"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

int diagonal_rank(MPI_Comm comm, const int coords[2], const int dims[2], int dx,
                  int dy) {
    const int x = coords[0] + dx;
    const int y = coords[1] + dy;

    if (x < 0 || x >= dims[0] || y < 0 || y >= dims[1]) {
        return MPI_PROC_NULL;
    }

    int diagonal_coords[2] = {x, y};
    int rank = MPI_PROC_NULL;
    MPI_Cart_rank(comm, diagonal_coords, &rank);
    return rank;
}

MpiGrid2D create_test_grid_2d() {
    MpiGrid2D grid;

    MPI_Comm_size(MPI_COMM_WORLD, &grid.size);
    grid.dims[0] = 0;
    grid.dims[1] = 0;
    MPI_Dims_create(grid.size, 2, grid.dims);

    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, grid.dims, periods, 0, &grid.comm);

    MPI_Comm_rank(grid.comm, &grid.rank);
    MPI_Cart_coords(grid.comm, grid.rank, 2, grid.coords);
    MPI_Cart_shift(grid.comm, 0, 1, &grid.left, &grid.right);
    MPI_Cart_shift(grid.comm, 1, 1, &grid.bottom, &grid.top);

    grid.bottom_left =
        diagonal_rank(grid.comm, grid.coords, grid.dims, -1, -1);
    grid.bottom_right =
        diagonal_rank(grid.comm, grid.coords, grid.dims, +1, -1);
    grid.top_left = diagonal_rank(grid.comm, grid.coords, grid.dims, -1, +1);
    grid.top_right = diagonal_rank(grid.comm, grid.coords, grid.dims, +1, +1);

    return grid;
}

bool has_four_ranks() {
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size == 4;
}

void initialize_uniform_equilibrium(field3_t f, int width, int height) {
    Kokkos::parallel_for(
        "test_initialize_uniform_equilibrium",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {width, height}),
        KOKKOS_LAMBDA(const int x, const int y) {
            for (int i = 0; i < v_dim; ++i) {
                f(x, y, i) = weight(i);
            }
        });
    Kokkos::fence();
}

scalar_t equilibrium_value(scalar_t rho, scalar_t ux, scalar_t uy, int i) {
    const scalar_t cu = cx(i) * ux + cy(i) * uy;
    const scalar_t u_squared = ux * ux + uy * uy;
    return weight(i) * rho *
           (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_squared);
}

} // namespace

TEST(MPI2D, x_halo_exchange) {
    if (!has_four_ranks()) {
        GTEST_SKIP() << "This test requires exactly 4 MPI ranks (2x2 grid).";
    }

    MpiGrid2D grid = create_test_grid_2d();
    EXPECT_EQ(grid.dims[0], 2);
    EXPECT_EQ(grid.dims[1], 2);

    constexpr int width = 6;
    constexpr int height = 6;

    field3_t f("x_halo_f", width, height, v_dim);
    Kokkos::deep_copy(f, 0.0);

    auto h_f = Kokkos::create_mirror_view(f);
    Kokkos::deep_copy(h_f, f);

    for (int y = 1; y < height - 1; ++y) {
        // Populations that have streamed into the left/right ghost columns.
        h_f(0, y, 3) = 1000.0 + 100.0 * grid.rank + y;
        h_f(width - 1, y, 1) = 2000.0 + 100.0 * grid.rank + y;
    }
    Kokkos::deep_copy(f, h_f);

    GhostBuffers2D buffers(width, height);
    share_ghost_cells_2d(f, width, height, grid, buffers);
    Kokkos::fence();

    h_f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f);

    for (int y = 1; y < height - 1; ++y) {
        if (grid.left != MPI_PROC_NULL) {
            const scalar_t expected = 2000.0 + 100.0 * grid.left + y;
            EXPECT_DOUBLE_EQ(h_f(1, y, 1), expected);
        }

        if (grid.right != MPI_PROC_NULL) {
            const scalar_t expected = 1000.0 + 100.0 * grid.right + y;
            EXPECT_DOUBLE_EQ(h_f(width - 2, y, 3), expected);
        }
    }

    MPI_Comm_free(&grid.comm);
}

TEST(MPI2D, y_halo_exchange) {
    if (!has_four_ranks()) {
        GTEST_SKIP() << "This test requires exactly 4 MPI ranks (2x2 grid).";
    }

    MpiGrid2D grid = create_test_grid_2d();
    EXPECT_EQ(grid.dims[0], 2);
    EXPECT_EQ(grid.dims[1], 2);

    constexpr int width = 6;
    constexpr int height = 6;

    field3_t f("y_halo_f", width, height, v_dim);
    Kokkos::deep_copy(f, 0.0);

    auto h_f = Kokkos::create_mirror_view(f);
    Kokkos::deep_copy(h_f, f);

    for (int x = 1; x < width - 1; ++x) {
        // Populations that have streamed into the bottom/top ghost rows.
        h_f(x, 0, 4) = 3000.0 + 100.0 * grid.rank + x;
        h_f(x, height - 1, 2) = 4000.0 + 100.0 * grid.rank + x;
    }
    Kokkos::deep_copy(f, h_f);

    GhostBuffers2D buffers(width, height);
    share_ghost_cells_2d(f, width, height, grid, buffers);
    Kokkos::fence();

    h_f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f);

    for (int x = 1; x < width - 1; ++x) {
        if (grid.bottom != MPI_PROC_NULL) {
            const scalar_t expected = 4000.0 + 100.0 * grid.bottom + x;
            EXPECT_DOUBLE_EQ(h_f(x, 1, 2), expected);
        }

        if (grid.top != MPI_PROC_NULL) {
            const scalar_t expected = 3000.0 + 100.0 * grid.top + x;
            EXPECT_DOUBLE_EQ(h_f(x, height - 2, 4), expected);
        }
    }

    MPI_Comm_free(&grid.comm);
}

TEST(MPI2D, diagonal_corner_exchange) {
    if (!has_four_ranks()) {
        GTEST_SKIP() << "This test requires exactly 4 MPI ranks (2x2 grid).";
    }

    MpiGrid2D grid = create_test_grid_2d();
    EXPECT_EQ(grid.dims[0], 2);
    EXPECT_EQ(grid.dims[1], 2);

    constexpr int width = 6;
    constexpr int height = 6;

    field3_t f("corner_halo_f", width, height, v_dim);
    Kokkos::deep_copy(f, 0.0);

    auto h_f = Kokkos::create_mirror_view(f);
    Kokkos::deep_copy(h_f, f);

    // One distinct value for each diagonal population/corner.
    h_f(width - 1, height - 1, 5) = 5000.0 + grid.rank; // top-right
    h_f(0, height - 1, 6) = 6000.0 + grid.rank;         // top-left
    h_f(0, 0, 7) = 7000.0 + grid.rank;                  // bottom-left
    h_f(width - 1, 0, 8) = 8000.0 + grid.rank;          // bottom-right
    Kokkos::deep_copy(f, h_f);

    GhostBuffers2D buffers(width, height);
    share_ghost_cells_2d(f, width, height, grid, buffers);
    Kokkos::fence();

    h_f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f);

    if (grid.bottom_left != MPI_PROC_NULL) {
        EXPECT_DOUBLE_EQ(h_f(1, 1, 5), 5000.0 + grid.bottom_left);
    }
    if (grid.bottom_right != MPI_PROC_NULL) {
        EXPECT_DOUBLE_EQ(h_f(width - 2, 1, 6), 6000.0 + grid.bottom_right);
    }
    if (grid.top_right != MPI_PROC_NULL) {
        EXPECT_DOUBLE_EQ(h_f(width - 2, height - 2, 7),
                         7000.0 + grid.top_right);
    }
    if (grid.top_left != MPI_PROC_NULL) {
        EXPECT_DOUBLE_EQ(h_f(1, height - 2, 8), 8000.0 + grid.top_left);
    }

    MPI_Comm_free(&grid.comm);
}

TEST(MPI2D, two_by_two_matches_serial) {
    if (!has_four_ranks()) {
        GTEST_SKIP() << "This test requires exactly 4 MPI ranks (2x2 grid).";
    }

    MpiGrid2D grid = create_test_grid_2d();
    EXPECT_EQ(grid.dims[0], 2);
    EXPECT_EQ(grid.dims[1], 2);

    constexpr int global_width = 12;
    constexpr int global_height = 12;
    constexpr int steps = 4;
    constexpr scalar_t omega = 1.7;

    const int local_nx = global_width / grid.dims[0];
    const int local_ny = global_height / grid.dims[1];
    const int local_width = local_nx + 2;
    const int local_height = local_ny + 2;

    field2_t density("mpi_regression_density", local_width, local_height);
    field3_t velocity("mpi_regression_velocity", local_width, local_height, 2);
    field3_t f("mpi_regression_f", local_width, local_height, v_dim);
    field3_t post_f("mpi_regression_post_f", local_width, local_height, v_dim);
    GhostBuffers2D buffers(local_width, local_height);

    initialize_uniform_equilibrium(f, local_width, local_height);

    // Place a non-trivial perturbation close to the internal 2x2 corner so
    // that x, y and diagonal communication are all exercised within 4 steps.
    constexpr int perturb_global_x = 4;
    constexpr int perturb_global_y = 4;
    constexpr scalar_t perturb_rho = 1.1;
    constexpr scalar_t perturb_ux = 0.04;
    constexpr scalar_t perturb_uy = 0.03;

    const int owner_x = perturb_global_x / local_nx;
    const int owner_y = perturb_global_y / local_ny;

    if (grid.coords[0] == owner_x && grid.coords[1] == owner_y) {
        const int local_x = perturb_global_x - owner_x * local_nx + 1;
        const int local_y = perturb_global_y - owner_y * local_ny + 1;

        auto h_f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f);
        for (int i = 0; i < v_dim; ++i) {
            h_f(local_x, local_y, i) =
                equilibrium_value(perturb_rho, perturb_ux, perturb_uy, i);
        }
        Kokkos::deep_copy(f, h_f);
    }

    for (int step = 0; step < steps; ++step) {
        execute_time_step_2d(f, post_f, density, velocity, local_width,
                             local_height, omega, grid, buffers);
    }

    // The last streaming step may have left populations in ghost cells. Move
    // them into the neighbouring owned cells before comparing with serial.
    share_ghost_cells_2d(f, local_width, local_height, grid, buffers);
    Kokkos::fence();

    auto h_f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f);

    const int local_count = local_nx * local_ny * v_dim;
    std::vector<scalar_t> packed(static_cast<std::size_t>(local_count));

    for (int local_x = 0; local_x < local_nx; ++local_x) {
        for (int local_y = 0; local_y < local_ny; ++local_y) {
            for (int i = 0; i < v_dim; ++i) {
                const std::size_t idx =
                    (static_cast<std::size_t>(local_x) * local_ny + local_y) *
                        v_dim +
                    i;
                packed[idx] = h_f(local_x + 1, local_y + 1, i);
            }
        }
    }

    std::vector<scalar_t> gathered;
    if (grid.rank == 0) {
        gathered.resize(static_cast<std::size_t>(local_count) * grid.size);
    }

    MPI_Gather(packed.data(), local_count, MPI_DOUBLE,
               grid.rank == 0 ? gathered.data() : nullptr, local_count,
               MPI_DOUBLE, 0, grid.comm);

    scalar_t max_error = 0.0;

    if (grid.rank == 0) {
        std::vector<scalar_t> global_mpi(
            static_cast<std::size_t>(global_width) * global_height * v_dim);

        for (int process = 0; process < grid.size; ++process) {
            int process_coords[2] = {0, 0};
            MPI_Cart_coords(grid.comm, process, 2, process_coords);

            const std::size_t tile_offset =
                static_cast<std::size_t>(process) * local_count;

            for (int local_x = 0; local_x < local_nx; ++local_x) {
                for (int local_y = 0; local_y < local_ny; ++local_y) {
                    const int global_x = process_coords[0] * local_nx + local_x;
                    const int global_y = process_coords[1] * local_ny + local_y;

                    for (int i = 0; i < v_dim; ++i) {
                        const std::size_t local_idx =
                            (static_cast<std::size_t>(local_x) * local_ny +
                             local_y) *
                                v_dim +
                            i;
                        const std::size_t global_idx =
                            (static_cast<std::size_t>(global_x) * global_height +
                             global_y) *
                                v_dim +
                            i;
                        global_mpi[global_idx] =
                            gathered[tile_offset + local_idx];
                    }
                }
            }
        }

        const int serial_width = global_width + 2;
        const int serial_height = global_height + 2;

        field2_t serial_density("serial_regression_density", serial_width,
                                serial_height);
        field3_t serial_velocity("serial_regression_velocity", serial_width,
                                 serial_height, 2);
        field3_t serial_f("serial_regression_f", serial_width, serial_height,
                          v_dim);
        field3_t serial_post_f("serial_regression_post_f", serial_width,
                               serial_height, v_dim);
        GhostBuffers2D serial_buffers(serial_width, serial_height);

        initialize_uniform_equilibrium(serial_f, serial_width, serial_height);

        auto h_serial_f =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, serial_f);
        for (int i = 0; i < v_dim; ++i) {
            h_serial_f(perturb_global_x + 1, perturb_global_y + 1, i) =
                equilibrium_value(perturb_rho, perturb_ux, perturb_uy, i);
        }
        Kokkos::deep_copy(serial_f, h_serial_f);

        MpiGrid2D serial_grid{};
        serial_grid.comm = MPI_COMM_SELF;
        serial_grid.rank = 0;
        serial_grid.size = 1;
        serial_grid.dims[0] = 1;
        serial_grid.dims[1] = 1;
        serial_grid.left = MPI_PROC_NULL;
        serial_grid.right = MPI_PROC_NULL;
        serial_grid.bottom = MPI_PROC_NULL;
        serial_grid.top = MPI_PROC_NULL;
        serial_grid.bottom_left = MPI_PROC_NULL;
        serial_grid.bottom_right = MPI_PROC_NULL;
        serial_grid.top_left = MPI_PROC_NULL;
        serial_grid.top_right = MPI_PROC_NULL;

        for (int step = 0; step < steps; ++step) {
            execute_time_step_2d(serial_f, serial_post_f, serial_density,
                                 serial_velocity, serial_width, serial_height,
                                 omega, serial_grid, serial_buffers);
        }
        Kokkos::fence();

        h_serial_f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          serial_f);

        for (int x = 0; x < global_width; ++x) {
            for (int y = 0; y < global_height; ++y) {
                for (int i = 0; i < v_dim; ++i) {
                    const std::size_t idx =
                        (static_cast<std::size_t>(x) * global_height + y) *
                            v_dim +
                        i;
                    max_error = std::max(
                        max_error,
                        std::abs(global_mpi[idx] - h_serial_f(x + 1, y + 1, i)));
                }
            }
        }
    }

    MPI_Bcast(&max_error, 1, MPI_DOUBLE, 0, grid.comm);
    EXPECT_LE(max_error, 1e-12)
        << "2x2 MPI result differs from the single-rank serial reference";

    MPI_Comm_free(&grid.comm);
}