#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "latticeBoltzmann.h"

//____________________________________________________________________________
scalar_t total_mass(field3_t f, int width, int height) {
    scalar_t mass = 0.0;
    Kokkos::parallel_reduce(
        "total_mass",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                               {width, height, v_dim}),
        KOKKOS_LAMBDA(const int x, const int y, const int i,
                      scalar_t &local_mass) { local_mass += f(x, y, i); },
        mass);
    return mass;
}

//____________________________________________________________________________
void calc_density(field3_t f, field2_t density, int width, int height) {
    Kokkos::parallel_for(
        "density",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {width, height}),
        KOKKOS_LAMBDA(const int x, const int y) {
            scalar_t sum = 0;
            for (int i = 0; i < v_dim; i++) {
                sum += f(x, y, i);
            }
            density(x, y) = sum;
        });
}

//____________________________________________________________________________
void calc_velocity(field3_t f, field2_t density, field3_t velocity, int width,
                   int height) {
    Kokkos::parallel_for(
        "velocity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {width, height}),
        KOKKOS_LAMBDA(const int x, const int y) {
            scalar_t d = density(x, y);

            if (d == 0) {
                velocity(x, y, 0) = 0;
                velocity(x, y, 1) = 0;
                return;
            }
            scalar_t u_x = 0;
            scalar_t u_y = 0;
            for (int i = 0; i < v_dim; i++) {
                u_x += f(x, y, i) * cx(i);
                u_y += f(x, y, i) * cy(i);
            }
            velocity(x, y, 0) = u_x / d;
            velocity(x, y, 1) = u_y / d;
        });
}

//____________________________________________________________________________
void calc_macroscopic(field3_t f, field2_t density, field3_t velocity,
                      int width, int height) {
    Kokkos::parallel_for(
        "macroscopic",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {width, height}),
        KOKKOS_LAMBDA(const int x, const int y) {
            scalar_t rho = 0.0;
            scalar_t momentum_x = 0.0;
            scalar_t momentum_y = 0.0;

            for (int i = 0; i < v_dim; ++i) {
                const scalar_t fi = f(x, y, i);

                rho += fi;
                momentum_x += fi * cx(i);
                momentum_y += fi * cy(i);
            }

            density(x, y) = rho;

            if (rho != 0.0) {
                velocity(x, y, 0) = momentum_x / rho;
                velocity(x, y, 1) = momentum_y / rho;
            } else {
                velocity(x, y, 0) = 0.0;
                velocity(x, y, 1) = 0.0;
            }
        });
}

//____________________________________________________________________________
void calc_collision(field3_t f, field3_t post_f, field2_t density,
                    field3_t velocity, int width, int height, scalar_t omega,
                    bool relaxation) {
    Kokkos::parallel_for(
        "collision",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {width - 1, height}),
        KOKKOS_LAMBDA(const int x, const int y) {
            scalar_t rho = 0.0;
            scalar_t momentum_x = 0.0;
            scalar_t momentum_y = 0.0;

            // Compute macroscopic quantities
            for (int i = 0; i < v_dim; ++i) {
                const scalar_t fi = f(x, y, i);

                rho += fi;
                momentum_x += fi * cx(i);
                momentum_y += fi * cy(i);
            }

            scalar_t ux = 0.0;
            scalar_t uy = 0.0;

            if (rho != 0.0) {
                ux = momentum_x / rho;
                uy = momentum_y / rho;
            }

            density(x, y) = rho;
            velocity(x, y, 0) = ux;
            velocity(x, y, 1) = uy;

            const scalar_t u_squared = ux * ux + uy * uy;

            // Collision
            for (int i = 0; i < v_dim; ++i) {

                if (!relaxation) {
                    post_f(x, y, i) = f(x, y, i);
                    continue;
                }

                const scalar_t cu = cx(i) * ux + cy(i) * uy;

                const scalar_t f_eq =
                    weight(i) * rho *
                    (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_squared);

                const scalar_t fi = f(x, y, i);

                post_f(x, y, i) = fi - omega * (fi - f_eq);
            }
        });
}

//____________________________________________________________________________
void add_neighbor_input(field3_t f, int width, int height, int rank, int size) {

    const bool has_left = rank > 0;
    const bool has_right = rank < size - 1;

    if (has_left) {
        Kokkos::parallel_for(
            "left_boundary", Kokkos::RangePolicy<>(0, height),
            KOKKOS_LAMBDA(const int y) {
                f(1, y, 1) = f(0, y, 1);
                if (y > 0) {
                    f(1, y, 5) = f(0, y, 5);
                }
                if (y < height - 1) {
                    f(1, y, 8) = f(0, y, 8);
                }
            });
    }

    // Mirror of the above for the right boundary (directions 3, 6, 7).
    if (has_right) {
        Kokkos::parallel_for(
            "right_boundary", Kokkos::RangePolicy<>(0, height),
            KOKKOS_LAMBDA(const int y) {
                f(width - 2, y, 3) = f(width - 1, y, 3);

                if (y > 0) {
                    f(width - 2, y, 6) = f(width - 1, y, 6);
                }
                if (y < height - 1) {
                    f(width - 2, y, 7) = f(width - 1, y, 7);
                }
            });
    }
}

//____________________________________________________________________________
KOKKOS_INLINE_FUNCTION
bool is_bounce_back(int x, int y, int i, int width, int height, int rank,
                    int size) {
    const bool left = rank == 0 && x == 1;
    const bool right = rank == size - 1 && x == width - 2;
    const bool bottom = y == 0;
    const bool top = y == height - 1;

    switch (i) {
    case 1:
        return right;
    case 3:
        return left;
    case 4:
        return bottom;
    case 5:
        return right || (left && top);
    case 6:
        return left || (right && top);
    case 7:
        return left || bottom;
    case 8:
        return right || bottom;
    default:
        return false;
    }
}

KOKKOS_INLINE_FUNCTION
bool is_moving_wall(int y, int i, int height) {
    return (y == height - 1) && (i == 2 || i == 5 || i == 6);
}

KOKKOS_INLINE_FUNCTION
bool is_boundary(int x, int y, int i, int width, int height, int rank,
                 int size) {
    const bool left = rank == 0 && x == 1;
    const bool right = rank == size - 1 && x == width - 2;
    const bool bottom = y == 0;
    const bool top = y == height - 1;

    if (!left && !right && !bottom && !top)
        return false;

    switch (i) {
    case 1:
        return right;
    case 2:
        return top;
    case 3:
        return left;
    case 4:
        return bottom;
    case 5:
        return right || top;
    case 6:
        return left || top;
    case 7:
        return left || bottom;
    case 8:
        return right || bottom;
    default:
        return false;
    }
}

//____________________________________________________________________________
KOKKOS_INLINE_FUNCTION
void handle_boundary(field3_t f, field3_t post_f, field2_t density, int x,
                     int y, int i, int width, int height, int rank, int size,
                     bool periodic_bound = false) {
    if (periodic_bound) {
        int x_next = x;
        if (x == 1) {
            x_next = width - 2;
        } else if (x == width - 2) {
            x_next = 1;
        }
        const int y_next = (height + y + cy(i)) % height;
        f(x_next, y_next, i) = post_f(x, y, i);
    } else if (is_bounce_back(x, y, i, width, height, rank, size)) {
        f(x, y, opposite(i)) = post_f(x, y, i);
    } else if (is_moving_wall(y, i, height)) {
        constexpr scalar_t wall_velocity = 0.1;
        constexpr scalar_t cs2 = 1.0 / 3.0;
        f(x, y, opposite(i)) = post_f(x, y, i) - 2.0 * weight(i) * cx(i) *
        density(x, y) * wall_velocity / cs2;
    }
}

//____________________________________________________________________________
void streaming(field3_t f, field3_t post_f, field2_t density, int width,
               int height, int rank, int size) {
    Kokkos::parallel_for(
        "update_f",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 0, 0},
                                               {width - 1, height, v_dim}),
        KOKKOS_LAMBDA(const int x, const int y, const int i) {
            if (is_boundary(x, y, i, width, height, rank, size)) {
                handle_boundary(f, post_f, density, x, y, i, width, height,
                                rank, size);
            } else {
                int x_next = x + cx(i);
                int y_next = y + cy(i);
                f(x_next, y_next, i) = post_f(x, y, i);
            }
        });
}

//____________________________________________________________________________
void share_ghost_cells(field3_t f, int width, int height, int rank, int size,
                       GhostBuffers &buffers) {
    const int left = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int right = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
    const int buff_size = height * 3;

    auto send_device = buffers.send_device;
    auto recv_device = buffers.recv_device;

    // 1. Send to LEFT, receive from RIGHT
    Kokkos::parallel_for(
        "pack_left_ghost", Kokkos::RangePolicy<>(0, buff_size),
        KOKKOS_LAMBDA(const int idx) {
            const int y = idx / 3;
            const int d = idx % 3;
            send_device(idx) = f(0, y, left_dir(d));
        });

    // Device -> Host
    Kokkos::deep_copy(buffers.send_host, buffers.send_device);
    MPI_Sendrecv(buffers.send_host.data(), buff_size, MPI_DOUBLE, left, 0,
                 buffers.recv_host.data(), buff_size, MPI_DOUBLE, right, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Host -> Device
    Kokkos::deep_copy(buffers.recv_device, buffers.recv_host);

    Kokkos::parallel_for(
        "unpack_right_ghost", Kokkos::RangePolicy<>(0, buff_size),
        KOKKOS_LAMBDA(const int idx) {
            const int y = idx / 3;
            const int d = idx % 3;

            f(width - 1, y, left_dir(d)) = recv_device(idx);
        });

    // 2. Send to RIGHT, receive from LEFT
    Kokkos::parallel_for(
        "pack_right_ghost", Kokkos::RangePolicy<>(0, buff_size),
        KOKKOS_LAMBDA(const int idx) {
            const int y = idx / 3;
            const int d = idx % 3;

            send_device(idx) = f(width - 1, y, right_dir(d));
        });

    Kokkos::deep_copy(buffers.send_host, buffers.send_device);

    MPI_Sendrecv(buffers.send_host.data(), buff_size, MPI_DOUBLE, right, 1,
                 buffers.recv_host.data(), buff_size, MPI_DOUBLE, left, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    Kokkos::deep_copy(buffers.recv_device, buffers.recv_host);
    Kokkos::parallel_for(
        "unpack_left_ghost", Kokkos::RangePolicy<>(0, buff_size),
        KOKKOS_LAMBDA(const int idx) {
            const int y = idx / 3;
            const int d = idx % 3;
            f(0, y, right_dir(d)) = recv_device(idx);
        });
    add_neighbor_input(f, width, height, rank, size);
}

//____________________________________________________________________________
void calc_total_mass(double &total_mass, field2_t density, int width,
                     int height) {
    double local_mass = 0.0L;
    Kokkos::parallel_reduce(
        "sum_local_mass",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {width - 1, height}),
        KOKKOS_LAMBDA(const int x, const int y, double &lsum) {
            lsum += density(x, y);
        }, local_mass);
    Kokkos::fence();
    MPI_Reduce(&local_mass, &total_mass, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
}

//____________________________________________________________________________
void calc_total_kin_energy(double &total_kin_energy, field3_t velocity,
                           field2_t density, int width, int height) {
    double local_energy = 0.0;
    Kokkos::parallel_reduce(
        "sum_local_kin_energy",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {width - 1, height}),
        KOKKOS_LAMBDA(const int x, const int y, double &lsum) {
            lsum += 0.5 * density(x, y) *
                    (square(velocity(x, y, 0)) + square(velocity(x, y, 1)));
        }, local_energy);
    Kokkos::fence();
    MPI_Reduce(&local_energy, &total_kin_energy, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
}

//____________________________________________________________________________
void execute_time_step(field3_t f, field3_t post_f, field2_t density,
                       field3_t velocity, int width, int height, scalar_t omega,
                       int rank, int size, GhostBuffers &buffers) {
    if (size > 1) {
        share_ghost_cells(f, width, height, rank, size, buffers);
    }
    calc_collision(f, post_f, density, velocity, width, height, omega);
    streaming(f, post_f, density, width, height, rank, size);
}


// 2D MPI domain decomposition
//____________________________________________________________________________
KOKKOS_INLINE_FUNCTION
constexpr int top_dir(int d) {
    switch (d) {
    case 0:
        return 2;
    case 1:
        return 5;
    case 2:
        return 6;
    default:
        return 0;
    }
}

KOKKOS_INLINE_FUNCTION
constexpr int bottom_dir(int d) {
    switch (d) {
    case 0:
        return 4;
    case 1:
        return 7;
    case 2:
        return 8;
    default:
        return 0;
    }
}

//____________________________________________________________________________
void calc_collision_2d(field3_t f, field3_t post_f, field2_t density,
                       field3_t velocity, int width, int height,
                       scalar_t omega) {
    // Collide owned cells only, ghost cells are excluded
    Kokkos::parallel_for(
        "collision_2d",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {width - 1, height - 1}),
        KOKKOS_LAMBDA(const int x, const int y) {
            scalar_t rho = 0.0;
            scalar_t momentum_x = 0.0;
            scalar_t momentum_y = 0.0;

            for (int i = 0; i < v_dim; ++i) {
                const scalar_t fi = f(x, y, i);
                rho += fi;
                momentum_x += fi * cx(i);
                momentum_y += fi * cy(i);
            }

            scalar_t ux = 0.0;
            scalar_t uy = 0.0;

            if (rho != 0.0) {
                ux = momentum_x / rho;
                uy = momentum_y / rho;
            }
            density(x, y) = rho;
            velocity(x, y, 0) = ux;
            velocity(x, y, 1) = uy;
            const scalar_t u_squared = ux * ux + uy * uy;
            for (int i = 0; i < v_dim; ++i) {
                const scalar_t cu = cx(i) * ux + cy(i) * uy;
                const scalar_t f_eq =
                    weight(i) * rho *
                    (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_squared);
                const scalar_t fi = f(x, y, i);
                post_f(x, y, i) = fi - omega * (fi - f_eq);
            }
        });
}

//____________________________________________________________________________
KOKKOS_INLINE_FUNCTION
bool is_bounce_back_2d(int x, int y, int i, int width, int height,
                       bool physical_left, bool physical_right,
                       bool physical_bottom, bool physical_top) {
    const bool left = physical_left && x == 1;
    const bool right = physical_right && x == width - 2;
    const bool bottom = physical_bottom && y == 1;
    const bool top = physical_top && y == height - 2;

    // match 1D corner treatment
    switch (i) {
    case 1:
        return right;
    case 3:
        return left;
    case 4:
        return bottom;
    case 5:
        return right || (left && top);
    case 6:
        return left || (right && top);
    case 7:
        return left || bottom;
    case 8:
        return right || bottom;
    default:
        return false;
    }
}

//____________________________________________________________________________
KOKKOS_INLINE_FUNCTION
bool is_moving_wall_2d(int y, int i, int height, bool physical_top) {
    return physical_top && y == height - 2 && (i == 2 || i == 5 || i == 6);
}

//____________________________________________________________________________
KOKKOS_INLINE_FUNCTION
bool is_boundary_2d(int x, int y, int i, int width, int height,
                    bool physical_left, bool physical_right,
                    bool physical_bottom, bool physical_top) {
    const bool left = physical_left && x == 1;
    const bool right = physical_right && x == width - 2;
    const bool bottom = physical_bottom && y == 1;
    const bool top = physical_top && y == height - 2;
    switch (i) {
    case 1: return right;
    case 2: return top;
    case 3: return left;
    case 4: return bottom;
    case 5: return right || top;
    case 6: return left || top;
    case 7: return left || bottom;
    case 8: return right || bottom;
    default: return false;
    }
}

//____________________________________________________________________________
KOKKOS_INLINE_FUNCTION
void handle_boundary_2d(field3_t f, field3_t post_f, field2_t density, int x,
                        int y, int i, int width, int height, bool physical_left,
                        bool physical_right, bool physical_bottom,
                        bool physical_top) {
    if (is_bounce_back_2d(x, y, i, width, height, physical_left, physical_right,
                          physical_bottom, physical_top)) {
        f(x, y, opposite(i)) = post_f(x, y, i);
        return;
    }
    if (is_moving_wall_2d(y, i, height, physical_top)) {
        constexpr scalar_t wall_velocity = 0.1;
        constexpr scalar_t cs2 = 1.0 / 3.0;
        f(x, y, opposite(i)) = post_f(x, y, i) - 2.0 * weight(i) * cx(i) *
                    density(x, y) * wall_velocity / cs2;
    }
}

//____________________________________________________________________________
void streaming_2d(field3_t f, field3_t post_f, field2_t density, int width,
                  int height, const MpiGrid2D &grid) {
    // MPI boundaries stream into ghosts; physical boundaries apply walls
    const bool physical_left = grid.left == MPI_PROC_NULL;
    const bool physical_right = grid.right == MPI_PROC_NULL;
    const bool physical_bottom = grid.bottom == MPI_PROC_NULL;
    const bool physical_top = grid.top == MPI_PROC_NULL;
    Kokkos::parallel_for(
        "streaming_2d",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 0},
                                               {width - 1, height - 1, v_dim}),
        KOKKOS_LAMBDA(const int x, const int y, const int i) {
            if (is_boundary_2d(x, y, i, width, height, physical_left,
                               physical_right, physical_bottom, physical_top)) {
                handle_boundary_2d(f, post_f, density, x, y, i, width, height,
                                   physical_left, physical_right,
                                   physical_bottom, physical_top);
            } else {
                const int x_next = x + cx(i);
                const int y_next = y + cy(i);
                f(x_next, y_next, i) = post_f(x, y, i);
            }
        });
}

//____________________________________________________________________________
void share_ghost_cells_2d(field3_t f, int width, int height,
                          const MpiGrid2D &grid, GhostBuffers2D &buffers) {
    // Move ghost populations to neighbouring owned boundary cells
    // X-direction halos
    const bool physical_left = grid.left == MPI_PROC_NULL;
    const bool physical_right = grid.right == MPI_PROC_NULL;
    const bool physical_bottom = grid.bottom == MPI_PROC_NULL;
    const bool physical_top = grid.top == MPI_PROC_NULL;
    const int x_buffer_size = (height - 2) * 3;

    auto send_x_device = buffers.send_x_device;
    auto recv_x_device = buffers.recv_x_device;

    // Send left, receive from right
    Kokkos::parallel_for(
        "pack_left_2d", Kokkos::RangePolicy<>(0, x_buffer_size),
        KOKKOS_LAMBDA(const int idx) {
            const int y = 1 + idx / 3;
            const int d = idx % 3;

            send_x_device(idx) = f(0, y, left_dir(d));
        });
    Kokkos::deep_copy(buffers.send_x_host, buffers.send_x_device);

    MPI_Sendrecv(
        buffers.send_x_host.data(), x_buffer_size, MPI_DOUBLE, grid.left, 100,
        buffers.recv_x_host.data(), x_buffer_size, MPI_DOUBLE, grid.right, 100,
        grid.comm, MPI_STATUS_IGNORE);

    if (grid.right != MPI_PROC_NULL) {
        Kokkos::deep_copy(buffers.recv_x_device, buffers.recv_x_host);
        Kokkos::parallel_for(
            "unpack_from_right_2d", Kokkos::RangePolicy<>(0, x_buffer_size),
            KOKKOS_LAMBDA(const int idx) {
                const int y = 1 + idx / 3;
                const int d = idx % 3;
                const int i = left_dir(d);
                const int source_y = y - cy(i);
                const bool outside_y = (physical_bottom && source_y < 1) ||
                                       (physical_top && source_y > height - 2);
                if (!outside_y) {
                    f(width - 2, y, i) = recv_x_device(idx);
                }
            });
    }

    // Send right, receive from left
    Kokkos::parallel_for(
        "pack_right_2d", Kokkos::RangePolicy<>(0, x_buffer_size),
        KOKKOS_LAMBDA(const int idx) {
            const int y = 1 + idx / 3;
            const int d = idx % 3;
            send_x_device(idx) = f(width - 1, y, right_dir(d));
        });

    Kokkos::deep_copy(buffers.send_x_host, buffers.send_x_device);

    MPI_Sendrecv(
        buffers.send_x_host.data(), x_buffer_size, MPI_DOUBLE, grid.right, 101,
        buffers.recv_x_host.data(), x_buffer_size, MPI_DOUBLE, grid.left, 101,
        grid.comm, MPI_STATUS_IGNORE);

    if (grid.left != MPI_PROC_NULL) {
        Kokkos::deep_copy(buffers.recv_x_device, buffers.recv_x_host);
        Kokkos::parallel_for(
            "unpack_from_left_2d", Kokkos::RangePolicy<>(0, x_buffer_size),
            KOKKOS_LAMBDA(const int idx) {
                const int y = 1 + idx / 3;
                const int d = idx % 3;
                const int i = right_dir(d);
                const int source_y = y - cy(i);
                const bool outside_y = (physical_bottom && source_y < 1) ||
                                       (physical_top && source_y > height - 2);
                if (!outside_y) {
                    f(1, y, i) = recv_x_device(idx);
                }
            });
    }

    // Y direction halos
    const int y_buffer_size = (width - 2) * 3;
    auto send_y_device = buffers.send_y_device;
    auto recv_y_device = buffers.recv_y_device;
    
    // Send bottom, receive from top
    Kokkos::parallel_for(
        "pack_bottom_2d", Kokkos::RangePolicy<>(0, y_buffer_size),
        KOKKOS_LAMBDA(const int idx) {
            const int x = 1 + idx / 3;
            const int d = idx % 3;
            send_y_device(idx) = f(x, 0, bottom_dir(d));
        });
    Kokkos::deep_copy(buffers.send_y_host, buffers.send_y_device);
    MPI_Sendrecv(
        buffers.send_y_host.data(), y_buffer_size, MPI_DOUBLE, grid.bottom, 102,
        buffers.recv_y_host.data(), y_buffer_size, MPI_DOUBLE, grid.top, 102,
        grid.comm, MPI_STATUS_IGNORE);

    if (grid.top != MPI_PROC_NULL) {
        Kokkos::deep_copy(buffers.recv_y_device, buffers.recv_y_host);
        Kokkos::parallel_for(
            "unpack_from_top_2d", Kokkos::RangePolicy<>(0, y_buffer_size),
            KOKKOS_LAMBDA(const int idx) {
                const int x = 1 + idx / 3;
                const int d = idx % 3;
                const int i = bottom_dir(d);
                const int source_x = x - cx(i);
                const bool outside_x = (physical_left && source_x < 1) ||
                                       (physical_right && source_x > width - 2);
                if (!outside_x) {
                    f(x, height - 2, i) = recv_y_device(idx);
                }
            });
    }

    // Send top, receive from bottom
    Kokkos::parallel_for(
        "pack_top_2d", Kokkos::RangePolicy<>(0, y_buffer_size),
        KOKKOS_LAMBDA(const int idx) {
            const int x = 1 + idx / 3;
            const int d = idx % 3;
            send_y_device(idx) = f(x, height - 1, top_dir(d));
        });

    Kokkos::deep_copy(buffers.send_y_host, buffers.send_y_device);
    MPI_Sendrecv(
        buffers.send_y_host.data(), y_buffer_size, MPI_DOUBLE, grid.top, 103,
        buffers.recv_y_host.data(), y_buffer_size, MPI_DOUBLE, grid.bottom, 103,
        grid.comm, MPI_STATUS_IGNORE);
    if (grid.bottom != MPI_PROC_NULL) {
        Kokkos::deep_copy(buffers.recv_y_device, buffers.recv_y_host);
        Kokkos::parallel_for(
            "unpack_from_bottom_2d", Kokkos::RangePolicy<>(0, y_buffer_size),
            KOKKOS_LAMBDA(const int idx) {
                const int x = 1 + idx / 3;
                const int d = idx % 3;
                const int i = top_dir(d);
                const int source_x = x - cx(i);
                const bool outside_x = (physical_left && source_x < 1) ||
                                       (physical_right && source_x > width - 2);
                if (!outside_x) {
                    f(x, 1, i) = recv_y_device(idx);
                }
            });
    }

    // Corner halos
    auto send_corner_device = buffers.send_corner_device;
    auto recv_corner_device = buffers.recv_corner_device;
    Kokkos::parallel_for(
        "pack_corners_2d", Kokkos::RangePolicy<>(0, 4),
        KOKKOS_LAMBDA(const int corner) {
            switch (corner) {
            case 0:
                // top-right -> direction 5
                send_corner_device(0) = f(width - 1, height - 1, 5);
                break;
            case 1:
                // top-left -> direction 6
                send_corner_device(1) = f(0, height - 1, 6);
                break;
            case 2:
                // bottom-left -> direction 7
                send_corner_device(2) = f(0, 0, 7);
                break;
            case 3:
                // bottom-right -> direction 8
                send_corner_device(3) = f(width - 1, 0, 8);
                break;
            }
        });

    Kokkos::deep_copy(buffers.send_corner_host, buffers.send_corner_device);
    // top-right -> receive from bottom-left
    MPI_Sendrecv(buffers.send_corner_host.data() + 0, 1, MPI_DOUBLE,
                 grid.top_right, 104,
                 buffers.recv_corner_host.data() + 0, 1, MPI_DOUBLE,
                 grid.bottom_left, 104,
                 grid.comm, MPI_STATUS_IGNORE);
    // top-left -> receive from bottom-right
    MPI_Sendrecv(buffers.send_corner_host.data() + 1, 1, MPI_DOUBLE,
                 grid.top_left, 105,
                 buffers.recv_corner_host.data() + 1, 1, MPI_DOUBLE,
                 grid.bottom_right, 105,
                 grid.comm, MPI_STATUS_IGNORE);
    // bottom-left -> receive from top-right
    MPI_Sendrecv(buffers.send_corner_host.data() + 2, 1, MPI_DOUBLE,
                 grid.bottom_left, 106,
                 buffers.recv_corner_host.data() + 2, 1, MPI_DOUBLE,
                 grid.top_right, 106,
                 grid.comm, MPI_STATUS_IGNORE);

    // bottom-right -> receive from top-left
    MPI_Sendrecv(buffers.send_corner_host.data() + 3, 1, MPI_DOUBLE,
                 grid.bottom_right, 107,
                 buffers.recv_corner_host.data() + 3, 1, MPI_DOUBLE,
                 grid.top_left, 107,
                 grid.comm, MPI_STATUS_IGNORE);

    Kokkos::deep_copy(buffers.recv_corner_device, buffers.recv_corner_host);
    const bool from_bottom_left = grid.bottom_left != MPI_PROC_NULL;
    const bool from_bottom_right = grid.bottom_right != MPI_PROC_NULL;
    const bool from_top_right = grid.top_right != MPI_PROC_NULL;
    const bool from_top_left = grid.top_left != MPI_PROC_NULL;

    Kokkos::parallel_for(
        "unpack_corners_2d", Kokkos::RangePolicy<>(0, 4),
        KOKKOS_LAMBDA(const int corner) {
            switch (corner) {
            case 0:
                // direction 5 arriving from bottom-left
                if (from_bottom_left) {
                    f(1, 1, 5) = recv_corner_device(0);
                }
                break;
            case 1:
                // direction 6 arriving from bottom-right
                if (from_bottom_right) {
                    f(width - 2, 1, 6) = recv_corner_device(1);
                }
                break;
            case 2:
                // direction 7 arriving from top-right
                if (from_top_right) {
                    f(width - 2, height - 2, 7) = recv_corner_device(2);
                }
                break;
            case 3:
                // direction 8 arriving from top-left
                if (from_top_left) {
                    f(1, height - 2, 8) = recv_corner_device(3);
                }
                break;
            }
        });
}

//____________________________________________________________________________
void calc_total_mass_2d(double &total_mass, field2_t density, int width,
                        int height, MPI_Comm comm) {
    double local_mass = 0.0;
    Kokkos::parallel_reduce(
        "sum_local_mass_2d",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {width - 1, height - 1}),
        KOKKOS_LAMBDA(const int x, const int y, double &sum) {
            sum += density(x, y);
        },
        local_mass);
    Kokkos::fence();
    MPI_Allreduce(&local_mass, &total_mass, 1, MPI_DOUBLE, MPI_SUM, comm);
}

//____________________________________________________________________________
void calc_total_kin_energy_2d(double &total_kin_energy, field3_t velocity,
                              field2_t density, int width, int height,
                              MPI_Comm comm) {
    double local_energy = 0.0;

    Kokkos::parallel_reduce(
        "sum_local_kin_energy_2d",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {width - 1, height - 1}),
        KOKKOS_LAMBDA(const int x, const int y, double &sum) {
            const scalar_t ux = velocity(x, y, 0);
            const scalar_t uy = velocity(x, y, 1);
            sum += 0.5 * density(x, y) * (ux * ux + uy * uy);
        },
        local_energy);
    Kokkos::fence();
    MPI_Allreduce(&local_energy, &total_kin_energy, 1, MPI_DOUBLE, MPI_SUM,
                  comm);
}

void execute_time_step_2d(field3_t f, field3_t post_f, field2_t density,
                          field3_t velocity, int width, int height,
                          scalar_t omega, const MpiGrid2D &grid,
                          GhostBuffers2D &buffers) {
    // Exchange populations stored in ghost cells
    if (grid.size > 1) {
        share_ghost_cells_2d(f, width, height, grid, buffers);
    }

    calc_collision_2d(f, post_f, density, velocity, width, height, omega);

    streaming_2d(f, post_f, density, width, height, grid);
}