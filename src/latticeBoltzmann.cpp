#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "latticeBoltzmann.h"

// Remove?
scalar_t total_mass(field3_t f, int width, int height) {
    scalar_t mass = 0.0;
    Kokkos::parallel_reduce("total_mass", Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {0, 0, 0}, {width, height, v_dim}),
        KOKKOS_LAMBDA(const int x, const int y, const int i, scalar_t& local_mass) {
            local_mass += f(x, y, i);
        }, mass);
    return mass;
}

scalar_t checksum(field3_t f, int width, int height)
{
    scalar_t result = 0.0;

    Kokkos::parallel_reduce(
        "checksum",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {1, 0, 0},
            {width - 1, height, v_dim}),
        KOKKOS_LAMBDA(
            const int x,
            const int y,
            const int i,
            scalar_t& sum) {

            sum += f(x, y, i)
                 * (1.0 + 0.001 * x
                        + 0.00001 * y
                        + 0.0000001 * i);
        },
        result);

    return result;
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
                                                     density(x, y) *
                                                     wall_velocity / cs2;
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
void share_ghost_cells(field3_t f,
                       int width,
                       int height,
                       int rank,
                       int size,
                       GhostBuffers& buffers)
{
    const int left =
        (rank > 0) ? rank - 1 : MPI_PROC_NULL;

    const int right =
        (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    const int buff_size = height * 3;

    auto send_device = buffers.send_device;
    auto recv_device = buffers.recv_device;

    // 1. Send to LEFT, receive from RIGHT
    Kokkos::parallel_for(
        "pack_left_ghost",
        Kokkos::RangePolicy<>(0, buff_size),
        KOKKOS_LAMBDA(const int idx) {
            const int y = idx / 3;
            const int d = idx % 3;

            send_device(idx) =
                f(0, y, left_dir(d));
        });

    // Device -> Host
    Kokkos::deep_copy(
        buffers.send_host,
        buffers.send_device);

    MPI_Sendrecv(
        buffers.send_host.data(),
        buff_size,
        MPI_DOUBLE,
        left,
        0,

        buffers.recv_host.data(),
        buff_size,
        MPI_DOUBLE,
        right,
        0,

        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);

    // Host -> Device
    Kokkos::deep_copy(
        buffers.recv_device,
        buffers.recv_host);

    Kokkos::parallel_for(
        "unpack_right_ghost",
        Kokkos::RangePolicy<>(0, buff_size),
        KOKKOS_LAMBDA(const int idx) {
            const int y = idx / 3;
            const int d = idx % 3;

            f(width - 1, y, left_dir(d)) =
                recv_device(idx);
        });

    // 2. Send to RIGHT, receive from LEFT
    Kokkos::parallel_for(
        "pack_right_ghost",
        Kokkos::RangePolicy<>(0, buff_size),
        KOKKOS_LAMBDA(const int idx) {
            const int y = idx / 3;
            const int d = idx % 3;

            send_device(idx) =
                f(width - 1, y, right_dir(d));
        });

    Kokkos::deep_copy(
        buffers.send_host,
        buffers.send_device);

    MPI_Sendrecv(
        buffers.send_host.data(),
        buff_size,
        MPI_DOUBLE,
        right,
        1,

        buffers.recv_host.data(),
        buff_size,
        MPI_DOUBLE,
        left,
        1,

        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);

    Kokkos::deep_copy(
        buffers.recv_device,
        buffers.recv_host);

    Kokkos::parallel_for(
        "unpack_left_ghost",
        Kokkos::RangePolicy<>(0, buff_size),
        KOKKOS_LAMBDA(const int idx) {
            const int y = idx / 3;
            const int d = idx % 3;

            f(0, y, right_dir(d)) =
                recv_device(idx);
        });

    add_neighbor_input(
        f, width, height, rank, size);
}

void calc_total_mass(double &total_mass, field2_t density, int width,
                     int height) {
    double local_mass = 0.0L;

    Kokkos::parallel_reduce(
        "sum_local_mass",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {width - 1, height}),
        KOKKOS_LAMBDA(const int x, const int y, double &lsum) {
            lsum += density(x, y);
        },
        local_mass);
    Kokkos::fence();

    MPI_Reduce(&local_mass, &total_mass, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
}

void calc_total_kin_energy(double &total_kin_energy, field3_t velocity,
                           field2_t density, int width, int height) {
    double local_energy = 0.0;

    Kokkos::parallel_reduce(
        "sum_local_kin_energy",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {width - 1, height}),
        KOKKOS_LAMBDA(const int x, const int y, double &lsum) {
            lsum += 0.5 * density(x, y) *
                    (square(velocity(x, y, 0)) + square(velocity(x, y, 1)));
        },
        local_energy);
    Kokkos::fence();

    MPI_Reduce(&local_energy, &total_kin_energy, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
}

void execute_time_step(field3_t f, field3_t post_f, field2_t density,
                       field3_t velocity, int width, int height, scalar_t omega,
                       int rank, int size, GhostBuffers &buffers) {
    if (size > 1) {
        share_ghost_cells(f, width, height, rank, size, buffers);
    }
    calc_collision(f, post_f, density, velocity, width, height, omega);
    streaming(f, post_f, density, width, height, rank, size);
}