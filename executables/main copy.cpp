#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <vector>

using View3D = Kokkos::View<double ***>;
using View2D = Kokkos::View<double **>;

const int nx = 512;
const int ny = 128;
const int q = 9;

const int cx[q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int cy[q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

const double w[q] = {4.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0, 1.0 / 9.0,
                     1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

const int opposite[q] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

void init_cavity(View3D f, int local_nx) {
    Kokkos::deep_copy(f, 0.0);

    auto h_f = Kokkos::create_mirror_view(f);

    for (int x = 1; x <= local_nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            double rho = 1.0;
            // shear wave double ux = epsilon * std::sin(2.0 * M_PI * y / ny);
            // cavity
            double ux = 0.0;
            double uy = 0.0;

            // if (x >= nx / 2 - 1 && x <= nx / 2 + 1 && y >= ny / 2 - 1 &&
            //     y <= ny / 2 + 1) {
            //     rho = 0.7;
            // }

            double u2 = ux * ux + uy * uy;

            for (int j = 0; j < q; ++j) {
                double cu = cx[j] * ux + cy[j] * uy;
                h_f(x, y, j) =
                    w[j] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
            }
        }
    }

    Kokkos::deep_copy(f, h_f);
}
void calc_density(View3D f, View2D rho, int local_nx) {
    Kokkos::parallel_for(
        "calc_density",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {local_nx + 1, ny}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double sum = 0.0;

            for (int j = 0; j < q; j++) {
                sum += f(x, y, j);
            }
            rho(x, y) = sum;
        });
}

double total_mass(View3D f) {
    auto h_f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);

    double mass = 0.0;

    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            for (int j = 0; j < q; ++j) {
                mass += h_f(x, y, j);
            }
        }
    }
    return mass;
}

double local_mass(View2D rho, int local_nx) {
    double mass = 0.0;

    Kokkos::parallel_reduce(
        "local_mass",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {local_nx + 1, ny}),
        KOKKOS_LAMBDA(const int x, const int y, double &local_sum) {
            local_sum += rho(x, y);
        },
        mass);

    return mass;
}

double local_kinetic_energy(View2D rho, View2D ux, View2D uy, int local_nx) {
    double ke = 0.0;

    Kokkos::parallel_reduce(
        "local_kinetic_energy",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {local_nx + 1, ny}),
        KOKKOS_LAMBDA(const int x, const int y, double &local_sum) {
            double u2 = ux(x, y) * ux(x, y) + uy(x, y) * uy(x, y);
            local_sum += 0.5 * rho(x, y) * u2;
        },
        ke);

    return ke;
}

double max_velocity_change(View2D ux, View2D uy, View2D ux_old, View2D uy_old,
                           int local_nx) {
    double max_change = 0.0;

    Kokkos::parallel_reduce(
        "max_velocity_change",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {local_nx + 1, ny}),
        KOKKOS_LAMBDA(const int x, const int y, double &local_max) {
            double dux = ux(x, y) - ux_old(x, y);
            double duy = uy(x, y) - uy_old(x, y);
            double change = sqrt(dux * dux + duy * duy);

            if (change > local_max) {
                local_max = change;
            }
        },
        Kokkos::Max<double>(max_change));
    return max_change;
}

void calc_velocity(View3D f, View2D rho, View2D ux, View2D uy, int local_nx) {
    Kokkos::parallel_for(
        "calc_velocity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {local_nx + 1, ny}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double sum_x = 0.0;
            double sum_y = 0.0;

            for (int j = 0; j < q; j++) {
                sum_x += cx[j] * f(x, y, j);
                sum_y += cy[j] * f(x, y, j);
            }

            double r = rho(x, y);

            if (r > 0.0) {
                ux(x, y) = sum_x / r;
                uy(x, y) = sum_y / r;
            } else {
                ux(x, y) = 0.0;
                uy(x, y) = 0.0;
            }
        });
}

void streaming(View3D f, View3D f_new, int local_nx) {
    double u_lid = 0.1;

    Kokkos::parallel_for(
        "streaming",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {local_nx + 1, ny}),
        KOKKOS_LAMBDA(const int x, const int y) {
            for (int j = 0; j < q; j++) {
                int x_from = x - cx[j];
                int y_from = y - cy[j];

                if (y_from < 0) {
                    f_new(x, y, j) = f(x, y, opposite[j]);
                } else if (y_from >= ny) {
                    double rho_wall = 1.0;

                    f_new(x, y, j) = f(x, y, opposite[j]) +
                                     6.0 * w[j] * rho_wall * cx[j] * u_lid;
                } else {
                    f_new(x, y, j) = f(x_from, y_from, j);
                }
            }
        });
}

void write_fields(View2D rho, View2D ux, View2D uy, int step) {
    auto h_rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho);
    auto h_ux = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ux);
    auto h_uy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), uy);

    std::string filename = "output_" + std::to_string(step) + ".csv";
    std::ofstream file(filename);

    file << "x,y,rho,ux,uy\n";

    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            file << x << "," << y << "," << h_rho(x, y) << "," << h_ux(x, y)
                 << "," << h_uy(x, y) << "\n";
        }
    }
}

void collision(View3D f, View2D rho, View2D ux, View2D uy, double omega,
               int local_nx) {
    Kokkos::parallel_for(
        "collision",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {local_nx + 1, ny}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double r = rho(x, y);
            double u_x = ux(x, y);
            double u_y = uy(x, y);
            double u2 = (u_x * u_x) + (u_y * u_y);

            for (int j = 0; j < q; j++) {
                double cu = cx[j] * u_x + cy[j] * u_y;

                double feq =
                    w[j] * r * (1.0 + 3.0 * cu + 4.5 * (cu * cu) - 1.5 * u2);
                f(x, y, j) += omega * (feq - f(x, y, j));
            }
        });
}

double measure_amplitude(View2D ux) {
    auto h_ux = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ux);

    double sin_sum = 0.0;
    double norm = 0.0;

    for (int y = 0; y < ny; ++y) {
        double s = std::sin(2.0 * M_PI * y / ny);
        double ux_avg = 0.0;

        for (int x = 0; x < nx; ++x) {
            ux_avg += h_ux(x, y);
        }
        ux_avg /= nx;

        sin_sum += ux_avg * s;
        norm += s * s;
    }
    return sin_sum / norm;
}

void ghostCells(View3D f, int local_nx, int rank, int size) {
    auto h_f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);

    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    const int count = ny * q;

    std::vector<double> sendLeft(count);
    std::vector<double> sendRight(count);
    std::vector<double> recLeft(count);
    std::vector<double> recRight(count);

    for (int y = 0; y < ny; ++y) {
        for (int j = 0; j < q; ++j) {
            sendLeft[y * q + j] = h_f(1, y, j);
            sendRight[y * q + j] = h_f(local_nx, y, j);
        }
    }
    MPI_Sendrecv(sendLeft.data(), count, MPI_DOUBLE, left, 0, recRight.data(),
                 count, MPI_DOUBLE, right, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    MPI_Sendrecv(sendRight.data(), count, MPI_DOUBLE, right, 1, recLeft.data(),
                 count, MPI_DOUBLE, left, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int y = 0; y < ny; ++y) {
        for (int j = 0; j < q; ++j) {
            h_f(0, y, j) = recLeft[y * q + j];
            h_f(local_nx + 1, y, j) = recRight[y * q + j];
        }
    }
    Kokkos::deep_copy(f, h_f);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Kokkos::initialize(argc, argv);
    {
        int local_nx = nx / size;
        View3D f("f", local_nx + 2, ny, q);
        View3D f_new("f_new", local_nx + 2, ny, q);

        View2D rho("rho", local_nx + 2, ny);
        View2D ux("ux", local_nx + 2, ny);
        View2D uy("uy", local_nx + 2, ny);
        View2D ux_old("ux_old", local_nx + 2, ny);
        View2D uy_old("uy_old", local_nx + 2, ny);

        Kokkos::deep_copy(ux_old, 0.0);
        Kokkos::deep_copy(uy_old, 0.0);

        double omega = 1.7;
        const double steady_threshold = 1e-6;

        if (argc > 1) {
            omega = std::stod(argv[1]);
        }

        init_cavity(f, local_nx);
        Kokkos::fence();

        double change = 0.0;
        double global_mass = 0.0;
        double global_ke = 0.0;
        auto start = std::chrono::high_resolution_clock::now();
        int executed_steps = 0;
        for (int step = 0; step <= 30000; ++step) {

            calc_density(f, rho, local_nx);
            calc_velocity(f, rho, ux, uy, local_nx);

            double local_change =
                max_velocity_change(ux, uy, ux_old, uy_old, local_nx);

            MPI_Allreduce(&local_change, &change, 1, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD);
            if (step > 0 && change < steady_threshold) {
                if (rank == 0)
                    std::cout << "Steady state reached at step " << step
                              << ", max_change = " << change << "\n";
                executed_steps = step + 1;
                break;
            }

            Kokkos::deep_copy(ux_old, ux);
            Kokkos::deep_copy(uy_old, uy);

            if (step % 1000 == 0) {
                double lm = local_mass(rho, local_nx);
                
                MPI_Allreduce(&lm, &global_mass, 1, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                double lke = local_kinetic_energy(rho, ux, uy, local_nx);
             
                MPI_Allreduce(&lke, &global_ke, 1, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
            }

            if (step % 1000 == 0 && rank == 0) {
                std::cout << "step " << step << ", max_change = " << change
                          << ", mass = " << global_mass << ", kinetic_energy = " << global_ke << "\n";
            }

            collision(f, rho, ux, uy, omega, local_nx);

            ghostCells(f, local_nx, rank, size);
            streaming(f, f_new, local_nx);
            std::swap(f, f_new);
            executed_steps = step + 1;
        }
        Kokkos::fence();

        auto end = std::chrono::high_resolution_clock::now();

        double seconds = std::chrono::duration<double>(end - start).count();
        double mlups =
            static_cast<double>(nx) * ny * executed_steps / seconds / 1e6;

        if (rank == 0) {
            std::cout << "Runtime: " << seconds << " s\n";
            std::cout << "Steps: " << executed_steps << "\n";
            std::cout << "Performance: " << mlups << " MLUPS\n";
            // write_fields(rho, ux, uy, executed_steps);
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
}
