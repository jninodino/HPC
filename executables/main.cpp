#include "hello.h"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>

using View2D = Kokkos::View<double **>;
using View3D = Kokkos::View<double ***>;

void initialize(View3D f, int nx, int ny, double w[9]) {
    Kokkos::parallel_for(
        "initialize", nx * ny * 9, KOKKOS_LAMBDA(const int i) {
            int j = i % 9;
            int cell = i / 9;
            int x = cell / ny;
            int y = cell % ny;

            double rho0 = 1.0;

            if (x == nx / 2 && y == ny / 2) {
                rho0 = 1.01;
            }

            f(x, y, j) = w[j] * rho0;
        });
}

void initializeShearWave(View3D f, int nx, int ny, int cx[9], int cy[9],
                         double w[9]) {
    double epsilon = 0.05;

    Kokkos::parallel_for(
        "initialize_shearwave", nx * ny * 9, KOKKOS_LAMBDA(const int i) {
            int j = i % 9;
            int cell = i / 9;

            int x = cell / ny;
            int y = cell % ny;

            double rho = 1.0;

            double ux = epsilon * sin(2.0 * M_PI * y / ny);
            double uy = 0;

            double cu = cx[j] * ux + cy[j] * uy;
            double u2 = ux * ux + uy * uy;

            double feq = w[j] * rho * (1 + 3 * cu + 4.5 * (cu * cu) - 1.5 * u2);

            f(x, y, j) = feq;
        });
}

void calc_density(View3D f, View2D rho, int nx, int ny) {
    // p(x,y) = integral (d 9 * f(x,y,9))
    Kokkos::parallel_for(
        "density", nx * ny, KOKKOS_LAMBDA(const int i) {
            int x = i / ny;
            int y = i % ny;

            double sum = 0.0;
            for (int j = 0; j < 9; j++) {
                sum += f(x, y, j);
            }
            rho(x, y) = sum;
        });
}

void calc_velocity(View3D f, View2D rho, View2D ux, View2D uy, int nx, int ny,
                   int cx[9], int cy[9], double u_lid) {
    Kokkos::parallel_for(
        "velocity", nx * ny, KOKKOS_LAMBDA(const int i) {
            int x = i / ny;
            int y = i % ny;

            double sum_x = 0.0;
            double sum_y = 0.0;

            for (int j = 0; j < 9; j++) {
                sum_x += f(x, y, j) * cx[j];
                sum_y += f(x, y, j) * cy[j];
            }

            double r = rho(x, y);
            if (r <= 1e-12 || !Kokkos::isfinite(r)) {
                ux(x, y) = 0.0;
                uy(x, y) = 0.0;
            } else {
                ux(x, y) = sum_x / r;
                uy(x, y) = sum_y / r;
            }

            if (y == 0 || x == 0 || x == nx - 1) {
                ux(x, y) = 0.0;
                uy(x, y) = 0.0;
            }

            if (y == ny - 1) {
                ux(x, y) = u_lid;
                uy(x, y) = 0.0;
            }
        });
}

double measure_amplitude(View2D ux, int nx, int ny) {
    auto ux_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ux);

    double max_amp = 0.0;

    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            max_amp = std::max(max_amp, std::abs(ux_host(x, y)));
        }
    }
    return max_amp;
}

void collision(View3D f, View2D rho, View2D ux, View2D uy, double omega, int nx,
               int ny, int cx[9], int cy[9], double w[9]) {
    Kokkos::parallel_for(
        "collision", nx * ny * 9, KOKKOS_LAMBDA(const int i) {
            int j = i % 9;
            int cell = i / 9;

            int x = cell / ny;
            int y = cell % ny;

            double cu = cx[j] * ux(x, y) + cy[j] * uy(x, y);
            double u2 = (ux(x, y) * ux(x, y)) + (uy(x, y) * uy(x, y));

            double feq =
                w[j] * rho(x, y) * (1.0 + 3 * cu + 4.5 * (cu * cu) - 1.5 * u2);

            f(x, y, j) = f(x, y, j) - omega * (f(x, y, j) - feq);
        });
}

void streaming(View3D f, View3D f_new, int nx, int ny, int cx[9], int cy[9]) {
    Kokkos::deep_copy(f_new, 0.0);

    Kokkos::parallel_for(
        "streaming", nx * ny * 9, KOKKOS_LAMBDA(const int i) {
            int j = i % 9;
            int cell = i / 9;

            int x = cell / ny;
            int y = cell % ny;

            int x_new = (x + cx[j] + nx) % nx;
            int y_new = (y + cy[j] + ny) % ny;

            f_new(x_new, y_new, j) = f(x, y, j);
        });
}

void applyBoundaries(View3D f, View3D f_new, int nx, int ny, double u_lid) {
    Kokkos::parallel_for(
        "boundaries", nx * ny, KOKKOS_LAMBDA(const int i) {
            int x = i / ny;
            int y = i % ny;

            double rho = 1.0;

            // bottom wall
            if (y == 0) {
                f_new(x, y, 2) = f(x, y, 4);
                f_new(x, y, 5) = f(x, y, 7);
                f_new(x, y, 6) = f(x, y, 8);
            }

            // moving top wall
            else if (y == ny - 1) {
                f_new(x, y, 4) = f(x, y, 2);
                f_new(x, y, 7) = f(x, y, 5) - (1.0 / 6.0) * rho * u_lid;
                f_new(x, y, 8) = f(x, y, 6) + (1.0 / 6.0) * rho * u_lid;
            }

            // left wall
            else if (x == 0) {
                f_new(x, y, 1) = f(x, y, 3);
                f_new(x, y, 5) = f(x, y, 7);
                f_new(x, y, 8) = f(x, y, 6);
            }

            // right wall
            else if (x == nx - 1) {
                f_new(x, y, 3) = f(x, y, 1);
                f_new(x, y, 7) = f(x, y, 5);
                f_new(x, y, 6) = f(x, y, 8);
            }
        });
}

double compute_max_velocity(View2D ux, View2D uy, View2D ux_old, View2D uy_old,
                            int nx, int ny) {
    auto ux_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ux);
    auto uy_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), uy);
    auto ux_old_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ux_old);
    auto uy_old_h =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), uy_old);

    double max_change = 0.0;

    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            double dux = ux_h(x, y) - ux_old_h(x, y);
            double duy = uy_h(x, y) - uy_old_h(x, y);

            double change = std::sqrt(dux * dux + duy * duy);
            max_change = std::max(max_change, change);
        }
    }
    return max_change;
}

void write_vtk_velocity(const std::string &filename, View2D ux, View2D uy,
                        int nx, int ny) {
    auto ux_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ux);
    auto uy_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), uy);

    std::ofstream file(filename);

    file << "# vtk DataFile Version 3.0\n";
    file << "LBM velocity field\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << nx << " " << ny << " 1\n";
    file << "ORIGIN 0 0 0\n";
    file << "SPACING 1 1 1\n";
    file << "POINT_DATA " << nx * ny << "\n";

    file << "VECTORS velocity double\n";

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            file << ux_h(x, y) << " " << uy_h(x, y) << " " << 0.0 << "\n";
        }
    }

    file.close();
}

/**
 * 1. streaming: f -> f_new
 * 2. copy: f_new -> f
 * 3. density: f -> rho
 * 4. velocity: f, rho -> ux, uy
 * 5. collision: f, rho, ux, uy -> f
 * 6. repeat
 */

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    {
        int nx = 128;
        int ny = 128;

        int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
        int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

        double w[9] = {4.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0, 1.0 / 9.0,
                       1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

        double omega = 1.7;

        View3D f("f", nx, ny, 9);
        View3D f_new("f_new", nx, ny, 9);

        View2D rho("rho", nx, ny);

        View2D ux("ux", nx, ny);
        View2D uy("uy", nx, ny);

        View2D ux_old("ux_old", nx, ny);
        View2D uy_old("uy_old", nx, ny);

        double u_lid = 0.1;

        initialize(f, nx, ny, w);

        double threshold = 1e-6;
        double max_change = 1.0;
        int step = 0;
        int max_steps = 100000;

        while (max_change > threshold && step < max_steps) {
            Kokkos::deep_copy(ux_old, ux);
            Kokkos::deep_copy(uy_old, uy);

            streaming(f, f_new, nx, ny, cx, cy);
            applyBoundaries(f, f_new, nx, ny, u_lid);
            Kokkos::deep_copy(f, f_new);


            calc_density(f, rho, nx, ny);
            calc_velocity(f, rho, ux, uy, nx, ny, cx, cy, u_lid);
            // double amp = measure_amplitude(ux, nx, ny);
            // std::cout << step << " " << amp << " " << std::log(amp)
            //           << std::endl;
            max_change = compute_max_velocity(ux, uy, ux_old, uy_old, nx, ny);
            if (!std::isfinite(max_change)) {
                std::cout << "NaN detected at step: " << step << std::endl;
                break;
            }
            collision(f, rho, ux, uy, omega, nx, ny, cx, cy, w);

            step++;
            if (step % 1000 == 0) {
                std::cout << "step: " << step << " max_change: " << max_change
                          << std::endl;
            }
        }

        std::cout << "finished at step: " << step
                  << " with max_change: " << max_change << std::endl;
        write_vtk_velocity("velocity.vtk", ux, uy, nx, ny);
    }
    Kokkos::finalize();
}
