#include <iostream>
#include <filesystem>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <string>
#include <fstream>

int C[2][9] = {
	{0, 1, 0, -1, 0, 1, -1, -1, 1},
	{0, 0, 1, 0, -1, 1, 1, -1, -1}};


float D4_9 = 4.0f / 9.0f;
float D1_9 = 1.0f / 9.0f;
float D1_36 = 1.0f / 36.0f;

float W[9] = {D4_9, D1_9, D1_9, D1_9, D1_9, D1_36, D1_36, D1_36, D1_36};


int VELO_DIM = 9;

void calc_density(Kokkos::View<float***> pdf,
				  Kokkos::View<float**> density,
				  int x,
				  int y){
	float sum = 0;
	for (int i=0; i<VELO_DIM; i++) {
		sum += pdf(x, y, i);
	}
	density(x, y) = sum;
}

void calc_velocity(Kokkos::View<float***> pdf,
				   Kokkos::View<float**> density,
				   Kokkos::View<float***> velocity,
				   int x,
				   int y) {
	calc_density(pdf, density, x, y);
	float d = density(x, y);
	if (d == 0) {
		velocity(x, y, 0) = 0;
		velocity(x, y, 1) = 0;
		return;
	}
	float vx = 0;
	float vy = 0;
	for (int i=0; i<VELO_DIM; i++) {
		vx += pdf(x, y, i) * C[0][i];
		vy += pdf(x, y, i) * C[1][i];
	}
	velocity(x, y, 0) = vx / d;
	velocity(x, y, 1) = vy / d;
}

float square(float x) {
	return x * x;
}

float calc_pdf_eq(Kokkos::View<float**> density,
				 Kokkos::View<float***> velocity,
				 int x,
				 int y,
				 int i
				) {
	float c_times_u = C[0][i] * velocity(x, y, 0) + C[1][i] * velocity(x, y, 1);
	float v_abs_square = square(velocity(x, y, 0)) + square(velocity(x, y, 1));

	float pfdi_eq = W[i] * density(x, y) * (1.0f + 3.0f * c_times_u + 4.5f * square(c_times_u) - 1.5f * v_abs_square);
	return pfdi_eq;
}

void streaming(Kokkos::View<float***> pdf,
			   Kokkos::View<float**> density,
			   Kokkos::View<float***> velocity,
			   int width,
			   int height,
			   float tau
			){
	auto pdf_old = Kokkos::View<float***>("pdf_old", width, height, VELO_DIM);
	Kokkos::deep_copy(pdf_old, pdf);   // copy values from pdf into pdf_old

	Kokkos::parallel_for("update_pdf", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {width, height, VELO_DIM}),
		KOKKOS_LAMBDA(const int x, const int y, const int i) {
			int new_x;
			int new_y;
			// calculate new x, y (r -> r + c*t)
			new_x = (width + x + C[0][i]) % width;
			new_y = (height + y + C[1][i]) % height;
			
			// Recalculate density and velocity, calc_denstiy will be called in calc_velocity
			calc_velocity(pdf_old, density, velocity, x, y);

			// Calculate pdf_eq
			float pdf_eq = calc_pdf_eq(density, velocity, x, y, i);

			pdf(new_x, new_y, i) = pdf_old(x, y, i) + (1.0f / tau) * (pdf_eq - pdf_old(x, y, i));
		}
	);
}			   
