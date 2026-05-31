#include <iostream>
#include <filesystem>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <string>
#include <fstream>

int C[2][9] = {
	{0, 1, 0, -1, 0, 1, -1, -1, 1},
	{0, 0, 1, 0, -1, 1, 1, -1, -1}};

int VELO_DIM = 9;

void calc_density(Kokkos::View<int***> pdf,
				  Kokkos::View<int**> density,
				  int x,
				  int y){
	int sum = 0;
	for (int i=0; i<VELO_DIM; i++) {
		sum += pdf(x, y, i);
	}
	density(x, y) = sum;
}

void calc_velocity(Kokkos::View<int***> pdf,
				   Kokkos::View<int**> density,
				   Kokkos::View<int***> velocity,
				   int x,
				   int y) {
	calc_density(pdf, density, x, y);
	int d = density(x, y);
	if (d == 0) {
		velocity(x, y, 0) = 0;
		velocity(x, y, 1) = 0;
		return;
	}
	int vx = 0;
	int vy = 0;
	for (int i=0; i<VELO_DIM; i++) {
		vx += pdf(x, y, i) * C[0][i];
		vy += pdf(x, y, i) * C[1][i];
	}
	velocity(x, y, 0) = vx / d;
	velocity(x, y, 1) = vy / d;
}

void streaming(Kokkos::View<int***> pdf,
			   Kokkos::View<int**> density,
			   Kokkos::View<int***> velocity,
			   int width,
			   int height){
	auto pdf_old = Kokkos::View<int***>("pdf_old", width, height, VELO_DIM);
	Kokkos::deep_copy(pdf_old, pdf);   // copy values from pdf into pdf_old

	Kokkos::parallel_for("update_pdf", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {width, height, VELO_DIM}),
		KOKKOS_LAMBDA(const int x, const int y, const int i) {
			int new_x;
			int new_y;
			new_x = (width + x + C[0][i]) % width;
			new_y = (height + y + C[1][i]) % height;
			pdf(new_x, new_y, i) = pdf_old(x, y, i);
		}
	);
}			   
