#include "streaming.cpp"
#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>
#include <Kokkos_Core.hpp>


TEST(CALC_DENSITY, testing) {
	Kokkos::View<float**> density("density", 3, 3);
	Kokkos::View<float***> pdf("pdf", 3, 3, VELO_DIM);
	Kokkos::deep_copy(pdf, 0);
	pdf(1, 1, 0) = 1;

	for (int x=0; x<3; x++) {
		for (int y=0; y<3; y++) {
			calc_density(pdf, density, x, y);
		}
	}

	ASSERT_EQ(density(0, 0), 0);
	ASSERT_EQ(density(0, 1), 0);
	ASSERT_EQ(density(1, 0), 0);
	ASSERT_EQ(density(1, 1), 1);
	ASSERT_EQ(density(0, 2), 0);
	ASSERT_EQ(density(1, 2), 0);
	ASSERT_EQ(density(2, 0), 0);
	ASSERT_EQ(density(2, 1), 0);
	ASSERT_EQ(density(2, 2), 0);
}

TEST(CALC_VELOCITY, testing) {
	Kokkos::View<float**> density("density", 3, 3);
	Kokkos::View<float***> pdf("pdf", 3, 3, VELO_DIM);
	Kokkos::View<float***> velocity("velocity", 3, 3, 2);
	Kokkos::deep_copy(pdf, 0);
	pdf(1, 1, 1) = 1;

	for (int x=0; x<3; x++) {
		for (int y=0; y<3; y++) {
			calc_velocity(pdf, density, velocity, x, y);
		}
	}

	ASSERT_EQ(velocity(1, 1, 0), 1);
	ASSERT_EQ(velocity(1, 1, 1), 0);
}

TEST(CALC_PDF_EQ, testing) {
	Kokkos::View<float**> density("density", 3, 3);
	Kokkos::View<float***> velocity("Velocity", 3, 3);
	
	Kokkos::deep_copy(density, 0);
	Kokkos::deep_copy(velocity, 0);
	density(1, 1) = 1.0f;

	float eq = calc_pdf_eq(density, velocity, 1, 1, 0);
	ASSERT_FLOAT_EQ(eq, 4.0f/9.0f);

	velocity(1, 1, 0) = 1.0f;
	eq = calc_pdf_eq(density, velocity, 1, 1, 0);
	ASSERT_FLOAT_EQ(eq, -2.0f/9.0f);
	eq = calc_pdf_eq(density, velocity, 1, 1, 1);
	ASSERT_FLOAT_EQ(eq, 7.0f/9.0f);
}

TEST(STREAMING, testing) {
	Kokkos::View<float**> density("density", 3, 3);
	Kokkos::View<float***> pdf("pdf", 3, 3, VELO_DIM);
	Kokkos::View<float***> velocity("velocity", 3, 3, 2);

	float tau = 1.0f;
	Kokkos::deep_copy(pdf, 0);
	pdf(1, 1, 1) = 1;

	calc_density(pdf, density, 1, 1);
	calc_density(pdf, density, 2, 1);
	ASSERT_EQ(density(1, 1), 1);
	ASSERT_EQ(density(2, 1), 0);
	calc_velocity(pdf, density, velocity, 1, 1);
	ASSERT_EQ(velocity(1, 1, 0), 1);
	ASSERT_EQ(velocity(1, 1, 1), 0);
	float eq = calc_pdf_eq(density, velocity, 1, 1, 1);
	ASSERT_FLOAT_EQ(eq, 7.0f/9.0f);

	streaming(pdf, density, velocity, 3, 3, tau);
	ASSERT_FLOAT_EQ(pdf(2, 1, 1), 7.0f/9.0f);
}

