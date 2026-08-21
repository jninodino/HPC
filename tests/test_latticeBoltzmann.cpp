#include "latticeBoltzmann.h"
#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>
#include <Kokkos_Core.hpp>

TEST(LATTICEBOLTZMANN, propagation_all_directions) {
	// A signle interior cell (3, 3) has all 9 distinct distribution values.
	// After one streaming step each value mus land at (3+cx[i], 3+cy[i], i).
	const int width = 7;
	const int height = 7;
	field3_t f("f", width, height, v_dim);
	field3_t post_f("post_f", width, height, v_dim);
	field2_t density("density", width, height);

	for (int i=0; i<v_dim; i++) {
		post_f(3, 3, i) = static_cast<scalar_t>(i + 1); 
	}
	
	streaming(f, post_f, density, width, height, 0, 1);

    EXPECT_DOUBLE_EQ(f(3, 3, 0), 1.0);  // rest       -> (3,3)
    EXPECT_DOUBLE_EQ(f(4, 3, 1), 2.0);  // right      -> (4,3)
    EXPECT_DOUBLE_EQ(f(3, 4, 2), 3.0);  // up         -> (3,4)
    EXPECT_DOUBLE_EQ(f(2, 3, 3), 4.0);  // left       -> (2,3)
    EXPECT_DOUBLE_EQ(f(3, 2, 4), 5.0);  // down       -> (3,2)
    EXPECT_DOUBLE_EQ(f(4, 4, 5), 6.0);  // right-up   -> (4,4)
    EXPECT_DOUBLE_EQ(f(2, 4, 6), 7.0);  // left-up    -> (2,4)
    EXPECT_DOUBLE_EQ(f(2, 2, 7), 8.0);  // left-down  -> (2,2)
    EXPECT_DOUBLE_EQ(f(4, 2, 8), 9.0);  // right-down -> (4,2)
}

TEST(LATTICEBOLTZMANN, calc_density) {
    const int width = 3;
    const int height = 3;

    field2_t density("density", width, height);
    field3_t f("f", width, height, v_dim);

    f(1, 1, 1) = 1.0;
    f(1, 1, 2) = 1.0;
    f(1, 1, 3) = 1.0;

    calc_density(f, density, width, height);
    Kokkos::fence();

    EXPECT_DOUBLE_EQ(density(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(density(1, 1), 3.0);
}

TEST(LATTICEBOLTZMANN, total_mass) {
	const int width = 3;
	const int height = 3;
	field3_t f("f", width, height, v_dim);

	f(0, 0, 1) = 1.0;
	f(2, 0, 4) = 1.0;
	f(0, 1, 2) = 1.0;

	EXPECT_DOUBLE_EQ(total_mass(f, width, height), 3.0L);
}

// Test of Milestone 3: Mass conversation, the total mass must be conserved
// at every time step
TEST(LATTICEBOLTZMANN, mass_converation) {
	const int width = 9;
	const int height = 9;
	field2_t density("density", width, height);
	field3_t velocity("velocity", width, height, 2);
	field3_t f("f", width, height, v_dim);
	field3_t post_f("post_f", width, height, v_dim);
	scalar_t omega = 1.7L;

	f(4, 4, 1) = 1.0L;

	double mass_prev = total_mass(f, width, height);
	// Execute 3 steps mass need to be constant
	for (int step=0; step<3; step++) {
		calc_collision(f, post_f, density, velocity, width, height, omega);
		streaming(f, post_f, density, width, height, 1, 1);
		EXPECT_DOUBLE_EQ(mass_prev, total_mass(f, width, height));
	}
}

// Test of Milestone 3: Momentum converation, without external forcing, the
// collision step must not change the local momentum.
TEST(LATTICEBOLZMANN, momentum_conservation) {
	const int width = 9;
	const int height = 9;
	field2_t density("density", width, height);
	field3_t velocity("velocity", width, height, 2);
	field3_t f("f", width, height, v_dim);
	field3_t post_f("post_f", width, height, v_dim);
	scalar_t omega = 1.7L;

	int x = 4;
	int y = 4;
	f(x, y, 1) = 1.0L;

	// calc_density(f, density, width, height);
	// calc_velocity(f, density, velocity, width, height);
	calc_macroscopic(f, density, velocity, width, height);
	Kokkos::fence();

	const double momentum_x_before =
		density(x, y) * velocity(x, y, 0);
	const double momentum_y_before =
		density(x, y) * velocity(x, y, 1);

	calc_collision(
		f,
		post_f,
		density,
		velocity,
		width,
		height,
		omega
	);

	// calc_density(post_f, density, width, height);
	// calc_velocity(post_f, density, velocity, width, height);
	calc_macroscopic(post_f, density, velocity, width, height);
	Kokkos::fence();

	const double momentum_x_after =
		density(x, y) * velocity(x, y, 0);
	const double momentum_y_after =
		density(x, y) * velocity(x, y, 1);

	EXPECT_DOUBLE_EQ(momentum_x_before, momentum_x_after);
	EXPECT_DOUBLE_EQ(momentum_y_before, momentum_y_after);
}

// Test of Milestone 3: Equilibrium is a fixed point, if f = f_eq nothing
// must change
TEST(LATTICEBOLTZMANN, equilibrium_is_a_fixed_point)
{
    const int width = 9;
    const int height = 9;

    field2_t density("density", width, height);
    field3_t velocity("velocity", width, height, 2);
    field3_t f("f", width, height, v_dim);
    field3_t post_f("post_f", width, height, v_dim);

    const scalar_t omega = 1.7;

    // Uniform equilibrium state
    Kokkos::parallel_for(
        "initialize_equilibrium",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0}, {width, height}),
        KOKKOS_LAMBDA(const int x, const int y) {
            density(x, y) = 1.0;
            velocity(x, y, 0) = 0.0;
            velocity(x, y, 1) = 0.0;

            for (int i = 0; i < v_dim; ++i) {
                f(x, y, i) =
                    calc_f_eq(density, velocity, x, y, i);
            }
        });

    Kokkos::fence();

    calc_collision(
        f, post_f,
        density, velocity,
        width, height,
        omega);

    Kokkos::fence();

    auto h_f =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, f);

    auto h_post_f =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, post_f);

    for (int x = 1; x < width - 1; ++x) {
        for (int y = 0; y < height; ++y) {
            for (int i = 0; i < v_dim; ++i) {
                EXPECT_NEAR(
                    h_post_f(x, y, i),
                    h_f(x, y, i),
                    1e-14);
            }
        }
    }
}

TEST(LATTICEBOLTZMANN, calc_f_eq) {
	int width = 1;
	int height = 1;
	
	field2_t density("density", width, height);
	field3_t velocity("velocity", width, height, 2);

	density(0, 0) = 2.0;
	velocity(0, 0, 0) = 1.0;
	velocity(0, 0, 1) = 1.0;
	
	scalar_t expected_res[v_dim] = {
		-16.0L/9.0L, 11.0L/9.0L, 11.0L/9.0L, -1.0L/9.0L, -1.0L/9.0L, 11.0L/9.0L,
		-1.0L/9.0L, 5.0L/9.0L, -1.0L/9.0L};

	for (int i = 0; i < v_dim; i++) {
		EXPECT_DOUBLE_EQ(calc_f_eq(density, velocity, 0, 0, i), 
		expected_res[i]);
	}
}

TEST(LATTICEBOLTZMANN, calc_collision) {
	int width = 5;
	int height = 3;

	field2_t density("density", width, height);
	field3_t velocity("velocity", width, height, 2);
	field3_t f("f", width, height, v_dim);
	field3_t post_f("post_f", width, height, v_dim);
	scalar_t omega = 1.5L;
	bool relaxation = true;

	f(2, 1, 1) = 1.0;

	scalar_t expected_res[v_dim] = {
		-1.0L / 3.0L, 2.0L / 3.0L, -1.0L / 12.0L, 1.0L / 6.0L, -1.0L / 12.0L,
		7.0L / 24.0L, 1.0L / 24.0L, 1.0L / 24.0L, 7.0L / 24.0L
	};

	calc_collision(f, post_f, density, velocity, width, height, omega, 
		relaxation);
	for (int i = 0; i < v_dim; i++) {
		EXPECT_DOUBLE_EQ(post_f(1, 0, i), 0.0L);
		EXPECT_DOUBLE_EQ(post_f(1, 1, i), 0.0L);
		EXPECT_DOUBLE_EQ(post_f(1, 2, i), 0.0L);
		EXPECT_DOUBLE_EQ(post_f(2, 0, i), 0.0L);
		EXPECT_DOUBLE_EQ(post_f(2, 1, i), expected_res[i]);
		EXPECT_DOUBLE_EQ(post_f(2, 2, i), 0.0L);
		EXPECT_DOUBLE_EQ(post_f(3, 0, i), 0.0L);
		EXPECT_DOUBLE_EQ(post_f(3, 1, i), 0.0L);
		EXPECT_DOUBLE_EQ(post_f(3, 2, i), 0.0L);
	}
}

TEST(LATTICEBOLTZMANN, collision_streaming) {
	int width = 5;
	int height = 3;

	field2_t density("density", width, height);
	field3_t velocity("velocity", width, height, 2);
	field3_t f("f", width, height, v_dim);
	field3_t post_f("post_f", width, height, v_dim);
	scalar_t omega = 1.5L;
	bool relaxation = true;

	f(2, 1, 1) = 1.0;

	scalar_t expected_res[v_dim] = {
		-1.0L / 3.0L, 2.0L / 3.0L, -1.0L / 12.0L, 1.0L / 6.0L, -1.0L / 12.0L,
		7.0L / 24.0L, 1.0L / 24.0L, 1.0L / 24.0L, 7.0L / 24.0L
	};

	calc_collision(f, post_f, density, velocity, width, height, omega, 
		relaxation);
	streaming(f, post_f, density, width, height, 1, 1);
	
	EXPECT_DOUBLE_EQ(f(1, 0, 7), expected_res[7]);
	EXPECT_DOUBLE_EQ(f(1, 1, 3), expected_res[3]);
	EXPECT_DOUBLE_EQ(f(1, 2, 6), expected_res[6]);
	EXPECT_DOUBLE_EQ(f(2, 0, 4), expected_res[4]);
	EXPECT_DOUBLE_EQ(f(2, 1, 0), expected_res[0]);
	EXPECT_DOUBLE_EQ(f(2, 2, 2), expected_res[2]);
	EXPECT_DOUBLE_EQ(f(3, 0, 8), expected_res[8]);
	EXPECT_DOUBLE_EQ(f(3, 1, 1), expected_res[1]);
	EXPECT_DOUBLE_EQ(f(3, 2, 5), expected_res[5]);
}

TEST(LATTICEBOLTZMANN, bounce_back_right_wall)
{
    const int width = 5;
    const int height = 5;

    field3_t f("f", width, height, v_dim);
    field3_t post_f("post_f", width, height, v_dim);
    field2_t density("density", width, height);

    // rank 1 of 2 => right side is the physical x-boundary
    const int rank = 1;
    const int size = 2;

    auto h_post_f = Kokkos::create_mirror_view(post_f);

    const int x = width - 2;

    h_post_f(x, 2, 1) = 1.0; // right
    h_post_f(x, 2, 5) = 2.0; // right-up
    h_post_f(x, 2, 8) = 3.0; // right-down

    Kokkos::deep_copy(post_f, h_post_f);

    streaming(
        f, post_f, density,
        width, height,
        rank, size);

    auto h_f =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, f);

    // 1 -> 3
    EXPECT_DOUBLE_EQ(h_f(x, 2, 3), 1.0);

    // 5 -> 7
    EXPECT_DOUBLE_EQ(h_f(x, 2, 7), 2.0);

    // 8 -> 6
    EXPECT_DOUBLE_EQ(h_f(x, 2, 6), 3.0);

    // Nothing may leave through right ghost column
    EXPECT_DOUBLE_EQ(h_f(width - 1, 2, 1), 0.0);
    EXPECT_DOUBLE_EQ(h_f(width - 1, 3, 5), 0.0);
    EXPECT_DOUBLE_EQ(h_f(width - 1, 1, 8), 0.0);
}

TEST(LATTICEBOLTZMANN, bounce_back_left_wall)
{
    const int width = 5;
    const int height = 5;

    field3_t f("f", width, height, v_dim);
    field3_t post_f("post_f", width, height, v_dim);
    field2_t density("density", width, height);

    // rank 0 of 2 => only left side is a physical x-boundary
    const int rank = 0;
    const int size = 2;

    auto h_post_f = Kokkos::create_mirror_view(post_f);

    // Interior y-coordinate, so no top/bottom interference
    h_post_f(1, 2, 3) = 1.0; // left
    h_post_f(1, 2, 6) = 2.0; // left-up
    h_post_f(1, 2, 7) = 3.0; // left-down

    Kokkos::deep_copy(post_f, h_post_f);

    streaming(
        f, post_f, density,
        width, height,
        rank, size);

    auto h_f =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, f);

    // 3 -> 1
    EXPECT_DOUBLE_EQ(h_f(1, 2, 1), 1.0);

    // 6 -> 8
    EXPECT_DOUBLE_EQ(h_f(1, 2, 8), 2.0);

    // 7 -> 5
    EXPECT_DOUBLE_EQ(h_f(1, 2, 5), 3.0);

    // Nothing may leave through the left ghost column
    EXPECT_DOUBLE_EQ(h_f(0, 2, 3), 0.0);
    EXPECT_DOUBLE_EQ(h_f(0, 3, 6), 0.0);
    EXPECT_DOUBLE_EQ(h_f(0, 1, 7), 0.0);
}

TEST(LATTICEBOLTZMANN, moving_wall_top)
{
    const int width = 5;
    const int height = 5;

    field3_t f("f", width, height, v_dim);
    field3_t post_f("post_f", width, height, v_dim);
    field2_t density("density", width, height);

    const int x = 2;
    const int y = height - 1;

    auto h_post_f = Kokkos::create_mirror_view(post_f);
    auto h_density = Kokkos::create_mirror_view(density);

    h_post_f(x, y, 2) = 1.0;  // straight up
    h_post_f(x, y, 5) = 1.0;  // right-up
    h_post_f(x, y, 6) = 1.0;  // left-up

    h_density(x, y) = 1.0;

    Kokkos::deep_copy(post_f, h_post_f);
    Kokkos::deep_copy(density, h_density);

    streaming(
        f, post_f, density,
        width, height,
        0, 1);

    auto h_f =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, f);

    constexpr scalar_t wall_velocity = 0.1;
    constexpr scalar_t cs2 = 1.0 / 3.0;

    const scalar_t correction =
        2.0 * weight(5) * wall_velocity / cs2;

    // i=2: cx(2)=0 -> no moving-wall correction
    EXPECT_DOUBLE_EQ(
        h_f(x, y, opposite(2)),
        1.0);

    // i=5: cx(5)=+1
    EXPECT_NEAR(
        h_f(x, y, opposite(5)),
        1.0 - correction,
        1e-14);

    // i=6: cx(6)=-1 -> correction changes sign
    EXPECT_NEAR(
        h_f(x, y, opposite(6)),
        1.0 + correction,
        1e-14);
}


//TEST(LATTICEBOLTZMANN, mass_conservation_interior) {
    // For an interior particle (not near walls) mass must be conserved in one
    // streaming step. We seed every direction at a central cell and verify
    // the total sum is unchanged across the whole domain.
//    const int width = 7, height = 7;
//    Kokkos::View<scalar_t***> f("f", width, height, v_dim);
//    Kokkos::View<scalar_t***> post_f("post_f", width, height, v_dim);

    // Equilibrium weights at unit density at cell (3, 3) — well away from all walls
//    for (int i = 0; i < v_dim; i++)
//        post_f(3, 3, i) = W[i];  // sum = 1 (unit density)

//    streaming(f, post_f, width, height, 0, 1);

//    scalar_t mass = 0.0;
//    for (int x = 0; x < width; x++)
//        for (int y = 0; y < height; y++)
//            for (int i = 0; i < v_dim; i++)
//                mass += f(x, y, i);

//    EXPECT_NEAR(mass, 1.0, 1e-14);
//}
