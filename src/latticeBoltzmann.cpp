#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>

#include "latticeBoltzmann.h"


scalar_t D4_9 = 4.0L / 9.0L;
scalar_t D1_9 = 1.0L / 9.0L;
scalar_t D1_36 = 1.0L / 36.0L;

scalar_t W[9] = {D4_9, D1_9, D1_9, D1_9, D1_9, D1_36, D1_36, D1_36, D1_36};

scalar_t c_s = 0.5777L;
scalar_t uw = 0.1;

// Remove?
scalar_t total_mass(field3_t f, int width, int height) {
	scalar_t mass = 0.0;
	for (int w = 0; w < width; w++) {
		for (int h = 0; h < height; h++) {
			for (int i = 0; i < v_dim; i++) {
				mass += f(w, h, i);
			}
		}
	}
	return mass;
}

void calc_density(field3_t f,
				  field2_t density,
				  int x,
				  int y){
	scalar_t sum = 0;
	for (int i=0; i<v_dim; i++) {
		sum += f(x, y, i);
	}
	density(x, y) = sum;
}

void calc_velocity(field3_t f,
				   field2_t density,
				   field3_t velocity,
				   int x,
				   int y) {
	scalar_t d = density(x, y);
	if (d == 0) {
		velocity(x, y, 0) = 0;
		velocity(x, y, 1) = 0;
		return;
	}
	scalar_t vx = 0;
	scalar_t vy = 0;
	for (int i=0; i<v_dim; i++) {
		vx += f(x, y, i) * cx(i);
		vy += f(x, y, i) * cy(i);
	}
	velocity(x, y, 0) = vx / d;
	velocity(x, y, 1) = vy / d;
}

scalar_t square(scalar_t x) {
	return x * x;
}

scalar_t calc_f_eq(field2_t density,
				 field3_t velocity,
				 int x,
				 int y,
				 int i
				) {
	scalar_t c_times_u = cx(i) * velocity(x, y, 0) + cy(i) * velocity(x, y, 1);
	scalar_t v_abs_square = square(velocity(x, y, 0)) + 
		square(velocity(x, y, 1));
	scalar_t fi_eq = W[i] * density(x, y) * (1.0L + 3.0L * c_times_u + 4.5L *
		square(c_times_u) - 1.5L * v_abs_square);
	return fi_eq;
}

void calc_collision(field3_t f,
			   field3_t post_f,
			   field2_t density,
			   field3_t velocity,
			   int width,
			   int height,
			   scalar_t omega,
			   bool relaxation
			) {
	// calculate density
	for (int x=0; x<width; x++) {
		for (int y=0; y<height; y++) {
			calc_density(f, density, x, y);
		}
	}


	// calculate velocity
	for (int x=0; x<width; x++) {
		for (int y=0; y<height; y++) {
			calc_velocity(f, density, velocity, x, y);
		}
	}

	// calculate next step
	Kokkos::parallel_for("update_f", 
		Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
			{1, 0, 0}, {width - 1, height, v_dim}),
		KOKKOS_LAMBDA(const int x, const int y, const int i) {

			// Calculate f_eq
			scalar_t f_eq = calc_f_eq(density, velocity, x, y, i);
			if (relaxation) {
				post_f(x, y, i) = f(x, y, i) - omega * (f(x, y, i) - f_eq);
			} else {
				post_f(x, y, i) = f(x, y, i);
			}

		}
	);
}

bool is_bounce_back(int x, int y, int i, int width, int height, int rank, int size) {
	// top-left corner: direction 6 is already bounce-backed by the left-wall
	// check below; also bounce-back its diagonal partner, direction 5,
	// instead of letting the moving-wall branch apply a one-sided (and
	// therefore mass-changing) lid correction to it alone
	if (x == 1 && rank == 0 && y == height-1 && i == 5) {
		return true;
	// top-right corner: mirror of the above (direction 5 is already
	// bounce-backed by the right-wall check below; also bounce-back 6)
	} else if (x == width-2 && rank == size-1 && y == height-1 && i == 6) {
		return true;
	// right end of the total map
	} else if ((x == width-2) && (i==1 || i==5 || i==8) && (rank == size - 1)) {
		return true;
	// left end of the total map
	} else if (x == 1  && (i==3 || i==6 || i==7) && (rank == 0)) {
		return true;
	} else if (y == 0 && (i==4 || i==7 || i==8)) {
		return true;
	} else {
		return false;
	}
}

bool is_moving_wall(int x, int y, int i, int width, int height) {
	if (y == height-1 && (i==2 || i==5 || i==6)) {
		return true;
	} else {
		return false;
	}
}

void add_neighbor_input(field3_t f,
				        int width,
						int height,
						int rank,
						int size
					) {


	bool has_left = (rank != 0);
	bool has_right = (rank != size - 1);

	if (has_left) {
		for (int y = 0; y < height; y++) { // TODO use Kokkos
			f(1, y, 1) = f(0, y, 1);
		}
		for (int y = 1; y < height; y++) { // TODO use Kokkos
			f(1, y, 5) = f(0, y, 5);
		}
		for (int y = 0; y < height - 1; y++) { // TODO use Kokkos
			f(1, y, 8) = f(0, y, 8);
		}
	}

	// Mirror of the above for the right boundary (directions 3, 6, 7).
	if (has_right) {
		for (int y = 0; y < height; y++) { // TODO use Kokkos
			f(width-2, y, 3) = f(width-1, y, 3);
		}
		for (int y = 1; y < height; y++) { // TODO use Kokkos
			f(width-2, y, 6) = f(width-1, y, 6);
		}
		for (int y = 0; y < height - 1; y++) { // TODO use Kokkos
			f(width-2, y, 7) = f(width-1, y, 7);
		}
	}
}

bool is_boundery(int x, int y, int i, int width, int height, int rank, int size) {
	// right end of the total map
	if ((x == width-2) && (i==1 || i==5 || i==8) && (rank == size - 1)) {
		return true;
	// left end of the total map
	} else if (x == 1  && (i==3 || i==6 || i==7) && (rank == 0)) {
		return true;
	// bottom end
	} else if (y == 0 && (i==4 || i==7 || i==8)) {
		return true;
	// up end
	} else if (y == height-1 && (i==2 || i==5 || i==6)) {
		return true;
	} else {
		return false;
	}
}

void handle_boundary(field3_t f,
			   field3_t post_f,
			   field2_t density,
			   int x,
			   int y,
			   int i,
			   int width,
			   int height,
			   int rank,
			   int size,
			   bool periodic_bound = false
			){
	if (periodic_bound) {
		int x_next = x;
		if (x == 1) {
			x_next = width - 2;	
		} else if (x == width - 2) {
			x_next = 1;
		}
		int y_next = (height + y + cy(i)) % height;

		f(x_next, y_next, i) = post_f(x, y, i);
	} else if (is_bounce_back(x, y, i, width, height,  rank, size)) {
		f(x, y, opposite_i[i]) = post_f(x, y, i);
	} else if(is_moving_wall(x, y, i, width, height)) {
		f(x, y, opposite_i[i]) = post_f(x, y, i) 
		- 2 * W[i] * cx(i) * density(x, y) * uw / (square(c_s));
	}
}

void streaming(field3_t f, field3_t post_f, field2_t density, int width, 
	int height, int rank, int size){
	Kokkos::parallel_for("update_f", Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
		{1, 0, 0}, {width - 1, height, v_dim}),
		KOKKOS_LAMBDA(const int x, const int y, const int i) {
			
			if (is_boundery(x, y, i, width, height, rank, size)) {
				handle_boundary(f, post_f, density, x, y, i, width, height, rank, size);
			} else {
				int x_next = x + cx(i);
				int y_next = y + cy(i);
				f(x_next, y_next, i) = post_f(x, y, i);
			}
		}
	);
	Kokkos::fence();
}

void share_ghost_cells(field3_t f, int width, int height, int rank, int size) {
		// Neighbor ranks; MPI_PROC_NULL turns the exchange into a no-op
	int left = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
	int right = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

	int buf_size = height * 3;
	std::vector<double> send_buf(buf_size);
	std::vector<double> recv_buf(buf_size);
	// Send buffer to left neighbor and receive from right neighbor
	for (int y=0; y<height; y++) {
		for (int d=0; d<3; d++) {
			send_buf[y * 3 + d] = f(0, y, left_dirs[d]);
		}
	}
	MPI_Sendrecv(send_buf.data(), buf_size, MPI_DOUBLE, left, 0,
				 recv_buf.data(), buf_size, MPI_DOUBLE, right, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	// Unpack into right ghost column
	for (int y=0; y<height; y++) {
		for (int d=0; d<3; d++) {
			f(width-1, y, left_dirs[d]) = recv_buf[y * 3 + d];
		}
	}

	// Send buffer to right neighbor and receive from left neighbor
	for (int y=0; y<height; y++) {
		for (int d=0; d<3; d++) {
			send_buf[y * 3 + d] = f(width-1, y, right_dirs[d]);				}
	}
	MPI_Sendrecv(send_buf.data(), buf_size, MPI_DOUBLE, right, 1,
				 recv_buf.data(), buf_size, MPI_DOUBLE, left, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	// Unpack into right ghost column
	for (int y=0; y<height; y++) { // Use Kokkos
		for (int d=0; d<3; d++) {
			f(0, y, right_dirs[d]) = recv_buf[y * 3 + d];
		}
	}
	add_neighbor_input(f, width, height, rank, size);
}

void calc_total_mass(double &total_mass, field2_t density, int width, 
	int height) {
	double local_mass = 0.0L;
	
	Kokkos::parallel_reduce("sum_local_mass",
		Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {width-1, height}),
		KOKKOS_LAMBDA(const int x, const int y, double& lsum) {
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

    Kokkos::parallel_reduce("sum_local_kin_energy",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {width-2, height}),
        KOKKOS_LAMBDA(const int x, const int y, double& lsum) {
            lsum += 0.5 * density(x, y) * (square(velocity(x, y, 0)) + 
			square(velocity(x, y, 1)));
        },
        local_energy);
    Kokkos::fence();

    MPI_Reduce(&local_energy, &total_kin_energy, 1, MPI_DOUBLE, MPI_SUM, 0, 
		MPI_COMM_WORLD);
}

void execute_time_step(field3_t f, field3_t post_f, field2_t density, 
	field3_t velocity, int width, int height, scalar_t omega, int rank, int size)
	{
		if (size > 1) {
			share_ghost_cells(f, width, height, rank, size);
		}
		calc_collision(f, post_f, density, velocity, width, 
			height, omega);
		streaming(f, post_f, density, width, height, rank, size);
	}