#include <Kokkos_Core.hpp>



int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    int nx = 15;
    int ny = 10;

    int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

    Kokkos::View<double***> f("f", nx, ny, 9);
    Kokkos::View<double***> f_new("f_new", nx, ny, 9);
    
    Kokkos::finalize();
}

