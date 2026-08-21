add_executable(main main.cpp)
add_executable(relaxation relaxation.cpp)
target_link_libraries(main PRIVATE lib)
target_link_libraries(relaxation PRIVATE lib)