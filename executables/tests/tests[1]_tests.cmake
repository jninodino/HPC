add_test( HelloTest.BasicAssertions [==[/mnt/c/Users/Janine/Desktop/Semester 1/HPC/HPC/executables/tests/tests]==] [==[--gtest_filter=HelloTest.BasicAssertions]==] --gtest_also_run_disabled_tests)
set_tests_properties( HelloTest.BasicAssertions PROPERTIES WORKING_DIRECTORY [==[/mnt/c/Users/Janine/Desktop/Semester 1/HPC/HPC/executables/tests]==] SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set( tests_TESTS HelloTest.BasicAssertions)
