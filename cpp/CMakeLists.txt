include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.10.0
)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(Eigen3)

add_library(axe_core STATIC tensor.cpp variable.cpp op.cpp autodiff.cpp jit.cpp optimizer.cpp codegen.cpp dynamic_compiler.cpp jit_engine.cpp)
target_include_directories(axe_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(axe_core PUBLIC Eigen3::Eigen)

add_library(axe_pybind MODULE pybind.cpp)
target_link_libraries(axe_pybind PRIVATE axe_core pybind11::module dl)

# This is important: it creates `_axe.so` which is what python/axe.py expects
set_target_properties(axe_pybind PROPERTIES PREFIX "" OUTPUT_NAME "_axe")

# --- Google Test Integration ---
include(GoogleTest)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG    v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_executable(run_cpp_tests tests/main.cpp tests/test_autodiff.cpp tests/test_advanced_optimizer.cpp tests/test_codegen.cpp)
target_link_libraries(run_cpp_tests PRIVATE axe_core GTest::gtest_main dl)

add_test(NAME CppTests COMMAND run_cpp_tests)