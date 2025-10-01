include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.10.0
)

FetchContent_MakeAvailable(pybind11)

add_library(axe_core STATIC tensor.cpp)
target_include_directories(axe_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(axe_pybind MODULE pybind.cpp)
target_link_libraries(axe_pybind PRIVATE axe_core pybind11::module)

# This is important: it creates `_axe.so` which is what python/axe.py expects
set_target_properties(axe_pybind PROPERTIES PREFIX "" OUTPUT_NAME "_axe")