target_include_directories(axe_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(pybind11)
add_library(axe_core STATIC tensor.cpp)
target_include_directories(axe_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(axe_pybind MODULE pybind.cpp)
target_link_libraries(axe_pybind PRIVATE axe_core pybind11::module)
set_target_properties(axe_pybind PROPERTIES PREFIX "" OUTPUT_NAME "_axe")
