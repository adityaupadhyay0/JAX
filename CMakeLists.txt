cmake_minimum_required(VERSION 3.18)
project(axe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_testing()
include(CTest)

add_subdirectory(cpp)
