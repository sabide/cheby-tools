cmake_minimum_required(VERSION 3.15)
project(myproject LANGUAGES CXX)

# Requiert pybind11
#find_package(pybind11 REQUIRED)
add_subdirectory(external/pybind11)
add_subdirectory(external/tecio/teciosrc/)


add_subdirectory(tecio_wrapper)

