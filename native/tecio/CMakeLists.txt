if(NOT DEFINED CHEBY_TECIO_TARGET)
  message(FATAL_ERROR "CHEBY_TECIO_TARGET is not set by the top-level project.")
endif()

if(NOT DEFINED CHEBY_TECIO_INCLUDE_DIR OR CHEBY_TECIO_INCLUDE_DIR STREQUAL "")
  message(FATAL_ERROR "CHEBY_TECIO_INCLUDE_DIR is not set by the top-level project.")
endif()

pybind11_add_module(_tecio tecio.cpp)
target_compile_features(_tecio PRIVATE cxx_std_14)
target_include_directories(_tecio PRIVATE "${CHEBY_TECIO_INCLUDE_DIR}")
target_link_libraries(_tecio PRIVATE ${CHEBY_TECIO_TARGET} pybind11::module)

install(TARGETS _tecio
  LIBRARY DESTINATION ${CHEBY_PYTHON_INSTALL_DIR}/cheby_tools
)
