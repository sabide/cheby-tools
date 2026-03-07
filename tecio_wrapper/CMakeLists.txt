if(NOT DEFINED CHEBY_TECIO_TARGET)
  message(FATAL_ERROR "CHEBY_TECIO_TARGET is not set by the top-level CMakeLists.txt.")
endif()

if(NOT DEFINED CHEBY_TECIO_INCLUDE_DIR OR CHEBY_TECIO_INCLUDE_DIR STREQUAL "")
  message(FATAL_ERROR "CHEBY_TECIO_INCLUDE_DIR is not set by the top-level CMakeLists.txt.")
endif()

pybind11_add_module(tecio_wrapper tecio_wrapper.cpp)
target_compile_features(tecio_wrapper PRIVATE cxx_std_14)
target_include_directories(tecio_wrapper PRIVATE "${CHEBY_TECIO_INCLUDE_DIR}")
target_link_libraries(tecio_wrapper PRIVATE ${CHEBY_TECIO_TARGET} pybind11::module)

include(GNUInstallDirs)
install(TARGETS tecio_wrapper
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/tecio_wrapper
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/tecio_wrapper
)
