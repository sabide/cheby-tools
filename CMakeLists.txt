cmake_minimum_required(VERSION 3.18)
project(cheby_tools_adastra LANGUAGES CXX)

option(CHEBY_USE_BUNDLED_PYBIND11 "Build with the git submodule in external/pybind11." ON)
option(CHEBY_USE_BUNDLED_TECIO "Build TecIO from external/tecio/teciosrc." ON)

include(GNUInstallDirs)
set(CHEBY_PYTHON_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/python" CACHE STRING
  "Install directory for Python modules, relative to CMAKE_INSTALL_PREFIX.")

set(CHEBY_BOOST_INCLUDE_DIR "" CACHE PATH "Boost include directory (used when building bundled TecIO).")
set(CHEBY_TECIO_INCLUDE_DIR "" CACHE PATH "TecIO include directory when using an external TecIO build.")
set(CHEBY_TECIO_LIBRARY "" CACHE FILEPATH "Path to libtecio.a/.so/.dylib when using an external TecIO build.")

if(CHEBY_USE_BUNDLED_PYBIND11)
  if(EXISTS "${CMAKE_SOURCE_DIR}/external/pybind11/CMakeLists.txt")
    add_subdirectory(external/pybind11)
  else()
    message(WARNING
      "external/pybind11 is missing. Falling back to find_package(pybind11).")
    find_package(pybind11 CONFIG REQUIRED)
  endif()
else()
  find_package(pybind11 CONFIG REQUIRED)
endif()

if(CHEBY_USE_BUNDLED_TECIO)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/external/tecio/teciosrc/CMakeLists.txt")
    message(FATAL_ERROR "Bundled TecIO not found at external/tecio/teciosrc.")
  endif()

  if(CHEBY_BOOST_INCLUDE_DIR)
    set(Boost_INCLUDE_DIR "${CHEBY_BOOST_INCLUDE_DIR}" CACHE PATH "Boost include directory" FORCE)
  endif()

  add_subdirectory(external/tecio/teciosrc)
  set(CHEBY_TECIO_TARGET tecio)
  set(CHEBY_TECIO_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/tecio/teciosrc")
else()
  if(NOT CHEBY_TECIO_INCLUDE_DIR OR NOT CHEBY_TECIO_LIBRARY)
    message(FATAL_ERROR
      "Using external TecIO requires both:\n"
      "  -DCHEBY_TECIO_INCLUDE_DIR=/path/to/teciosrc\n"
      "  -DCHEBY_TECIO_LIBRARY=/path/to/libtecio.a")
  endif()

  add_library(tecio_external UNKNOWN IMPORTED)
  set_target_properties(tecio_external PROPERTIES
    IMPORTED_LOCATION "${CHEBY_TECIO_LIBRARY}"
  )
  set(CHEBY_TECIO_TARGET tecio_external)
endif()

add_subdirectory(tecio_wrapper)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/discr
  DESTINATION ${CHEBY_PYTHON_INSTALL_DIR}
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/stats
  DESTINATION ${CHEBY_PYTHON_INSTALL_DIR}
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/cheby_tool
  DESTINATION ${CHEBY_PYTHON_INSTALL_DIR}
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)

# Namespace install so everything is available under `cheby_tool`.
install(DIRECTORY ${CMAKE_SOURCE_DIR}/discr
  DESTINATION ${CHEBY_PYTHON_INSTALL_DIR}/cheby_tool
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/stats
  DESTINATION ${CHEBY_PYTHON_INSTALL_DIR}/cheby_tool
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)
