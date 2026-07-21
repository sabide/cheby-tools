cmake_minimum_required(VERSION 3.18)
project(cheby_tools_adastra LANGUAGES CXX)

# Prefer the local Miniconda interpreter when available (macOS dev setup).
if(NOT DEFINED Python_EXECUTABLE AND EXISTS "/opt/miniconda3/bin/python")
  set(Python_EXECUTABLE "/opt/miniconda3/bin/python" CACHE FILEPATH
    "Preferred Python interpreter for pybind11/FindPython.")
endif()

option(CHEBY_USE_BUNDLED_TECIO "Build TecIO from external/tecio/teciosrc." ON)
option(CHEBY_INSTALL_TECIO_WRAPPER "Build and install the tecio_wrapper Python extension." ON)
option(CHEBY_INSTALL_POSTPROCESSING_TOOLS "Install the Python post-processing tools (discr, stats, spec_forge)." ON)

include(GNUInstallDirs)
set(CHEBY_PYTHON_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/python" CACHE STRING
  "Install directory for Python modules, relative to CMAKE_INSTALL_PREFIX.")

set(CHEBY_BOOST_INCLUDE_DIR "" CACHE PATH
  "Boost headers root (the directory containing boost/).")
set(CHEBY_TECIO_INCLUDE_DIR "" CACHE PATH "TecIO include directory when using an external TecIO build.")
set(CHEBY_TECIO_LIBRARY "" CACHE FILEPATH "Path to libtecio.a/.so/.dylib when using an external TecIO build.")

if(NOT CHEBY_INSTALL_TECIO_WRAPPER AND NOT CHEBY_INSTALL_POSTPROCESSING_TOOLS)
  message(FATAL_ERROR
    "Nothing to install: enable CHEBY_INSTALL_TECIO_WRAPPER and/or "
    "CHEBY_INSTALL_POSTPROCESSING_TOOLS.")
endif()

if(CHEBY_INSTALL_TECIO_WRAPPER)
  if(EXISTS "${CMAKE_SOURCE_DIR}/external/pybind11/CMakeLists.txt")
    add_subdirectory(external/pybind11)
  else()
    message(FATAL_ERROR
      "Bundled pybind11 not found at external/pybind11. "
      "Initialize submodules with:\n"
      "  git submodule update --init --recursive")
  endif()

  if(CHEBY_USE_BUNDLED_TECIO)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/external/tecio/teciosrc/CMakeLists.txt")
      message(FATAL_ERROR "Bundled TecIO not found at external/tecio/teciosrc.")
    endif()

    # Prefer explicitly supplied or repository-local headers. Otherwise let
    # TecIO discover a modern system Boost and consume Boost::headers.
    if(NOT CHEBY_BOOST_INCLUDE_DIR
       AND EXISTS "${CMAKE_SOURCE_DIR}/external/boost/boost/version.hpp")
      set(CHEBY_BOOST_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/boost")
    endif()

    if(CHEBY_BOOST_INCLUDE_DIR)
      if(NOT EXISTS "${CHEBY_BOOST_INCLUDE_DIR}/boost/version.hpp")
        message(FATAL_ERROR
          "CHEBY_BOOST_INCLUDE_DIR must contain boost/version.hpp: "
          "${CHEBY_BOOST_INCLUDE_DIR}")
      endif()
      set(Boost_INCLUDE_DIR "${CHEBY_BOOST_INCLUDE_DIR}" CACHE PATH
        "Boost headers root" FORCE)
      set(Boost_NO_SYSTEM_PATHS ON CACHE BOOL
        "Use only the explicitly selected Boost headers" FORCE)
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
endif()

if(CHEBY_INSTALL_POSTPROCESSING_TOOLS)
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/discr
    DESTINATION ${CHEBY_PYTHON_INSTALL_DIR}
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE
    PATTERN "._*" EXCLUDE
  )
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/stats
    DESTINATION ${CHEBY_PYTHON_INSTALL_DIR}
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE
    PATTERN "._*" EXCLUDE
  )
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/spec_forge
    DESTINATION ${CHEBY_PYTHON_INSTALL_DIR}
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE
    PATTERN "._*" EXCLUDE
  )
endif()
