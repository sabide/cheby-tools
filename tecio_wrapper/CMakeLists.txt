pybind11_add_module(tecio_wrapper tecio_wrapper.cpp)

include(GNUInstallDirs)



# 1) Installer le module compilé (.so)
install(TARGETS tecio_wrapper
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/tecio_wrapper
)

# 2) Installer les fichiers Python du package (__init__.py)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/tecio_wrapper
)


# Trouver pybind11
#set(PYBIND11_FINDPYTHON ON)
#find_package(pybind11 REQUIRED)

# === Définit où est TECIO ===
# (Tu peux aussi passer ces variables avec -D à cmake)
#set(TECIO_INCLUDE_DIR "/chemin/vers/tecio/include")
#set(TECIO_LIBRARY "/chemin/vers/tecio/lib/libtecio.a")  # ou .so

set(TECIO_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/tecio/teciosrc/")
set(LIB_DEST_DIR "${CMAKE_SOURCE_DIR}/tecio_wrapper/")




target_include_directories(tecio_wrapper PRIVATE ${TECIO_INCLUDE_DIR})

target_link_libraries(tecio_wrapper PRIVATE tecio pybind11::module)

# Installation dans le répertoire Python
set_target_properties(tecio_wrapper PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${LIB_DEST_DIR}
)
