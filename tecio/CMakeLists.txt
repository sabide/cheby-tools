pybind11_add_module(tecio_wrapper tecio_wrapper.cpp)


# Trouver pybind11
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

# === Définit où est TECIO ===
# (Tu peux aussi passer ces variables avec -D à cmake)
#set(TECIO_INCLUDE_DIR "/chemin/vers/tecio/include")
#set(TECIO_LIBRARY "/chemin/vers/tecio/lib/libtecio.a")  # ou .so

set(LIB_DEST_DIR "${CMAKE_SOURCE_DIR}/tecio/")

target_include_directories(tecio_wrapper PRIVATE ${TECIO_INCLUDE_DIR})

target_link_libraries(tecio_wrapper PRIVATE ${TECIO_LIBRARY})

# Installation dans le répertoire Python
set_target_properties(tecio_wrapper PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${LIB_DEST_DIR}
)
