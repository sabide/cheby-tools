cmake_minimum_required(VERSION 3.18)

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release CACHE STRING "CMake build type (Debug|Release)" FORCE)
endif ()

project (tecio C CXX)

if (Boost_INCLUDE_DIR)
    if (NOT EXISTS "${Boost_INCLUDE_DIR}/boost/version.hpp")
        message(FATAL_ERROR
            "Boost_INCLUDE_DIR must contain boost/version.hpp: ${Boost_INCLUDE_DIR}")
    endif ()
    add_library(tecio_boost_headers INTERFACE)
    target_include_directories(tecio_boost_headers SYSTEM INTERFACE
        "${Boost_INCLUDE_DIR}")
    set(TECIO_BOOST_HEADERS_TARGET tecio_boost_headers)
else ()
    find_package(Boost 1.69.0 CONFIG REQUIRED)
    if (TARGET Boost::headers)
        set(TECIO_BOOST_HEADERS_TARGET Boost::headers)
    elseif (TARGET Boost::boost)
        set(TECIO_BOOST_HEADERS_TARGET Boost::boost)
    else ()
        message(FATAL_ERROR
            "Boost was found but provides neither Boost::headers nor Boost::boost")
    endif ()
endif ()

include_directories(.)

set (CMAKE_CXX_STANDARD 14)

IF (WIN32)
    set(BaseFlags "/EHsc /MP /wd\"4996\" /D MSWIN /D TP_PROJECT_USES_BOOST /D BOOST_ALL_NO_LIB /D MAKEARCHIVE /D NO_THIRD_PARTY_LIBS /D NO_ASSERTS /D tecio_EXPORTS")
ELSE ()
	set(BaseFlags "-DLINUX -DLINUX64 -DTP_PROJECT_USES_BOOST -DBOOST_ALL_NO_LIB -DMAKEARCHIVE -DNO_THIRD_PARTY_LIBS -DNO_ASSERTS -Dtecio_EXPORTS -fmessage-length=0 -fPIC -fvisibility=hidden -w")
ENDIF ()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BaseFlags}")

FILE(GLOB mainFiles "*.cpp" "*.h")
LIST(REMOVE_ITEM mainFiles "${CMAKE_CURRENT_SOURCE_DIR}/szcombine.cpp")

add_library(tecio STATIC ${mainFiles})

# This interface target only propagates the Boost include directory. No
# compiled Boost library is linked.
target_link_libraries(tecio PRIVATE ${TECIO_BOOST_HEADERS_TARGET})

add_executable(szcombine "szcombine.cpp")
if (WIN32)
    target_link_libraries(szcombine tecio)
else ()
    target_link_libraries(szcombine tecio pthread)
endif ()

# Had to add this so building tecio care package within loci-chem works...
set_target_properties(tecio PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(szcombine PROPERTIES LINKER_LANGUAGE CXX)
