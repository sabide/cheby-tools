#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "TECIO.h"

namespace py = pybind11;

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

void check_dimension(py::ssize_t size) {
    if (size <= 0) {
        throw std::invalid_argument("array dimensions must be positive");
    }
    if (size > std::numeric_limits<INTEGER4>::max()) {
        throw std::overflow_error("array dimension exceeds TecIO INTEGER4");
    }
}

std::string parent_directory(const std::string& filename) {
#ifdef _WIN32
    const std::size_t separator = filename.find_last_of("/\\");
#else
    const std::size_t separator = filename.find_last_of('/');
#endif
    if (separator == std::string::npos) {
        return ".";
    }
    if (separator == 0) {
        return filename.substr(0, 1);
    }
#ifdef _WIN32
    if (separator == 2 && filename.size() > 2 && filename[1] == ':') {
        return filename.substr(0, 3);
    }
#endif
    return filename.substr(0, separator);
}

void write_plt(const std::string& filename,
               const std::vector<std::string>& names,
               const std::vector<Array>& arrays) {
    if (arrays.empty()) {
        throw std::invalid_argument("arrays must not be empty");
    }
    if (names.size() != arrays.size()) {
        throw std::invalid_argument("names and arrays must have equal lengths");
    }

    std::vector<py::buffer_info> buffers;
    buffers.reserve(arrays.size());
    for (const auto& array : arrays) {
        buffers.push_back(array.request());
    }

    const py::ssize_t ndim = buffers.front().ndim;
    if (ndim < 1 || ndim > 3) {
        throw std::invalid_argument("arrays must be 1D, 2D, or 3D");
    }
    for (const auto& buffer : buffers) {
        if (buffer.ndim != ndim || buffer.shape != buffers.front().shape) {
            throw std::invalid_argument("all arrays must have the same shape");
        }
        for (const py::ssize_t size : buffer.shape) {
            check_dimension(size);
        }
    }

    const auto& shape = buffers.front().shape;
    INTEGER4 imax = static_cast<INTEGER4>(shape[ndim - 1]);
    INTEGER4 jmax = ndim >= 2 ? static_cast<INTEGER4>(shape[ndim - 2]) : 1;
    INTEGER4 kmax = ndim == 3 ? static_cast<INTEGER4>(shape[ndim - 3]) : 1;
    const long long point_count = 1LL * imax * jmax * kmax;
    if (point_count > std::numeric_limits<INTEGER4>::max()) {
        throw std::overflow_error("grid contains too many points for TecIO");
    }

    std::string variables;
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (names[index].empty()) {
            throw std::invalid_argument("variable names must not be empty");
        }
        if (names[index].find('\0') != std::string::npos ||
            names[index].find('\n') != std::string::npos) {
            throw std::invalid_argument(
                "variable names must not contain NUL or newline characters");
        }
        if (names[index].size() > 128) {
            throw std::invalid_argument(
                "variable name at index " + std::to_string(index) +
                " exceeds TecIO's 128-byte limit");
        }
        if (index != 0) {
            // TecIO chooses newline as the list delimiter when present.  It
            // therefore preserves spaces and commas inside variable names.
            variables += "\n";
        }
        variables += names[index];
    }

    INTEGER4 file_format = FILEFORMAT_PLT;
    INTEGER4 file_type = 0;
    INTEGER4 debug = 0;
    INTEGER4 is_double = 1;
    const std::string scratch_directory = parent_directory(filename);
    if (TECINI142(
            "cheby-tools",
            variables.c_str(),
            filename.c_str(),
            scratch_directory.c_str(),
            &file_format,
            &file_type,
            &debug,
            &is_double) != 0) {
        throw std::runtime_error(
            "TECINI142 failed for output '" + filename + "'");
    }

    try {
        INTEGER4 zone_type = 0;
        INTEGER4 icellmax = 0;
        INTEGER4 jcellmax = 0;
        INTEGER4 kcellmax = 0;
        double solution_time = 0.0;
        INTEGER4 strand_id = 0;
        INTEGER4 parent_zone = 0;
        INTEGER4 is_block = 1;
        INTEGER4 num_face_connections = 0;
        INTEGER4 face_neighbor_mode = 0;
        INTEGER4 total_num_face_nodes = 0;
        INTEGER4 num_connected_boundary_faces = 0;
        INTEGER4 total_num_boundary_connections = 0;
        INTEGER4 share_connectivity = 0;

        if (TECZNE142(
                "Zone 1",
                &zone_type,
                &imax,
                &jmax,
                &kmax,
                &icellmax,
                &jcellmax,
                &kcellmax,
                &solution_time,
                &strand_id,
                &parent_zone,
                &is_block,
                &num_face_connections,
                &face_neighbor_mode,
                &total_num_face_nodes,
                &num_connected_boundary_faces,
                &total_num_boundary_connections,
                nullptr,
                nullptr,
                nullptr,
                &share_connectivity) != 0) {
            throw std::runtime_error(
                "TECZNE142 failed for output '" + filename + "'");
        }

        INTEGER4 count = static_cast<INTEGER4>(point_count);
        for (std::size_t index = 0; index < buffers.size(); ++index) {
            if (TECDAT142(&count, buffers[index].ptr, &is_double) != 0) {
                throw std::runtime_error(
                    "TECDAT142 failed for variable '" + names[index] +
                    "' in output '" + filename + "'");
            }
        }
    } catch (...) {
        TECEND142();
        throw;
    }

    if (TECEND142() != 0) {
        throw std::runtime_error(
            "TECEND142 failed for output '" + filename + "'");
    }
}

}  // namespace

PYBIND11_MODULE(_tecio, module) {
    module.doc() = "Private classic Tecplot .plt backend for cheby-tools.";
    module.def(
        "write_plt",
        &write_plt,
        py::arg("filename"),
        py::arg("names"),
        py::arg("arrays"));
}
