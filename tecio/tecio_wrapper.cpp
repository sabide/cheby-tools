// io_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>   // conversions std::vector/std::string
#include <string>
#include <vector>

#include <iostream>

#include "TECIO.h"          // TecIO 142 API

namespace py = pybind11;

void write_ndarray_1d(
    const std::string &filename,
    const std::vector<std::string> &var_names,
    const std::vector<py::array_t<double,py::array::c_style | py::array::forcecast>> &vars)
{
    if (vars.empty())
        throw std::runtime_error("no variable in argument.");

    if (var_names.size() != vars.size())
        throw std::runtime_error("the number of variables is not equal  .");

    auto b0 = vars[0].request(); // c'est un py::array_t
    if (b0.ndim != 1)
        throw std::runtime_error("plot only one-dimensional array");

    INTEGER4 n = static_cast<INTEGER4>(b0.shape[0]); // TECIO.h
    
    for (size_t i = 1; i < vars.size(); ++i) {
        auto bi = vars[i].request();
        if (bi.ndim != 1 || bi.shape[0] != n )
            throw std::runtime_error(" all the variables must have the same size n.");
    }

    // TecIO "Variables" list (separated by spaces)
    std::string var_list;
    for (size_t i = 0; i < var_names.size(); ++i) {
        var_list += var_names[i];
        if (i + 1 < var_names.size()) var_list += " ";
    }
    
    // File settings
    INTEGER4 debug     = 0;
    INTEGER4 is_double = 1; // écrire en double
    INTEGER4 file_type = 0; // 0 = Grid & solution
    INTEGER4 file_fmt  = 0; // 1 = SZL (format moderne)
    INTEGER4 result    = 0;

    // Ouverture du dataset
    result = TECINI142(
        "PyTecIO Output",
        var_list.c_str(),
        filename.c_str(),
        "./",
        &file_fmt,   // FileFormat
        &file_type,  // FileType
        &debug,      // Debug
        &is_double   // VIsDouble
    );
    if (result != 0) throw std::runtime_error("TECINI142 has failed.");

    // Déclaration de la zone "Ordered", layout bloc
    INTEGER4 zone_type = 0; // Ordered
    INTEGER4 one       = 1;
    INTEGER4 zero      = 0;
    double   sol_time  = 0.0;

    // Ces champs existent dans la signature TECZNE142 et doivent être fournis;
    // on ne veut PAS utiliser les 3 tableaux "par variable" => passer NULL.
    INTEGER4 ICellMax = 0, JCellMax = 0, KCellMax = 0;
    INTEGER4 StrandID = 0, ParentZn = 0;
    INTEGER4 IsBlock  = 1;
    INTEGER4 NFConns  = 0, FNMode = 0, TotalNumFaceNodes = 0;
    INTEGER4 TotalNumBndryFaces = 0, TotalNumBndryConns = 0;
    INTEGER4 ShareConnFromZone = 0; // 0 => pas de partage de connectivité

    result = TECZNE142(
        "Zone 1",
        &zone_type,
        &n, &one, &one,                    // I, J, K (K=1 en 2D)
        &ICellMax, &JCellMax, &KCellMax,   // ICellMx, JCellMx, KCellMx
        &sol_time,                         // SolutionTime
        &StrandID, &ParentZn,              // StrandID, ParentZone
        &IsBlock,                          // IsBlock (1 = données en blocs par variable)
        &NFConns, &FNMode, &TotalNumFaceNodes,
        &TotalNumBndryFaces, &TotalNumBndryConns,
        /* Tableaux par variable — NE PAS passer &zero : laisser NULL */
        nullptr,   // ValueLocation[]  (NULL => nodal par défaut)
        nullptr,   // IsPassive[]      (NULL => toutes actives)
        nullptr,   // ShareVarFromZone[] (NULL => aucun partage)
        &ShareConnFromZone                // ShareConnectivityFromZone (0 => none)
    );
    if (result != 0) { TECEND142(); throw std::runtime_error("TECZNE142 a échoué."); }

    // Écriture bloc par bloc
    INTEGER4 num_pts = n ;
    for (const auto &arr : vars) {
        auto br  = arr.request(); // c_style garanti contigu
        auto ptr = static_cast<const double*>(br.ptr);
        result = TECDAT142(&num_pts, ptr, &is_double);
        if (result != 0) { TECEND142(); throw std::runtime_error("TECDAT142 a échoué."); }
    }

    //TECLAB142("Créé via pybind11");
    TECEND142();
}




void hello(
    const std::string &filename,
    const std::vector<std::string> &var_names,
    const std::vector<py::array_t<double,py::array::c_style | py::array::forcecast>> &vars)
{

    std::cout << filename << std::endl;

    auto b0 = vars[0].request();
    if (b0.ndim != 2)
        throw std::runtime_error("Chaque variable doit être un ndarray 2D (ny, nx).");
    std::cout << b0.ndim << std::endl;
}

void write_szplt(
    const std::string &filename,
    const std::vector<std::string> &var_names,
    const std::vector<py::array_t<double,py::array::c_style | py::array::forcecast>> &vars)
{
    if (vars.empty())
        throw std::runtime_error("Aucune variable fournie.");
    if (var_names.size() != vars.size())
        throw std::runtime_error("var_names et vars doivent avoir la même taille.");

    // Dimensions cohérentes (toutes 2D ny x nx)
    auto b0 = vars[0].request();
    if (b0.ndim != 2)
        throw std::runtime_error("Chaque variable doit être un ndarray 2D (ny, nx).");

    INTEGER4 ny = static_cast<INTEGER4>(b0.shape[0]);
    INTEGER4 nx = static_cast<INTEGER4>(b0.shape[1]);
    for (size_t i = 1; i < vars.size(); ++i) {
        auto bi = vars[i].request();
        if (bi.ndim != 2 || bi.shape[0] != ny || bi.shape[1] != nx)
            throw std::runtime_error("Toutes les variables doivent partager les mêmes dimensions (ny, nx).");
    }

    // Liste "Variables" TecIO (séparées par des espaces)
    std::string var_list;
    for (size_t i = 0; i < var_names.size(); ++i) {
        var_list += var_names[i];
        if (i + 1 < var_names.size()) var_list += " ";
    }

    // Paramètres fichier
    INTEGER4 debug     = 0;
    INTEGER4 is_double = 1; // écrire en double
    INTEGER4 file_type = 0; // 0 = Grid & solution
    INTEGER4 file_fmt  = 0; // 1 = SZL (format moderne)
    INTEGER4 result    = 0;

    // Ouverture du dataset
    result = TECINI142(
        "PyTecIO Output",
        var_list.c_str(),
        filename.c_str(),
        ".",
        &file_fmt,   // FileFormat
        &file_type,  // FileType
        &debug,      // Debug
        &is_double   // VIsDouble
    );
    if (result != 0) throw std::runtime_error("TECINI142 a échoué.");

    // Déclaration de la zone "Ordered", layout bloc
    INTEGER4 zone_type = 0; // Ordered
    INTEGER4 one       = 1;
    INTEGER4 zero      = 0;
    double   sol_time  = 0.0;

    // Ces champs existent dans la signature TECZNE142 et doivent être fournis;
    // on ne veut PAS utiliser les 3 tableaux "par variable" => passer NULL.
    INTEGER4 ICellMax = 0, JCellMax = 0, KCellMax = 0;
    INTEGER4 StrandID = 0, ParentZn = 0;
    INTEGER4 IsBlock  = 1;
    INTEGER4 NFConns  = 0, FNMode = 0, TotalNumFaceNodes = 0;
    INTEGER4 TotalNumBndryFaces = 0, TotalNumBndryConns = 0;
    INTEGER4 ShareConnFromZone = 0; // 0 => pas de partage de connectivité

    result = TECZNE142(
        "Zone 1",
        &zone_type,
        &nx, &ny, &one,                    // I, J, K (K=1 en 2D)
        &ICellMax, &JCellMax, &KCellMax,   // ICellMx, JCellMx, KCellMx
        &sol_time,                         // SolutionTime
        &StrandID, &ParentZn,              // StrandID, ParentZone
        &IsBlock,                          // IsBlock (1 = données en blocs par variable)
        &NFConns, &FNMode, &TotalNumFaceNodes,
        &TotalNumBndryFaces, &TotalNumBndryConns,
        /* Tableaux par variable — NE PAS passer &zero : laisser NULL */
        nullptr,   // ValueLocation[]  (NULL => nodal par défaut)
        nullptr,   // IsPassive[]      (NULL => toutes actives)
        nullptr,   // ShareVarFromZone[] (NULL => aucun partage)
        &ShareConnFromZone                // ShareConnectivityFromZone (0 => none)
    );
    if (result != 0) { TECEND142(); throw std::runtime_error("TECZNE142 a échoué."); }

    // Écriture bloc par bloc
    INTEGER4 num_pts = nx * ny;
    for (const auto &arr : vars) {
        auto br  = arr.request(); // c_style garanti contigu
        auto ptr = static_cast<const double*>(br.ptr);
        result = TECDAT142(&num_pts, ptr, &is_double);
        if (result != 0) { TECEND142(); throw std::runtime_error("TECDAT142 a échoué."); }
    }

    //TECLAB142("Créé via pybind11");
    TECEND142();
}

PYBIND11_MODULE(tecio_wrapper, m) {
    m.doc() = "Wrapper Python TecIO 142 (écriture SZPLT depuis ndarrays 2D).";

    m.def("write_szplt", &write_szplt,
          py::arg("filename"), py::arg("var_names"), py::arg("vars"),
          "Écrit un .szplt avec des variables 2D (ny, nx) en layout bloc.");

    m.def("write_ndarray_1d", &write_ndarray_1d,
          py::arg("filename"), py::arg("var_names"), py::arg("vars"),
          "Écrit un .szplt avec des variables 2D (ny, nx) en layout bloc.");

}



