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

void write_szplt_2d(
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



void write_szplt_3d(
    const std::string &filename,
    const std::vector<std::string> &var_names,
    const std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>> &vars)
{
    if (vars.empty())
        throw std::runtime_error("Aucune variable fournie.");
    if (var_names.size() != vars.size())
        throw std::runtime_error("var_names et vars doivent avoir la même taille.");

    // Dimensions cohérentes (toutes 3D nz x ny x nx)
    auto b0 = vars[0].request();
    if (b0.ndim != 3)
        throw std::runtime_error("Chaque variable doit être un ndarray 3D (nz, ny, nx).");

    INTEGER4 nz = static_cast<INTEGER4>(b0.shape[0]);
    INTEGER4 ny = static_cast<INTEGER4>(b0.shape[1]);
    INTEGER4 nx = static_cast<INTEGER4>(b0.shape[2]);

    for (size_t i = 1; i < vars.size(); ++i) {
        auto bi = vars[i].request();
        if (bi.ndim != 3 || bi.shape[0] != nz || bi.shape[1] != ny || bi.shape[2] != nx)
            throw std::runtime_error("Toutes les variables doivent partager les mêmes dimensions (nz, ny, nx).");
    }

    // Liste de variables TecIO
    std::string var_list;
    for (size_t i = 0; i < var_names.size(); ++i) {
        var_list += var_names[i];
        if (i + 1 < var_names.size()) var_list += " ";
    }

    // Paramètres fichier
    INTEGER4 debug     = 0;
    INTEGER4 is_double = 1;
    INTEGER4 file_type = 0; // Grid & solution
    INTEGER4 file_fmt  = 0; // SZL/SZPLT (selon ton TecIO; tu avais 0/1 commenté)
    INTEGER4 result    = 0;

    result = TECINI142(
        "PyTecIO Output",
        var_list.c_str(),
        filename.c_str(),
        ".",
        &file_fmt,
        &file_type,
        &debug,
        &is_double
    );
    if (result != 0) throw std::runtime_error("TECINI142 a échoué.");

    // Zone
    INTEGER4 zone_type = 0; // Ordered
    double   sol_time  = 0.0;
    INTEGER4 ICellMax = 0, JCellMax = 0, KCellMax = 0;
    INTEGER4 StrandID = 0, ParentZn = 0;
    INTEGER4 IsBlock  = 1;
    INTEGER4 NFConns  = 0, FNMode = 0, TotalNumFaceNodes = 0;
    INTEGER4 TotalNumBndryFaces = 0, TotalNumBndryConns = 0;
    INTEGER4 ShareConnFromZone = 0;

    // I, J, K = nx, ny, nz
    result = TECZNE142(
        "Zone 1",
        &zone_type,
        &nx, &ny, &nz,
        &ICellMax, &JCellMax, &KCellMax,
        &sol_time,
        &StrandID, &ParentZn,
        &IsBlock,
        &NFConns, &FNMode, &TotalNumFaceNodes,
        &TotalNumBndryFaces, &TotalNumBndryConns,
        nullptr, nullptr, nullptr,
        &ShareConnFromZone
    );
    if (result != 0) { TECEND142(); throw std::runtime_error("TECZNE142 a échoué."); }

    // Écriture en blocs
    const long long npts64 = 1LL * nx * ny * nz;
    if (npts64 > std::numeric_limits<INTEGER4>::max()) {
        TECEND142();
        throw std::runtime_error("Trop de points pour INTEGER4 (nx*ny*nz dépasse 2^31-1).");
    }
    INTEGER4 num_pts = static_cast<INTEGER4>(npts64);

    // IMPORTANT :
    // vars[i] est forcé C-contiguous (c_style|forcecast).
    // Pour shape (nz,ny,nx), l’ordre mémoire est x fastest => compatible Ordered I,J,K.
    for (const auto &arr : vars) {
        auto br  = arr.request();
        auto ptr = static_cast<const double*>(br.ptr);
        result = TECDAT142(&num_pts, ptr, &is_double);
        if (result != 0) { TECEND142(); throw std::runtime_error("TECDAT142 a échoué."); }
    }

    TECEND142();
}


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <limits>
#include <stdexcept>

#include "TECIO.h"

namespace py = pybind11;

class Szplt3DWriter {
public:
    Szplt3DWriter(const std::string& filename,
                  const std::vector<std::string>& var_names, // ["x","y","z","w","T",...]
                  py::array_t<double, py::array::c_style | py::array::forcecast> x,
                  py::array_t<double, py::array::c_style | py::array::forcecast> y,
                  py::array_t<double, py::array::c_style | py::array::forcecast> z)
        : filename_(filename), var_names_(var_names)
    {
        if (var_names_.size() < 3)
            throw std::runtime_error("var_names doit contenir au moins x y z.");

        bx_ = x.request();
        by_ = y.request();
        bz_ = z.request();

        if (bx_.ndim != 3 || by_.ndim != 3 || bz_.ndim != 3)
            throw std::runtime_error("x,y,z doivent être 3D (nz,ny,nx).");

        nz_ = static_cast<INTEGER4>(bx_.shape[0]);
        ny_ = static_cast<INTEGER4>(bx_.shape[1]);
        nx_ = static_cast<INTEGER4>(bx_.shape[2]);

        if (by_.shape[0]!=nz_ || by_.shape[1]!=ny_ || by_.shape[2]!=nx_ ||
            bz_.shape[0]!=nz_ || bz_.shape[1]!=ny_ || bz_.shape[2]!=nx_)
            throw std::runtime_error("x,y,z doivent partager (nz,ny,nx).");

        // var list
        std::string var_list;
        for (size_t i=0;i<var_names_.size();++i){
            var_list += var_names_[i];
            if (i+1<var_names_.size()) var_list += " ";
        }

        // TecIO init
        INTEGER4 debug=0, is_double=1, file_type=0, file_fmt=0, result=0;
        result = TECINI142("PyTecIO 3D series (shared grid, streaming)",
                           var_list.c_str(),
                           filename_.c_str(),
                           ".",
                           &file_fmt, &file_type, &debug, &is_double);
        if (result != 0) throw std::runtime_error("TECINI142 a échoué.");

        opened_ = true;

        // points/zone
        long long npts64 = 1LL * nx_ * ny_ * nz_;
        if (npts64 > std::numeric_limits<INTEGER4>::max())
            throw std::runtime_error("nx*ny*nz dépasse INTEGER4.");
        num_pts_ = static_cast<INTEGER4>(npts64);

        // share array size = nvars
        share_from_.assign(var_names_.size(), 0);
    }

    ~Szplt3DWriter() {
        if (opened_) {
            TECEND142();
            opened_ = false;
        }
    }

    // fields: list of 3D arrays (nz,ny,nx) corresponding to var_names[3:]
    void add_zone(double sol_time,
                  const std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>>& fields)
    {
        if (!opened_) throw std::runtime_error("Writer fermé.");
        const size_t n_fields_expected = var_names_.size() - 3;
        if (fields.size() != n_fields_expected)
            throw std::runtime_error("Nombre de champs != var_names.size()-3.");

        // check shapes
        for (size_t i=0;i<fields.size();++i){
            auto b = fields[i].request();
            if (b.ndim != 3 ||
                b.shape[0]!=nz_ || b.shape[1]!=ny_ || b.shape[2]!=nx_)
                throw std::runtime_error("Chaque champ doit être 3D (nz,ny,nx) et matcher le mesh.");
        }

        // Zone parameters
        INTEGER4 zone_type=0;
        INTEGER4 ICellMax=0, JCellMax=0, KCellMax=0;
        INTEGER4 ParentZn=0;
        INTEGER4 IsBlock=1;
        INTEGER4 NFConns=0, FNMode=0, TotalNumFaceNodes=0;
        INTEGER4 TotalNumBndryFaces=0, TotalNumBndryConns=0;
        INTEGER4 ShareConnFromZone=0;
        const INTEGER4 StrandID = 1;

        // Share x,y,z from zone 1 for zones >=2
        INTEGER4* share_ptr = nullptr;
        if (zone_index_ >= 2) {
            std::fill(share_from_.begin(), share_from_.end(), 0);
            share_from_[0] = 1;
            share_from_[1] = 1;
            share_from_[2] = 1;
            share_ptr = share_from_.data();
        }

        std::string zname = "Zone " + std::to_string(zone_index_);

        INTEGER4 is_double = 1;
        INTEGER4 result = TECZNE142(
            zname.c_str(),
            &zone_type,
            &nx_, &ny_, &nz_,
            &ICellMax, &JCellMax, &KCellMax,
            &sol_time,
            const_cast<INTEGER4*>(&StrandID), &ParentZn,
            &IsBlock,
            &NFConns, &FNMode, &TotalNumFaceNodes,
            &TotalNumBndryFaces, &TotalNumBndryConns,
            nullptr, nullptr, share_ptr,
            &ShareConnFromZone
        );
        if (result != 0) { TECEND142(); opened_ = false; throw std::runtime_error("TECZNE142 a échoué."); }

        auto write_block = [&](const double* ptr){
            INTEGER4 r = TECDAT142(&num_pts_, ptr, &is_double);
            if (r != 0) { TECEND142(); opened_ = false; throw std::runtime_error("TECDAT142 a échoué."); }
        };

        // Zone 1: write x,y,z once
        if (zone_index_ == 1) {
            write_block(static_cast<const double*>(bx_.ptr));
            write_block(static_cast<const double*>(by_.ptr));
            write_block(static_cast<const double*>(bz_.ptr));
        }

        // Always write fields
        for (const auto& a : fields) {
            auto b = a.request();
            write_block(static_cast<const double*>(b.ptr));
        }

        zone_index_ += 1;
    }

    void close() {
        if (opened_) {
            TECEND142();
            opened_ = false;
        }
    }

private:
    std::string filename_;
    std::vector<std::string> var_names_;

    py::buffer_info bx_, by_, bz_;
    INTEGER4 nx_{0}, ny_{0}, nz_{0};
    INTEGER4 num_pts_{0};

    bool opened_{false};
    int zone_index_{1}; // 1-based

    std::vector<INTEGER4> share_from_;
};





PYBIND11_MODULE(tecio_wrapper, m) {
    m.doc() = "Wrapper Python TecIO 142 (écriture SZPLT depuis ndarrays 2D).";

    m.def("write_szplt_2d", &write_szplt_2d,
          py::arg("filename"), py::arg("var_names"), py::arg("vars"),
          "Écrit un .szplt avec des variables 2D (ny, nx) en layout bloc.");

    m.def("write_ndarray_1d", &write_ndarray_1d,
          py::arg("filename"), py::arg("var_names"), py::arg("vars"),
          "Écrit un .szplt avec des variables 2D (ny, nx) en layout bloc.");

    m.def("write_szplt_3d", &write_szplt_3d,
      py::arg("filename"), py::arg("var_names"), py::arg("vars"),
      "Écrit un .szplt avec des variables 3D (nz, ny, nx) en layout bloc.");


    py::class_<Szplt3DWriter>(m, "Szplt3DWriter")
  .def(py::init<const std::string&,
                const std::vector<std::string>&,
                py::array_t<double, py::array::c_style | py::array::forcecast>,
                py::array_t<double, py::array::c_style | py::array::forcecast>,
                py::array_t<double, py::array::c_style | py::array::forcecast>>(),
       py::arg("filename"), py::arg("var_names"), py::arg("x"), py::arg("y"), py::arg("z"))
  .def("add_zone", &Szplt3DWriter::add_zone, py::arg("sol_time"), py::arg("fields"))
  .def("close", &Szplt3DWriter::close);


}



