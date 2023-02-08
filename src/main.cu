#include "../include/kin_1d.cuh"
using namespace std;
int main(int argc, char *argv[])
{
    using namespace std::complex_literals;

    uint32_t nx = stoi(string(argv[1]));
    uint32_t nv = stoi(string(argv[2]));

    Constants cc;

    double Tref = 1.0e4 * cc.ev_; // in erg
    double den_ref = 1e12; // in cm-3
    uint ND = 10;
    uint Nth = 2;
    double wa = 1.0; // antenna frequency;
    double source_r0 = 0.04;  // source position (in coordinates [0,1]);
    double source_w = 0.01;  // source width;

    KW dd(nx, nv, Tref, den_ref, ND, Nth, wa, false);
    dd.init_device();

    dd.set_background_profiles("flat"s);
    dd.save_xv_background_profs_to_hdf5();

    dd.form_rhs(source_r0, source_w);
    dd.save_rhs_to_hdf5();

    dd.form_matrices();
    dd.print_matrix();
    dd.save_matrices();

    return 0;
}




// // For original matrices (without taking into account sparsity)
// // and for the fast-inverse method ()
// #include "../include/kinetic_system.h"
// using namespace std;
// int main(int argc, char *argv[])
// {
//     using namespace std::complex_literals;

//     uint32_t nx = stoi(string(argv[1]));
//     uint32_t nv = stoi(string(argv[2]));

//     Constants cc;

//     double Tref = 1.0e4 * cc.ev_; // in erg
//     double den_ref = 1e12; // in cm-3
//     uint ND = 10;
//     uint Nth = 2;
//     double wa = 1.222; // antenna frequency;

//     KW dd(nx, nv, Tref, den_ref, ND, Nth, wa);

//     dd.take_flat_profiles();
//     // dd.take_gaussian_profile();

//     dd.form_matrices();
//     // dd.print_grids();
//     // dd.print_matrix();

//     // dd.find_svd();

//     dd.save_matrices();

//     return 0;
// }
