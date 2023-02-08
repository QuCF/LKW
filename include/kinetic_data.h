#pragma once
#include "mix.h"


// ---------------------------------------------------------
// --- Data describing a stationary kinetic system ---
// ---------------------------------------------------------
struct KDATA
{
    uint32_t nx, ny, nz; //log2(Nx)
    uint32_t Nx, Ny, Nz; 

    uint32_t nvx, nvy, nvz;
    uint32_t Nvx, Nvy, Nvz;

    uint32_t Nvx_h, Nvy_h, Nvz_h; // half of Nv

    uint32_t Nvars;

    uint32_t N_discr; // a constant defining the discretization;

    double w;  // normalized frequency;
    double h;  // normalized spatial step;
    double dv; // normalized velocity step;

    bool flag_with_copies;

    double xmax; // maximum x-coordinate normalized to the Debye length;
    double vmax; // maximum x-velocity normalized to the thermal velocity;

    SpMatrixC A; // Matrix describing the kinetic problem (Ax = b); [on device].
    cuDoubleComplex* b; // Right-hand-side vector; on device]/
    cuDoubleComplex* psi; // Solution of the system A*psi=b; [on device].
    double* FB; // background distribution function [x,v];
    double* Y;  // combined background profiles [x,v];

    void set_to_zero()
    {
        nx = 0; ny = 0; nz = 0;
        Nx = 0; Ny = 0; Nz = 0;

        nvx = 0; nvy = 0; nvz = 0;
        Nvx = 0; Nvy = 0; Nvz = 0;

        Nvx_h = 0; Nvy_h = 0; Nvz_h = 0;
    }
};