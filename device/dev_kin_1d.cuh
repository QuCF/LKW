#pragma once

#include <cuComplex.h>
#include "../include/kinetic_data.h"

__constant__ KDATA dev_dd_;


/** Get a velocity-point on the velocity grid. */
__device__ __host__ __forceinline__
double get_v1(YCD vmax, YCD dv, YCU iv){ return (-vmax + dv * iv); }

/** Get a space-point on the spatial grid. */
__device__ __host__ __forceinline__
double get_x1(YCD h, YCU ix){ return (h*ix); }



/**
 * Initialize the (x,v)-background profiles.
 * F[ix, iv] --> F[ix*Nv + iv] (column-major format).
*/
__global__ void init_background_distribution(double* T, double* n, double* dT, double* dn)
{
    uint32_t iv = threadIdx.x;  // velocity id;
    uint32_t ix = blockIdx.x;   // space id;

    // row-id of the resulting (Nx*Nv)\times(Nx*Nv) matrix;
    uint32_t ir = ix * dev_dd_.Nvx + iv; 

    double T1 = T[ix];
    double n1 = n[ix];

    double v1 = get_v1(dev_dd_.vmax , dev_dd_.dv, iv);
    double v2 = v1*v1;

    double temp1 = 2*T1;
    double temp2 = n1 / (sqrt(M_PI * temp1) * dev_dd_.dv * T1);
    
    dev_dd_.FB[ir] = sqrt(temp2 * exp(-v2/temp1)); // sqrt(Maxwellian / (dv*T))
    dev_dd_.Y[ir]  = 0.25 * (2*dn[ix]/n1 - dT[ix]/T1 * (3. - v2/T1));
}

/**
 * Initialize the right-hand-side vector: Guassian-shaped source for E at x0; 
 * case WITH copies of E;
*/
__global__ void init_rhs_with_copies(double r0, double wg)
{
    uint32_t iv = threadIdx.x;  // velocity id;
    uint32_t ix = blockIdx.x;   // space id;

    // shift because the source is set for the electric field;
    uint32_t sh_E = dev_dd_.Nx * dev_dd_.Nvx;

    // row-id of the resulting (Nx*Nv)\times(Nx*Nv) matrix;
    uint32_t ir = ix * dev_dd_.Nvx + iv + sh_E; 

    double r1 = get_x1(dev_dd_.h, ix) / dev_dd_.xmax;
    double r2 = (r1-r0)*(r1-r0);
    double wg2 = 2*wg*wg;

    dev_dd_.b[ir].x = exp(-r2/wg2) / (wg*sqrt(2.*M_PI));
    dev_dd_.b[ir].y = 0.0;
}


/**
 * Initialize the right-hand-side vector: Guassian-shaped source for E at x0; 
 * case WITHOUT copies of E;
 * here, iv = 0 always;
*/
__global__ void init_rhs_without_copies(double r0, double wg)
{
    uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x; // space id;

    // shift because the source is set for the electric field;
    uint32_t sh_E = dev_dd_.Nx * dev_dd_.Nvx;

    // row-id of the resulting (Nx*Nv)\times(Nx*Nv) matrix;
    uint32_t ir = ix * dev_dd_.Nvx + sh_E; 

    double r1 = get_x1(dev_dd_.h, ix) / dev_dd_.xmax;
    double r2 = (r1-r0)*(r1-r0);
    double wg2 = 2*wg*wg;

    dev_dd_.b[ir].x = exp(-r2/wg2) / (wg*sqrt(2.*M_PI));
    dev_dd_.b[ir].y = 0.0;
}


/** 
 * Initialize the submatrices F and CE;
 * each thread sets nonzero values on a single row;
*/
__device__ void init_F_CE(cuDoubleComplex* values, int* columns, int* rows)
{
    uint32_t iv = threadIdx.x;
    uint32_t ix = blockIdx.x;
    double ih = 1./(2.*dev_dd_.h);
    double ih3 = 3.*ih;
    double ih4 = 4.*ih;

    uint32_t ir = ix*dev_dd_.Nvx + iv; // row id;
    double v1 = get_v1(dev_dd_.vmax , dev_dd_.dv, iv);

    uint32_t sh, sh_next;

    // N of nonzero elements at ix = 0;
    uint32_t sh_ix0  = (dev_dd_.N_discr+3)*dev_dd_.Nvx_h; 

    // N of nonzero elements at ix = 1,2,...Nx-2
    uint32_t sh_bulk = (dev_dd_.N_discr+1)*dev_dd_.Nvx * (dev_dd_.Nx-2);

    // column shift between variables:
    uint32_t sh_column_var  = dev_dd_.Nvx * dev_dd_.Nx;

    // ******************************************************
    // *** matrix F ***
    // --- F: left boundary ---
    if(ix == 0) 
    {
        if(iv < dev_dd_.Nvx_h)
        {
            sh = (dev_dd_.N_discr+1)*iv;

            columns[sh]  = ir;
            values[sh].x = v1 * (ih3 - dev_dd_.Y[ir]);
            values[sh].y = dev_dd_.w;
            
            columns[sh+1]  = ir + dev_dd_.Nvx;
            values[sh+1].x = -ih4 * v1;
            values[sh+1].y =  0;
            
            columns[sh+2]  = ir + 2*dev_dd_.Nvx;
            values[sh+2].x = ih * v1;
            values[sh+2].y = 0;

            sh_next = sh+3;
        }
        else
        {
            // here, iv >= Nvx_h; also, do not forget the contribution from the matrix S;
            sh = (dev_dd_.N_discr-1)*dev_dd_.Nvx_h + 2*iv; 

            columns[sh]  = ir;
            values[sh].x = - v1 * dev_dd_.Y[ir];
            values[sh].y = dev_dd_.w;

            sh_next = sh+1;
        }
    }
    // --- F: right boundary ---
    else if(ix == dev_dd_.Nx-1) 
    {
        if(iv < dev_dd_.Nvx_h)
        {
            sh = sh_ix0 + sh_bulk + 2*iv; 
            
            columns[sh]  = ir;
            values[sh].x = - v1 * dev_dd_.Y[ir];
            values[sh].y = dev_dd_.w;

            sh_next = sh+1;
        }
        else
        {
            sh = sh_ix0 + sh_bulk + dev_dd_.Nvx + (dev_dd_.N_discr+1)*(iv - dev_dd_.Nvx_h);

            columns[sh]  = ir - 2*dev_dd_.Nvx;
            values[sh].x = -ih * v1;
            values[sh].y = 0;

            columns[sh+1]  = ir - dev_dd_.Nvx;
            values[sh+1].x = ih4 * v1;
            values[sh+1].y =  0;

            columns[sh+2]  = ir;
            values[sh+2].x = - v1 * (ih3 + dev_dd_.Y[ir]);
            values[sh+2].y = dev_dd_.w;

            sh_next = sh+3;
        }
    }
    // --- F: bulk points ---
    else
    {
        sh = sh_ix0 + (dev_dd_.N_discr+1) * (ir - dev_dd_.Nvx); 

        columns[sh]  = ir - dev_dd_.Nvx;
        values[sh].x = ih * v1;
        values[sh].y = 0.0;

        columns[sh+1]  = ir;
        values[sh+1].x = - v1*dev_dd_.Y[ir];
        values[sh+1].y = dev_dd_.w;

        columns[sh+2]  = ir + dev_dd_.Nvx;
        values[sh+2].x = - ih * v1;
        values[sh+2].y = 0.0;

        sh_next = sh+3;
    }

    // ******************************************************
    // *** matrix CE ***
    if(dev_dd_.flag_with_copies)
    {
        columns[sh_next]  = ir + sh_column_var; // diagonal;
    }
    else 
    {
        columns[sh_next]  = ix*dev_dd_.Nvx + sh_column_var; // column;
    }
    values[sh_next].x = 0.0;
    values[sh_next].y = v1*dev_dd_.FB[ir];

    // --- row position ---
    rows[ir] = sh;
}


/** 
 * Initialize the submatrix S;
*/
__device__ void init_S(cuDoubleComplex* values, int* columns, int* rows)
{
    uint32_t iv = threadIdx.x;
    uint32_t ix = blockIdx.x;
    uint32_t ir_loc = ix*dev_dd_.Nvx + iv; 
    uint32_t ir     = ir_loc + dev_dd_.Nvx * dev_dd_.Nx; 
    uint32_t sh;

    // number of nonzero elements in the first half of the matrix A (in matrices F and CE):
    uint32_t sh_start = dev_dd_.Nvx * (dev_dd_.Nx*(dev_dd_.N_discr+1) + (1-dev_dd_.N_discr));

    if(dev_dd_.flag_with_copies)
    {
        sh = sh_start + ir_loc * (dev_dd_.Nvx + 1) + dev_dd_.Nvx;
        rows[ir] = sh - dev_dd_.Nvx;
        columns[sh]  = ir; 
        values[sh].x = 0.0;
        values[sh].y = dev_dd_.w;
    }
    else
    {
        sh = sh_start + 2* ix * dev_dd_.Nvx + dev_dd_.Nvx + iv;
        if(iv == 0)
            rows[ir] = sh - dev_dd_.Nvx;
        else
            rows[ir] = sh;
        columns[sh]  = ir; 
        values[sh].x = 0.0;
        values[sh].y = dev_dd_.w;
        // printf("iv, ix, ir, sh: %d, %d, %d, %d\n", iv, ix, ir, sh);
        printf("rows[%d] = %d\n", ir, rows[ir]);
    }
    // else if(iv == 0)
    // {
    //     sh = sh_start + ix * (dev_dd_.Nvx + 1) + dev_dd_.Nvx;
    //     rows[ir] = sh - dev_dd_.Nvx;
    //     columns[sh]  = ir; 
    //     values[sh].x = 0.0;
    //     values[sh].y = dev_dd_.w;
    //     printf("iv, ix, ir, sh: %d, %d, %d, %d\n", iv, ix, ir, sh);
    // }
    
}


/** 
 * Initialize the submatrices Cf;
 * each thread sets one nonzero value in a row;
 * each CUDA block sets the velocity submatrix for a single ix.
*/
__device__ void init_Cf(cuDoubleComplex* values, int* columns)
{
    uint32_t iv_col = threadIdx.x; // here, it is a column index within Nv x Nv submatrix;
    uint32_t ix = blockIdx.x; 
    uint32_t ir_loc = ix*dev_dd_.Nvx; 
    double v1 = get_v1(dev_dd_.vmax , dev_dd_.dv, iv_col);
    uint32_t sh;

    // number of nonzero elements in the first half of the matrix A (in matrices F and CE):
    uint32_t sh_start = dev_dd_.Nvx * (dev_dd_.Nx*(dev_dd_.N_discr+1) + (1-dev_dd_.N_discr));

    if(dev_dd_.flag_with_copies)
    {
        for(uint32_t iv_row = 0; iv_row < dev_dd_.Nvx; iv_row++)
        {
            sh = sh_start + (ir_loc + iv_row) * (dev_dd_.Nvx + 1) + iv_col;  
            columns[sh]  = ir_loc + iv_col;
            values[sh].x = 0.0;
            values[sh].y = v1 * dev_dd_.FB[ir_loc + iv_col];
        }
    }
    else
    {
        sh = sh_start + 2 * ix * dev_dd_.Nvx + iv_col; 
        columns[sh]  = ir_loc + iv_col;
        values[sh].x = 0.0;
        values[sh].y = v1 * dev_dd_.FB[ir_loc + iv_col];
    }
    // else
    // {
    //     sh = sh_start + ix * (dev_dd_.Nvx+1) + iv_col; 
    //     columns[sh]  = ir_loc + iv_col;
    //     values[sh].x = 0.0;
    //     values[sh].y = v1 * dev_dd_.FB[ir_loc + iv_col];
    // }
}


/** 
 * Initialize the submatrices F, CE and S of the matrix A;
 * each thread sets nonzero values on a single row;
*/
__global__ void init_A_F_CE_S()
{
    init_F_CE(dev_dd_.A.values, dev_dd_.A.columns, dev_dd_.A.rows);
    init_S(dev_dd_.A.values, dev_dd_.A.columns, dev_dd_.A.rows);
}


/** 
 * Initialize the submatrix Cf of the matrix A;
 * each thread sets one nonzero value in a row;
 * each CUDA block sets the velocity submatrix for a single ix.
*/
__global__ void init_A_Cf()
{
    init_Cf(dev_dd_.A.values, dev_dd_.A.columns);
}

__global__ void complete_A()
{
    dev_dd_.A.rows[dev_dd_.A.N] = dev_dd_.A.Nnz;
}


__global__ void init_check()
{
    printf("-----------------------------\n");
    printf("Nx = %d\n", dev_dd_.Nx);
    printf("Nv = %d\n", dev_dd_.Nvx);
    printf("N = %d\n", dev_dd_.A.N);
    printf("A[0].real = %0.3e\n", dev_dd_.A.values[0].x);
}


