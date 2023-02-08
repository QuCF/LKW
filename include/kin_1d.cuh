#pragma once

#include "LA.cuh"
#include "../device/dev_kin_1d.cuh"

// -------------------------------------------------------------------
// --- Linear kinetic waves in a 1-D electron plasma (skin effect) ---
// -------------------------------------------------------------------
struct KW
{
protected:
    YMatrix<ycomplex> L_; // Preconditioner 
    KDATA dd_; // plasma parameters

    double coef_superposition_;

    std::shared_ptr<double[]> x_; // normalized to kx spatial grid;
    std::shared_ptr<double[]> v_; // normalized velocity grid;

    std::shared_ptr<double[]> rx_; // normalized to 1 spatial grid;
    std::shared_ptr<double[]> rv_; // normalized to 1 velocity grid;

    std::shared_ptr<double[]> T_;   // normalized temperature profile;
    std::shared_ptr<double[]> den_; // normalized density profile;

    double Tref_;    // reference electron temperature (erg);
    double den_ref_; // reference density profile (cm-3);
    double wp_;  // reference plasma frequency (1/s);
    double ld_;  // reference Debye length (cm);
    double vth_; // reference thermal speed (cm/s);

    std::string gl_path_out_;   // path to the output files;
    std::string hdf5_name_out_; // name of the output .hdf5 file;

    YHDF5 f_;

public:

    /**
     * nx = log2 of spatial grid size;
     * nv = log2 of velocity grid size;
     * Tref - reference electron temperature (erg);
     * den_ref - reference density profile (cm-3);
     * ND - number of Debye lengths in the spatial grid (defines the length of the grid);
     * Nth - number of thermal velocities in the velocity grid;
     * wa - normalized (to wp_) antenna frequency;
     * flag_with_copies: true means that the state vector will be filled with the copies of the electric field;
    */
    KW(
        YCU nx, YCU nv, 
        YCD Tref, YCD den_ref, 
        YCU ND, YCU Nth, 
        YCD wa, 
        YCB flag_with_copies = true
    ) : Tref_(Tref), 
        den_ref_(den_ref),
        gl_path_out_("../../results/LKW-1D-results/"),
        hdf5_name_out_("out.hdf5")
    {
        using namespace std;
        Constants cc;

        dd_.set_to_zero();
        dd_.w = wa;
        dd_.flag_with_copies = flag_with_copies;

        dd_.nx = nx;  dd_.Nx  = 1 << dd_.nx;
        dd_.nvx = nv; dd_.Nvx = 1 << dd_.nvx;
        
        dd_.Nvx_h = 1 << (dd_.nvx - 1);

        dd_.Nvars = 2;
        dd_.A.N = dd_.Nx * dd_.Nvx * dd_.Nvars;

        dd_.N_discr = 3;

        // // number of nonzero elements:
        // dd_.A.Nnz = 3*dd_.Nx*dd_.Nvx - 2*dd_.Nvx; // submatrix F
        // dd_.A.Nnz += 2*dd_.Nx*dd_.Nvx; // submatrices S and CE;
        // if(dd_.flag_with_copies) // submatrices Cf;
        //     dd_.A.Nnz += dd_.Nx*dd_.Nvx*dd_.Nvx; 
        // else
        //     dd_.A.Nnz += dd_.Nx*dd_.Nvx; 

        // number of nonzero elements:
        dd_.A.Nnz = dd_.Nvx * (dd_.Nx*(dd_.N_discr+1) + (1 - dd_.N_discr)); 
        if(dd_.flag_with_copies) 
            dd_.A.Nnz += (dd_.Nvx+1) * dd_.Nx * dd_.Nvx; 
        else
            dd_.A.Nnz += 2 * dd_.Nx * dd_.Nvx;
            // dd_.A.Nnz += dd_.Nx * (dd_.Nvx+1);

        double temp = 4*cc.pi_*pow(cc.e_,2)*den_ref_;
        wp_  = sqrt(temp/cc.me_);
        ld_  = sqrt(Tref_/temp);
        vth_ = ld_ * wp_;

        // --- spatial and velocity grids ---
        dd_.xmax = ND;  // for the normalized to ld_  x-grid;
        dd_.vmax = Nth; // for the normalized to vth_ v-grid;
        double h1, dv1; // steps for the normalized to 1 grids;

        dd_.h  = dd_.xmax / (dd_.Nx - 1);
        dd_.dv = 2.*dd_.vmax / (dd_.Nvx - 1); // negative and positive velocities;

        h1  = dd_.h / dd_.xmax;
        dv1 = dd_.dv / dd_.vmax;

        x_  = shared_ptr<double[]>(new double[dd_.Nx]);
        rx_ = shared_ptr<double[]>(new double[dd_.Nx]);
        for(uint32_t ii = 0; ii < dd_.Nx; ii++){
            x_[ii]  = get_x1(dd_.h, ii);
            rx_[ii] = get_x1(   h1, ii);
        }

        v_  = shared_ptr<double[]>(new double[dd_.Nvx]);
        rv_ = shared_ptr<double[]>(new double[dd_.Nvx]);
        for(uint32_t ii = 0; ii < dd_.Nvx; ii++){
            v_[ii]  = get_v1(dd_.vmax, dd_.dv, ii);
            rv_[ii] = get_v1(       1,    dv1, ii);
        }

        printf("--------------------------------------------------\n");
        printf("------------------ Matrix parameters -------------\n");
        printf("--------------------------------------------------\n");
        printf("\tNx = %d\n", dd_.Nx);
        printf("\tNv = %d\n", dd_.Nvx);
        printf("Number of rows in the matrix: %d\n", dd_.A.N);
        printf("Number of nonzero elements: %d\n", dd_.A.Nnz);
        printf("------------------ Plasma parameters -------------\n");
        printf("\tTref[erg] = %0.3e,   Tref[K] = %0.3e\n", Tref_, Tref_/cc.kB_);
        printf("\tden-ref[cm-3] = %0.3e\n", den_ref_);
        printf("\twp[s-1] = %0.3e\n", wp_);
        printf("\tld[cm] = %0.3e\n", ld_);
        printf("\tvth[cm/s] = %0.3e,   vth/c = %0.3e\n", vth_, vth_/cc.c_light_);
        printf("------------------ Normalized parameters -------------\n");
        printf("antenna frequency (norm. to wp): \t%0.3f\n", dd_.w);
        printf("spatial step (norm. to ld): \t%0.3e\n", dd_.h);
        printf("velocity step (norm. to vth): \t%0.3e\n", dd_.dv);
        printf("\txmax/ld = %0.1f,  \tfull size [cm] = %0.3e\n",  x_[dd_.Nx-1], x_[dd_.Nx-1] * ld_);
        printf("\tvmax/vth = %0.1f, \tfull size [cm/s] = %0.3e\n", v_[dd_.Nvx-1], v_[dd_.Nvx-1] * vth_);
        printf("--------------------------------------------------\n\n");

        // --- Create the HDF5 file ---
        f_.create(gl_path_out_ + hdf5_name_out_);
        f_.add_group("basic");
        f_.add_group("grids");
        f_.add_group("profiles");
        f_.add_group("matrices");

        // date of a simulation:
        string str_date_time;
        YMIX::get_current_date_time(str_date_time);
        f_.add_scalar(str_date_time, "date-of-simulation", "basic");
        f_.add_scalar(filesystem::current_path(), "launch-path", "basic");
        f_.add_scalar(dd_.flag_with_copies, "flag-copies", "basic");

        // save the grids:
        f_.add_array(x_.get(), dd_.Nx, std::string("x"), "grids");
        f_.add_array(v_.get(), dd_.Nvx, std::string("v"), "grids");
        f_.add_array(rx_.get(), dd_.Nx, std::string("rx"), "grids");
        f_.add_array(rv_.get(), dd_.Nvx, std::string("rv"), "grids");

        // close the file:
        f_.close();
    }


    ~KW()
    {
        clean_device();
    }


    void clean_device()
    {
        dd_.A.clean();
        CUDA_CHECK(cudaFree(dd_.b));
        CUDA_CHECK(cudaFree(dd_.psi));
        CUDA_CHECK(cudaFree(dd_.FB));
        CUDA_CHECK(cudaFree(dd_.Y));
    }


    void init_device()
    {
        printf("--- GPU initialization... ---\n");
        cudaSetDevice(0);

        printf("-> allocating matrices...\n");
        dd_.A.allocate();
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&dd_.b),  
            sizeof(cuDoubleComplex) * dd_.A.N
        ));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&dd_.psi),  
            sizeof(cuDoubleComplex) * dd_.A.N
        ));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&dd_.FB),  
            sizeof(double) * dd_.Nx * dd_.Nvx
        ));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&dd_.Y),  
            sizeof(double) * dd_.Nx * dd_.Nvx
        ));

        printf("-> initializing vectors...\n");
        CUDA_CHECK(cudaMemset(dd_.b, 0, sizeof(cuDoubleComplex) * dd_.A.N));
        CUDA_CHECK(cudaMemset(dd_.psi, 0, sizeof(cuDoubleComplex) * dd_.A.N));

        printf("-> copying constant parameters...\n");
        CUDA_CHECK(cudaMemcpyToSymbol(dev_dd_, &dd_, sizeof(KDATA)));
    }


    void set_background_profiles(YCS id_profile_input)
    {
        bool flag_found = false;
        printf("--- Setting background profiles... ---\n");
        YTimer timer;
        timer.Start();

        std::string id_profile = id_profile_input;
        std::transform(
            id_profile.begin(), id_profile.end(), 
            id_profile.begin(), 
            ::tolower
        );

        T_  = std::shared_ptr<double[]>(new double[dd_.Nx]);
        den_ = std::shared_ptr<double[]>(new double[dd_.Nx]);
        if(id_profile.compare("flat") == 0)
        {
            printf("-> Forming flat profiles...\n");
            for(uint32_t ii = 0; ii < dd_.Nx; ii++){
                T_[ii] = 1.0;
                den_[ii] = 1.0;
            }
            flag_found = true;
        }
        if(id_profile.compare("gauss") == 0)
        {
            double sigma_T = 0.2;
            double sigma_n = 0.1;
            double sigma_T2 = 2.*sigma_T*sigma_T;
            double sigma_n2 = 2.*sigma_n*sigma_n;
            printf("-> Forming Gaussian profiles...\n");
            for(uint32_t ii = 0; ii < dd_.Nx; ii++){
                T_[ii]   = exp(-pow(rx_[ii] - 0.50,2)/sigma_T2);
                den_[ii] = exp(-pow(rx_[ii] - 0.50,2)/sigma_n2);
            }
            flag_found = true;
        }
        if(flag_found)
            build_background_distribution();
        else
        {
            std::cerr << "\t>>> Error: No background profiles found." << std::endl;
        }

        cudaDeviceSynchronize();
        timer.Stop();
        printf("\tDone: total elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }

    /** Form right-hand-side vector. */
    void form_rhs(YCD r0, YCD wg)
    {
        printf("--- Setting the right-hand-side vector... ---\n");
        YTimer timer;
        timer.Start();

        if(dd_.flag_with_copies)
        {
            init_rhs_with_copies<<<dd_.Nx, dd_.Nvx>>>(r0, wg);
        }
        else
        {   
            uint32_t N_threads, N_blocks;
            if(nq_THREADS < dd_.nx)
            {
                N_threads = 1 << nq_THREADS;
                N_blocks = 1 << (dd_.nx - nq_THREADS);
            }
            else
            {
                N_threads = 1 << dd_.nx;
                N_blocks = 1;
            }
            init_rhs_without_copies<<<N_blocks, N_threads>>>(r0, wg);
        }
        
        cudaDeviceSynchronize();
        timer.Stop();
        printf("\tDone: elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }


    void form_matrices()
    {
        using namespace std::complex_literals;

        // using namespace std::string_literals;
        YTimer timer;

        printf("--- Creating matrices... ---\n");
        timer.Start();

        init_A_F_CE_S<<<dd_.Nx, dd_.Nvx>>>();
        init_A_Cf<<<dd_.Nx, dd_.Nvx>>>();  
        complete_A<<<1,1>>>();
        
        cudaDeviceSynchronize();    
        timer.Stop();
        printf("--- Done: elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }


    void find_svd()
    {
        // printf("\n-> Find the condition number of A...\n");
        // LA::cond_number_cusolver(dd_.A);

        // printf("\n-> Find the condition number of AG...\n");
        // LA::cond_number_cusolver(AG_);

        // printf("\n-> Find the condition number of AS...\n");
        // LA::cond_number_cusolver(AS_);
    }


    void save_xv_background_profs_to_hdf5()
    {
        printf("--- Saving the (x,v) background profiles to %s... ---", hdf5_name_out_.c_str()); 
        std::cout << std::endl;
        YTimer timer;
        timer.Start();

        uint32_t NxNv = dd_.Nx*dd_.Nvx;
        auto size_v = sizeof(double) * NxNv;

        double* F = new double[NxNv];
        double* Y = new double[NxNv];
        
        // transfer data from GPU to host:
        CUDA_CHECK(cudaMemcpy(F, dd_.FB, size_v, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(Y, dd_.Y, size_v, cudaMemcpyDeviceToHost));

        // save data in the .hdf5 file:
        f_.open_w();
        f_.add_array(F, NxNv, std::string("F"), "profiles");
        f_.add_array(Y, NxNv, std::string("Y"), "profiles");
        f_.close();

        // remove temporary arrays;
        delete [] F;
        delete [] Y;
        cudaDeviceSynchronize();
        timer.Stop();
        printf("\tDone: elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }


    void save_rhs_to_hdf5()
    {
        printf("--- Saving the right-hand-side vector to %s... ---", hdf5_name_out_.c_str()); 
        std::cout << std::endl;
        YTimer timer;
        timer.Start();

        auto size_v = sizeof(ycomplex) * dd_.A.N;
        ycomplex* b = new ycomplex[dd_.A.N];

        CUDA_CHECK(cudaMemcpy(b, dd_.b, size_v, cudaMemcpyDeviceToHost));

        f_.open_w();
        f_.add_array(b, dd_.A.N, std::string("b"), "profiles");
        f_.close();

        delete [] b;
        cudaDeviceSynchronize();
        timer.Stop();
        printf("\tDone: elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }


    void save_matrices()
    {
        printf("--- Saving the matrices...---"); std::cout << std::endl;
        YTimer timer;
        timer.Start();

        save_one_matrix(dd_.A,   "A");
        // save_one_matrix(AG_, "AG");
        // save_one_matrix(AS_, "AS");

        timer.Stop();
        printf("\tDone: elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }


    void print_grids()
    {
        printf("\n\n1/(2h) = %0.3e\n", 1./(2.*dd_.h));
        printf("--- x-grid ---\n");
        for(uint32_t ix = 0; ix < dd_.Nx; ix++)
        {
            printf("%0.3e  ", x_[ix]);
            if(ix != 0 && ix%18 == 0) printf("\n");
        }

        printf("\n--- v-grid ---\n");
        for(uint32_t iv = 0; iv < dd_.Nvx; iv++)
        {
            printf("%0.3e  ", v_[iv]);
            if(iv != 0 && iv%18 == 0) printf("\n");
        }
        std::cout << "\n" << std::endl;
    }


    void print_matrix()
    {
        YMatrix<ycomplex> A;
        dd_.A.form_dense_matrix(A);

        // double ih = 1./(2.*dd_.h);
        // double ih3 = 3.*ih;
        // double ih4 = 4.*ih;
        // printf("sigma*vmax:  %0.3e\n",   ih*dd_.vmax);
        // printf("4sigma*vmax: %0.3e\n", ih4*dd_.vmax);
        // printf("\n");

        uint32_t idx;
        uint64_t sh_r;
        uint64_t sh_var = dd_.Nx*dd_.Nvx;
        printf("\n");

        // printf("\n --- matrix F_EL ---\n");
        // A.print(0, dd_.Nvx, 0, dd_.Nvx);

        // printf("\n --- matrix F_EL_0 ---\n");
        // A.print(0, dd_.Nvx, dd_.Nvx, 2*dd_.Nvx);

        // printf("\n --- matrix F_EL_1 ---\n");
        // A.print(0, dd_.Nvx, 2*dd_.Nvx, 3*dd_.Nvx);



        // sh_r = (dd_.Nx-1)*dd_.Nvx;
        // printf("\n --- matrix F_ER_1 ---\n");
        // A.print(sh_r, sh_r + dd_.Nvx, sh_r - 2*dd_.Nvx, sh_r - dd_.Nvx);

        // sh_r = (dd_.Nx-1)*dd_.Nvx;
        // printf("\n --- matrix F_ER_0 ---\n");
        // A.print(sh_r, sh_r + dd_.Nvx, sh_r - dd_.Nvx, sh_r);

        // sh_r = (dd_.Nx-1)*dd_.Nvx;
        // printf("\n --- matrix F_ER ---\n");
        // A.print(sh_r, sh_r + dd_.Nvx, sh_r, sh_r + dd_.Nvx);



        // idx = 1;
        // sh_r = idx * dd_.Nvx;
        // printf("\n --- matrix left F_BD ---\n");
        // A.print(sh_r, sh_r + dd_.Nvx, sh_r - dd_.Nvx, sh_r);

        // idx = 1;
        // sh_r = idx * dd_.Nvx;
        // printf("\n --- matrix F_B ---\n");
        // A.print(sh_r, sh_r + dd_.Nvx, sh_r, sh_r + dd_.Nvx);

        // idx = 1;
        // sh_r = idx * dd_.Nvx;
        // printf("\n --- matrix right F_BD ---\n");
        // A.print(sh_r, sh_r + dd_.Nvx, sh_r + dd_.Nvx, sh_r + 2*dd_.Nvx);




        // idx = 7;
        // sh_r = idx * dd_.Nvx;
        // printf("\n --- matrix CE ---\n");
        // A.print(sh_r, sh_r + dd_.Nvx, sh_var + sh_r, sh_var + sh_r + dd_.Nvx);



        idx = 0;
        sh_r = idx * dd_.Nvx;
        printf("\n --- matrix CF ---\n");
        A.print(
            sh_var + sh_r, sh_var + sh_r + dd_.Nvx, 
            sh_r, sh_r + dd_.Nvx
        );
    


        idx = 7;
        sh_r = idx * dd_.Nvx;
        printf("\n --- matrix S ---\n");
        A.print(
            sh_var + sh_r, sh_var + sh_r + dd_.Nvx, 
            sh_var + sh_r, sh_var + sh_r + dd_.Nvx
        );
    }

protected:

    void build_background_distribution()
    {
        printf("-> Building (x,v) background distributions...\n");
        YTimer timer;
        timer.Start();

        double *dT = new double[dd_.Nx]; 
        double *dn = new double[dd_.Nx]; 
        
        // find derivatives:
        YMATH::find_der(T_.get(),   dd_.h, dd_.Nx, dT);
        YMATH::find_der(den_.get(), dd_.h, dd_.Nx, dn);
        
        // save the x-profiles:
        f_.open_w();
        f_.add_array(T_.get(),   dd_.Nx, "T", "profiles");
        f_.add_array(den_.get(), dd_.Nx, "n", "profiles");
        f_.add_array(dT, dd_.Nx, "der-T", "profiles");
        f_.add_array(dn, dd_.Nx, "der-n", "profiles");
        f_.close();

        // save the x-background profiles on the GPU:
        double* dev_T;
        double* dev_n;
        double* dev_dT;
        double* dev_dn;
        auto size_v = sizeof(double) * dd_.Nx;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_T), size_v));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_n), size_v));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_dT), size_v));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_dn), size_v));

        CUDA_CHECK(cudaMemcpy(dev_T,   T_.get(), size_v, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_n, den_.get(), size_v, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_dT,      dT, size_v, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_dn,      dn, size_v, cudaMemcpyHostToDevice));

        // initialize the (x,v)-profiles on the GPU:
        init_background_distribution<<<dd_.Nx, dd_.Nvx>>>(dev_T, dev_n, dev_dT, dev_dn);

        // remove the x-background profiles from the GPU:
        CUDA_CHECK(cudaFree(dev_T));
        CUDA_CHECK(cudaFree(dev_n));
        CUDA_CHECK(cudaFree(dev_dT)); 
        CUDA_CHECK(cudaFree(dev_dn)); 
        delete [] dT;
        delete [] dn;

        cudaDeviceSynchronize();
        timer.Stop();
        printf("\tDone: elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }


    void save_one_matrix(SpMatrixC& A, YCS name)
    {
        using namespace std;

        uint32_t Nr = A.N+1;

        ycomplex* values_host = new ycomplex[A.Nnz];
        int* columns_host     = new int[A.Nnz];
        int* rows_host        = new int[Nr];

        auto size_complex = sizeof(ycomplex) * A.Nnz;
        auto size_columns = sizeof(int) * A.Nnz;
        auto size_rows    = sizeof(int) * Nr;

        CUDA_CHECK(cudaMemcpy(values_host,  A.values,  size_complex, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(columns_host, A.columns, size_columns, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(rows_host,    A.rows,    size_rows,    cudaMemcpyDeviceToHost));

        f_.open_w();
        f_.add_scalar(A.N,   name + "-N"s,   "matrices"s);
        f_.add_scalar(A.Nnz, name + "-Nnz"s, "matrices"s);
        f_.add_array(values_host,  A.Nnz, name + "-values"s,  "matrices"s);
        f_.add_array(columns_host, A.Nnz, name + "-columns"s, "matrices"s);
        f_.add_array(rows_host,       Nr, name + "-rows"s,    "matrices"s);
        f_.close();

        delete [] values_host;
        delete [] columns_host;
        delete [] rows_host;
    }


    void form_submatrix_F(YMatrix<ycomplex> &A)
    {
        // using namespace std::complex_literals;
        // // using namespace std::string_literals;

        // uint32_t idx;
        // uint64_t sh_r;
        // ycomplex wi = 1i*dd_.w;
        // double ih = 1./(2.*dd_.h);
        // double ih3 = 3.*ih;
        // double ih4 = 4.*ih;

        // // F_EL, F_EL_0, F_EL_1:
        // idx = 0;
        // for(uint32_t iv = 0; iv < dd_.Nvx_h; iv++)
        // {
        //     A(iv, iv)         =   wi + v_[iv] * (ih3 - Y(idx, iv));
        //     A(iv, dd_.Nvx + iv)   = -ih4 * v_[iv];
        //     A(iv, 2*dd_.Nvx + iv) =   ih * v_[iv];
        // }
        // for(uint32_t iv = dd_.Nvx_h; iv < dd_.Nvx; iv++)
        //     A(iv, iv) = wi - v_[iv] * dd_.Y(idx,iv);

        // // F_ER, F_ER_0, F_ER_1:
        // idx = dd_.Nx-1;
        // sh_r = idx * dd_.Nvx;
        // for(uint32_t iv = 0; iv < dd_.Nvx_h; iv++)
        //     A(sh_r + iv, sh_r + iv) = wi - v_[iv] * dd_.Y(idx, iv);
        // for(uint32_t iv = dd_.Nvx_h; iv < dd_.Nvx; iv++)
        // {
        //     A(sh_r + iv, sh_r - 2*dd_.Nvx + iv) = -ih * v_[iv];
        //     A(sh_r + iv, sh_r - dd_.Nvx + iv)   = ih4 * v_[iv];
        //     A(sh_r + iv, sh_r + iv)         =  wi - v_[iv] * (ih3 + dd_.Y(idx, iv));
        // }

        // // left FBD, FB, right FBD:
        // for(uint ix = 1; ix < (dd_.Nx-1); ix++)
        // {
        //     sh_r = ix * dd_.Nvx;
        //     for(uint32_t iv = 0; iv < dd_.Nvx; iv++)
        //     {
        //         A(sh_r + iv, sh_r - dd_.Nvx + iv) =  ih * v_[iv];
        //         A(sh_r + iv, sh_r + iv)       =  wi - v_[iv] * dd_.Y(ix, iv);
        //         A(sh_r + iv, sh_r + dd_.Nvx + iv) = -ih * v_[iv];
        //     }
        // }
    }


    void form_submatrix_CE(YMatrix<ycomplex> &A)
    {   
        // using namespace std::complex_literals;
        // uint64_t sh_r;
        // uint64_t sh_var = dd_.Nx*dd_.Nvx;

        // for(uint ix = 0; ix < dd_.Nx; ix++)
        // {
        //     sh_r = ix * dd_.Nvx;
        //     for(uint32_t iv = 0; iv < dd_.Nvx; iv++)
        //         A(sh_r + iv, sh_var + sh_r + iv) = 1i*v_[iv]*dd_.FB(ix,iv);
        // }
    }


    void form_submatrix_CF(YMatrix<ycomplex> &A)
    {
        // using namespace std::complex_literals;
        // uint64_t sh_r;
        // uint64_t sh_var = dd_.Nx*dd_.Nvx;
        // for(uint ix = 0; ix < dd_.Nx; ix++)
        // {
        //     sh_r = ix * dd_.Nvx;
        //     for(uint32_t ivr = 0; ivr < dd_.Nvx; ivr++)
        //         for(uint32_t ivc = 0; ivc < dd_.Nvx; ivc++)
        //             A(sh_var + sh_r + ivr, sh_r + ivc) = 1i*v_[ivc]*dd_.FB(ix,ivc);
        // }
    }


    void form_submatrix_S(YMatrix<ycomplex> &A)
    {
        // using namespace std::complex_literals;
        // ycomplex wi = 1i*dd_.w;
        // uint64_t sh_r;
        // uint64_t sh_var = dd_.Nx*dd_.Nvx;
        // for(uint ix = 0; ix < dd_.Nx; ix++)
        // {
        //     sh_r = ix * dd_.Nvx;
        //     for(uint32_t iv = 0; iv < dd_.Nvx; iv++)
        //         A(sh_var + sh_r + iv, sh_var + sh_r + iv) = wi;
        // }
    }


    void add_correction(YMatrix<ycomplex> &A, YCCo coef_corr)
    {
        // uint64_t sh = 0;
        // uint64_t shx = 0;
        // for(uint ivar = 0; ivar < dd_.Nvars; ivar++)
        // {
        //     sh = ivar * dd_.Nx * dd_.Nvx;
        //     for(uint ix = 0; ix < dd_.Nx; ix++)
        //     {
        //         shx = ix * dd_.Nvx;
        //         for(uint32_t iv = 0; iv < dd_.Nvx; iv++)
        //             A(sh + shx + iv, sh + shx + iv) += coef_corr;
        //     }
        // }
    }


};