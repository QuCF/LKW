#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <stdexcept>

#include <filesystem>

#include <memory>
#include <complex>
#include <math.h>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <iterator>
#include <algorithm>

#include <time.h>
#include <stdlib.h> 
#include <stdio.h> 

#include <limits>
#include <chrono>
#include <stdarg.h>
#include <numeric>
#include <unistd.h>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include "cusolver_utils.h"
#include "H5Cpp.h"

// ------------------------------------------
// --- Type synonyms --- 
// ------------------------------------------
#define YCB const bool&

#define YCsh  const short&
#define YCVsh const std::vector<short>&
#define YVsh  std::vector<short>&
#define YVshv std::vector<short>

#define YCU  const uint32_t&
#define YCVU const std::vector<uint32_t>&
#define YVU  std::vector<uint32_t>&
#define YVUv std::vector<uint32_t>

#define YCUL const uint64_t&

#define YCI  const int&
#define YCVI const std::vector<int>&
#define YVI  std::vector<int>&
#define YVIv std::vector<int>

#define YCVT const std::vector<T>&

#define YCS  const std::string&
#define YS   std::string&
#define YCVS const std::vector<std::string>&
#define YVSv std::vector<std::string>

#define YCD const double&
#define YCVD const std::vector<double>&
#define YVD  std::vector<double>&
#define YVDv std::vector<double>

#define ycomplex std::complex<double>
#define YCCo const ycomplex&

#define YCCM const std::shared_ptr<const YMatrix>

// ------------------------------------------
// --- Constants --- 
// ------------------------------------------
#define ZERO_ERROR 1e-14
#define nq_THREADS  7
struct Constants
{
    double c_light_ = 299792458 * 1e2; // speed of light (cm/s);
    double me_ = 9.1093837e-28; // electro mass (g);
    double e_ = 4.8032e-10; // electron charge (statcoul);
    double ev_ = 1.602176634e-12; // electronvolt (erg);
    double kB_ = 1.3807e-16; // Boltzmann constant (erg/K);
    double pi_ = M_PI;
    double pi2_ = 2*M_PI;
};

// ------------------------------------------
// --- CPU Timer --- 
// ------------------------------------------
struct YTimer{
public:           
    void Start(){
        start_ = std::chrono::steady_clock::now();
    }
    void Stop(){
        end_ = std::chrono::steady_clock::now();
    }
    // void StartPrint(YCS mess)
    // {
    //     Start();
    //     print_log_flush(mess);
    // }
    // void StopPrint()
    // {
    //     Stop();

    //     std::ostringstream oss;
    //     oss << std::scientific << std::setprecision(3) 
    //         << get_dur_s() << " s" << std::endl;
    //     print_log_flush(oss.str());
    // }
    double get_dur(){
        std::chrono::duration<double> dur_seconds = end_ - start_;
        return 1000.*dur_seconds.count(); // in ms
    }
    double get_dur_s(){
        std::chrono::duration<double> dur_seconds = end_ - start_;
        return dur_seconds.count(); // in seconds
    }
    std::string get_dur_str_ms(){
        std::ostringstream ostr;
        ostr << std::scientific << std::setprecision(3) << get_dur() << " ms";
        return ostr.str();
    }
    std::string get_dur_str_s(){
        std::ostringstream ostr;
        ostr << std::scientific << std::setprecision(3) << get_dur()/1000. << " s";
        return ostr.str();
    }
protected:
    std::chrono::time_point<std::chrono::steady_clock> start_;
    std::chrono::time_point<std::chrono::steady_clock> end_;
};


// ------------------------------------------
// --- Matrix --- 
// ------------------------------------------
template<typename T>
class YMatrix{
    protected:
        uint64_t Nr_, Nc_;
        T** a_ = nullptr;
        std::shared_ptr<T[]> a_1d_ = nullptr;

        /**
         *  Nonzero matrix elements as a 1-D array ordered in the column-major format.
         *  nz_[k] = A_{ir, ic}
         *  ids_[k] = ir + ic*Nr_;
         */
        T* nz_ = nullptr;

        /** 
         * Indices to nonzero matrix elements.
        */
        uint64_t* ids_ = nullptr;

        /**
         * Number of nonzero elements;
        */
       uint64_t Nnz_;

    public:
        /**
         * @brief Create an empty matrix object whithout reserving any memory.
         */
        YMatrix(){
            set_prec();
            Nr_ = 0;
            Nc_ = 0;
        }

        /**
         * @brief Create a zero matrix (\p Nrows, \p Ncols).
         * @param Nrows is a number of rows in a new matrix.
         * @param Ncols is a number of columns.
         * @param flag_zero if true, set elements to zero.
         */
        YMatrix(YCU Nrows, YCU Ncols, YCB flag_zero=false)
            :Nr_(Nrows), Nc_(Ncols)
        {
            set_prec();
            create_new();
            if(flag_zero) set_zeros();
        }

        /**
         * @brief Copy a matrix \p M. 
         */
        YMatrix(YCCM oo)
        {
            set_prec();
            Nr_ = oo->Nr_;
            Nc_ = oo->Nc_;
            create_new();
            for(uint64_t i = 0; i < Nr_; ++i)
                for(uint64_t k = 0; k < Nc_; ++k)
                    a_[i][k] = oo->a_[i][k];
        }

        ~YMatrix(){
            clear();
        }


        /**
         * @brief Free the memory occupied by the matrix.
         */
        void clear()
        {
            Nnz_ = 0;
            if(Nc_ != 0 || Nr_ != 0)
            {
                // cout << "delete a matrix" << endl;
                for(uint64_t i = 0; i < Nr_; ++i) 
                    delete [] a_[i];
                delete [] a_;
                Nc_ = 0;
                Nr_ = 0;
                a_1d_.reset();
            }
            if(!nz_ || !ids_)
            {
                delete [] nz_;   nz_  = nullptr;
                delete [] ids_;  ids_ = nullptr;
            }
        }


        /**
         * @brief Create an empty matrix.
         */
        void create(YCU Nrows, YCU Ncols){
            clear();
            Nc_ = Ncols;
            Nr_ = Nrows;
            create_new();
        }

        /**
         * @brief Create a zero matrix.
         */
        void zeros(YCU Nrows, YCU Ncols){
            clear();
            Nc_ = Ncols;
            Nr_ = Nrows;
            create_new();

            set_zeros();
        }

        /**
         * @brief Create an identity matrix.
         */
        void create_identity_matrix(YCU Nrows, YCU Ncols){
            clear();
            Nc_ = Ncols;
            Nr_ = Nrows;
            create_new();

            for(uint64_t i = 0; i < Nr_; ++i)
                for(uint64_t k = 0; k < Nc_; ++k)
                    if(i == k) a_[i][k] = 1.0;
                    else       a_[i][k] = 0.0;
        }

        /**
         * @brief Gives a raw pointer of the matrix.
         */
        T** get_pointer(){ return a_; }

        /**
         * @brief Create 1-D vector that represents the matrix in the column-major order.
         * i_column * Nrow + i_row
         */
        T* get_1d_column_major(){
            if(!a_1d_ && Nc_ > 0 && Nr_ > 0)
            {
                a_1d_ = std::shared_ptr<T[]>(new T[Nc_*Nr_]);
                for(unsigned ic = 0; ic < Nc_; ic++)
                    for(unsigned ir = 0; ir < Nr_; ir++)
                        a_1d_[ic*Nr_ + ir] = a_[ir][ic];
            }
            return a_1d_.get();
        }

        inline uint64_t get_nr(){return Nr_;}
        inline uint64_t get_nc(){return Nc_;}

        inline
        T& operator()(YCU id_row, YCU id_col)
        {
            if (id_row >= Nr_)
            {
                std::cerr << "\nError: id-row = " << id_row << ", while N-rows = " << Nr_ << std::endl;
                exit(-1);
            }
            if (id_col >= Nc_)
            {
                std::cerr << "\nError: id-column = " << id_col << ", while N-columns = " << Nc_ << std::endl;
                exit(-1);
            }
            return a_[id_row][id_col];
        }

        inline
        T operator()(YCU id_row, YCU id_col) const
        {
            if (id_row >= Nr_)
            {
                std::cerr << "\nError: id-row = " << id_row << ", while N-rows = " << Nr_ << std::endl;
                exit(-1);
            }
            if (id_col >= Nc_)
            {
                std::cerr << "\nError: id-column = " << id_col << ", while N-columns = " << Nc_ << std::endl;
                exit(-1);
            }
            return a_[id_row][id_col];
        }

        /**
         * @param prec precision of matrix elements.
         * @param flag_scientific if true, then print in the scientific notation.
         * @param wc extra width of every column.
         */
        void set_prec(bool flag_scientific=false, YCI prec=3, YCI wc=2){
            if(flag_scientific)
                std::cout << std::scientific << std::setprecision(prec) << std::setw(prec+wc);
            else
                std::cout << std::fixed << std::setprecision(prec) << std::setw(prec+wc + 1);
        }

        /**
         * @brief Print the matrix.
         */
        void print(){
            for(unsigned i = 0; i < Nr_; ++i)
            {
                for(unsigned k = 0; k < Nc_; ++k)
                    std::cout << a_[i][k] << " ";
                std::cout << "\n";
            }
            std::cout << std::endl;
        }

        /**
         * Print a submatrix A[i_row1:i_row2, i_col1:i_col2] without including i_row2 and i_col2.
        */
        void print(YCU i_row1, YCU i_row2, YCU i_col1, YCU i_col2){
            for(unsigned i = i_row1; i < i_row2; ++i)
            {
                for(unsigned k = i_col1; k < i_col2; ++k)
                    std::cout << a_[i][k] << " ";
                std::cout << "\n";
            }
            std::cout << std::endl;
        }


        /**
         * Count the current number of nonzero elements in the matrix.
        */
        void count_nz()
        {
            Nnz_ = 0;
            for(uint64_t ir = 0; ir < Nr_; ir++)
                for(uint64_t ic = 0; ic < Nc_; ic++)
                    if(abs(a_[ir][ic]) > ZERO_ERROR)
                        ++Nnz_;
        }

        uint64_t get_Nnz(){return Nnz_;}

        /**
         * Save an array with nonzero values of the matrix and 
         * an array with indices to these elements;
        */
        void form_sparse_format()
        {
            if(!nz_ || !ids_)
            {
                delete [] nz_;
                delete [] ids_;
            }

            nz_ = new T[Nnz_];
            ids_ = new uint64_t[Nnz_];

            uint64_t counter = 0;
            for(uint64_t ic = 0; ic < Nc_; ic++)
                for(uint64_t ir = 0; ir < Nr_; ir++)
                    if(abs(a_[ir][ic]) > ZERO_ERROR)
                    {
                        nz_[counter] = a_[ir][ic];
                        ids_[counter] = ir + Nr_ * ic;
                        ++counter;
                    }
        }

        T* get_nz_array(){return nz_;}
        uint64_t* get_ids_array(){return ids_;}

    protected:
        /**
         * @brief Reserve memory for a new matrix of a known size.
         * The function does not check whether the matrix has been already initialized.
         */
        void create_new()
        {
            Nnz_ = 0;
            a_ = new T*[Nr_];
            for(uint64_t i = 0; i < Nr_; ++i)
                a_[i] = new T[Nc_];
        }

        /**
         * @brief Set elements to zeros.
         * The function does not check where the matrix has been already initialized.
         */
        void set_zeros()
        {
            Nnz_ = 0;
            for(uint64_t i = 0; i < Nr_; ++i)
                for(uint64_t k = 0; k < Nc_; ++k)
                    a_[i][k] = 0.0;
        }
};


// ------------------------------------------
// --- Compact finite difference method --- 
// ------------------------------------------
class YMATH
{
public:
    /**
     * @param y original function;
     * @param h spatial step;
     * @param N number of points in the spatial grid;
     * @param der_y resulting derivaritive of @param y. 
     * The array @param der_y must be initialized outside the function.
    */
    static void find_der(const double* y, YCD h, YCU N, double* der_y)
    {
        double ih = 1./(2.*h);
        uint32_t l = N-1;

        der_y[0] = ih * (-3.*y[0] + 4.*y[1] - y[2]);
        der_y[l] = ih * (y[l-2] - 4.*y[l-1] + 3*y[l]);

        for(uint32_t ii = 1; ii < l; ii++)
            der_y[ii] = ih*(y[ii+1] - y[ii-1]);
    }

};


// ------------------------------------------
// --- To deal with hdf5 files --- 
// ------------------------------------------
struct YHDF5
{
    /**
     * @brief Create and open an .hdf5 file with a name \p fname. 
     */
    void create(YCS fname);
    void close();

    inline void set_name(YCS fname){ name_ = fname; }

    /**
     * @brief Open an .hdf5 file with a name \p fname only to read it. 
     */
    void open_r();

    /**
     * @brief Open an .hdf5 file with a name \p fname to write-read it. 
     */
    void open_w();

    /**
     * @brief Add a group (folder) with a name \p gname to an already opened file.
     */
    void add_group(YCS gname);

    /**
     * @brief Add a dataset with a name \p dname, where a scalar \p v is to be written.
     * The dataset is put to a group \p gname.
     */
    template<class T>
    void add_scalar(const T& v, YCS dname, YCS gname)
    {
        if(!flag_opened) 
            throw "HDF5 File " + name_ + 
                " is not opened to add a dataset " + dname + " to a group " + gname;

        H5::Group grp(f_->openGroup(gname));
        write(v, dname, grp);
    }

    template<class T>
    void add_vector(const std::vector<T>& v, YCS dname, YCS gname)
    {
        if(!flag_opened) 
            throw "HDF5 File " + name_ + 
                " is not opened to add a dataset " + dname + " to a group " + gname;

        H5::Group grp(f_->openGroup(gname));
        write(v, dname, grp);
    }

    template<class T>
    void add_array(const T* v, YCUL N, YCS dname, YCS gname)
    {
        if(!flag_opened) 
            throw "HDF5 File " + name_ + 
                " is not opened to add a dataset " + dname + " to a group " + gname;

        H5::Group grp(f_->openGroup(gname));
        write(v, N, dname, grp);
    }

    void add_array(ycomplex* v, YCUL N, YCS dname, YCS gname)
    {
        if(!flag_opened) 
            throw "HDF5 File " + name_ + 
                " is not opened to add a dataset " + dname + " to a group " + gname;

        H5::Group grp(f_->openGroup(gname));
        write(v, N, dname, grp);
    }


    template<class T>
    void read_scalar(T& v, YCS dname, YCS gname)
    {
        if(!flag_opened) 
            throw "HDF5 File " + name_ + 
                " is not opened to read a dataset " + dname + " from a group " + gname;
        H5::Group grp(f_->openGroup(gname));
        read(v, dname, grp);
    }

    template<class T>
    void read_vector(std::vector<T>& v, YCS dname, YCS gname)
    {
        if(!flag_opened) 
            throw "HDF5 File " + name_ + 
                " is not opened to add a dataset " + dname + " to a group " + gname;
        H5::Group grp(f_->openGroup(gname));
        read(v, dname, grp);
    }


    protected:
        inline void write(YCS v, YCS dname, H5::Group& grp)
        {
            auto dspace = H5::DataSpace(H5S_SCALAR);
            H5::StrType dtype(H5::PredType::C_S1, v.size()+1);
            H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
            dataset.write(v, dtype);
        }
        inline void write(YCI v, YCS dname, H5::Group& grp)
        {
            auto dspace = H5::DataSpace(H5S_SCALAR);
            auto dtype = H5::PredType::NATIVE_INT;
            H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
            dataset.write((int*) &v, dtype);
        }
        inline void write(YCU v, YCS dname, H5::Group& grp)
        {
            auto dspace = H5::DataSpace(H5S_SCALAR);
            auto dtype = H5::PredType::NATIVE_UINT32;
            H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
            dataset.write((uint32_t*) &v, dtype);
        }
        inline void write(YCUL v, YCS dname, H5::Group& grp)
        {
            auto dspace = H5::DataSpace(H5S_SCALAR);
            auto dtype = H5::PredType::NATIVE_UINT64;
            H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
            dataset.write((uint64_t*) &v, dtype);
        }
        inline void write(const double& v, YCS dname, H5::Group& grp)
        {
            auto dspace = H5::DataSpace(H5S_SCALAR);
            auto dtype = H5::PredType::NATIVE_DOUBLE;
            H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
            dataset.write((int*) &v, dtype);
        }

        inline void write(YCVU v, YCS dname, H5::Group& grp)
        {
            hsize_t dims[] = {v.size()};
            H5::DataSpace dspace(1, dims);
            auto dtype = H5::PredType::NATIVE_UINT;
            H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
            dataset.write(&v[0], dtype);
        }
        inline void write(YCVI v, YCS dname, H5::Group& grp)
        {
            hsize_t dims[] = {v.size()};
            H5::DataSpace dspace(1, dims);
            auto dtype = H5::PredType::NATIVE_INT;
            H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
            dataset.write(&v[0], dtype);
        }
        inline void write(const std::vector<double>& v, YCS dname, H5::Group& grp)
        {
            hsize_t dims[] = {v.size()};
            H5::DataSpace dspace(1, dims);
            auto dtype = H5::PredType::NATIVE_DOUBLE;
            H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
            dataset.write(&v[0], dtype);
        }
        inline void write(const std::vector<ycomplex>& v, YCS dname, H5::Group& grp)
        {
            hsize_t dims[] = {v.size()};
            H5::DataSpace dspace(1, dims);

            hid_t dtype = H5Tcreate(H5T_COMPOUND, sizeof(ycomplex));
            H5Tinsert (dtype, "real", 0, H5T_NATIVE_DOUBLE);
            H5Tinsert (dtype, "imag", sizeof(double), H5T_NATIVE_DOUBLE);

            H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
            dataset.write(&v[0], dtype);
        }

        inline void write(const double* v, YCUL N, YCS dname, H5::Group& grp)
        {
            hsize_t dims[] = {N};
            H5::DataSpace dspace(1, dims);
            auto dtype = H5::PredType::NATIVE_DOUBLE;
            H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
            dataset.write(v, dtype);
        }

        inline void write(const uint64_t* v, YCUL N, YCS dname, H5::Group& grp)
        {
            hsize_t dims[] = {N};
            H5::DataSpace dspace(1, dims);
            auto dtype = H5::PredType::NATIVE_UINT64;
            H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
            dataset.write(v, dtype);
        }

        inline void write(const int* v, YCU N, YCS dname, H5::Group& grp)
        {
            hsize_t dims[] = {N};
            H5::DataSpace dspace(1, dims);
            auto dtype = H5::PredType::NATIVE_INT32;
            H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
            dataset.write(v, dtype);
        }

        inline void write(const ycomplex* v, YCUL N, YCS dname, H5::Group& grp)
        {
            hsize_t dims[] = {N};
            H5::DataSpace dspace(1, dims);
  
            hid_t dtype = H5Tcreate(H5T_COMPOUND, sizeof(ycomplex));
            H5Tinsert (dtype, "real", 0, H5T_NATIVE_DOUBLE);
            H5Tinsert (dtype, "imag", sizeof(double), H5T_NATIVE_DOUBLE);

            H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
            dataset.write(v, dtype);
        }

        inline void write(const cuDoubleComplex* v, YCUL N, YCS dname, H5::Group& grp)
        {
            hsize_t dims[] = {N};
            H5::DataSpace dspace(1, dims);
  
            hid_t dtype = H5Tcreate(H5T_COMPOUND, sizeof(ycomplex));
            H5Tinsert (dtype, "real", HOFFSET(cuDoubleComplex,x), H5T_NATIVE_DOUBLE);
            H5Tinsert (dtype, "imag", HOFFSET(cuDoubleComplex,y), H5T_NATIVE_DOUBLE);

            H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
            dataset.write(v, dtype);
        }

        inline void write(short* v, const unsigned long& nr, const unsigned long& nc, YCS dname, H5::Group& grp)
        {
            hsize_t dims[] = {nr, nc};
            H5::DataSpace dspace(2, dims);
            auto dtype = H5::PredType::NATIVE_SHORT;
            H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
            dataset.write(v, dtype);
        }


        template<class T>
        inline void read(T& v, YCS dname, H5::Group& grp)
        {
            H5::DataSet dataset = grp.openDataSet(dname);
            H5::DataType dtype = dataset.getDataType();
            dataset.read(&v, dtype);
        }
        inline void read(YS v, YCS dname, H5::Group& grp)
        {
            H5::DataSet dataset = grp.openDataSet(dname);
            H5::DataType dtype = dataset.getDataType();
            v="";
            dataset.read(v, dtype);
        }
        template<class T>
        inline void read(std::vector<T>& v, YCS dname, H5::Group& grp)
        {
            H5::DataSet dataset = grp.openDataSet(dname);

            H5::DataSpace dataspace = dataset.getSpace();
            int rank = dataspace.getSimpleExtentNdims();
            hsize_t dims_out[rank];
            int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);

            unsigned long long N = 1;
            for(unsigned i_dim = 0; i_dim < rank; i_dim++)
                N *= dims_out[i_dim];
            v = std::vector<T>(N);

            H5::DataType dtype = dataset.getDataType();

            dataset.read(&v[0], dtype, dataspace, dataspace);
        }

    protected:
        std::string name_;
        bool flag_opened;
        H5::H5File* f_;
};


// ------------------------------------------
// --- Different help functions --- 
// ------------------------------------------
class YMIX
{
public:
    static void get_current_date_time(YS line_date_time){
        time_t rawtime;
        struct tm * timeinfo;
        char buffer[80];

        time (&rawtime);
        timeinfo = localtime(&rawtime);
        strftime(buffer, sizeof(buffer), "%m-%d-%Y %H:%M:%S", timeinfo);
        line_date_time = std::string(buffer);
    }
};


// ------------------------------------------
// --- Sparse matrix in the CSR format with complex elements --- 
// ------------------------------------------
struct SpMatrixC{
    uint32_t N; // full size of the matrix (the number of rows);
    uint32_t Nnz; // the total number of nonzero values in the matrix;

    /**
     * [Nnz] nonzero values in the row-major format:
     * all nonzero elements in the first row, 
     * then all nz elements in the second row and so on.
     */  
    cuDoubleComplex* values; 
        
    /**
     * [Nnz] columns of the nonzero values.
    */
    int* columns; 

    /**
     * of size [N+1]; rows[i] indicates where i-th row starts in the array "values";
     * rows[N] = Nnz;
    */
    int* rows; 
                
    void clean()
    {
        CUDA_CHECK(cudaFree(values));
        CUDA_CHECK(cudaFree(columns));
        CUDA_CHECK(cudaFree(rows));
    }

    void allocate()
    {
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&values),  
            sizeof(cuDoubleComplex) * Nnz
        ));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&columns),  
            sizeof(int) * Nnz
        ));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&rows),  
            sizeof(int) * (N + 1)
        ));
    }

    void form_dense_matrix(YMatrix<ycomplex> &A)
    {
        // -------------------------------------------------
        // --- Download the matrix A from GPU ---
        // -------------------------------------------------
        printf("--- Forming dense matrix A from the sparse one... ---"); 
        std::cout << std::endl;
        YTimer timer;
        timer.Start();

        uint32_t Nr = N+1;

        ycomplex* values_host = new ycomplex[Nnz];
        int* columns_host     = new int[Nnz];
        int* rows_host        = new int[Nr];

        auto size_complex = sizeof(ycomplex) * Nnz;
        auto size_columns = sizeof(int) * Nnz;
        auto size_rows    = sizeof(int) * Nr;

        CUDA_CHECK(cudaMemcpy(values_host,  values,  size_complex, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(columns_host, columns, size_columns, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(rows_host,    rows,    size_rows,    cudaMemcpyDeviceToHost));

        // -------------------------------------------------------
        // --- Transform the sparse format to the dense format ---
        // -------------------------------------------------------
        A.zeros(N, N);
        for(uint32_t ir = 0; ir < N; ir++)
            for(uint32_t i_nz = rows_host[ir]; i_nz < rows_host[ir+1]; i_nz++)
                A(ir, columns_host[i_nz]) =  values_host[i_nz];
        
        delete [] values_host;
        delete [] columns_host;
        delete [] rows_host;
        timer.Stop();
        printf("\tDone: elapsed time [s]: %0.3e ---\n", timer.get_dur_s());
    }

};