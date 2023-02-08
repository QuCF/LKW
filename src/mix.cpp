#include "../include/mix.h"

using namespace std;


void YHDF5::create(YCS fname)
{

    if(!fname.empty()) set_name(fname);
    f_ = new H5::H5File(name_.c_str(), H5F_ACC_TRUNC);
    flag_opened = true;
}

void YHDF5::open_r()
{
    f_ = new H5::H5File(name_.c_str(), H5F_ACC_RDONLY);
    flag_opened = true;
}

void YHDF5::open_w()
{
    f_ = new H5::H5File(name_.c_str(), H5F_ACC_RDWR);
    flag_opened = true;
}

void YHDF5::close()
{
    delete f_;
    flag_opened = false;
}

void YHDF5::add_group(YCS gname)
{
    // if(!flag_opened) throw "HDF5 File " + name_ + " is not opened. One cannot add a group " + gname;

    // if(find(grp_names_.begin(), grp_names_.end(), gname) == grp_names_.end())
    // {
    //     grp_names_.push_back(gname);
    //     H5::Group grp(f_->createGroup(gname));
    // }
    H5::Group grp(f_->createGroup(gname));
}





