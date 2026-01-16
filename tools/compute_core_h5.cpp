#include <hdf5.h>
#include <iostream>
#include <vector>

std::vector<double> read_dataset(const std::string &file, const std::string &name, hsize_t &rows, hsize_t &cols) {
    hid_t f = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dset = H5Dopen2(f, name.c_str(), H5P_DEFAULT);
    hid_t space = H5Dget_space(dset);
    int ndims = H5Sget_simple_extent_ndims(space);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, NULL);
    if(ndims!=2) throw std::runtime_error("Dataset not 2D");
    rows = dims[0]; cols = dims[1];
    std::vector<double> data(rows*cols);
    herr_t status = H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Sclose(space); H5Dclose(dset); H5Fclose(f);
    if(status<0) throw std::runtime_error("Failed to read dataset");
    return data;
}

int main(int argc, char **argv) {
    if(argc<2) { std::cerr<<"Usage: compute_core_h5 <h5file>\n"; return 2; }
    std::string f=argv[1];
    hsize_t Hrows,Hcols;
    auto Hdata = read_dataset(f, "H0", Hrows, Hcols);
    hsize_t Carows,Cacols;
    auto Cdata = read_dataset(f, "Ca", Carows, Cacols);
    // take first two columns
    size_t nocc = std::min<hsize_t>(2, Cacols);
    // compute P = Cocc * I * Cocc^T = sum_k c_k c_k^T
    std::vector<double> P(Hrows*Hcols, 0.0);
    for(size_t k=0;k<nocc;k++){
        for(size_t i=0;i<Hrows;i++) for(size_t j=0;j<Hcols;j++){
            P[i*Hcols + j] += Cdata[i*Cacols + k] * Cdata[j*Cacols + k];
        }
    }
    // compute trace(P*H)
    double tr = 0.0;
    for(size_t i=0;i<Hrows;i++) for(size_t j=0;j<Hcols;j++) tr += P[i*Hcols+j] * Hdata[j*Hcols + i];
    std::cout << "core energy (trace P H0) = " << tr << std::endl;
    return 0;
}
