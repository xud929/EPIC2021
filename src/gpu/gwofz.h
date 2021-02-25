#ifndef GWOFZ_H
#define GWOFZ_H
#include<thrust/complex.h>

namespace gEPIC{
    __host__ __device__
    extern thrust::complex<double> wofz(const thrust::complex<double> &);
}

#endif
