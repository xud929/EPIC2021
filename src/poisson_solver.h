#ifndef POISSON_SOLVER_H
#define POISSON_SOLVER_H
#include<vector>
#include<array>
#include<mpi.h>

namespace Poisson_Solver{
class solver_base{
public:
    virtual double operator()(std::vector<double> &, const std::array<double,4> &, MPI_Comm, std::vector<double> &, const std::array<double,4> &, MPI_Comm) const=0;
    virtual ~solver_base(){}
};

class soft_gaussian : public solver_base{
public:
    soft_gaussian(double bb1, double lum1, double bb2, double lum2, unsigned which=2):kbb1(bb1),klum1(lum1),kbb2(bb2),klum2(lum2),gaussian_when_luminosity(which){}
    double operator()(std::vector<double> &, const std::array<double,4> &, MPI_Comm, std::vector<double> &, const std::array<double,4> &, MPI_Comm) const;
    // beam1: coord, weight, S1, 
    // beam2: {mean_x, sigma_x, ... }, S2
    double slice_slice_kick(std::vector<double> &, const std::vector<double> &, double, const std::array<double,10> &, double, const std::tuple<unsigned,double> &) const;
    double slice_slice_kick_no_interpolation(std::vector<double> &, const std::array<double,10> &, double, const std::tuple<unsigned,double> &) const;
private:
    double kbb1,klum1,kbb2,klum2;
    unsigned gaussian_when_luminosity;
};

}

#endif
