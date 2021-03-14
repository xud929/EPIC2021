#ifndef ROTATED_LORENTZ_BOOST_H
#define ROTATED_LORENTZ_BOOST_H
#include"acc_base.h"
#include<cmath>
#include<tuple>
#include<Eigen/Dense>
#include<stdexcept>

class RotatedLorentzBoost: public AccBase{
public:
    using mat44=Eigen::Matrix<double,4,4>;
    using mat33=Eigen::Matrix<double,3,3>;
    using vec3=Eigen::Matrix<double,3,1>;
    using vec4=Eigen::Matrix<double,4,1>;
    RotatedLorentzBoost(const mat44 &M1, const mat44 &M2):MX(M1),MP(M2){}
    virtual double Pass(double&, double&, double&, double&, double&, double&) const;
    virtual double RPass(double&, double&, double&, double&, double&, double&) const{
        throw std::runtime_error("RPass method is not implemented for RotatedLorentzBoost");
        return 0.0;
    }
private:
    mat44 MX,MP;
};

class RotatedRevLorentzBoost: public AccBase{
public:
    using mat44=Eigen::Matrix<double,4,4>;
    using mat33=Eigen::Matrix<double,3,3>;
    using vec3=Eigen::Matrix<double,3,1>;
    using vec4=Eigen::Matrix<double,4,1>;
    RotatedRevLorentzBoost(const mat44 &M1, const mat44 &M2):MX(M1),MP(M2){}
    virtual double Pass(double&, double&, double&, double&, double&, double&) const;
    virtual double RPass(double&, double&, double&, double&, double&, double&) const{
        throw std::runtime_error("RPass method is not implemented for RotatedRevLorentzBoost");
        return 0.0;
    }
private:
    mat44 MX,MP;
};

std::tuple<RotatedLorentzBoost::mat44,
    RotatedLorentzBoost::mat44,
    RotatedLorentzBoost::mat44,
    RotatedLorentzBoost::mat44> rotate_matrix(double, double, double);
#endif
