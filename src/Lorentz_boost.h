#ifndef LORENTZ_BOOST_H
#define LORENTZ_BOOST_H
#include"acc_base.h"
#include<cmath>
#include<string>
#include<tuple>

class LorentzBoost: public AccBase{
public:
  friend std::ostream& operator<<(std::ostream &, const LorentzBoost &);
public:
  LorentzBoost()=default;
  LorentzBoost(double theta, const std::string &name=std::string()):angle(theta),cos_ang(cos(angle)),sin_ang(sin(angle)),tan_ang(sin_ang/cos_ang),AccBase(0.0,name){}
  virtual double Pass(double&, double&, double&, double&, double&, double&) const;
  virtual double RPass(double&, double&, double&, double&, double&, double&) const;
  std::tuple<double,double,double> getParams() const{return std::make_tuple(cos_ang,sin_ang,tan_ang);}
private:
  double angle=0.0;
  double cos_ang=1.0;
  double sin_ang=0.0;
  double tan_ang=0.0;
};

class RevLorentzBoost: public AccBase{
public:
  friend std::ostream& operator<<(std::ostream &, const RevLorentzBoost &);
public:
  RevLorentzBoost()=default;
  RevLorentzBoost(double theta, const std::string &name=std::string()):angle(theta),cos_ang(cos(angle)),sin_ang(sin(angle)),tan_ang(sin_ang/cos_ang),AccBase(0.0,name){}
  virtual double Pass(double&, double&, double&, double&, double&, double&) const;
  virtual double RPass(double&, double&, double&, double&, double&, double&) const;
  std::tuple<double,double,double> getParams() const{return std::make_tuple(cos_ang,sin_ang,tan_ang);}
private:
  double angle=0.0;
  double cos_ang=1.0;
  double sin_ang=0.0;
  double tan_ang=0.0;
};

#endif
