#ifndef ACC_BASE_H
#define ACC_BASE_H
#include<string>
#include"constants.h"
//#define SYMPLECTIC_ONE_TURN_PASS
//#define LINEAR_LORENTZ_BOOST_PASS
//#define HIRATA_MAP
//#define LINEAR_BEAMBEAM_KICK

class Beam;

class AccBase{
public:
  AccBase()=default;
  AccBase(double L, const std::string &key):length(L),name(key){}
  virtual double Pass(double &, double &, double &, double &, double &, double &) const=0;
  virtual double RPass(double &, double &, double &, double &, double &, double &) const=0;
  virtual double Pass(Beam&) const; //defined in beam.cpp
  virtual double RPass(Beam&) const; //defined in beam.cpp
  virtual ~AccBase(){}
protected:
  double length=0.0;
  std::string name;
};

#endif
