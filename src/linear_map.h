#ifndef LINEAR_MAP_H
#define LINEAR_MAP_H
#include"acc_base.h"
#include<vector>
#include<string>
#include<array>
#include<tuple>

class Linear6D: public AccBase{
public:
  friend std::ostream& operator<<(std::ostream &, const Linear6D &);
public:
  Linear6D()=default;
  Linear6D(const std::array<double,3> &, const std::array<double,2> &, const std::array<double,3> &, const std::array<double,2> &, double L=0.0, const std::string &name=std::string());
  virtual double Pass(double&, double&, double&, double&, double&, double&) const;
  virtual double RPass(double&, double&, double&, double&, double&, double&) const;
  std::tuple<double,double,double,double,double,double,double,double,double,double,double,double> getParams() const{
      return std::make_tuple(betx,bety,betz,alfx,alfy,gamx,gamy,xix,xiy,mux,muy,muz);
  }
private:
  double betx,bety,betz,alfx,alfy,gamx,gamy,xix,xiy,mux,muy,muz;
};

class Linear2D: public AccBase{
public:
  std::tuple<double,double,double,double> getTM() const{return std::make_tuple(m11,m12,m21,m22);}
protected:
  Linear2D()=default;
  Linear2D(double,double,double,double,double,double,const std::string &);
  //virtual int Pass(double&, double&, double&, double&, double&, double&) const=0;
  double m11,m12,m21,m22;
};

class LinearX: public Linear2D{
public:
  friend std::ostream& operator<<(std::ostream &, const LinearX &);
public:
  LinearX():Linear2D(){}
  LinearX(double bet1,double alf1,double bet2,double alf2,double dphase, double L=0.0, const std::string &name=std::string()):Linear2D(bet1,alf1,bet2,alf2,dphase,L,name){}
  virtual double Pass(double&, double&, double&, double&, double&, double&) const;
  virtual double RPass(double&, double&, double&, double&, double&, double&) const;
};

class LinearY: public Linear2D{
public:
  friend std::ostream& operator<<(std::ostream &, const LinearY &);
public:
  LinearY():Linear2D(){}
  LinearY(double bet1,double alf1,double bet2,double alf2,double dphase, double L=0.0, const std::string &name=std::string()):Linear2D(bet1,alf1,bet2,alf2,dphase,L,name){}
  virtual double Pass(double&, double&, double&, double&, double&, double&) const;
  virtual double RPass(double&, double&, double&, double&, double&, double&) const;
};

#endif
