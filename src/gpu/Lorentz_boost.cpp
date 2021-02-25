#include"Lorentz_boost.h"
#include<iostream>

using std::cin;using std::cout;using std::cerr;using std::endl;
double LorentzBoost::Pass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
#ifndef LINEAR_LORENTZ_BOOST_PASS
  double ps=1.0+pz;ps=ps*ps-px*px-py*py;ps=sqrt(ps);
  double h=1.0+pz-ps;

  py/=cos_ang;h/=cos_ang*cos_ang;
  px=px/cos_ang-h*sin_ang;
  pz-=px*sin_ang;
  ps=1.0+pz-h;

  double ds=x*sin_ang;
  x+=z*tan_ang+px/ps*ds;
  y+=py/ps*ds;
  z=z/cos_ang-h/ps*ds;
  //cout<<"non-linear lorentz boost pass"<<endl;
#else
  x+=z*tan_ang;
  pz-=px*tan_ang;
  px/=cos_ang;
  py/=cos_ang;
  z/=cos_ang;
  //cout<<"linear lorentz boost pass"<<endl;
#endif

  return 0;
}

double LorentzBoost::RPass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
#ifndef LINEAR_LORENTZ_BOOST_PASS
  double ps=1.0+pz;ps=ps*ps-px*px-py*py;ps=sqrt(ps);
  double h=1.0+pz-ps;

  x+=z*sin_ang;
  x/=1.0+(-px+h*sin_ang)*sin_ang/ps;
  z=(z-h/ps*x*sin_ang)*cos_ang;
  y+=py/ps*x*sin_ang;

  pz-=px*sin_ang;
  px=(px-h*sin_ang)*cos_ang;
  py*=cos_ang;
  //cout<<"non-linear lorentz boost pass"<<endl;
#else
  x+=z*sin_ang;
  pz-=px*sin_ang;
  px*=cos_ang;
  py*=cos_ang;
  z*=cos_ang;
  //cout<<"linear lorentz boost pass"<<endl;
#endif
  return 0;
}

double RevLorentzBoost::Pass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
#ifndef LINEAR_LORENTZ_BOOST_PASS
  double ps=1.0+pz;ps=ps*ps-px*px-py*py;ps=sqrt(ps);
  double h=1.0+pz-ps;

  x-=z*sin_ang;
  x/=1.0+(px+h*sin_ang)*sin_ang/ps;
  z=(z+h/ps*x*sin_ang)*cos_ang;
  y-=py/ps*x*sin_ang;

  pz+=px*sin_ang;
  px=(px+h*sin_ang)*cos_ang;
  py*=cos_ang;
  //cout<<"non-linear lorentz boost pass"<<endl;
#else
  x-=z*sin_ang;
  pz+=px*sin_ang;
  px*=cos_ang;
  py*=cos_ang;
  z*=cos_ang;
  //cout<<"linear lorentz boost pass"<<endl;
#endif

  return 0.0;
}

double RevLorentzBoost::RPass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
#ifndef LINEAR_LORENTZ_BOOST_PASS
  double ps=1.0+pz;ps=ps*ps-px*px-py*py;ps=sqrt(ps);
  double h=1.0+pz-ps;

  py/=cos_ang;h/=cos_ang*cos_ang;
  px=px/cos_ang+h*sin_ang;
  pz+=px*sin_ang;
  ps=1.0+pz-h;

  double ds=x*sin_ang;
  x-=z*tan_ang+px/ps*ds;
  y-=py/ps*ds;
  z=z/cos_ang+h/ps*ds;
  //cout<<"non-linear lorentz boost pass"<<endl;
#else
  x-=z*tan_ang;
  pz+=px*tan_ang;
  px/=cos_ang;
  py/=cos_ang;
  z/=cos_ang;
#endif

  return 0.0;
}
