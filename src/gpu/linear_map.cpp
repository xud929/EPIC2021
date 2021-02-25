#include"linear_map.h"
#include<stdexcept>
#include<cmath>
#include<iostream>

using std::cin;using std::cout;using std::endl;using std::cerr;
/*dispersion is not included yet*/
Linear6D::Linear6D(const std::array<double,3> &beta, const std::array<double,2> &alpha, const std::array<double,3> &nu, const std::array<double,2> &xi, double L, const std::string &name):AccBase(L,name){
  if(beta.size()!=3 || alpha.size()!=2 || nu.size()!=3 || xi.size()!=2)
	throw std::length_error("Linear6D initialization failed.");
  betx=beta[0];bety=beta[1];betz=beta[2];
  alfx=alpha[0];alfy=alpha[1];
  gamx=(1.0+alfx*alfx)/betx;gamy=(1.0+alfy*alfy)/bety;
  mux=nu[0]*math_const::twopi;muy=nu[1]*math_const::twopi;muz=nu[2]*math_const::twopi;
  xix=xi[0];xiy=xi[1];
}

double Linear6D::Pass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
  /*
  double x0=x, px0=px, y0=y, py0=py, z0=z, pz0=pz;
  double cos_t=cos(muz), sin_t=sin(muz);
  double A=(muz==0.0?0.0:(1.0-cos_t)/muz/betz), B=(muz==0.0?1.0:sin_t/muz);
  double z1=z0*cos_t+pz0*betz*sin_t, pz1=-z0*sin_t/betz+pz0*cos_t;
  double Jx=0.5*(x0*x0/betx+betx*px0*px0), Jy=0.5*(y0*y0/bety+bety*py0*py0);

  double dmux=A*z1+B*pz1, dmuy=math_const::twopi*xiy*dmux; dmux*=math_const::twopi*xix;
  cos_t=cos(mux+dmux);sin_t=sin(mux+dmux);
  x=x0*cos_t+px0*betx*sin_t;
  px=-x0*sin_t/betx+px0*cos_t;

  cos_t=cos(muy+dmuy);sin_t=sin(muy+dmuy);
  y=y0*cos_t+py0*bety*sin_t;
  py=-y0*sin_t/bety+py0*cos_t;

  z=math_const::twopi*(xix*Jx+xiy*Jy);
  pz=pz1-z*A;
  z=z1+z*B;
  */
  double x0=x, px0=px, y0=y, py0=py, z0=z, pz0=pz;
  double angle, cos_t, sin_t;

  angle=mux+math_const::twopi*xix*pz0;
  cos_t=cos(angle);sin_t=sin(angle);
  x=x0*(cos_t+alfx*sin_t)+px0*betx*sin_t;
  px=-x0*sin_t*gamx+px0*(cos_t-alfx*sin_t);

  angle=muy+math_const::twopi*xiy*pz0;
  cos_t=cos(angle);sin_t=sin(angle);
  y=y0*(cos_t+alfy*sin_t)+py0*bety*sin_t;
  py=-y0*sin_t*gamy+py0*(cos_t-alfy*sin_t);

#ifdef SYMPLECTIC_ONE_TURN_PASS
  double Jx=0.5*(x0*x0*gamx+2.0*alfx*x0*px0+betx*px0*px0), Jy=0.5*(y0*y0*gamy+2.0*alfy*y0*py0+bety*py0*py0);
  cos_t=cos(muz);sin_t=sin(muz);
  double tz=z0+math_const::twopi*(xix*Jx+xiy*Jy);
  z=tz*cos_t+pz0*betz*sin_t;
  pz=-tz*sin_t/betz+pz0*cos_t;
  //cout<<"symplectic one turn pass"<<endl;
#else
  cos_t=cos(muz);sin_t=sin(muz);
  z=z0*cos_t+pz0*betz*sin_t;
  pz=-z0*sin_t/betz+pz0*cos_t;
  //cout<<"non-symplectic one turn pass"<<endl;
#endif

  return 0;
}

double Linear6D::RPass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
  double x0=x, px0=px, y0=y, py0=py, z0=z, pz0=pz;
  double angle, cos_t, sin_t;

  double Jx=0.5*(x0*x0*gamx+2.0*alfx*x0*px0+betx*px0*px0), Jy=0.5*(y0*y0*gamy+2.0*alfy*y0*py0+bety*py0*py0);
  cos_t=cos(muz);sin_t=sin(muz);
  z=z0*cos_t+pz0*betz*sin_t;
  pz=-z0*sin_t/betz+pz0*cos_t;
#ifdef SYMPLECTIC_ONE_TURN_PASS
  z+=math_const::twopi*(xix*Jx+xiy*Jy);
#endif

  angle=mux+math_const::twopi*xix*pz;
  cos_t=cos(angle);sin_t=sin(angle);
  x=x0*(cos_t-alfx*sin_t)+px0*betx*sin_t;
  px=-x0*sin_t*gamx+px0*(cos_t+alfx*sin_t);

  angle=muy+math_const::twopi*xiy*pz;
  cos_t=cos(angle);sin_t=sin(angle);
  y=y0*(cos_t-alfy*sin_t)+py0*bety*sin_t;
  py=-y0*sin_t*gamy+py0*(cos_t+alfy*sin_t);

  return 0;
}

/*************************************************************************************************/
Linear2D::Linear2D(double bet1, double alf1, double bet2, double alf2, double dphase, double L, const std::string &name):AccBase(L,name){
  double cos_t=cos(dphase), sin_t=sin(dphase), K1=sqrt(bet1*bet2), K2=K1/bet1;

  m11=K2*(cos_t+alf1*sin_t);
  m12=K1*sin_t;
  m21=(-(1.0+alf1*alf2)*sin_t+(alf1-alf2)*cos_t)/K1;
  m22=(cos_t-alf2*sin_t)/K2;

  //cout<<m11<<"\t"<<m12<<"\n"<<m21<<"\t"<<m22<<"\n"<<m11*m22-m12*m21<<endl;
}

double LinearX::Pass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
  double u0=x, pu0=px;
  x=m11*u0+m12*pu0;
  px=m21*u0+m22*pu0;
  return 0;
}
double LinearX::RPass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
  double u0=x, pu0=px;
  x=m22*u0+m12*pu0;
  px=m21*u0+m11*pu0;
  return 0;
}
std::ostream& operator<<(std::ostream& out, const LinearX &m){
  out<<"&"<<m.name<<"\n";
  out<<" type=\"LinearX\",\n";
  out<<" M="<<m.m11<<","<<m.m12<<","<<m.m21<<","<<m.m22<<",\n";
  out<<"&end";
  return out;
}


double LinearY::Pass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
  double u0=y, pu0=py;
  y=m11*u0+m12*pu0;
  py=m21*u0+m22*pu0;
  return 0;
}
double LinearY::RPass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
  double u0=y, pu0=py;
  y=m22*u0+m12*pu0;
  py=m21*u0+m11*pu0;
  return 0;
}

std::ostream& operator<<(std::ostream& out, const LinearY &m){
  out<<"&"<<m.name<<"\n";
  out<<" type=\"LinearY\",\n";
  out<<" M="<<m.m11<<","<<m.m12<<","<<m.m21<<","<<m.m22<<",\n";
  out<<"&end";
  return out;
}
