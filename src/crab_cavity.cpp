#include"crab_cavity.h"
#include<cmath>

ThinCrabCavity::ThinCrabCavity(double theta, double f0, double phis, const std::string &key):AccBase(0.0,key),strength(theta),frequency(f0),phase(phis){
  kcc=2.0*M_PI*frequency/phys_const::clight;
}

double ThinCrabCavity::Pass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
  double dpx=0.0, dpz=0.0;
  for(const auto & m : harmonic){
	const unsigned &h=m.first;
	const double &r=m.second;
	dpx-=r*strength*sin(h*kcc*z+phase)/kcc/h;
	dpz-=r*strength*cos(h*kcc*z+phase)*x;
	//dpx-=r*strength*z;
	//dpz-=r*strength*x;
  }
  px+=dpx;
  pz+=dpz;
  return 0;
}

double ThinCrabCavity::RPass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
  double dpx=0.0, dpz=0.0;
  for(const auto & m : harmonic){
	const unsigned &h=m.first;
	const double &r=m.second;
	dpx+=r*strength*sin(h*kcc*z-phase)/kcc/h;
	dpz+=r*strength*cos(h*kcc*z-phase)*x;
	//dpx-=r*strength*z;
	//dpz-=r*strength*x;
  }
  px+=dpx;
  pz+=dpz;
  return 0;
}

std::ostream& operator<<(std::ostream &out, const ThinCrabCavity &tcc){
  out<<"&"<<tcc.name<<"\n";
  out<<"  type=\"ThinCrabCavity\",\n";
  out<<"  length=0.0,\n";
  out<<"  frequency="<<tcc.frequency<<",\n";
  out<<"  strength="<<tcc.strength<<",\n";
  out<<"  phase="<<tcc.phase<<",\n";
  out<<"  harmonic=";
  for(const auto & i : tcc.harmonic){
	out<<i.first<<",";
  }
  out<<"\n";
  out<<"  relative=";
  for(const auto & i : tcc.harmonic){
	out<<i.second<<",";
  }
  out<<"\n&end";
  return out;
}

//*************************************************************************************************************
ThinCrabCavity2D::ThinCrabCavity2D(double s1, double s2, double f0, double phis, const std::string &key):AccBase(0.0,key),strengthX(s1),strengthY(s2),frequency(f0),phase(phis){
  kcc=2.0*M_PI*frequency/phys_const::clight;
}

double ThinCrabCavity2D::Pass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
  double dpx=0.0, dpy=0.0, dpz=0.0;
  for(const auto & m : harmonic){
	const unsigned &h=m.first;
	const double &r=m.second;
	dpx-=r*strengthX*sin(h*kcc*z+phase)/kcc/h;
	dpy-=r*strengthY*sin(h*kcc*z+phase)/kcc/h;
	dpz-=r*(strengthX*x+strengthY*y)*cos(h*kcc*z+phase);
  }
  px+=dpx;
  py+=dpy;
  pz+=dpz;
  return 0;
}

double ThinCrabCavity2D::RPass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
  double dpx=0.0, dpy=0.0, dpz=0.0;
  for(const auto & m : harmonic){
	const unsigned &h=m.first;
	const double &r=m.second;
	dpx+=r*strengthX*sin(h*kcc*z-phase)/kcc/h;
	dpy+=r*strengthY*sin(h*kcc*z-phase)/kcc/h;
	dpz+=r*(strengthX*x+strengthY*y)*cos(h*kcc*z-phase);
  }
  px+=dpx;
  py+=dpy;
  pz+=dpz;
  return 0;
}
