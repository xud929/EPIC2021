#ifndef BEAM_H
#define BEAM_H
#include<vector>
#include<mpi.h>
#include<random>
#include<trng/mt19937.hpp>
#include<trng/truncated_normal_dist.hpp>
#include<trng/normal_dist.hpp>
#include<array>
#include<iostream>
#include<tuple>
#include<map>
#include"acc_base.h"
#include"poisson_solver.h"

class ThinStrongBeam;
class GaussianStrongBeam;
const double round_beam_threshold=1e-6;

class Beam{
public:
  using slice_type=std::tuple<double,std::vector<double>,std::vector<double>,std::vector<double>,std::vector<std::vector<unsigned>>>;
  Beam()=default;
  Beam(unsigned, const std::array<double,3>&, const std::array<double,2>&, const std::array<double,3>&, MPI_Comm comm=MPI_COMM_NULL);
  Beam(const Beam &tb)=delete;
  Beam& operator=(const Beam &)=delete;
  Beam& generate();
  Beam& normalize(const std::array<double,3>&, const std::array<double,2>&, const std::array<double,3>&);
  //Beam(unsigned, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, unsigned, int);
  double get_mean(unsigned) const;
  double get_variance(unsigned,unsigned) const;
  double get_emittance(unsigned) const;
  std::vector<double> get_statistics() const;
  double& _x_(unsigned i){return x[i];}
  double& _px_(unsigned i){return px[i];}
  double& _y_(unsigned i){return y[i];}
  double& _py_(unsigned i){return py[i];}
  double& _z_(unsigned i){return z[i];}
  double& _pz_(unsigned i){return pz[i];}
  const double& _x_(unsigned i) const{return x[i];}
  const double& _px_(unsigned i) const{return px[i];}
  const double& _y_(unsigned i) const{return y[i];}
  const double& _py_(unsigned i) const{return py[i];}
  const double& _z_(unsigned i) const{return z[i];}
  const double& _pz_(unsigned i) const{return pz[i];}
  const std::vector<double> &get_coordinate(unsigned d) const{return *(ptr_coord[d]);}
  const double &get_coordinate(unsigned d, unsigned ind) const{return (ptr_coord[d])->at(ind);}
  void set_coordinate(unsigned d, unsigned ind, double xx){(ptr_coord[d])->at(ind)=xx;}
  Beam& add_offset(unsigned d, double inc){
      std::vector<double> &v=*(ptr_coord[d]);
      for(auto & i : v)
          i+=inc;
      return *this;
  }
  Beam& add_offset(const std::array<double,6> &inc){
      for(unsigned i=0;i<6;++i)
          add_offset(i,inc[i]);
      return *this;
  }
  double Pass(const AccBase &ele){return ele.Pass(*this);}
  double RPass(const AccBase &ele){return ele.RPass(*this);}
  template<typename...Args>
  double Pass(const AccBase&, Args&&...);
  template<typename...Args>
  double RPass(const AccBase&, Args&&...);
  void reverse(){
      for(unsigned i=0;i<x.size();++i){
          px[i]*=-1.0;py[i]*=-1.0;z[i]*=-1.0;
      }
  }
public://beam-beam section
  std::tuple<double,double> min_max(unsigned) const;
  std::tuple<double,std::vector<double>,std::vector<double>> hist(unsigned, unsigned) const;
  slice_type set_longitudinal_slice_equal_area(unsigned zslice, unsigned resolution=10, const std::string &slice_center_pos="centroid") const;
  slice_type set_longitudinal_slice_specified(const std::vector<double> &, const std::string &slice_center_pos="centroid") const;
public:
  double luminosity_without_beambeam(const ThinStrongBeam &);
  double luminosity_without_beambeam(const GaussianStrongBeam &);
  std::vector<double> luminosity_beambeam_parameter(const ThinStrongBeam &, const std::vector<double> &);
  std::vector<double> luminosity_beambeam_parameter(const GaussianStrongBeam &, const std::vector<double> &);
  void write_to_file(const std::string &, unsigned) const;
public:
  Beam& set_comm_null(){_comm=MPI_COMM_NULL;return *this;}
  Beam& set_comm_world(){_comm=MPI_COMM_WORLD;return *this;}
  Beam& set_comm(MPI_Comm comm){_comm=comm;return *this;}
  MPI_Comm get_comm()const{return _comm;}
  unsigned get_beg()const{return _beg;}
  unsigned get_end()const{return _end;}
  unsigned get_total()const{return Nmacro;}
  static void set_cutoff(double);
  static void set_rng(unsigned long seed){mt.seed(seed);}
  static void set_rng(trng::mt19937 e){mt=e;}
  static trng::mt19937& get_rng(){return mt;}
private:
  std::vector<double> x,px,y,py,z,pz;
  unsigned Nmacro;
  std::vector<double>* ptr_coord[6];
  MPI_Comm _comm;
  int _rank=-1,_size=-1;
  unsigned _beg=0,_end=0;
  static double cutoff;
  static trng::mt19937 mt;
};

template<typename...Args>
double Beam::Pass(const AccBase& ele, Args&&... args){
  double luminosity=0.0;
  luminosity+=Pass(ele);
  luminosity+=Pass(std::forward<Args>(args)...);
  return luminosity;
}
template<typename...Args>
double Beam::RPass(const AccBase& ele, Args&&... args){
  double luminosity=0.0;
  luminosity+=RPass(ele);
  luminosity+=RPass(std::forward<Args>(args)...);
  return luminosity;
}

class ThinStrongBeam: public AccBase{
public:
  friend std::ostream& operator<<(std::ostream &, const ThinStrongBeam &);
public:
  ThinStrongBeam()=default;
  ThinStrongBeam(double strength, const std::vector<double> &beta, const std::vector<double> &alpha, 
	  const std::vector<double> &sigma, double zbeam=0.0, double L=0.0, const std::string &name=std::string()):
	AccBase(L,name),sigxo(sigma[0]),sigyo(sigma[1]),betxo(beta[0]),betyo(beta[1]),alfxo(alpha[0]),alfyo(alpha[1]),kbb(strength),zo(zbeam){
	  gamxo=(1.0+alfxo*alfxo)/betxo;
	  gamyo=(1.0+alfyo*alfyo)/betyo;
	  emitx=sigxo*sigxo/betxo;
	  emity=sigyo*sigyo/betyo;
	}
  std::tuple<double,double,double,double,double,double> get_beam_sigma() const{
      return std::make_tuple(betxo*emitx,-alfxo*emitx,gamxo*emitx,betyo*emity,-alfyo*emity,gamyo*emity);
  }
  void set_beam_size(const std::string &input, const std::vector<double> &sigma, const std::vector<double> &params, int n);
  void set_beam_size(double sx, double sy){
      sigxo=sx;sigyo=sy;
	  emitx=sigxo*sigxo/betxo;
	  emity=sigyo*sigyo/betyo;
  }
  void set_beam_centroid(const std::string &input, const std::vector<double> &params, int n);
  void set_beam_centroid(double xx, double yy){xo=xx;yo=yy;}
  double get_beam_size(int d) const{
      switch(d){
          case 0:
              return sigxo;
          case 1:
              return sigyo;
          default:
              return 0.0;;
      }
  }
  double get_beam_centroid(int d) const{
      switch(d){
          case 0:
              return xo;
          case 1:
              return yo;
          default:
              return 0.0;;
      }
  }
  ThinStrongBeam & set_slice_strength(double k){
      kbb=k;
      return *this;
  }
  double get_slice_strength() const{
      return kbb;
  }
  ThinStrongBeam & set_slice_center(double x, double y, double z){
      xo=x;yo=y;zo=z;
      return *this;
  }
  ThinStrongBeam & set_slice_center_x(double x){
      xo=x;
      return *this;
  }
  ThinStrongBeam & set_slice_center_y(double y){
      yo=y;
      return *this;
  }
  ThinStrongBeam & set_slice_center_z(double z){
      zo=z;
      return *this;
  }
  virtual double Pass(double &, double &, double &, double &, double &, double &) const;
  virtual double RPass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
      return Pass(x,px,y,py,z,pz);
  }
  virtual double Pass(Beam&) const;
  virtual double RPass(Beam&) const;
  double luminosity_without_beambeam(const double&, const double &, const double &, const double &, const double &, const double &) const;
  std::vector<double> luminosity_beambeam_parameter(const double&, const double &, const double &, const double &, const double &, const double &,
          const std::vector<double> &) const;
private:
  double sigxo,sigyo,betxo,alfxo,betyo,alfyo,gamxo,gamyo,emitx,emity;
  double kbb;
  double xo=0.0,yo=0.0,zo=0.0;
  enum SizeGrowthOption{SGO_Invalid,SGO_Linear,SGO_SIN,SGO_WhiteNoise};
  enum CentroidGrowthOption{CGO_Invalid,CGO_Linear,CGO_SIN,CGO_COS,CGO_WhiteNoise};
  std::map<std::string,SizeGrowthOption> SGO_dict{{"linear",SGO_Linear},{"sin",SGO_SIN},{"white-noise",SGO_WhiteNoise}};
  std::map<std::string,CentroidGrowthOption> CGO_dict{{"linear",CGO_Linear},{"sin",CGO_SIN},{"cos",CGO_COS},{"white-noise",CGO_WhiteNoise}};
public:
  SizeGrowthOption resolveSGO(const std::string &input);
  CentroidGrowthOption resolveCGO(const std::string &input);
};

double erfinv(double);

class GaussianStrongBeam: public AccBase{
public:
  friend std::ostream& operator<<(std::ostream &, const GaussianStrongBeam &);
  GaussianStrongBeam()=default;
  GaussianStrongBeam(int n, const ThinStrongBeam &b):ns(n),slice_center(n),slice_hoffset(n,0.0),slice_voffset(n,0.0),slice_weight(n),tsb(b){}
  GaussianStrongBeam & set_equal_area(double);
  GaussianStrongBeam & set_equal_width(double, double);
  GaussianStrongBeam & set_hvoffset(double coef, double freq, int);
  ThinStrongBeam& get_tsb(){return tsb;}
  const ThinStrongBeam& get_tsb() const{return tsb;}
  const std::vector<double> & get_slice_center() const{return slice_center;}
  const std::vector<double> & get_slice_weight() const{return slice_weight;}
  virtual double Pass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
      double kbb=tsb.get_slice_strength();
      double x0=tsb.get_beam_centroid(0), y0=tsb.get_beam_centroid(1);
      double lum=0.0;
      for(int i=ns-1;i>=0;--i){
          //tsb.set_slice_center_z(slice_center[i]);
          tsb.set_slice_center(x0+slice_hoffset[i],y0+slice_voffset[i],slice_center[i]);
          tsb.set_slice_strength(kbb*slice_weight[i]);
          lum+=tsb.Pass(x,px,y,py,z,pz)*slice_weight[i];
      }
      tsb.set_slice_strength(kbb);
      //tsb.set_slice_center_z(0.0);
      tsb.set_slice_center(x0,y0,0.0);
      return lum;
  }
  virtual double RPass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
      double kbb=tsb.get_slice_strength();
      double x0=tsb.get_beam_centroid(0), y0=tsb.get_beam_centroid(1);
      double lum=0.0;
      for(int i=0;i<ns;++i){
          //tsb.set_slice_center_z(-slice_center[i]);
          tsb.set_slice_center(x0+slice_hoffset[i],y0+slice_voffset[i],slice_center[i]);
          tsb.set_slice_strength(kbb*slice_weight[i]);
          lum+=tsb.RPass(x,px,y,py,z,pz)*slice_weight[i];
      }
      tsb.set_slice_strength(kbb);
      //tsb.set_slice_center_z(0.0);
      tsb.set_slice_center(x0,y0,0.0);
      return lum;
  }
  virtual double Pass(Beam&) const;
  virtual double RPass(Beam&) const;
  double luminosity_without_beambeam(const double &x, const double &px, const double &y, const double &py, const double &z, const double &pz) const{
      double kbb=tsb.get_slice_strength();
      double x0=tsb.get_beam_centroid(0), y0=tsb.get_beam_centroid(1);
      double lum=0.0;
      for(int i=ns-1;i>=0;--i){
          //tsb.set_slice_center_z(slice_center[i]);
          tsb.set_slice_center(x0+slice_hoffset[i],y0+slice_voffset[i],slice_center[i]);
          tsb.set_slice_strength(kbb*slice_weight[i]);
          lum+=tsb.luminosity_without_beambeam(x,px,y,py,z,pz)*slice_weight[i];
      }
      tsb.set_slice_strength(kbb);
      //tsb.set_slice_center_z(0.0);
      tsb.set_slice_center(x0,y0,0.0);
      return lum;
  }
  std::vector<double> luminosity_beambeam_parameter(const double &x, const double &px, const double &y, const double &py, const double &z, const double &pz,
          const std::vector<double> &twiss) const{
      double kbb=tsb.get_slice_strength();
      double x0=tsb.get_beam_centroid(0), y0=tsb.get_beam_centroid(1);
      std::vector<double> ret={0.0,0.0,0.0};
      for(int i=ns-1;i>=0;--i){
          //tsb.set_slice_center_z(slice_center[i]);
          tsb.set_slice_center(x0+slice_hoffset[i],y0+slice_voffset[i],slice_center[i]);
          tsb.set_slice_strength(kbb*slice_weight[i]);
          auto v=tsb.luminosity_beambeam_parameter(x,px,y,py,z,pz,twiss);
          ret[0]+=slice_weight[i]*v[0];
          ret[1]+=v[1];
          ret[2]+=v[2];
      }
      tsb.set_slice_strength(kbb);
      //tsb.set_slice_center_z(0.0);
      tsb.set_slice_center(x0,y0,0.0);
      return ret;
  }
private:
  int ns;
  std::vector<double> slice_center;
  std::vector<double> slice_weight;
  std::vector<double> slice_hoffset;
  std::vector<double> slice_voffset;
  mutable ThinStrongBeam tsb;
};

int gaussian_beambeam_kick(double &, double &, double, double, double, double);

double beam_beam(Beam &, const Beam::slice_type &, Beam&, const Beam::slice_type &, const Poisson_Solver::solver_base &);

#endif
