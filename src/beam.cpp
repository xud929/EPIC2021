#include"beam.h"
#include<stdexcept>
#include<algorithm>
#include<complex>
#include<typeinfo>
#include<cmath>
#include<functional>
#include"faddeeva.h"
/************************************************************************************************************************************************/

using std::vector;
using std::complex;

double Beam::cutoff=5.0;
trng::mt19937 Beam::mt;
/* if |(sigy-sigx)/(sigy+sigx)|<round_beam_threshold
 * then use round beam formula to calculate beam-beam kick
 * and second order derivative */

void Beam::set_cutoff(double c){
  cutoff=c;
}
/************************************************************************************************************************************************/
std::array<double,2> moments(const vector<double> &v, unsigned totalN, MPI_Comm comm){
  std::array<double,2> mom={0.0,0.0};
  for(const auto & i : v){
	mom[0]+=i;mom[1]+=i*i;
  }
  if(comm!=MPI_COMM_NULL)
      MPI_Allreduce(MPI_IN_PLACE,mom.data(),2,MPI_DOUBLE,MPI_SUM,comm);
  for(double & i : mom)
      i/=totalN;
  mom[1]-=mom[0]*mom[0];
  if(mom[1]>0.0)
      mom[1]=std::sqrt(mom[1]);
  else
      mom[1]=0.0;
  return mom;
}
/************************************************************************************************************************************************/

Beam& Beam::generate(){
  if(_comm==MPI_COMM_NULL)
      return *this;
  MPI_Comm_rank(_comm, &_rank);
  MPI_Comm_size(_comm, &_size);
  _beg=_rank*Nmacro/_size;_end=(_rank+1)*Nmacro/_size;
  unsigned N=_end-_beg;
  x=vector<double>(N,0.0);ptr_coord[0]=&x;
  px=vector<double>(N,0.0);ptr_coord[1]=&px;
  y=vector<double>(N,0.0);ptr_coord[2]=&y;
  py=vector<double>(N,0.0);ptr_coord[3]=&py;
  z=vector<double>(N,0.0);ptr_coord[4]=&z;
  pz=vector<double>(N,0.0);ptr_coord[5]=&pz;

  trng::truncated_normal_dist<double> normal_gaussian(0.0,1.0,-cutoff,cutoff);
  auto gaussian=std::bind(normal_gaussian,std::ref(mt));
  for(unsigned i=0;i<6;++i){
      std::vector<double> &v=*ptr_coord[i];
      mt.discard(_beg);
      std::generate(v.begin(),v.end(),gaussian);
      mt.discard(Nmacro-_end);
      const auto & [mean,sigma]=moments(v,Nmacro,_comm);
      std::for_each(v.begin(),v.end(),[=](double &t){t=(t-mean)/sigma;});
  }
  return *this;
}

Beam& Beam::normalize(const std::array<double,3> &beta, const std::array<double,2> &alpha, const std::array<double,3> &sigma){
  for(unsigned i=0;i<x.size();++i){
	px[i]=sigma[0]/beta[0]*(px[i]-alpha[0]*x[i]);
	x[i]*=sigma[0];

	py[i]=sigma[1]/beta[1]*(py[i]-alpha[1]*y[i]);
	y[i]*=sigma[1];

	pz[i]*=sigma[2]/beta[2];
	z[i]*=sigma[2];
  }
  return *this;
}

Beam::Beam(unsigned totalN, const std::array<double,3> &beta, const std::array<double,2> &alpha, const std::array<double,3> &sigma, MPI_Comm comm):
    Nmacro(totalN),_comm(comm){
  generate().normalize(beta,alpha,sigma);
}

/*
Beam::Beam(unsigned totalN, const std::vector<double> &beta, const std::vector<double> &alpha, const std::vector<double> &sigma, unsigned seed, int is_world):Nmacro(totalN){
  if(is_world)
      _comm=MPI_COMM_WORLD;
  else
      _comm=MPI_COMM_NULL;

  if(beta.size()!=3 || alpha.size()!=2 || sigma.size()!=3)
	throw std::length_error("Beam initialization failed.");

  int rank=0,size=1;
  if(_comm!=MPI_COMM_NULL){
      MPI_Comm_rank(_comm, &rank);
      MPI_Comm_size(_comm, &size);
  }


  std::mt19937 mt;
  std::random_device rd;
  if(seed==0 && rank==0){
	seed=rd();
  }
  MPI_Bcast(&seed,1,MPI_UNSIGNED,0,_comm);
  mt.seed(seed);

  unsigned beg=rank*totalN/size, end=(rank+1)*totalN/size;
  unsigned N=end-beg;

  x=vector<double>(N,0.0);
  px=vector<double>(N,0.0);
  y=vector<double>(N,0.0);
  py=vector<double>(N,0.0);
  z=vector<double>(N,0.0);
  pz=vector<double>(N,0.0);

  std::normal_distribution<double> normal_gaussian(0.0,1.0);
  auto gaussian=[&](double &g)->void{g=normal_gaussian(mt);};

  for(int i=0;i<beg;++i)
      normal_gaussian(mt);
  std::for_each(x.begin(),x.end(),gaussian);
  for(int i=end;i<Nmacro;++i)
      normal_gaussian(mt);

  for(int i=0;i<beg;++i)
      normal_gaussian(mt);
  std::for_each(px.begin(),px.end(),gaussian);
  for(int i=end;i<Nmacro;++i)
      normal_gaussian(mt);

  for(int i=0;i<beg;++i)
      normal_gaussian(mt);
  std::for_each(y.begin(),y.end(),gaussian);
  for(int i=end;i<Nmacro;++i)
      normal_gaussian(mt);

  for(int i=0;i<beg;++i)
      normal_gaussian(mt);
  std::for_each(py.begin(),py.end(),gaussian);
  for(int i=end;i<Nmacro;++i)
      normal_gaussian(mt);
  
  for(int i=0;i<beg;++i)
      normal_gaussian(mt);
  std::for_each(z.begin(),z.end(),gaussian);
  for(int i=end;i<Nmacro;++i)
      normal_gaussian(mt);

  for(int i=0;i<beg;++i)
      normal_gaussian(mt);
  std::for_each(pz.begin(),pz.end(),gaussian);

  auto normalize=[=](vector<double> &v, const vector<double> &m)->void{
	double mean=m[0]/static_cast<double>(totalN),std=m[1]/static_cast<double>(totalN);
	std-=mean*mean;
	if(std>0.0){
	  std=sqrt(std);
	  for(auto & i : v)
		i=(i-mean)/std;
	}else{
	  for(auto & i : v)
		i=0.0;
	}
  };
  auto mom_x=moments(x,_comm);normalize(x,mom_x);
  auto mom_px=moments(px,_comm);normalize(px,mom_px);
  auto mom_y=moments(y,_comm);normalize(y,mom_y);
  auto mom_py=moments(py,_comm);normalize(py,mom_py);
  auto mom_z=moments(z,_comm);normalize(z,mom_z);
  auto mom_pz=moments(pz,_comm);normalize(pz,mom_pz);

  for(unsigned i=0;i<N;++i){
	px[i]=sigma[0]/beta[0]*(px[i]-alpha[0]*x[i]);
	x[i]*=sigma[0];

	py[i]=sigma[1]/beta[1]*(py[i]-alpha[1]*y[i]);
	y[i]*=sigma[1];

	pz[i]*=sigma[2]/beta[2];
	z[i]*=sigma[2];
  }

  ptr_coord[0]=&x;
  ptr_coord[1]=&px;
  ptr_coord[2]=&y;
  ptr_coord[3]=&py;
  ptr_coord[4]=&z;
  ptr_coord[5]=&pz;

}
*/
/************************************************************************************************************************************************/


double Beam::get_mean(unsigned d) const{
  const auto &v=*(ptr_coord[d]);
  double sum=std::accumulate(v.begin(),v.end(),0.0);
  if(_comm!=MPI_COMM_NULL)
      MPI_Allreduce(MPI_IN_PLACE,&sum,1,MPI_DOUBLE,MPI_SUM,_comm);
  sum/=Nmacro;
  return sum;
}
double Beam::get_variance(unsigned d1, unsigned d2) const{
  const auto &v1=*(ptr_coord[d1]), &v2=*(ptr_coord[d2]);
  double sum=std::inner_product(v1.begin(),v1.end(),v2.begin(),0.0);
  if(_comm!=MPI_COMM_NULL)
      MPI_Allreduce(MPI_IN_PLACE,&sum,1,MPI_DOUBLE,MPI_SUM,_comm);
  sum/=Nmacro;
  double m1=get_mean(d1),m2;
  if(d1==d2)
	m2=m1;
  else
	m2=get_mean(d2);
  return sum-m1*m2;
}
double Beam::get_emittance(unsigned d) const{
  unsigned d1=2*d,d2=d1+1;
  double sum=get_variance(d1,d2);
  sum*=sum;
  sum=get_variance(d1,d1)*get_variance(d2,d2)-sum;
  if(sum<=0.0)
	return 0.0;
  else
	return sqrt(sum);
}

double AccBase::Pass(Beam &b) const{
    unsigned N=b.get_end()-b.get_beg();
    double luminosity=0.0;
    for(unsigned i=0;i<N;++i)
        luminosity+=Pass(b._x_(i),b._px_(i),b._y_(i),b._py_(i),b._z_(i),b._pz_(i));
    return luminosity;
}
double AccBase::RPass(Beam &b) const{
    unsigned N=b.get_end()-b.get_beg();
    double luminosity=0.0;
    for(unsigned i=0;i<N;++i)
        luminosity+=RPass(b._x_(i),b._px_(i),b._y_(i),b._py_(i),b._z_(i),b._pz_(i));
    return luminosity;
}
double ThinStrongBeam::Pass(Beam &b) const{
    unsigned N=b.get_end()-b.get_beg();
    double luminosity=0.0;
    for(unsigned i=0;i<N;++i)
        luminosity+=Pass(b._x_(i),b._px_(i),b._y_(i),b._py_(i),b._z_(i),b._pz_(i));
    MPI_Allreduce(MPI_IN_PLACE,&luminosity,1,MPI_DOUBLE,MPI_SUM,b.get_comm());
    return luminosity;
}
double ThinStrongBeam::RPass(Beam &b) const{
    unsigned N=b.get_end()-b.get_beg();
    double luminosity=0.0;
    for(unsigned i=0;i<N;++i)
        luminosity+=Pass(b._x_(i),b._px_(i),b._y_(i),b._py_(i),b._z_(i),b._pz_(i));
    MPI_Allreduce(MPI_IN_PLACE,&luminosity,1,MPI_DOUBLE,MPI_SUM,b.get_comm());
    return luminosity;
}
double GaussianStrongBeam::Pass(Beam &b) const{
    unsigned N=b.get_end()-b.get_beg();
    double luminosity=0.0;
    for(unsigned i=0;i<N;++i)
        luminosity+=Pass(b._x_(i),b._px_(i),b._y_(i),b._py_(i),b._z_(i),b._pz_(i));
    MPI_Allreduce(MPI_IN_PLACE,&luminosity,1,MPI_DOUBLE,MPI_SUM,b.get_comm());
    return luminosity;
}
double GaussianStrongBeam::RPass(Beam &b) const{
    unsigned N=b.get_end()-b.get_beg();
    double luminosity=0.0;
    for(unsigned i=0;i<N;++i)
        luminosity+=Pass(b._x_(i),b._px_(i),b._y_(i),b._py_(i),b._z_(i),b._pz_(i));
    MPI_Allreduce(MPI_IN_PLACE,&luminosity,1,MPI_DOUBLE,MPI_SUM,b.get_comm());
    return luminosity;
}
/*
double Beam::Pass(const AccBase& ele){
  unsigned N=x.size();
  double luminosity=0.0;
  for(unsigned i=0;i<N;++i){
	auto ret=ele.Pass(x[i],px[i],y[i],py[i],z[i],pz[i]);
    luminosity+=ret;
  }
  if(_comm!=MPI_COMM_NULL && ((typeid(ele)==typeid(ThinStrongBeam)) || (typeid(ele)==typeid(GaussianStrongBeam))))
      MPI_Allreduce(MPI_IN_PLACE,&luminosity,1,MPI_DOUBLE,MPI_SUM,_comm);
  return luminosity;
}
double Beam::RPass(const AccBase& ele){
  unsigned N=x.size();
  double luminosity=0.0;
  for(unsigned i=0;i<N;++i){
	auto ret=ele.RPass(x[i],px[i],y[i],py[i],z[i],pz[i]);
    luminosity+=ret;
  }
  if(_comm!=MPI_COMM_NULL && ((typeid(ele)==typeid(ThinStrongBeam)) || (typeid(ele)==typeid(GaussianStrongBeam))))
      MPI_Allreduce(MPI_IN_PLACE,&luminosity,1,MPI_DOUBLE,MPI_SUM,_comm);
  return luminosity;
}
*/

vector<double> Beam::luminosity_beambeam_parameter(const ThinStrongBeam & ele, const vector<double> &twiss){
  unsigned N=x.size();
  vector<double> ret={0.0,0.0,0.0};
  for(unsigned i=0;i<N;++i){
	auto v=ele.luminosity_beambeam_parameter(x[i],px[i],y[i],py[i],z[i],pz[i],twiss);
    std::transform(ret.begin(),ret.end(),v.begin(),ret.begin(),std::plus<double>());
  }
  if(_comm!=MPI_COMM_NULL){
      MPI_Allreduce(MPI_IN_PLACE,ret.data(),ret.size(),MPI_DOUBLE,MPI_SUM,_comm);
  }
  return ret;
}
double Beam::luminosity_without_beambeam(const ThinStrongBeam & ele){
  unsigned N=x.size();
  double luminosity=0.0;
  for(unsigned i=0;i<N;++i){
	auto ret=ele.luminosity_without_beambeam(x[i],px[i],y[i],py[i],z[i],pz[i]);
    luminosity+=ret;
  }
  if(_comm!=MPI_COMM_NULL)
      MPI_Allreduce(MPI_IN_PLACE,&luminosity,1,MPI_DOUBLE,MPI_SUM,_comm);
  return luminosity;
}

vector<double> Beam::luminosity_beambeam_parameter(const GaussianStrongBeam & ele, const vector<double> &twiss){
  unsigned N=x.size();
  vector<double> ret={0.0,0.0,0.0};
  for(unsigned i=0;i<N;++i){
	auto v=ele.luminosity_beambeam_parameter(x[i],px[i],y[i],py[i],z[i],pz[i],twiss);
    std::transform(ret.begin(),ret.end(),v.begin(),ret.begin(),std::plus<double>());
  }
  if(_comm!=MPI_COMM_NULL)
      MPI_Allreduce(MPI_IN_PLACE,ret.data(),ret.size(),MPI_DOUBLE,MPI_SUM,_comm);
  return ret;
}
double Beam::luminosity_without_beambeam(const GaussianStrongBeam & ele){
  unsigned N=x.size();
  double luminosity=0.0;
  for(unsigned i=0;i<N;++i){
	auto ret=ele.luminosity_without_beambeam(x[i],px[i],y[i],py[i],z[i],pz[i]);
    luminosity+=ret;
  }
  if(_comm!=MPI_COMM_NULL)
      MPI_Allreduce(MPI_IN_PLACE,&luminosity,1,MPI_DOUBLE,MPI_SUM,_comm);
  return luminosity;
}

vector<double> Beam::get_statistics() const{
    vector<double> ret(20);
    for(unsigned i=0;i<3;++i){
        unsigned dofx=2*i, dofp=2*i+1;
        double mx=get_mean(dofx), mp=get_mean(dofp);
        double vxx=get_variance(dofx,dofx),vpp=get_variance(dofp,dofp), vxp=get_variance(dofx,dofp);
        double ex=vxx*vpp-vxp*vxp;
        ex=(ex<0.0?0.0:std::sqrt(ex));
        double sx=(vxx<0.0?0.0:std::sqrt(vxx));
        double sp=(vpp<0.0?0.0:std::sqrt(vpp));
        ret[6*i]=mx;
        ret[6*i+1]=sx;
        ret[6*i+2]=mp;
        ret[6*i+3]=sp;
        ret[6*i+4]=vxp;
        ret[6*i+5]=ex;
    }
    ret[18]=get_variance(0,4);
    ret[19]=get_variance(2,4);
    return ret;
}

void Beam::write_to_file(const std::string &file, unsigned npart) const{
  if(npart>x.size())
      npart=x.size();
      //return;
  MPI_File fh;
  MPI_File_open(_comm,file.c_str(),MPI_MODE_CREATE|MPI_MODE_WRONLY|MPI_MODE_APPEND,MPI_INFO_NULL,&fh);
  MPI_Offset cur;
  MPI_File_get_position(fh, &cur);

  int rank=0,size=1;
  if(_comm!=MPI_COMM_NULL){
      MPI_Comm_rank(_comm, &rank);
      MPI_Comm_size(_comm, &size);
  }

  vector<double> data(npart*6);
  for(unsigned i=0;i<npart;++i){
	data[i*6]=x[i];
	data[i*6+1]=px[i];
	data[i*6+2]=y[i];
	data[i*6+3]=py[i];
	data[i*6+4]=z[i];
	data[i*6+5]=pz[i];
  }

  MPI_File_write_at_all(fh,cur+rank*npart*sizeof(double)*6,(void*)data.data(),data.size()*sizeof(double),MPI_CHAR,MPI_STATUS_IGNORE);

  MPI_File_close(&fh);
}
/************************************************************************************************************************************************/
int gaussian_beambeam_kick(double &Kx, double &Ky, double sigx, double sigy, double x, double y){
  if(sigx==0.0 || sigy==0.0){
	Kx=0.0;Ky=0.0;
	return 0;
  }
#ifndef LINEAR_BEAMBEAM_KICK
  double dsize=(sigx-sigy)/2.0, msize=sigx-dsize;
  bool negx=false, negy=false;
  if(x<0){
      negx=true;
      x=-x;
  }
  if(y<0){
      negy=true;
      y=-y;
  }
  if(dsize<0.0)
	dsize=-dsize;
  if(dsize/msize<round_beam_threshold){ //round beam
	double rr=x*x+y*y;
	if(rr==0.0){
	  Kx=0.0;Ky=0.0;
	}else{
	  double temp=2.0*(1.0-exp(-rr/2.0/msize/msize))/rr;
	  Kx=temp*x;Ky=temp*y;
	}
	return 0;
  }
  double sig1,sig2,x1,x2;
  if(sigx>sigy){
	sig1=sigx;x1=x;
	sig2=sigy;x2=y;
  }else{
	sig1=sigy;x1=y;
	sig2=sigx;x2=x;
  }
  double denominator=M_SQRT2*sqrt(sig1*sig1-sig2*sig2);
  auto z1=std::complex<double>(x1/denominator,x2/denominator), z2=std::complex<double>(sig2/sig1*x1/denominator, sig1/sig2*x2/denominator);
  double A=2.0*math_const::sqrtpi/denominator, B=exp(-x1*x1/sig1/sig1/2.0-x2*x2/sig2/sig2/2.0);
  auto ret=A*(Faddeeva::w(z1)-B*Faddeeva::w(z2));
  if(sigx>sigy){
	Ky=ret.real();Kx=ret.imag();
  }else{
	Ky=ret.imag();Kx=ret.real();
  }
  if(negx)
      Kx=-Kx;
  if(negy)
      Ky=-Ky;
#else
  double temp=2.0/(sigx+sigy);
  Kx=temp*x/sigx;
  Ky=temp*y/sigy;
#endif
  return 0;
}

/************************************************************************************************************************************************/
ThinStrongBeam::SizeGrowthOption ThinStrongBeam::resolveSGO(const std::string &input){
  auto itr=SGO_dict.find(input);
  if(itr!=SGO_dict.end())
      return itr->second;
  else
      return SGO_Invalid;
}
ThinStrongBeam::CentroidGrowthOption ThinStrongBeam::resolveCGO(const std::string &input){
  auto itr=CGO_dict.find(input);
  if(itr!=CGO_dict.end())
      return itr->second;
  else
      return CGO_Invalid;
}
void ThinStrongBeam::set_beam_size(const std::string &input, const std::vector<double> &sigma, const std::vector<double> &params, int n){
    double sx=sigma[0], sy=sigma[1];
    switch(resolveSGO(input)){
        case SGO_Linear:
            sx*=1.0+params[0]*n;
            sy*=1.0+params[1]*n;
            break;
        case SGO_SIN:
            sx*=1.0+params[0]*std::sin(2.0*M_PI*n*params[1]);
            sy*=1.0+params[2]*std::sin(2.0*M_PI*n*params[3]);
            break;
        case SGO_WhiteNoise:
            {
                trng::normal_dist<double> nX(params[0],params[1]),nY(params[2],params[3]);
                sx*=1.0+nX(Beam::get_rng());
                sy*=1.0+nY(Beam::get_rng());
            }
            break;
        case SGO_Invalid:
            break;
    }
    sigxo=sx;sigyo=sy;
    emitx=sigxo*sigxo/betxo;
    emity=sigyo*sigyo/betyo;
}
void ThinStrongBeam::set_beam_centroid(const std::string &input, const std::vector<double> &params, int n){
    switch(resolveCGO(input)){
        case CGO_Linear:
            xo=params[0]+params[1]*n;
            yo=params[2]+params[3]*n;
            break;
        case CGO_SIN:
            xo=params[0]*std::sin(2.0*M_PI*n*params[1]);
            yo=params[2]*std::sin(2.0*M_PI*n*params[3]);
            break;
        case CGO_COS:
            xo=params[0]*std::cos(2.0*M_PI*n*params[1]);
            yo=params[2]*std::cos(2.0*M_PI*n*params[3]);
            break;
        case CGO_WhiteNoise:
            {
                trng::normal_dist<double> nX(params[0],params[1]),nY(params[2],params[3]);
                xo=nX(Beam::get_rng());
                yo=nY(Beam::get_rng());
            }
            break;
        case CGO_Invalid:
            xo=0.0;yo=0.0;
            break;
    }
}

/************************************************************************************************************************************************/
vector<double> ThinStrongBeam::luminosity_beambeam_parameter(const double &x, const double &px, const double &y, const double &py, const double &z, const double &pz,
        const vector<double> &twiss) const{
  //assert(twiss.size()==6); //beta_x,alpha_x,gamma_x,beta_y,alpha_y,gamma_y
  if(sigxo==0.0 || sigyo==0.0){
	return {0.0,0.0,0.0};
  }
  double S=(z-zo)/2.0;
  double betx=betxo+2*S*alfxo+S*S*gamxo, alfx=alfxo+S*gamxo;
  double bety=betyo+2*S*alfyo+S*S*gamyo, alfy=alfyo+S*gamyo;
  double sigx=sqrt(emitx*betx), sigy=sqrt(emity*bety);

  double bx=twiss[0]-2.0*twiss[1]*S+twiss[2]*S*S;
  double by=twiss[3]-2.0*twiss[4]*S+twiss[5]*S*S;
  double coeff_of_bb_param=-kbb/math_const::twopi/(sigx+sigy);

  double x_xo=x+S*px-xo, y_yo=y+S*py-yo;
  double expterm=exp(-(x_xo*x_xo/sigx/sigx+y_yo*y_yo/sigy/sigy)/2.0);
  return {expterm/math_const::twopi/sigx/sigy,coeff_of_bb_param*bx/sigx,coeff_of_bb_param*by/sigy};
}
double ThinStrongBeam::luminosity_without_beambeam(const double &x, const double &px, const double &y, const double &py, const double &z, const double &pz) const{
  if(sigxo==0.0 || sigyo==0.0){
	return 0.0;
  }
  double S=(z-zo)/2.0;
  double betx=betxo+2*S*alfxo+S*S*gamxo, alfx=alfxo+S*gamxo;
  double bety=betyo+2*S*alfyo+S*S*gamyo, alfy=alfyo+S*gamyo;
  double sigx=sqrt(emitx*betx), sigy=sqrt(emity*bety);

  double x_xo=x+S*px-xo, y_yo=y+S*py-yo;
  double expterm=exp(-(x_xo*x_xo/sigx/sigx+y_yo*y_yo/sigy/sigy)/2.0);
  return expterm/math_const::twopi/sigx/sigy;;
}
double ThinStrongBeam::Pass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
  if(sigxo==0.0 || sigyo==0.0){
	return 0.0;
  }
  //double kcc=2.0*M_PI*200e6/299792458.0, temp_xo=-tan(12.5e-3)*(sin(kcc*z)/kcc-z),temp_yo=0.0;
  //double temp_xo=tan(12.5e-3)*(0.1)*z,temp_yo=0.0;
  double S=(z-zo)/2.0;
//#ifdef SYMPLECTIC_WEAK_STRONG_BEAMBEAM_PASS
#ifdef HIRATA_MAP
  double x1=x+S*px, y1=y+S*py, pz1=pz-0.25*(px*px+py*py);
  double px1=px, py1=py;
  double betx=betxo+2*S*alfxo+S*S*gamxo, alfx=alfxo+S*gamxo;
  double bety=betyo+2*S*alfyo+S*S*gamyo, alfy=alfyo+S*gamyo;
  double sigx=sqrt(emitx*betx), sigy=sqrt(emity*bety);

  double x2=x1, y2=y1, Kx, Ky;
  double x_xo=x2-xo, y_yo=y2-yo;
  gaussian_beambeam_kick(Kx,Ky,sigx,sigy,x_xo,y_yo);
  double px2=px1+Kx*kbb, py2=py1+Ky*kbb;

  double expterm=exp(-(x_xo*x_xo/sigx/sigx+y_yo*y_yo/sigy/sigy)/2.0);
  double Uxx,Uyy;
  double dsize=(sigx-sigy)/2.0, msize=sigx-dsize;
  if(dsize<0.0)
	dsize=-dsize;
  if(dsize/msize<round_beam_threshold){ //round beam
	//Uxx=0.0;Uyy=0.0;
    //
    //Uxx=-kbb*expterm/msize/sigx;
    //Uyy=-kbb*expterm/msize/sigy;
    //
    double rx=x_xo/sigx, ry=y_yo/sigy, r2=x_xo*x_xo+y_yo*y_yo;
    /*
    if((r2/msize/msize)<1e-12){
        Uxx=-kbb/sigx/sigx;
        Uyy=-kbb/sigy/sigy;
    }else{
        Uxx=rx*rx/r2*expterm;
        Uyy=ry*ry/r2*expterm;
        double temp=(rx*rx-ry*ry)*msize*msize/r2/r2*(1.0-expterm);
        Uxx-=temp;
        Uyy+=temp;
        Uxx*=-2.0*kbb;
        Uyy*=-2.0*kbb;
    }
    */
    Uxx=-kbb*expterm/msize/sigx;
    Uyy=-kbb*expterm/msize/sigy;
  }else{
	double temp1=kbb*(x_xo*Kx+y_yo*Ky), temp2=sigx*sigx-sigy*sigy, temp3=sigy/sigx;
	Uxx=temp1-2.0*kbb*(1.0-expterm*temp3);
	Uxx/=temp2;
	Uyy=-temp1+2.0*kbb*(1.0-expterm/temp3);
	Uyy/=temp2;
  }
  //std::cout<<Uxx<<std::endl;
  //std::cout<<Uyy<<std::endl;

  double dsigx2_dz_half=0.5*emitx*(alfx+S*gamxo), dsigy2_dz_half=0.5*emity*(alfy+S*gamyo);
//#ifdef HIRATA_MAP
  double Uz=Uxx*dsigx2_dz_half+Uyy*dsigy2_dz_half;
//#else
  //double Uz=Uxx*dsigx2_dz_half+Uyy*dsigy2_dz_half+kbb/msize*(dsigx2_dz_half/sigx+dsigy2_dz_half/sigy);
//#endif
  double pz2=pz1-Uz;
  //double Uz1=Uxx*dsigx2_dz_half+Uyy*dsigy2_dz_half;
  //double Uz2=Uxx*dsigx2_dz_half+Uyy*dsigy2_dz_half+kbb/msize*(dsigx2_dz_half/sigx+dsigy2_dz_half/sigy);

  x=x2-S*px2;
  px=px2;
  y=y2-S*py2;
  py=py2;
  pz=pz2+0.25*(px2*px2+py2*py2);

#else
  double x1=x+S*px, y1=y+S*py;
  double betx=betxo+2*S*alfxo+S*S*gamxo, alfx=alfxo+S*gamxo;
  double bety=betyo+2*S*alfyo+S*S*gamyo, alfy=alfyo+S*gamyo;
  double sigx=sqrt(emitx*betx), sigy=sqrt(emity*bety);
  double Kx,Ky;
  double x_xo=x1-xo, y_yo=y1-yo;
  gaussian_beambeam_kick(Kx,Ky,sigx,sigy,x_xo,y_yo);
  double expterm=exp(-(x_xo*x_xo/sigx/sigx+y_yo*y_yo/sigy/sigy)/2.0);

  px+=Kx*kbb;
  py+=Ky*kbb;
  x=x1-S*px;
  y=y1-S*py;
#endif

  return expterm/math_const::twopi/sigx/sigy;;
}

std::ostream& operator<<(std::ostream &out, const ThinStrongBeam &b){
  out<<"&"<<b.name<<"\n";
  out<<"  type=\"ThinStrongBeam\",\n";
  out<<"  size="<<b.sigxo<<","<<b.sigyo<<",\n";
  out<<"  beta="<<b.betxo<<","<<b.betyo<<",\n";
  out<<"  alpha="<<b.alfxo<<","<<b.alfyo<<",\n";
  out<<"  zpos="<<b.zo<<",\n";
  out<<"  bb_coefficient="<<b.kbb<<",\t\t //=C1*C2*N*r0/gamma0\n"; 
  out<<"&end";
  return out;
}

/************************************************************************************************************************************************/

GaussianStrongBeam & GaussianStrongBeam::set_equal_area(double sig_length){
  double rr=M_SQRT2*sig_length;
  if(ns%2){
      int w=(ns-1)/2;
      slice_weight[w]=1.0/ns;
      slice_center[w]=0.0;
      for(int i=1;i<=w;++i){
          slice_weight[i+w]=1.0/ns;
          slice_weight[-i+w]=1.0/ns;
          slice_center[i+w]=erfinv(2.0*i/ns)*rr;
          slice_center[-i+w]=-slice_center[i+w];
      }
  }else{
      int w=ns/2;
      for(int i=1;i<=w;++i){
          slice_weight[i+w-1]=1.0/ns;
          slice_weight[-i+w]=1.0/ns;
          slice_center[i+w-1]=erfinv((2.0*i-1.0)/ns)*rr;
          slice_center[-i+w]=-slice_center[i+w-1];
      }
  }
  return *this;
}
GaussianStrongBeam & GaussianStrongBeam::set_equal_width(double sig_length, double width){
    double normalized_width=width/M_SQRT2/sig_length;
    double sum_weight=std::erf(normalized_width*ns/2.0);
    if(ns%2){
        int w=(ns-1)/2;
        slice_center[w]=0.0;
        double upper,lower;
        upper=std::erf(0.5*normalized_width);
        slice_weight[w]=upper/sum_weight;
        for(int i=1;i<=w;++i){
            slice_center[i+w]=width*i;
            slice_center[-i+w]=-slice_center[i+w];
            lower=upper;
            upper=std::erf((i+0.5)*normalized_width);
            slice_weight[i+w]=(upper-lower)/2.0/sum_weight;
            slice_weight[-i+w]=slice_weight[i+w];
        }
    }else{
        int w=ns/2;
        double lower,upper;
        upper=0.0;
        for(int i=1;i<=w;++i){
            slice_center[i+w-1]=(i-0.5)*width;
            slice_center[-i+w]=-slice_center[i+w-1];
            lower=upper;
            upper=std::erf(i*normalized_width);
            slice_weight[i+w-1]=(upper-lower)/2.0/sum_weight;
            slice_weight[-i+w]=slice_weight[i+w-1];
        }
    }
    return *this;
}
GaussianStrongBeam & GaussianStrongBeam::set_hvoffset(double coef, double freq, int dim){
    if(freq<=0.0)
        return *this;
    double kcc=2.0*M_PI*freq/phys_const::clight;
    for(int i=0;i<ns;++i){
        double t=-coef*(std::sin(kcc*slice_center[i])/kcc-slice_center[i]);
        switch(dim){
            case 0:
                slice_hoffset[i]=t;
                break;
            case 1:
                slice_voffset[i]=t;
                break;
            default:
                throw std::runtime_error("Invalide dimension when setting GaussianStrongBeam offset");
                break;
        }
    }
    return *this;
}
