#include"gtrack.h"
//#include"../crab_cavity.h"
#include"crab_cavity.h"
//#include"../linear_map.h"
#include"linear_map.h"
//#include"../Lorentz_boost.h"
#include"Lorentz_boost.h"
//#include"../beam.h"
#include"beam.h"
#include<math.h>
#include<cassert>
#include<iostream>
#include<thrust/complex.h>
#include"gwofz.h"


/**************************************************************/
gThinCrabCavity::gThinCrabCavity(const ThinCrabCavity &tcc){
    phase=tcc.get_phase();
    kcc1=tcc.get_kcc();
    kcc2=kcc1*2.0;
    strength1=tcc.get_strength(1);
    strength2=tcc.get_strength(2);
}

__host__ __device__
void gThinCrabCavity::Pass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
    double a1=kcc1*z+phase, a2=kcc2*z+phase, sin_t, cos_t;

    sincos(a1,&sin_t,&cos_t);
    px-=strength1*sin_t/kcc1;
    pz-=strength1*cos_t*x;

    sincos(a2,&sin_t,&cos_t);
    px-=strength2*sin_t/kcc2;
    pz-=strength2*cos_t*x;

}

/**************************************************************/
gLinearX::gLinearX(const LinearX &m){
    auto t=m.getTM();
    m11=std::get<0>(t);
    m12=std::get<1>(t);
    m21=std::get<2>(t);
    m22=std::get<3>(t);
}

__host__ __device__
void gLinearX::Pass(double &x, double &px) const{
    double u0=x, pu0=px;
    x=m11*u0+m12*pu0;
    px=m21*u0+m22*pu0;
}

/**************************************************************/
gLorentzBoost::gLorentzBoost(const LorentzBoost &m){
    auto t=m.getParams();
    cos_ang=std::get<0>(t);
    sin_ang=std::get<1>(t);
    tan_ang=std::get<2>(t);
}

gRevLorentzBoost::gRevLorentzBoost(const RevLorentzBoost &m){
    auto t=m.getParams();
    cos_ang=std::get<0>(t);
    sin_ang=std::get<1>(t);
    tan_ang=std::get<2>(t);
}

__host__ __device__
void gLorentzBoost::Pass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
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
}

__host__ __device__
void gRevLorentzBoost::Pass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
    double ps=1.0+pz;ps=ps*ps-px*px-py*py;ps=sqrt(ps);
    double h=1.0+pz-ps;

    x-=z*sin_ang;
    x/=1.0+(px+h*sin_ang)*sin_ang/ps;
    z=(z+h/ps*x*sin_ang)*cos_ang;
    y-=py/ps*x*sin_ang;

    pz+=px*sin_ang;
    px=(px+h*sin_ang)*cos_ang;
    py*=cos_ang;
}

/**************************************************************/
gLinear6D::gLinear6D(const Linear6D &m){
    auto t=m.getParams();
    betx=std::get<0>(t);
    bety=std::get<1>(t);
    betz=std::get<2>(t);
    alfx=std::get<3>(t);
    alfy=std::get<4>(t);
    gamx=std::get<5>(t);
    gamy=std::get<6>(t);
     xix=std::get<7>(t);
     xiy=std::get<8>(t);
     mux=std::get<9>(t);
     muy=std::get<10>(t);
     muz=std::get<11>(t);
}

__host__ __device__
void gLinear6D::Pass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
    double x0=x, px0=px, y0=y, py0=py, z0=z, pz0=pz;
    double angle, cos_t, sin_t;

    angle=mux+math_const::twopi*xix*pz0;
    //cos_t=cos(angle);sin_t=sin(angle);
    sincos(angle,&sin_t,&cos_t);
    x=x0*(cos_t+alfx*sin_t)+px0*betx*sin_t;
    px=-x0*sin_t*gamx+px0*(cos_t-alfx*sin_t);

    angle=muy+math_const::twopi*xiy*pz0;
    //cos_t=cos(angle);sin_t=sin(angle);
    sincos(angle,&sin_t,&cos_t);
    y=y0*(cos_t+alfy*sin_t)+py0*bety*sin_t;
    py=-y0*sin_t*gamy+py0*(cos_t-alfy*sin_t);

#ifdef SYMPLECTIC_ONE_TURN_PASS
    double Jx=0.5*(x0*x0*gamx+2.0*alfx*x0*px0+betx*px0*px0), Jy=0.5*(y0*y0*gamy+2.0*alfy*y0*py0+bety*py0*py0);
    cos_t=cos(muz);sin_t=sin(muz);
    double tz=z0+math_const::twopi*(xix*Jx+xiy*Jy);
    z=tz*cos_t+pz0*betz*sin_t;
    pz=-tz*sin_t/betz+pz0*cos_t;
#else
    //cos_t=cos(muz);sin_t=sin(muz);
    sincos(muz,&sin_t,&cos_t);
    z=z0*cos_t+pz0*betz*sin_t;
    pz=-z0*sin_t/betz+pz0*cos_t;
#endif
}

/**************************************************************/
gGaussianStrongBeam::gGaussianStrongBeam(const GaussianStrongBeam &gsb){
    const auto &tsb=gsb.get_tsb();
    double kbb=tsb.get_slice_strength();
    xo=tsb.get_beam_centroid(0);
    yo=tsb.get_beam_centroid(1);
    auto t=tsb.get_beam_sigma();
    s11=std::get<0>(t);
    s12=std::get<1>(t);
    s22=std::get<2>(t);
    s33=std::get<3>(t);
    s34=std::get<4>(t);
    s44=std::get<5>(t);

    const auto &vc=gsb.get_slice_center();
    const auto &vw=gsb.get_slice_weight();
    size_t num_of_slices=vc.size();
    assert(num_of_slices<=MAX_SLICES);

    for(size_t i=0;i<num_of_slices;++i){
        slice_strength[num_of_slices-i-1]=vw[i]*kbb;
        slice_center[num_of_slices-i-1]=vc[i];
    }
    for(size_t i=vc.size();i<MAX_SLICES;++i){
        slice_strength[i]=nan("");
        slice_center[i]=nan("");
    }

    /*
    for(size_t i=0;i<MAX_SLICES;++i){
        std::cout<<slice_center[i]<<"\t"<<slice_strength[i]<<std::endl;
    }
    */
}

//Bassetti-Erskine formula
__host__ __device__ 
void Bassetti_Erskine(double &Kx, double &Ky, double &expterm, double sigx2, double sigy2, double x, double y){
    if(sigx2<=0.0 || sigy2<=0.0){
        Kx=0.0;Ky=0.0;expterm=1.0;
        return;
    }
    expterm=exp(-(x*x/sigx2+y*y/sigy2)/2.0);

    bool negx=(x<0), negy=(y<0), switchxy=(sigy2>sigx2);
    double sig11,sig22,sig1,sig2,x1,x2;//sig1>sig2
    if(switchxy){
        sig11=sigy2;
        sig22=sigx2;
        sig1=sqrt(sigy2);
        sig2=sqrt(sigx2);
        x1=fabs(y);
        x2=fabs(x);
    }else{
        sig11=sigx2;
        sig22=sigy2;
        sig1=sqrt(sigx2);
        sig2=sqrt(sigy2);
        x1=fabs(x);
        x2=fabs(y);
    }


    double dsize=(sig1-sig2)/2.0, msize=(sig1+sig2)/2.0;
    if(dsize/msize<round_beam_threshold){
        double temp=2.0*(1.0-expterm)/(x1*x1+x2*x2);
        Kx=temp*x;Ky=temp*y;
        return;
    }

    double denominator=1.4142135623730951*sqrt(sig11-sig22);
    auto z1=thrust::complex<double>(x1/denominator,x2/denominator), z2=thrust::complex<double>(sig2/sig1*x1/denominator,sig1/sig2*x2/denominator);
    //2*sqrt(pi)...
    auto ret=3.5449077018110318/denominator*(gEPIC::wofz(z1)-expterm*gEPIC::wofz(z2));
    Kx=(switchxy?(negx?-ret.real():ret.real()):(negx?-ret.imag():ret.imag()));
    Ky=(switchxy?(negy?-ret.imag():ret.imag()):(negy?-ret.real():ret.real()));

    return;
}

__host__ __device__ 
void gGaussianStrongBeam::Pass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
    for(size_t i=0;i<MAX_SLICES;++i){
        double zo=slice_center[i], kbb=slice_strength[i];
        if(isnan(z) || isnan(kbb))
            break;
        double S=(z-zo)/2.0;
        double sigma11=s11+S*(s22*S-2.0*s12), sigma33=s33+S*(s44*S-2.0*s34);
        double Kx,Ky,expterm;
        Bassetti_Erskine(Kx,Ky,expterm,sigma11,sigma33,x+S*px-xo,y+S*py-yo);
        double dpx=Kx*kbb, dpy=Ky*kbb;
        px+=dpx;py+=dpy;
        x-=S*dpx;y-=S*dpy;
    }
}
