//#include"../crab_cavity.h"
#include"crab_cavity.h"
//#include"../linear_map.h"
#include"linear_map.h"
//#include"../Lorentz_boost.h"
#include"Lorentz_boost.h"
#include"../beam.h"
#include"beam.h"
#include"gtrack.h"
#include<math.h>
#include<thrust/device_vector.h>
#include<thrust/copy.h>
#include<thrust/reduce.h>
#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/tuple.h>
#include<iostream>
#include<sys/time.h>

using std::cout;using std::endl;

double now(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return (double)tp.tv_sec+(double)tp.tv_usec*1e-6;
}

int test(double *x, double *px, double *y, double *py, double *z, double *pz, size_t N){
    thrust::device_vector<double> vx(N), vpx(N), vy(N), vpy(N), vz(N), vpz(N);
    thrust::copy(x,x+N,vx.begin());
    thrust::copy(px,px+N,vpx.begin());
    thrust::copy(y,y+N,vy.begin());
    thrust::copy(py,py+N,vpy.begin());
    thrust::copy(z,z+N,vz.begin());
    thrust::copy(pz,pz+N,vpz.begin());

    auto first=thrust::make_zip_iterator(thrust::make_tuple(vx.begin(),vpx.begin(),vy.begin(),vpy.begin(),vz.begin(),vpz.begin()));
    auto last=thrust::make_zip_iterator(thrust::make_tuple(vx.end(),vpx.end(),vy.end(),vpy.end(),vz.end(),vpz.end()));

    gtrack_functor gfun;

    double strength=tan(12.5e-3)/sqrt(0.9*1300);
    ThinCrabCavity tcc(strength,200e6);
    gfun.thin_crab_cavity_before_IP=gThinCrabCavity(tcc);
    gfun.thin_crab_cavity_after_IP=gThinCrabCavity(tcc);

    LinearX MX1(0.9,0.0,1300.0,0.0,-M_PI/2.0);
    LinearX MX2(1300.0,0.0,0.9,0.0, M_PI/2.0);
    LinearX MX3(0.9,0.0,1300.0,0.0, M_PI/2.0);
    LinearX MX4(1300.0,0.0,0.9,0.0,-M_PI/2.0);
    gfun.MX1=gLinearX(MX1);gfun.MX2=gLinearX(MX2);
    gfun.MX3=gLinearX(MX3);gfun.MX4=gLinearX(MX4);

    LorentzBoost lb(12.5e-3);
    gfun.lb=gLorentzBoost(lb);

    RevLorentzBoost rlb(12.5e-3);
    gfun.rlb=gRevLorentzBoost(rlb);

    Linear6D ot({0.9,0.18,0.07/6.6e-4},{0.0,0.0},{0.228,0.210,0.01},{2.0,2.0});
    gfun.oneturn=gLinear6D(ot);

    ThinStrongBeam tsb(-9e-10,{0.7,0.07},{0.0,0.0},{120e-6,20e-6});
    GaussianStrongBeam gsb(4,tsb);
    gsb.set_equal_area(0.02);
    gfun.gsb=gGaussianStrongBeam(gsb);

    // test speed
    cout.precision(16);
    cout.flags(std::ios::scientific);

    double t0,t1;
    t0=now();
    /*
    for(size_t j=0;j<1000;++j){
        thrust::for_each(first,last,gfun);
        cudaDeviceSynchronize();
    }
    */
    thrust::for_each(first,last,gfun);
    cudaDeviceSynchronize();
    t1=now();
    cout<<"gpu: "<<t1-t0<<" seconds."<<endl;

    t0=now();
    for(size_t j=0;j<1000;++j){
        for(size_t i=0;i<N;++i){
            tcc.Pass(x[i],px[i],y[i],py[i],z[i],pz[i]);
            MX1.Pass(x[i],px[i],y[i],py[i],z[i],pz[i]);
            MX2.Pass(x[i],px[i],y[i],py[i],z[i],pz[i]);
             lb.Pass(x[i],px[i],y[i],py[i],z[i],pz[i]);
            gsb.Pass(x[i],px[i],y[i],py[i],z[i],pz[i]);
            rlb.Pass(x[i],px[i],y[i],py[i],z[i],pz[i]);
            MX3.Pass(x[i],px[i],y[i],py[i],z[i],pz[i]);
            MX4.Pass(x[i],px[i],y[i],py[i],z[i],pz[i]);
            tcc.Pass(x[i],px[i],y[i],py[i],z[i],pz[i]);
             ot.Pass(x[i],px[i],y[i],py[i],z[i],pz[i]);
        }
    }
    t1=now();
    cout<<"cpu: "<<t1-t0<<" seconds."<<endl;

    // test correctness
    double sumx=thrust::reduce(vx.begin(),vx.end());
    double sumpx=thrust::reduce(vpx.begin(),vpx.end());
    double sumy=thrust::reduce(vy.begin(),vy.end());
    double sumpy=thrust::reduce(vpy.begin(),vpy.end());
    double sumz=thrust::reduce(vz.begin(),vz.end());
    double sumpz=thrust::reduce(vpz.begin(),vpz.end());

    cout.precision(16);
    cout.flags(std::ios::scientific);
    cout<<sumx<<"\t"<<sumpx<<"\t"<<sumy<<"\t"<<sumpy<<"\t"<<sumz<<"\t"<<sumpz<<endl;

    sumx=0.0;sumpx=0.0;
    sumy=0.0;sumpy=0.0;
    sumz=0.0;sumpz=0.0;
    for(size_t i=0;i<N;++i){
        sumx+=x[i];sumpx+=px[i];
        sumy+=y[i];sumpy+=py[i];
        sumz+=z[i];sumpz+=pz[i];
    }

    cout<<sumx<<"\t"<<sumpx<<"\t"<<sumy<<"\t"<<sumpy<<"\t"<<sumz<<"\t"<<sumpz<<endl;
    return 0;
}
