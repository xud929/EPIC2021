#ifndef GTRACKH
#define GTRACKH
#include<thrust/tuple.h>
#include<thrust/device_vector.h>

class ThinCrabCavity;
class LinearX;
class LorentzBoost;
class RevLorentzBoost;
class Linear6D;
class GaussianStrongBeam;

struct gThinCrabCavity{
    gThinCrabCavity()=default;
    gThinCrabCavity(const ThinCrabCavity &);
    __host__ __device__ void Pass(double &, double &, double &, double &, double &, double &) const;
private:
    double phase,kcc1,kcc2,strength1,strength2;
};

struct gLinearX{
    gLinearX()=default;
    gLinearX(const LinearX&);
    __host__ __device__ void Pass(double &, double &) const;
private:
    double m11,m12,m21,m22;
};

struct gLorentzBoost{
    gLorentzBoost()=default;
    gLorentzBoost(const LorentzBoost&);
    __host__ __device__ void Pass(double &, double &, double &, double &, double &, double &) const;
private:
    double cos_ang,sin_ang,tan_ang;
};

struct gRevLorentzBoost{
    gRevLorentzBoost()=default;
    gRevLorentzBoost(const RevLorentzBoost&);
    __host__ __device__ void Pass(double &, double &, double &, double &, double &, double &) const;
private:
    double cos_ang,sin_ang,tan_ang;
};

struct gLinear6D{
    gLinear6D()=default;
    gLinear6D(const Linear6D&);
    __host__ __device__ void Pass(double &, double &, double &, double &, double &, double &) const;
private:
    double betx,bety,betz,alfx,alfy,gamx,gamy,xix,xiy,mux,muy,muz;
};

struct gGaussianStrongBeam{
    gGaussianStrongBeam()=default;
    gGaussianStrongBeam(const GaussianStrongBeam&);
    __host__ __device__ void Pass(double &, double &, double &, double &, double &, double &) const;
private:
#ifndef MAX_SLICES
#define MAX_SLICES 10
#endif
    //array in struct can be deeply copied (not address)
    //so we can pass it to gpu
    //the disadvantage is we have to reserve enough space to store the slice information
    double slice_center[MAX_SLICES];
    double slice_strength[MAX_SLICES];
    double xo,yo,s11,s12,s22,s33,s34,s44;
};

struct gtrack_functor{
    gThinCrabCavity thin_crab_cavity_before_IP, thin_crab_cavity_after_IP;
    // lattice = (cc1,IP,cc2), cc1 is thin crab cavity before IP
    // then MX1 is the inverse map from cc1 to IP
    //      MX4 is the inverse map from IP to cc2
    gLinearX MX1,MX2,MX3,MX4; 
    gLorentzBoost lb;
    gRevLorentzBoost rlb;
    //one turn map
    gLinear6D oneturn;
    //strong beam
    gGaussianStrongBeam gsb;

    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple t) const{
        double x=thrust::get<0>(t), px=thrust::get<1>(t);
        double y=thrust::get<2>(t), py=thrust::get<3>(t);
        double z=thrust::get<4>(t), pz=thrust::get<5>(t);

        for(int i=0;i<1000;++i){
            thin_crab_cavity_before_IP.Pass(x,px,y,py,z,pz);
            MX1.Pass(x,px);
            MX2.Pass(x,px);
            lb.Pass(x,px,y,py,z,pz);
            gsb.Pass(x,px,y,py,z,pz);
            rlb.Pass(x,px,y,py,z,pz);
            MX3.Pass(x,px);
            MX4.Pass(x,px);
            thin_crab_cavity_after_IP.Pass(x,px,y,py,z,pz);
            oneturn.Pass(x,px,y,py,z,pz);
        }

        //cudaDeviceSynchronize();

        thrust::get<0>(t)=x;
        thrust::get<1>(t)=px;
        thrust::get<2>(t)=y;
        thrust::get<3>(t)=py;
        thrust::get<4>(t)=z;
        thrust::get<5>(t)=pz;
    }
};

class gtrack{
public:
    gtrack(const Beam&, );
private:
    thrust::device_vector<double> vx,vpx,vy,vpy,vz,vpz;
    gtrack_functor gfun;
};

#endif
