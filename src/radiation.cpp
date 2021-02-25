#include"radiation.h"
#include"beam.h"
#include<iostream>

LumpedRad::LumpedRad(const std::array<double,3> &turns, const std::array<double,3> &beta, const std::array<double,2> &alpha, const std::array<double,3> &sigma):
    is_damping(true),is_excitation(true),excitation(9,0.0),damping(3,0.0){
    auto [tx,ty,tz]=turns;
    if(tx<0.0 || ty<0.0 || tz<0.0){
        is_damping=false;
        is_excitation=false;
        return;
    }

    tx=std::exp(-1.0/tx);
    ty=std::exp(-1.0/ty);
    tz=std::exp(-1.0/tz);
    damping[0]=tx;
    damping[1]=ty;
    damping[2]=tz;

    for(unsigned i=0;i<3;++i){
        if(sigma[i]<0.0 || beta[i]<0.0){
            is_excitation=false;
            return;
        }
        double t=1.0-damping[i]*damping[i];
        t=(t<0.0?0.0:std::sqrt(t));
        excitation[3*i]=sigma[i]*t;
        excitation[3*i+1]=excitation[3*i]/beta[i];
        excitation[3*i+2]=(i==2?0.0:-excitation[3*i+1]*alpha[i]);
    }
}

double LumpedRad::Pass(Beam &beam) const{
    trng::mt19937 &mt=beam.get_rng();
    if(beam.get_comm()==MPI_COMM_NULL)
        return 0.0;
    unsigned beg=beam.get_beg(),end=beam.get_end(),totalN=beam.get_total(),N=end-beg;

    if(is_damping){
        for(unsigned i=0;i<6;++i){
            const auto & v=beam.get_coordinate(i);
            for(unsigned j=0;j<v.size();++j)
                beam.set_coordinate(i,j,v[j]*damping[i/2]);
        }
    }
    if(is_excitation){
        trng::normal_dist<double> norm(0.0,1.0);
        std::vector<std::vector<double>> tv(6);
        for(unsigned i=0;i<6;++i){
            tv[i].resize(N);
            mt.discard(beg);
            for(unsigned j=0;j<tv[i].size();++j)
                tv[i][j]=norm(mt);
            mt.discard(totalN-end);
        }
        for(unsigned i=0;i<3;++i){
            const double &A=excitation[3*i], &B=excitation[3*i+1], &C=excitation[3*i+2];
            unsigned idx=2*i, idp=idx+1;
            const auto & vx=beam.get_coordinate(idx), & vp=beam.get_coordinate(idp);
            for(unsigned j=0;j<N;++j){
                beam.set_coordinate(idx,j,vx[j]+A*tv[idx][j]);
                beam.set_coordinate(idp,j,vp[j]+B*tv[idp][j]+C*tv[idx][j]);
            }
        }
    }

    return 0.0;
}
