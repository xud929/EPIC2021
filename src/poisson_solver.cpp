/* Suppose beam1 is distributed in comm1 while beam2 is distributed in comm2, where comm1 and comm2 are MPI communicators
 * only suitable when comm1==comm2
 * when comm1!=comm2, moments of beam2 (in comm2) should be broadcasted to comm1 and vice versa <--- maybe considered in the future
 */
#include"poisson_solver.h"
#include"beam.h"
#include<algorithm>
#include<functional>
#include<numeric>
#include<tuple>
#include<cmath>
#include<iostream>
#include<fstream>
#include<cassert>

std::tuple<double,double> center_moments(std::vector<double>::const_iterator beg, std::vector<double>::const_iterator end, MPI_Comm comm){
    double sum=0.0,ssum=0.0;
    for(std::vector<double>::const_iterator iter=beg;iter!=end;++iter){
        sum+=*iter;ssum+=(*iter)*(*iter);
    }
    unsigned n=end-beg;
    if(comm!=MPI_COMM_NULL){
        MPI_Allreduce(MPI_IN_PLACE,&sum,1,MPI_DOUBLE,MPI_SUM,comm);
        MPI_Allreduce(MPI_IN_PLACE,&ssum,1,MPI_DOUBLE,MPI_SUM,comm);
        MPI_Allreduce(MPI_IN_PLACE,&n,1,MPI_UNSIGNED,MPI_SUM,comm);
    }
    if(n!=0){
        sum/=n;
        ssum/=n;
        ssum-=sum*sum;
    }
    if(ssum>0.0)
        ssum=std::sqrt(ssum);
    else
        ssum=0.0;
    return std::make_tuple(sum,ssum);
}

double inner_product(std::vector<double>::const_iterator beg, std::vector<double>::const_iterator end,
        std::vector<double>::const_iterator beg2, MPI_Comm comm){
    double sum=std::inner_product(beg,end,beg2,0.0);
    unsigned n=end-beg;
    if(comm!=MPI_COMM_NULL){
        MPI_Allreduce(MPI_IN_PLACE,&sum,1,MPI_DOUBLE,MPI_SUM,comm);
        MPI_Allreduce(MPI_IN_PLACE,&n,1,MPI_UNSIGNED,MPI_SUM,comm);
    }
    if(n!=0)
        return sum/n;
    else
        return 0.0;
}

double Poisson_Solver::soft_gaussian::slice_slice_kick(std::vector<double> &coord1, const std::vector<double> &weight, double sc1,
        const std::array<double,10> &params2, double sc2, const std::tuple<unsigned,double> &info) const{

    double S1=(sc1-sc2)/2.0, S2=-S1;

    const auto &[which,ww]=info;
    double kbb_slice, klum_slice;
    switch(which){
        case 1:
            kbb_slice=ww*kbb2;
            klum_slice=ww*klum2;
            break;
        case 2:
            kbb_slice=ww*kbb1;
            klum_slice=ww*klum1;
            break;
        default:
            throw std::runtime_error("Unspecified strong beam.");
            break;
    }
    const unsigned n1=weight.size();
    assert(n1*6==coord1.size());
    std::vector<double>::iterator iter_x1=coord1.begin(), iter_px1=iter_x1+n1, iter_y1=iter_px1+n1, iter_py1=iter_y1+n1, iter_z1=iter_py1+n1, iter_pz1=iter_z1+n1;

    auto [m_x2,s_x2,m_px2,s_px2,s_x2_px2,m_y2,s_y2,m_py2,s_py2,s_y2_py2]=params2;
    m_x2+=m_px2*S2;
    m_y2+=m_py2*S2;
    s_x2=s_x2*s_x2+(2.0*s_x2_px2+s_px2*s_px2*S2)*S2;
    s_y2=s_y2*s_y2+(2.0*s_y2_py2+s_py2*s_py2*S2)*S2;
    s_x2=(s_x2>0?std::sqrt(s_x2):0.0);
    s_y2=(s_y2>0?std::sqrt(s_y2):0.0);
    double dsize=(s_x2-s_y2)/2.0, msize=s_x2-dsize;
    if(dsize<0)
        dsize=-dsize;
    bool is_round=(dsize/msize<round_beam_threshold);

    /*************************************/
    /*20200903*/
    /*
    int rank=-1;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    std::cout.precision(20);
    */
    /*--------*/
    /*
    if(rank==0 && which==1){
        std::cout<<"Slice sizes after drift "<<S2<<" are:\t"<<s_x2<<"\t"<<s_y2<<std::endl;
    }
    */
    /*************************************/

    double ret=0.0;
    for(unsigned i=0;i<n1;++i){
        const double sw=weight.at(i);
        double &dx=iter_x1[i], &dpx=iter_px1[i], &dy=iter_y1[i], &dpy=iter_py1[i], &dz=iter_z1[i], &dpz=iter_pz1[i];
        double x=dx, px=dpx, y=dy, py=dpy, z=dz, pz=dpz;
#ifdef HIRATA_MAP
        dpz=-0.25*(px*px+py*py);
#endif
        double Kx,Ky;
        double x_xo=x+S1*px-m_x2, y_yo=y+S1*py-m_y2;
        gaussian_beambeam_kick(Kx,Ky,s_x2,s_y2,x_xo,y_yo);
        dpx=Kx*kbb_slice;dpy=Ky*kbb_slice;
        double expterm=std::exp(-(x_xo*x_xo/s_x2/s_x2+y_yo*y_yo/s_y2/s_y2)/2.0);
#ifdef HIRATA_MAP
        double Uxx,Uyy;
        if(is_round){
            Uxx=-kbb_slice*expterm/msize/s_x2;
            Uyy=-kbb_slice*expterm/msize/s_y2;
        }else{
            double temp1=kbb_slice*(x_xo*Kx+y_yo*Ky), temp2=s_x2*s_x2-s_y2*s_y2, temp3=s_y2/s_x2;
            Uxx=temp1-2.0*kbb_slice*(1.0-expterm*temp3);
            Uxx/=temp2;
            Uyy=-temp1+2.0*kbb_slice*(1.0-expterm/temp3);
            Uyy/=temp2;
        }
        double dsigx2_dS2_half=s_x2_px2+S2*s_px2*s_px2, dsigy2_dS2_half=s_y2_py2+S2*s_py2*s_py2, dS2_dz=-0.5;
        dpz-=(Uxx*dsigx2_dS2_half+Uyy*dsigy2_dS2_half)*dS2_dz;
        double tpx=px+dpx, tpy=py+dpy;
        dpz+=0.25*(tpx*tpx+tpy*tpy);
        dpz*=sw;
#endif
        dpx*=sw;dpy*=sw;
        double tS=(z-sc2)/2.0;
        dx=-dpx*tS;
        dy=-dpy*tS;

        if(which==gaussian_when_luminosity){
            ret+=expterm*sw;
        }
    }
    /*
    int rank=-1;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(rank==0)
        std::cout<<ret<<std::endl;
        */
    return ret/math_const::twopi/s_x2/s_y2*klum_slice;
}
double Poisson_Solver::soft_gaussian::slice_slice_kick_no_interpolation(std::vector<double> &coord1,
        const std::array<double,10> &params2, double sc2, const std::tuple<unsigned,double> &info) const{

    const auto &[which,ww]=info;
    double kbb_slice, klum_slice;
    switch(which){
        case 1:
            kbb_slice=ww*kbb2;
            klum_slice=ww*klum2;
            break;
        case 2:
            kbb_slice=ww*kbb1;
            klum_slice=ww*klum1;
            break;
        default:
            throw std::runtime_error("Unspecified strong beam.");
            break;
    }
    const unsigned n1=coord1.size()/6;
    std::vector<double>::iterator iter_x1=coord1.begin(), iter_px1=iter_x1+n1, iter_y1=iter_px1+n1, iter_py1=iter_y1+n1, iter_z1=iter_py1+n1, iter_pz1=iter_z1+n1;

    auto [m0_x2,s0_x2,m0_px2,s0_px2,s0_x2_px2,m0_y2,s0_y2,m0_py2,s0_py2,s0_y2_py2]=params2;

    double ret=0.0;
    for(unsigned i=0;i<n1;++i){
        double &x=iter_x1[i], &px=iter_px1[i], &y=iter_y1[i], &py=iter_py1[i], &z=iter_z1[i], &pz=iter_pz1[i];

        double S1=(z-sc2)/2.0, S2=-S1;
        double m_x2=m0_x2+m0_px2*S2;
        double m_y2=m0_y2+m0_py2*S2;
        double s_x2=s0_x2*s0_x2+(2.0*s0_x2_px2+s0_px2*s0_px2*S2)*S2;
        double s_y2=s0_y2*s0_y2+(2.0*s0_y2_py2+s0_py2*s0_py2*S2)*S2;
        s_x2=(s_x2>0?std::sqrt(s_x2):0.0);
        s_y2=(s_y2>0?std::sqrt(s_y2):0.0);
        double dsize=(s_x2-s_y2)/2.0, msize=s_x2-dsize;
        if(dsize<0)
            dsize=-dsize;
        bool is_round=(dsize/msize<round_beam_threshold);

        x+=px*S1;y+=py*S1;
#ifdef HIRATA_MAP
        pz-=0.25*(px*px+py*py);
#endif
        double Kx,Ky;
        double x_xo=x-m_x2, y_yo=y-m_y2;
        gaussian_beambeam_kick(Kx,Ky,s_x2,s_y2,x_xo,y_yo);
        px+=Kx*kbb_slice;py+=Ky*kbb_slice;
        double expterm=std::exp(-(x_xo*x_xo/s_x2/s_x2+y_yo*y_yo/s_y2/s_y2)/2.0);
#ifdef HIRATA_MAP
        double Uxx,Uyy;
        if(is_round){
            Uxx=-kbb_slice*expterm/msize/s_x2;
            Uyy=-kbb_slice*expterm/msize/s_y2;
        }else{
            double temp1=kbb_slice*(x_xo*Kx+y_yo*Ky), temp2=s_x2*s_x2-s_y2*s_y2, temp3=s_y2/s_x2;
            Uxx=temp1-2.0*kbb_slice*(1.0-expterm*temp3);
            Uxx/=temp2;
            Uyy=-temp1+2.0*kbb_slice*(1.0-expterm/temp3);
            Uyy/=temp2;
        }
        double dsigx2_dS2_half=s_x2_px2+S2*s_px2*s_px2, dsigy2_dS2_half=s_y2_py2+S2*s_py2*s_py2, dS2_dz=-0.5;
        dpz-=(Uxx*dsigx2_dS2_half+Uyy*dsigy2_dS2_half)*dS2_dz;
        pz+=0.25*(px*px+py*py);
#endif
        x-=px*S1;y-=py*S1;

        if(which==gaussian_when_luminosity){
            ret+=expterm/s_x2/s_y2;
        }
    }
    return ret/math_const::twopi*klum_slice;
}

double Poisson_Solver::soft_gaussian::operator()(std::vector<double> &coord1, const std::array<double,4> &param1, MPI_Comm comm1,
        std::vector<double> &coord2, const std::array<double,4> &param2, MPI_Comm comm2) const{
    const auto &[w1,lb1,c1,rb1]=param1;
    const auto &[w2,lb2,c2,rb2]=param2;

    assert(coord1.size()%6==0);
    assert(coord2.size()%6==0);
    const unsigned n1=coord1.size()/6, n2=coord2.size()/6;
    std::vector<double>::const_iterator iter_beg_x1=coord1.cbegin(), iter_end_x1=iter_beg_x1+n1;
    std::vector<double>::const_iterator iter_beg_px1=iter_end_x1, iter_end_px1=iter_beg_px1+n1;
    std::vector<double>::const_iterator iter_beg_y1=iter_end_px1, iter_end_y1=iter_beg_y1+n1;
    std::vector<double>::const_iterator iter_beg_py1=iter_end_y1, iter_end_py1=iter_beg_py1+n1;
    std::vector<double>::const_iterator iter_beg_x2=coord2.cbegin(), iter_end_x2=iter_beg_x2+n2;
    std::vector<double>::const_iterator iter_beg_px2=iter_end_x2, iter_end_px2=iter_beg_px2+n2;
    std::vector<double>::const_iterator iter_beg_y2=iter_end_px2, iter_end_y2=iter_beg_y2+n2;
    std::vector<double>::const_iterator iter_beg_py2=iter_end_y2, iter_end_py2=iter_beg_py2+n2;

    auto [m_x1,s_x1]=center_moments(iter_beg_x1,iter_end_x1,comm1);
    auto [m_px1,s_px1]=center_moments(iter_beg_px1,iter_end_px1,comm1);
    double s_x1_px1=inner_product(iter_beg_x1,iter_end_x1,iter_beg_px1,comm1)-m_x1*m_px1;
    auto [m_y1,s_y1]=center_moments(iter_beg_y1,iter_end_y1,comm1);
    auto [m_py1,s_py1]=center_moments(iter_beg_py1,iter_end_py1,comm1);
    double s_y1_py1=inner_product(iter_beg_y1,iter_end_y1,iter_beg_py1,comm1)-m_y1*m_py1;
    auto [m_x2,s_x2]=center_moments(iter_beg_x2,iter_end_x2,comm2);
    auto [m_px2,s_px2]=center_moments(iter_beg_px2,iter_end_px2,comm2);
    double s_x2_px2=inner_product(iter_beg_x2,iter_end_x2,iter_beg_px2,comm2)-m_x2*m_px2;
    auto [m_y2,s_y2]=center_moments(iter_beg_y2,iter_end_y2,comm2);
    auto [m_py2,s_py2]=center_moments(iter_beg_py2,iter_end_py2,comm2);
    double s_y2_py2=inner_product(iter_beg_y2,iter_end_y2,iter_beg_py2,comm2)-m_y2*m_py2;

    const std::array<double,10> moments1={m_x1,s_x1,m_px1,s_px1,s_x1_px1,m_y1,s_y1,m_py1,s_py1,s_y1_py1};
    const std::array<double,10> moments2={m_x2,s_x2,m_px2,s_px2,s_x2_px2,m_y2,s_y2,m_py2,s_py2,s_y2_py2};

    /*20200903*/
    /*
    int rank=-1;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(rank==0){
        std::ofstream out("moments.txt",std::ofstream::app);
        out.precision(20);
        out.flags(std::ios::scientific);
        for(const auto & i : moments1)
            out<<i<<"\t";
        out<<std::endl;
        for(const auto & i : moments2)
            out<<i<<"\t";
        out<<std::endl;
        out.close();
    }
    */
    /*-------*/
    double slum2=slice_slice_kick_no_interpolation(coord1,moments2,c2,std::make_tuple(2U,w2));
    double slum1=slice_slice_kick_no_interpolation(coord2,moments1,c1,std::make_tuple(1U,w1));

    switch(gaussian_when_luminosity){
        case 1:
            MPI_Allreduce(MPI_IN_PLACE,&slum1,1,MPI_DOUBLE,MPI_SUM,comm2);
            break;
        case 2:
            MPI_Allreduce(MPI_IN_PLACE,&slum2,1,MPI_DOUBLE,MPI_SUM,comm1);
            break;
        default:
            throw std::runtime_error("Unspecified strong beam.");
            break;
    }

    return (gaussian_when_luminosity==2?slum2:slum1);

/* interpolation
    std::vector<double> wL1(n1), wR1(n1), wL2(n2), wR2(n2);
    for(unsigned i=0;i<n1;++i){
        wL1[i]=(rb1-coord1[4*n1+i])/(rb1-lb1);
        wR1[i]=1.0-wL1[i];
    }
    for(unsigned i=0;i<n2;++i){
        wL2[i]=(rb2-coord2[4*n2+i])/(rb2-lb2);
        wR2[i]=1.0-wL2[i];
    }

    double slice_luminosity=0.0, temp_luminosity;
    // collision of beam1 (left and right boundary) with beam2 (center)
    std::vector<double> L1=coord1,R1=coord1;
    temp_luminosity=slice_slice_kick(L1,wL1,lb1,moments2,c2,std::make_tuple(2U,w2));
    //temp_luminosity=slice_slice_kick3(L1,wL1,lb1,comm1,coord2,c2,comm2,std::make_tuple(2U,w2));
    if(gaussian_when_luminosity==2)
        slice_luminosity+=temp_luminosity;
    temp_luminosity=slice_slice_kick(R1,wR1,rb1,moments2,c2,std::make_tuple(2U,w2));
    //temp_luminosity=slice_slice_kick3(R1,wR1,rb1,comm1,coord2,c2,comm2,std::make_tuple(2U,w2));
    if(gaussian_when_luminosity==2)
        slice_luminosity+=temp_luminosity;

    // collision of beam2 (left and right boundary) with beam1 (center)
    std::vector<double> L2=coord2,R2=coord2;
    temp_luminosity=slice_slice_kick(L2,wL2,lb2,moments1,c1,std::make_tuple(1U,w1));
    //temp_luminosity=slice_slice_kick3(L2,wL2,lb2,comm2,coord1,c1,comm1,std::make_tuple(1U,w1));
    if(gaussian_when_luminosity==1)
        slice_luminosity+=temp_luminosity;
    temp_luminosity=slice_slice_kick(R2,wR2,rb2,moments1,c1,std::make_tuple(1U,w1));
    //temp_luminosity=slice_slice_kick3(R2,wR2,rb2,comm2,coord1,c1,comm1,std::make_tuple(1U,w1));
    if(gaussian_when_luminosity==1)
        slice_luminosity+=temp_luminosity;


    switch(gaussian_when_luminosity){
        case 1:
            MPI_Allreduce(MPI_IN_PLACE,&slice_luminosity,1,MPI_DOUBLE,MPI_SUM,comm2);
            break;
        case 2:
            MPI_Allreduce(MPI_IN_PLACE,&slice_luminosity,1,MPI_DOUBLE,MPI_SUM,comm1);
            break;
        default:
            throw std::runtime_error("Unspecified strong beam.");
            break;
    }

    // update coord1 and coord2
    for(unsigned i=0;i<n1;++i){
        for(unsigned j=0;j<4;++j){
            unsigned index=j*n1+i;
            coord1[index]+=L1[index]+R1[index];
        }
#ifdef HIRATA_MAP
        unsigned index=5*n1+i;
        coord1[index]+=L1[index]+R1[index];
#endif
    }
    for(unsigned i=0;i<n2;++i){
        for(unsigned j=0;j<4;++j){
            unsigned index=j*n2+i;
            coord2[index]+=L2[index]+R2[index];
        }
#ifdef HIRATA_MAP
        unsigned index=5*n2+i;
        coord2[index]+=L2[index]+R2[index];
#endif
    }

    return slice_luminosity;
*/
}
