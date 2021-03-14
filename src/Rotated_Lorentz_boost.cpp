#include"Rotated_Lorentz_boost.h"

std::tuple<
  RotatedLorentzBoost::mat44,
  RotatedLorentzBoost::mat44,
  RotatedLorentzBoost::mat44,
  RotatedLorentzBoost::mat44> rotate_matrix(double thc, double phi, double psi){
    using mat44=RotatedLorentzBoost::mat44;
    using mat33=RotatedLorentzBoost::mat33;
    using vec3=RotatedLorentzBoost::vec3;

    using Eigen::all;
    double cos_thc=std::cos(thc), sin_thc=std::sin(thc);
    double cos_psi=std::cos(psi), sin_psi=std::sin(psi);
    double cos_phi=std::cos(phi), sin_phi=std::sin(phi), t=1.0-cos(phi);

    vec3 N(sin_psi,0.0,cos_psi);
    vec3 Ez(sin_thc,0.0,cos_thc);
    vec3 Iz(sin_thc,0.0,-cos_thc);
    vec3 Ex(cos_thc,0.0,-sin_thc);
    vec3 Ix(cos_thc,0.0,sin_thc);
    vec3 Ey(0.0,1.0,0.0);
    vec3 Iy(0.0,1.0,0.0);

    mat33 T0,T1,T2,T3,R;
    T0<<Ex,Ey,Ez;
    T3<<Ix,Iy,Iz;

    double cp2=cos_psi*cos_psi, sp2=sin_psi*sin_psi, cs2=cos_psi*sin_psi;
    R<<cos_phi+t*sp2,-sin_phi*cos_psi,t*cs2,
       sin_phi*cos_psi,cos_phi,-sin_phi*sin_psi,
       t*cs2,sin_phi*sin_psi,cos_phi+t*cp2;
    T2=R*T0;
    vec3 Ez1=T2(all,2);

    auto Z1=(Ez1-Iz).normalized();
    auto X1=(Ez1+Iz).normalized();
    auto Y1=(Ez1.cross(Iz)).normalized();
    T1<<X1,Y1,Z1;

    double thc1=std::acos(-Ez1.transpose()*Iz)/2.0;
    double tc=std::cos(thc1), tt=std::tan(thc1);
    mat44 L,A,B,C,T14,T24,T34;
    L.setIdentity();A.setIdentity();B.setIdentity();C.setIdentity();
    T14.setIdentity();T24.setIdentity();T34.setIdentity();
    L(0,0)=L(1,1)=1.0/tc;L(0,1)=L(1,0)=-tt;
    A(0,0)=-1.0;A(0,3)=1;
    B(3,0)=1;B(3,3)=-1;
    C(3,3)=-1;
    for(int i=0;i<3;++i){
        for(int j=0;j<3;++j){
            T14(i+1,j+1)=T1(i,j);
            T24(i+1,j+1)=T2(i,j);
            T34(i+1,j+1)=T3(i,j);
        }
    }
    mat44 Ainv=A.inverse(),Binv=B.inverse(),Tinv=T14.inverse();
    mat44 EX=Ainv*L*Tinv*T24*A;
    mat44 EP=Binv*L*Tinv*T24*B/tc;
    mat44 IX=Ainv*L*C*Tinv*T34*A;
    mat44 IP=Binv*L*C*Tinv*T34*B/tc;

    return std::make_tuple(EX,EP,IX,IP);
}

double RotatedLorentzBoost::Pass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
    double t=1.0+pz;
    double ps=std::sqrt(t*t-px*px-py*py);
    vec4 vp,vx;
    vp<<pz,px,py,t-ps;
    vx<<z,x,y,0.0;
    vp=MP*vp;
    vx=MX*vx;

    pz=vp[0];px=vp[1];py=vp[2];ps=1.0+pz-vp[3];
    x=vx[1]-px/ps*vx[3];
    y=vx[2]-py/ps*vx[3];
    z=vx[0]+px/ps*vx[3];
    return 0.0;
}

double RotatedRevLorentzBoost::Pass(double &x, double &px, double &y, double &py, double &z, double &pz) const{
    double s=-(MX(3,0)*z+MX(3,1)*x+MX(3,2)*y)/MX(3,3);
    double t=1.0+pz;
    double ps=std::sqrt(t*t-px*px-py*py),h=t-ps;

    vec4 vx,vp;
    vx<<z-h/ps*s,x+px/ps*s,y+py/ps*s,s;
    vp<<pz,px,py,h;
    vx=MX*vx;
    vp=MP*vp;
    x=vx[1];y=vx[2];z=vx[0];
    px=vp[1];py=vp[2];pz=vp[0];
    return 0.0;
}
