#ifndef CONSTANTS_H
#define CONSTANTS_H
#include<cmath>

namespace phys_const{
  constexpr double clight=299792458.0;  // m/s
  constexpr double re=2.8179403227e-15; // m
  constexpr double me0=0.51099895e6;    // eV
}

namespace math_const{
  constexpr double pi=3.141592653589793238462643383279502884197169;
  //constexpr double twopi=2.0*M_PI;
  constexpr double twopi=6.283185307179586476925286766559005768394338;
  //constexpr double sqrt2pi=M_PI*M_2_SQRTPI;
  constexpr double sqrt2pi=2.506628274631000502415765284811045253006964;
  //constexpr double sqrtpi=sqrt2pi/2.0;
  constexpr double sqrtpi=1.772453850905516027298167483341145182797554;
  constexpr double sqrt2=1.414213562373095048801688724209698078569662;
}

#endif
