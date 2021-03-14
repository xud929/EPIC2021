#ifndef CRAB_CAVITY_H
#define CRAB_CAVITY_H

#include<map>
#include<utility>
#include<string>
#include<iostream>
#include"acc_base.h"


class ThinCrabCavity: public AccBase{
public:
  friend std::ostream& operator<<(std::ostream &, const ThinCrabCavity &);
public:
  ThinCrabCavity()=default;
  ThinCrabCavity(double, double, double=0.0, const std::string &key=std::string());
  ThinCrabCavity& SetHarmonic(unsigned h, double k){
	if(harmonic.find(h)!=harmonic.end())
	  std::cerr<<"Override harmonic <"<<h<<"> of crab cavity <"<<name<<">"<<" as "<<k<<std::endl;
	harmonic[h]=k;
	return *this;
  }
  ThinCrabCavity& SetHarmonic(const std::pair<unsigned, double> &p){
	if(harmonic.find(p.first)!=harmonic.end())
	  std::cerr<<"Override harmonic <"<<p.first<<"> of crab cavity <"<<name<<">"<<" as "<<p.second<<std::endl;
	harmonic[p.first]=p.second;
	return *this;
  }
  template<typename...Args>
  ThinCrabCavity& SetHarmonic(const std::pair<unsigned, double> &p, Args&&... args){
	if(harmonic.find(p.first)!=harmonic.end())
	  std::cerr<<"Override harmonic <"<<p.first<<"> of crab cavity <"<<name<<">"<<" as "<<p.second<<std::endl;
	harmonic[p.first]=p.second;
	SetHarmonic(std::forward<Args>(args)...);
	return *this;
  }
  template<typename...Args>
  ThinCrabCavity& SetHarmonic(unsigned h, double k, Args&&... args){
	if(harmonic.find(h)!=harmonic.end())
	  std::cerr<<"Override harmonic <"<<h<<"> of crab cavity <"<<name<<">"<<" as "<<k<<std::endl;
	harmonic[h]=k;
	SetHarmonic(std::forward<Args>(args)...);
	return *this;
  }
  double get_strength(unsigned order) const{
      auto it=harmonic.find(order);
      if(it!=harmonic.end())
          return strength*(it->second);
      else
          return 0.0;
  }
  double get_kcc() const{return kcc;}
  double get_phase() const{return phase;}
  virtual double Pass(double &, double &, double &, double &, double &, double &) const;
  virtual double RPass(double &, double &, double &, double &, double &, double &) const;
private:
  double strength=0.0;
  double frequency=1e8;
  double phase=0.0;
  double kcc=0.0;
  std::map<unsigned, double> harmonic={{1,1.0}};
};

class ThinCrabCavity2D: public AccBase{
public:
  ThinCrabCavity2D()=default;
  ThinCrabCavity2D(double, double, double, double=0.0, const std::string &key=std::string());
  ThinCrabCavity2D& SetHarmonic(unsigned h, double k){
	if(harmonic.find(h)!=harmonic.end())
	  std::cerr<<"Override harmonic <"<<h<<"> of crab cavity <"<<name<<">"<<" as "<<k<<std::endl;
	harmonic[h]=k;
	return *this;
  }
  ThinCrabCavity2D& SetHarmonic(const std::pair<unsigned, double> &p){
	if(harmonic.find(p.first)!=harmonic.end())
	  std::cerr<<"Override harmonic <"<<p.first<<"> of crab cavity <"<<name<<">"<<" as "<<p.second<<std::endl;
	harmonic[p.first]=p.second;
	return *this;
  }
  template<typename...Args>
  ThinCrabCavity2D& SetHarmonic(const std::pair<unsigned, double> &p, Args&&... args){
	if(harmonic.find(p.first)!=harmonic.end())
	  std::cerr<<"Override harmonic <"<<p.first<<"> of crab cavity <"<<name<<">"<<" as "<<p.second<<std::endl;
	harmonic[p.first]=p.second;
	SetHarmonic(std::forward<Args>(args)...);
	return *this;
  }
  template<typename...Args>
  ThinCrabCavity2D& SetHarmonic(unsigned h, double k, Args&&... args){
	if(harmonic.find(h)!=harmonic.end())
	  std::cerr<<"Override harmonic <"<<h<<"> of crab cavity <"<<name<<">"<<" as "<<k<<std::endl;
	harmonic[h]=k;
	SetHarmonic(std::forward<Args>(args)...);
	return *this;
  }
  double get_strengthX(unsigned order) const{
      auto it=harmonic.find(order);
      if(it!=harmonic.end())
          return strengthX*(it->second);
      else
          return 0.0;
  }
  double get_strengthY(unsigned order) const{
      auto it=harmonic.find(order);
      if(it!=harmonic.end())
          return strengthY*(it->second);
      else
          return 0.0;
  }
  double get_kcc() const{return kcc;}
  double get_phase() const{return phase;}
  virtual double Pass(double &, double &, double &, double &, double &, double &) const;
  virtual double RPass(double &, double &, double &, double &, double &, double &) const;
private:
  double strengthX=0.0, strengthY=0.0;
  double frequency=1e8;
  double phase=0.0;
  double kcc=0.0;
  std::map<unsigned, double> harmonic={{1,1.0}};
};
#endif
