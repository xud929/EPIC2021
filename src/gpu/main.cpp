#include<iostream>
#include<vector>
#include<random>

using std::cout;
using std::endl;
using std::vector;

void test(double *,double *,double *,double *,double *,double *,size_t);

int main(){
    std::mt19937 mt(123456789);
    std::normal_distribution<double> dist(0.0,1.0);

    size_t N=125000;
    vector<double> x(N),px(N),y(N),py(N),z(N),pz(N);
    for(size_t i=0;i<N;++i){
        x[i]=dist(mt)*100e-6;px[i]=dist(mt)*100e-6/0.9;
        y[i]=dist(mt)*20e-6;py[i]=dist(mt)*20e-6/0.18;
        z[i]=dist(mt)*0.07;pz[i]=dist(mt)*6.6e-4;
    }
    test(x.data(),px.data(),y.data(),py.data(),z.data(),pz.data(),N);
    return 0;
}
