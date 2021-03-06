#include"parser/bb.h"
#include"crab_cavity.h"
#include"Lorentz_boost.h"
#include"linear_map.h"
#include"beam.h"
#include"radiation.h"
#include"gtrack.h"
#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<random>
#include<string>
#include<algorithm>
#include<iterator>
#include<unordered_map>
#include<map>
#include<cassert>

using std::vector;
using std::cin;using std::cout;using std::cerr;using std::endl;
using std::string;
using std::unordered_map;

int input(const string &, const string &begin);
std::ostream& output(std::ostream &);


int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    if(argc==1){
        cerr<<"Missing input files."<<endl;
        return 1;
    }
    else if(argc==2)
        input(argv[1],"");
    else
        input(argv[1],argv[2]);


    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    int local_rank;
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&local_comm);
    MPI_Comm_rank(local_comm,&local_rank);

    cudaSetDevice(local_rank%deviceCount);

    //Global
    double angle=DBL0(Global,half_crossing_angle);
    unsigned seed=INT0(Global,seed);
    if(seed==0){
        if(rank==0){
            std::random_device rd;
            seed=rd();
        }
    }
    MPI_Bcast(&seed,1,MPI_UNSIGNED,0,MPI_COMM_WORLD);
    Beam::set_rng(seed);
    const int total_turns=INT0(Global,total_turns);
    string data_out=STR0(Global,output);
    assert(!data_out.empty());
    /*scan setup*/
    const double fit_ratio=DBL0(Global,fit_ratio);
    const int fit_step=INT0(Global,fit_step);
    const int fit_start=(1-fit_ratio)*total_turns;
    const double nux_start=DBL0(Global,nux), nux_end=DBL1(Global,nux), nux_step=DBL2(Global,nux);
    const double nuy_start=DBL0(Global,nuy), nuy_end=DBL1(Global,nuy), nuy_step=DBL2(Global,nuy);

    //WeakBeam
    double weak_n_particle=DBL0(WeakBeam,n_particle);
    unsigned weak_n_macro=INT0(WeakBeam,n_macro);
    std::array<double,3> weak_sigma={DBL0(WeakBeam,transverse_size),DBL1(WeakBeam,transverse_size),DBL0(WeakBeam,longitudinal_size)};
    std::array<double,3> weak_beta={DBL0(WeakBeam,beta),DBL1(WeakBeam,beta),DBL0(WeakBeam,longitudinal_size)/DBL1(WeakBeam,longitudinal_size)};
    std::array<double,2> weak_alpha={DBL0(WeakBeam,alpha),DBL1(WeakBeam,alpha)};
    double weak_charge=DBL0(WeakBeam,charge);
    double weak_lorentz_factor=DBL0(WeakBeam,energy)/DBL0(WeakBeam,mass);
    double weak_r0=phys_const::re*phys_const::me0/DBL0(WeakBeam,mass);
    Beam wb(weak_n_macro,weak_beta,weak_alpha,weak_sigma,MPI_COMM_WORLD);
    /*
    string weak_output="";
    int weak_output_start=0,weak_output_end=0,weak_output_step=1,weak_output_npart=0;
    IFINSTR0(weak_output,WeakBeam,output);
    if(!weak_output.empty()){
        IFININT0(weak_output_start,WeakBeam,start_end_step_npart);
        IFININT1(weak_output_end,WeakBeam,start_end_step_npart);
        IFININT2(weak_output_step,WeakBeam,start_end_step_npart);
        IFININT3(weak_output_npart,WeakBeam,start_end_step_npart);
    }
    double weak_rex_limit=-1.0,weak_rey_limit=-1.0;
    IFINDBL0(weak_rex_limit,WeakBeam,emittance_blowup_limit);
    IFINDBL1(weak_rey_limit,WeakBeam,emittance_blowup_limit);
    */

    //GaussianStrongBeam
    double strong_charge=DBL0(GaussianStrongBeam,charge);
    double strong_n_particle=DBL0(GaussianStrongBeam,n_particle);
    double kbb=weak_charge*strong_charge*strong_n_particle*weak_r0/weak_lorentz_factor;
    vector<double> strong_beta={DBL0(GaussianStrongBeam,beta),DBL1(GaussianStrongBeam,beta)};
    vector<double> strong_alpha={DBL0(GaussianStrongBeam,alpha),DBL1(GaussianStrongBeam,alpha)};
    vector<double> strong_sigma={DBL0(GaussianStrongBeam,sizes),DBL1(GaussianStrongBeam,sizes)};
    int strong_zslice=INT0(GaussianStrongBeam,zslice);
    double strong_sigz=DBL2(GaussianStrongBeam,sizes);
    /*
    string strong_growth_method="";
    vector<double> strong_growth_params;
    IFINSTR0(strong_growth_method,GaussianStrongBeam,growth_method);
    if(!strong_growth_method.empty()){
        string tscope="GaussianStrongBeam",tfield="growth_params";
        int num=bbp::count_index<double>(tscope,tfield);
        for(int i=0;i<num;++i)
            strong_growth_params.push_back(bbp::get<double>(tscope,tfield,i));
    }
    string strong_centroid_method="";
    vector<double> strong_centroid_params;
    IFINSTR0(strong_centroid_method,GaussianStrongBeam,centroid_method);
    if(!strong_centroid_method.empty()){
        string tscope="GaussianStrongBeam",tfield="centroid_params";
        int num=bbp::count_index<double>(tscope,tfield);
        for(int i=0;i<num;++i)
            strong_centroid_params.push_back(bbp::get<double>(tscope,tfield,i));
    }
    */
    string strong_method="equal_area";
    IFINSTR0(strong_method,GaussianStrongBeam,slice_method);
    ThinStrongBeam tsb(kbb,strong_beta,strong_alpha,strong_sigma);
    GaussianStrongBeam gsb(strong_zslice,tsb);

    unordered_map<string,int> all_slice_methods={{"equal_area",0},{"equal_width",1}};
    switch(all_slice_methods.at(strong_method)){
        case 0:
            gsb.set_equal_area(strong_sigz);
            break;
        case 1:
            gsb.set_equal_width(strong_sigz,DBL0(GaussianStrongBeam,slice_width));
            break;
        default:
            cerr<<"Slice method "<<strong_method<<" is not implemented."<<endl;
            return 1;
            break;
    }

    double strong_cc_strength=0.0, strong_cc_freq=-1.0;
    IFINDBL0(strong_cc_strength,GaussianStrongBeam,hoffset_params);
    IFINDBL1(strong_cc_freq,GaussianStrongBeam,hoffset_params);
    gsb.set_hvoffset(strong_cc_strength,strong_cc_freq,0);

    //LinearX
    double linear_beta1,linear_beta2,linear_alpha1,linear_alpha2,linear_phase;

    linear_beta1=DBL0(CC2IP,betax);
    linear_beta2=DBL1(CC2IP,betax);
    linear_alpha1=DBL0(CC2IP,alphax);
    linear_alpha2=DBL1(CC2IP,alphax);
    linear_phase=DBL0(CC2IP,dphasex);
    LinearX MX1(linear_beta2,linear_alpha2,linear_beta1,linear_alpha1,-linear_phase);
    LinearX MX2(linear_beta1,linear_alpha1,linear_beta2,linear_alpha2,linear_phase);

    linear_beta1=DBL0(IP2CC,betax);
    linear_beta2=DBL1(IP2CC,betax);
    linear_alpha1=DBL0(IP2CC,alphax);
    linear_alpha2=DBL1(IP2CC,alphax);
    linear_phase=DBL0(IP2CC,dphasex);
    LinearX MX3(linear_beta1,linear_alpha1,linear_beta2,linear_alpha2,linear_phase);
    LinearX MX4(linear_beta2,linear_alpha2,linear_beta1,linear_alpha1,-linear_phase);

    //ThinCrabCavity
    double tcc_frequency,tcc_strength,tcc_phase;

    tcc_frequency=DBL0(ThinCrabCavity_before_IP,frequency);
    tcc_strength=DBL0(ThinCrabCavity_before_IP,strength);
    tcc_phase=DBL0(ThinCrabCavity_before_IP,phase);
    ThinCrabCavity tccb(tcc_strength,tcc_frequency,tcc_phase);

    tcc_frequency=DBL0(ThinCrabCavity_after_IP,frequency);
    tcc_strength=DBL0(ThinCrabCavity_after_IP,strength);
    tcc_phase=DBL0(ThinCrabCavity_after_IP,phase);
    ThinCrabCavity tcca(tcc_strength,tcc_frequency,tcc_phase);

    int nh,nr;
    string scope,fieldh,fieldr;

    scope="ThinCrabCavity_before_IP";fieldh="harmonic";fieldr="relative_strength";
    nh=bbp::count_index<int>(scope,fieldh);
    nr=bbp::count_index<double>(scope,fieldr);
    if(nh!=nr){
        cerr<<"Harmonics and relative strength are not matched"<<endl;
        return 1;
    }
    for(int i=0;i<nh;++i){
        unsigned tcc_h=bbp::get<int>(scope,fieldh,i);
        double tcc_k=bbp::get<double>(scope,fieldr,i);
        tccb.SetHarmonic(tcc_h,tcc_k);
    }

    scope="ThinCrabCavity_after_IP";fieldh="harmonic";fieldr="relative_strength";
    nh=bbp::count_index<int>(scope,fieldh);
    nr=bbp::count_index<double>(scope,fieldr);
    if(nh!=nr){
        cerr<<"Harmonics and relative strength are not matched"<<endl;
        return 1;
    }
    for(int i=0;i<nh;++i){
        unsigned tcc_h=bbp::get<int>(scope,fieldh,i);
        double tcc_k=bbp::get<double>(scope,fieldr,i);
        tcca.SetHarmonic(tcc_h,tcc_k);
    }

    //Lorentz boost and reverse Lorentz boost
    LorentzBoost lb(angle);
    RevLorentzBoost rlb(angle);

    //Linear6D (one turn map)
    std::array<double,3> ot_beta={DBL0(OneTurn,beta),DBL1(OneTurn,beta),DBL2(OneTurn,beta)};
    std::array<double,2> ot_alpha={DBL0(OneTurn,alpha),DBL1(OneTurn,alpha)};
    std::array<double,3> ot_tune={DBL0(OneTurn,tune),DBL1(OneTurn,tune),DBL2(OneTurn,tune)};
    std::array<double,2> ot_xi={DBL0(OneTurn,chromaticity),DBL1(OneTurn,chromaticity)};
    Linear6D ot(ot_beta,ot_alpha,ot_tune,ot_xi);

    // Radiation and excitation
    std::array<double,3> weak_damping_turns={-1.0,-1.0,-1.0}, weak_excitation_sizes={-1.0,-1.0,-1.0};
    IFINDBL0(weak_damping_turns[0],OneTurn,damping_turns);
    IFINDBL1(weak_damping_turns[1],OneTurn,damping_turns);
    IFINDBL2(weak_damping_turns[2],OneTurn,damping_turns);
    IFINDBL0(weak_excitation_sizes[0],OneTurn,equilibrium_sizes);
    IFINDBL1(weak_excitation_sizes[1],OneTurn,equilibrium_sizes);
    IFINDBL2(weak_excitation_sizes[2],OneTurn,equilibrium_sizes);

    LumpedRad rad(weak_damping_turns,ot_beta,ot_alpha,weak_excitation_sizes);

    int finished_cases=0;

    std::ofstream fout;
    if(rank==0){
        fout.open(data_out,std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
        vector<string> fmt={"nux","nuy","luminosity","sigma_x","sigma_y","growth_rate_x","growth_rate_y"};
        std::ostringstream sout;
        std::copy(fmt.begin(),fmt.end(),std::ostream_iterator<string>(sout,","));
        string temp=sout.str();
        temp.pop_back();
        int len=temp.size();

        fout.write((char*)&finished_cases,sizeof(int));
        fout.write((char*) &len,sizeof(int));
        fout.write(temp.c_str(),temp.size());

    }

    double *d_beam_data;
    int output_times=(total_turns-fit_start+fit_step-1)/fit_step;
    unsigned bytes=sizeof(double)*5*output_times;
    cudaMallocManaged((void**)&d_beam_data,bytes);

    vector<double> turns(output_times), h_beam_data(output_times*5);
    for(unsigned i=0;i<output_times;++i)
        turns[i]=fit_start+i*fit_step;

    gtrack gg(wb,tccb,tcca,MX1,MX2,MX3,MX4,lb,rlb,ot,gsb);
    double current_nux=nux_start, current_nuy;
    while(current_nux<nux_end){
        current_nuy=nuy_start;
        while(current_nuy<nuy_end){
            /****track for (current_nux, current_nuy)****/
            gg.gfun.oneturn.mux=current_nux*math_const::twopi;
            gg.gfun.oneturn.muy=current_nuy*math_const::twopi;

            //vector<vector<double>> beam_data(output_times);
            for(unsigned i=0;i<output_times;++i){
                if(i)
                    gg.track3(fit_step,d_beam_data+i*5);
                else
                    gg.track3(fit_start,d_beam_data+i*5);
            }

            //copy to host
            cudaMemcpy(h_beam_data.data(),d_beam_data,bytes,cudaMemcpyDeviceToHost);
            // Reduce all to process 0
            //MPI_Reduce(h_beam_data.data(),h_beam_data.data(),h_beam_data.size(),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE,h_beam_data.data(),h_beam_data.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            for(unsigned i=0;i<h_beam_data.size();i+=5){
                h_beam_data[i]*=weak_n_particle*strong_n_particle/weak_n_macro;
                for(unsigned j=1;j<5;++j)
                    h_beam_data[i+j]/=weak_n_macro;
                for(unsigned j=0;j<2;++j){
                    double &um=h_beam_data[i+1+j*2], &us=h_beam_data[i+2+j*2];
                    us-=um*um;
                    us=us>0?sqrt(us):0;
                }
            }

            //track finished, next calculating the growth rate
            /* fitting x=turns, and y=sigx (beam_data[:,2]) or sigy (beam_data[:,4])
             * y = bet0 + bet1 * x
             * bet1 = (<x * y> -<x> * <y>)/(<x*x>- <x> * <x> )
             * bet0 = <y> -bet1 * <x>
             * growth rate = bet1 / bet0 = (<x * y> - <x> * <y>)/(<y> * <x * x> - <x> * <x * y>)
             */
            if(rank==0){
                double x_mean=0.0, x2_mean=0.0, sx_mean=0.0, sy_mean=0.0, x_sx_mean=0.0, x_sy_mean=0.0;
                for(unsigned i=0;i<output_times;++i){
                    x_mean+=turns[i];
                    x2_mean+=turns[i]*turns[i];
                    sx_mean+=h_beam_data[i*5+2];
                    sy_mean+=h_beam_data[i*5+4];
                    x_sx_mean+=turns[i]*h_beam_data[i*5+2];
                    x_sy_mean+=turns[i]*h_beam_data[i*5+4];
                }
                x_mean/=output_times;
                x2_mean/=output_times;
                sx_mean/=output_times;
                sy_mean/=output_times;
                x_sx_mean/=output_times;
                x_sy_mean/=output_times;
                double gx=(x_sx_mean-x_mean*sx_mean)/(sx_mean*x2_mean-x_mean*x_sx_mean);
                double gy=(x_sy_mean-x_mean*sy_mean)/(sy_mean*x2_mean-x_mean*x_sy_mean);

                const auto &vback=h_beam_data.end()-5;
                vector<double> data2file={current_nux,current_nuy,vback[0],vback[2],vback[4],gx,gy};
                ++finished_cases;
                auto pos=fout.tellp();
                fout.seekp(0);
                fout.write((char*)&finished_cases,sizeof(int));
                fout.seekp(pos);
                fout.write((char*)data2file.data(),data2file.size()*sizeof(double));
            }
            current_nuy+=nuy_step;
        }
        current_nux+=nux_step;
    }
    if(rank==0)
        fout.close();

    cudaFree(&d_beam_data);

    MPI_Finalize();
    return 0;
}
