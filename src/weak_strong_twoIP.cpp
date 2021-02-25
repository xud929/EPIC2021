#include"parser/bb.h"
#include"crab_cavity.h"
#include"Lorentz_boost.h"
#include"linear_map.h"
#include"beam.h"
#include"radiation.h"
#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<random>
#include<vector>
#include<string>
#include<algorithm>
#include<iterator>
#include<unordered_map>

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
    const int output_turns=INT0(Global,output_turns);
    string data_out;
    IFINSTR0(data_out,Global,output);

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

    //GaussianStrongBeam
    double strong_charge=DBL0(GaussianStrongBeam,charge);
    double strong_n_particle=DBL0(GaussianStrongBeam,n_particle);
    double kbb=weak_charge*strong_charge*strong_n_particle*weak_r0/weak_lorentz_factor;
    vector<double> strong_beta={DBL0(GaussianStrongBeam,beta),DBL1(GaussianStrongBeam,beta)};
    vector<double> strong_alpha={DBL0(GaussianStrongBeam,alpha),DBL1(GaussianStrongBeam,alpha)};
    vector<double> strong_sigma={DBL0(GaussianStrongBeam,sizes),DBL1(GaussianStrongBeam,sizes)};
    int strong_zslice=INT0(GaussianStrongBeam,zslice);
    double strong_sigz=DBL2(GaussianStrongBeam,sizes);
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

    double ip2ip_beta1, ip2ip_alpha1, ip2ip_beta2, ip2ip_alpha2, ip2ip_phase;
    ip2ip_beta1=DBL0(IP2IPX,beta);
    ip2ip_beta2=DBL1(IP2IPX,beta);
    ip2ip_alpha1=DBL0(IP2IPX,alpha);
    ip2ip_alpha2=DBL1(IP2IPX,alpha);
    ip2ip_phase=DBL0(IP2IPX,dphase);
    LinearX ip2ipX(ip2ip_beta1,ip2ip_alpha1,ip2ip_beta2,ip2ip_alpha2,ip2ip_phase);
    ip2ip_beta1=DBL0(IP2IPY,beta);
    ip2ip_beta2=DBL1(IP2IPY,beta);
    ip2ip_alpha1=DBL0(IP2IPY,alpha);
    ip2ip_alpha2=DBL1(IP2IPY,alpha);
    ip2ip_phase=DBL0(IP2IPY,dphase);
    LinearY ip2ipY(ip2ip_beta1,ip2ip_alpha1,ip2ip_beta2,ip2ip_alpha2,ip2ip_phase);

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

    vector<double> beam_data_first=wb.get_statistics();

    std::ofstream fout;
    if(!data_out.empty() && rank==0){
        fout.open(data_out,std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
        vector<string> fmt={"turn","luminosity",
                            "mean_x","std_x","mean_px","std_px","var_x_px","ex",
                            "mean_y","std_y","mean_py","std_py","var_y_py","ey",
                            "mean_z","std_z","mean_pz","std_pz","var_z_pz","ez",
                            "var_x_z","var_y_z"};
        std::ostringstream sout;
        std::copy(fmt.begin(),fmt.end(),std::ostream_iterator<string>(sout,","));
        string temp=sout.str();
        temp.pop_back();
        int tlen=1, len=temp.size();

        fout.write((char*)&tlen,sizeof(int));
        fout.write((char*) &len,sizeof(int));
        fout.write(temp.c_str(),temp.size());

        double n=0.0,lum=0.0;
        fout.write((char*)&n,sizeof(double));
        fout.write((char*)&lum,sizeof(double));
        fout.write((char*)beam_data_first.data(),sizeof(double)*beam_data_first.size());
    }

    int current_turn=0;
    vector<double> turns(output_turns), luminosity(output_turns);
    vector<vector<double>> beam_data(output_turns);

    while(current_turn<total_turns){
        for(int i=0;i<output_turns;++i){
            wb.Pass(MX1,tccb,MX2,lb);
            double lum=wb.Pass(gsb);
            wb.Pass(rlb,MX3,tcca,MX4);

            wb.Pass(ip2ipX,ip2ipY);

            wb.Pass(MX1,tccb,MX2,lb);
            lum+=wb.Pass(gsb);
            wb.Pass(rlb,MX3,tcca,MX4);

            wb.Pass(ot,rad);

            turns[i]=current_turn+i+1;
            luminosity[i]=lum*weak_n_particle*strong_n_particle/weak_n_macro;
            beam_data[i]=wb.get_statistics();
            int temp_turn=current_turn+i;
            if(!weak_output.empty()){
                if((temp_turn>=weak_output_start) && (temp_turn<weak_output_end) && ((temp_turn-weak_output_start)%weak_output_step==0))
                    wb.write_to_file(weak_output,weak_output_npart);
            }
        }
        current_turn+=output_turns;
        if(rank==0){
            int write_turn=current_turn+1;
            auto pos=fout.tellp();
            fout.seekp(0);
            fout.write((char*)&write_turn,sizeof(int));
            fout.seekp(pos);
            for(int i=0;i<output_turns;++i){
                fout.write((char*)&turns[i],sizeof(double));
                fout.write((char*)&luminosity[i],sizeof(double));
                fout.write((char*)beam_data[i].data(),beam_data[i].size()*sizeof(double));
            }
        }

        if(weak_rex_limit>0.0 && weak_rey_limit>0.0){
            const auto & bd1=beam_data.back(), & bd0=beam_data.front();
            double ex_blowup=(bd1[5]-bd0[5])/bd0[5], ey_blowup=(bd1[11]-bd0[11])/bd0[11];
            if(ex_blowup>weak_rex_limit || ex_blowup<-weak_rex_limit || ey_blowup>weak_rey_limit || ey_blowup<-weak_rey_limit){
                if(rank==0){
                    cout<<"Emittance blow-up happens."<<endl;
                }
                break;
            }
        }
    }

    if(!data_out.empty() && rank==0){
        fout.close();
    }
    MPI_Finalize();
    return 0;
}
