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
    unsigned long seed=INT0(Global,seed);
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
    const int gaussian_when_luminosity=INT0(Global,gaussian_when_luminosity);
    string lumi_out;
    IFINSTR0(lumi_out,Global,output_luminosity);

    //Beam1
    double beam1_n_particle=DBL0(Beam1,n_particle);
    unsigned beam1_n_macro=INT0(Beam1,n_macro);
    std::array<double,3> beam1_sigma={DBL0(Beam1,transverse_size),DBL1(Beam1,transverse_size),DBL0(Beam1,longitudinal_size)};
    std::array<double,3> beam1_beta={DBL0(Beam1,beta),DBL1(Beam1,beta),DBL0(Beam1,longitudinal_size)/DBL1(Beam1,longitudinal_size)};
    std::array<double,2> beam1_alpha={DBL0(Beam1,alpha),DBL1(Beam1,alpha)};
    double beam1_charge=DBL0(Beam1,charge);
    double beam1_lorentz_factor=DBL0(Beam1,energy)/DBL0(Beam1,mass);
    double beam1_r0=phys_const::re*phys_const::me0/DBL0(Beam1,mass);
    string mom_out1=STR0(Beam1,output_moments);
    int beam1_replace_turn=-1;
    IFININT0(beam1_replace_turn,Beam1,replace_turn);
    std::array<double,6> beam1_replace={0.0};
    IFINDBL0(beam1_replace[0],Beam1,replace_offset);
    IFINDBL1(beam1_replace[2],Beam1,replace_offset);
    IFINDBL2(beam1_replace[4],Beam1,replace_offset);
    IFINDBL0(beam1_replace[1],Beam1,replace_angle);
    IFINDBL1(beam1_replace[3],Beam1,replace_angle);
    IFINDBL2(beam1_replace[5],Beam1,replace_angle);

    int zslice1=INT0(Beam1,zslice),res1=10;
    vector<double> beam1_normalized_boundary;
    string beam1_slice_center=STR0(Beam1,slice_center);
    string beam1_slice_method="equal-area";
    IFINSTR0(beam1_slice_method,Beam1,slice_method);
    if(beam1_slice_method=="equal-area"){
        IFININT0(res1,Beam1,resolution);
    }else if(beam1_slice_method=="erfinv"){
        for(int i=1;i<zslice1;++i)
            beam1_normalized_boundary.push_back(math_const::sqrt2*erfinv((i*2.0)/zslice1-1.0));
    }else
        throw std::runtime_error("Unknown slice method.");

    Beam wb1(beam1_n_macro,beam1_beta,beam1_alpha,beam1_sigma,MPI_COMM_WORLD);

    string beam1_output="";
    int beam1_output_start=0,beam1_output_end=0,beam1_output_step=1,beam1_output_npart=0;
    IFINSTR0(beam1_output,Beam1,output);
    if(!beam1_output.empty()){
        IFININT0(beam1_output_start,Beam1,start_end_step_npart);
        IFININT1(beam1_output_end,Beam1,start_end_step_npart);
        IFININT2(beam1_output_step,Beam1,start_end_step_npart);
        IFININT3(beam1_output_npart,Beam1,start_end_step_npart);
    }

    //Beam2
    double beam2_n_particle=DBL0(Beam2,n_particle);
    unsigned beam2_n_macro=INT0(Beam2,n_macro);
    std::array<double,3> beam2_sigma={DBL0(Beam2,transverse_size),DBL1(Beam2,transverse_size),DBL0(Beam2,longitudinal_size)};
    std::array<double,3> beam2_beta={DBL0(Beam2,beta),DBL1(Beam2,beta),DBL0(Beam2,longitudinal_size)/DBL1(Beam2,longitudinal_size)};
    std::array<double,2> beam2_alpha={DBL0(Beam2,alpha),DBL1(Beam2,alpha)};
    double beam2_charge=DBL0(Beam2,charge);
    double beam2_lorentz_factor=DBL0(Beam2,energy)/DBL0(Beam2,mass);
    double beam2_r0=phys_const::re*phys_const::me0/DBL0(Beam2,mass);
    string mom_out2=STR0(Beam2,output_moments);
    int beam2_replace_turn=-1;
    IFININT0(beam2_replace_turn,Beam2,replace_turn);
    std::array<double,6> beam2_replace={0.0};
    IFINDBL0(beam2_replace[0],Beam2,replace_offset);
    IFINDBL1(beam2_replace[1],Beam2,replace_offset);
    IFINDBL2(beam2_replace[2],Beam2,replace_offset);
    IFINDBL0(beam2_replace[3],Beam2,replace_angle);
    IFINDBL1(beam2_replace[4],Beam2,replace_angle);
    IFINDBL2(beam2_replace[5],Beam2,replace_angle);

    int zslice2=INT0(Beam2,zslice),res2=10;
    vector<double> beam2_normalized_boundary;
    string beam2_slice_center=STR0(Beam2,slice_center);
    string beam2_slice_method="equal-area";
    IFINSTR0(beam2_slice_method,Beam2,slice_method);
    if(beam2_slice_method=="equal-area"){
        IFININT0(res2,Beam2,resolution);
    }else if(beam2_slice_method=="erfinv"){
        for(int i=1;i<zslice2;++i)
            beam2_normalized_boundary.push_back(math_const::sqrt2*erfinv((i*2.0)/zslice2-1.0));
    }else
        throw std::runtime_error("Unknown slice method.");

    Beam wb2(beam2_n_macro,beam2_beta,beam2_alpha,beam2_sigma,MPI_COMM_WORLD);

    string beam2_output="";
    int beam2_output_start=0,beam2_output_end=0,beam2_output_step=1,beam2_output_npart=0;
    IFINSTR0(beam2_output,Beam2,output);
    if(!beam2_output.empty()){
        IFININT0(beam2_output_start,Beam2,start_end_step_npart);
        IFININT1(beam2_output_end,Beam2,start_end_step_npart);
        IFININT2(beam2_output_step,Beam2,start_end_step_npart);
        IFININT3(beam2_output_npart,Beam2,start_end_step_npart);
    }

    //LinearX
    double linear_beta1,linear_beta2,linear_alpha1,linear_alpha2,linear_phase;

    linear_beta1=DBL0(CC2IP1,betax);
    linear_beta2=DBL1(CC2IP1,betax);
    linear_alpha1=DBL0(CC2IP1,alphax);
    linear_alpha2=DBL1(CC2IP1,alphax);
    linear_phase=DBL0(CC2IP1,dphasex);
    LinearX MX11(linear_beta2,linear_alpha2,linear_beta1,linear_alpha1,-linear_phase);
    LinearX MX21(linear_beta1,linear_alpha1,linear_beta2,linear_alpha2,linear_phase);

    linear_beta1=DBL0(IP2CC1,betax);
    linear_beta2=DBL1(IP2CC1,betax);
    linear_alpha1=DBL0(IP2CC1,alphax);
    linear_alpha2=DBL1(IP2CC1,alphax);
    linear_phase=DBL0(IP2CC1,dphasex);
    LinearX MX31(linear_beta1,linear_alpha1,linear_beta2,linear_alpha2,linear_phase);
    LinearX MX41(linear_beta2,linear_alpha2,linear_beta1,linear_alpha1,-linear_phase);

    linear_beta1=DBL0(CC2IP2,betax);
    linear_beta2=DBL1(CC2IP2,betax);
    linear_alpha1=DBL0(CC2IP2,alphax);
    linear_alpha2=DBL1(CC2IP2,alphax);
    linear_phase=DBL0(CC2IP2,dphasex);
    LinearX MX12(linear_beta2,linear_alpha2,linear_beta1,linear_alpha1,-linear_phase);
    LinearX MX22(linear_beta1,linear_alpha1,linear_beta2,linear_alpha2,linear_phase);

    linear_beta1=DBL0(IP2CC2,betax);
    linear_beta2=DBL1(IP2CC2,betax);
    linear_alpha1=DBL0(IP2CC2,alphax);
    linear_alpha2=DBL1(IP2CC2,alphax);
    linear_phase=DBL0(IP2CC2,dphasex);
    LinearX MX32(linear_beta1,linear_alpha1,linear_beta2,linear_alpha2,linear_phase);
    LinearX MX42(linear_beta2,linear_alpha2,linear_beta1,linear_alpha1,-linear_phase);

    //ThinCrabCavity
    auto set_hcc=[](const string &scope, ThinCrabCavity &tcc) -> void {
        const string fieldh="harmonic", fieldr="relative_strength";
        int nh=bbp::count_index<int>(scope,fieldh);
        int nr=bbp::count_index<double>(scope,fieldr);
        if(nh!=nr){
            cerr<<"Harmonics and relative strength are not matched. No harmonics are set."<<endl;
            return;
        }
        for(int i=0;i<nh;++i){
            unsigned tcc_h=bbp::get<int>(scope,fieldh,i);
            double tcc_k=bbp::get<double>(scope,fieldr,i);
            tcc.SetHarmonic(tcc_h,tcc_k);
        }
    };

    double tcc_frequency,tcc_strength,tcc_phase;

    tcc_frequency=DBL0(ThinCrabCavity_before_IP1,frequency);
    tcc_strength=DBL0(ThinCrabCavity_before_IP1,strength);
    tcc_phase=DBL0(ThinCrabCavity_before_IP1,phase);
    ThinCrabCavity tccb1(tcc_strength,tcc_frequency,tcc_phase);
    set_hcc("ThinCrabCavity_before_IP1",tccb1);

    tcc_frequency=DBL0(ThinCrabCavity_after_IP1,frequency);
    tcc_strength=DBL0(ThinCrabCavity_after_IP1,strength);
    tcc_phase=DBL0(ThinCrabCavity_after_IP1,phase);
    ThinCrabCavity tcca1(tcc_strength,tcc_frequency,tcc_phase);
    set_hcc("ThinCrabCavity_after_IP1",tcca1);

    tcc_frequency=DBL0(ThinCrabCavity_before_IP2,frequency);
    tcc_strength=DBL0(ThinCrabCavity_before_IP2,strength);
    tcc_phase=DBL0(ThinCrabCavity_before_IP2,phase);
    ThinCrabCavity tccb2(tcc_strength,tcc_frequency,tcc_phase);
    set_hcc("ThinCrabCavity_before_IP2",tccb2);

    tcc_frequency=DBL0(ThinCrabCavity_after_IP2,frequency);
    tcc_strength=DBL0(ThinCrabCavity_after_IP2,strength);
    tcc_phase=DBL0(ThinCrabCavity_after_IP2,phase);
    ThinCrabCavity tcca2(tcc_strength,tcc_frequency,tcc_phase);
    set_hcc("ThinCrabCavity_after_IP2",tcca2);


    //Lorentz boost and reverse Lorentz boost
    LorentzBoost lb(angle);
    RevLorentzBoost rlb(angle);

    //Linear6D (one turn map)
    std::array<double,3> ot_beta1={DBL0(OneTurn1,beta),DBL1(OneTurn1,beta),DBL2(OneTurn1,beta)};
    std::array<double,2> ot_alpha1={DBL0(OneTurn1,alpha),DBL1(OneTurn1,alpha)};
    std::array<double,3> ot_tune1={DBL0(OneTurn1,tune),DBL1(OneTurn1,tune),DBL2(OneTurn1,tune)};
    std::array<double,2> ot_xi1={DBL0(OneTurn1,chromaticity),DBL1(OneTurn1,chromaticity)};
    Linear6D ot1(ot_beta1,ot_alpha1,ot_tune1,ot_xi1);

    std::array<double,3> ot_beta2={DBL0(OneTurn2,beta),DBL1(OneTurn2,beta),DBL2(OneTurn2,beta)};
    std::array<double,2> ot_alpha2={DBL0(OneTurn2,alpha),DBL1(OneTurn2,alpha)};
    std::array<double,3> ot_tune2={DBL0(OneTurn2,tune),DBL1(OneTurn2,tune),DBL2(OneTurn2,tune)};
    std::array<double,2> ot_xi2={DBL0(OneTurn2,chromaticity),DBL1(OneTurn2,chromaticity)};
    Linear6D ot2(ot_beta2,ot_alpha2,ot_tune2,ot_xi2);

    // Radiation and excitation
    std::array<double,3> damping_turns1={-1.0,-1.0,-1.0},damping_turns2={-1.0,-1.0,-1.0};
    std::array<double,3> excitation_sizes1={-1.0,-1.0,-1.0},excitation_sizes2={-1.0,-1.0,-1.0};
    IFINDBL0(damping_turns1[0],OneTurn1,damping_turns);
    IFINDBL1(damping_turns1[1],OneTurn1,damping_turns);
    IFINDBL2(damping_turns1[2],OneTurn1,damping_turns);
    IFINDBL0(excitation_sizes1[0],OneTurn1,equilibrium_sizes);
    IFINDBL1(excitation_sizes1[1],OneTurn1,equilibrium_sizes);
    IFINDBL2(excitation_sizes1[2],OneTurn1,equilibrium_sizes);
    IFINDBL0(damping_turns2[0],OneTurn2,damping_turns);
    IFINDBL1(damping_turns2[1],OneTurn2,damping_turns);
    IFINDBL2(damping_turns2[2],OneTurn2,damping_turns);
    IFINDBL0(excitation_sizes2[0],OneTurn2,equilibrium_sizes);
    IFINDBL1(excitation_sizes2[1],OneTurn2,equilibrium_sizes);
    IFINDBL2(excitation_sizes2[2],OneTurn2,equilibrium_sizes);

    LumpedRad rad1(damping_turns1,ot_beta1,ot_alpha1,excitation_sizes1);
    LumpedRad rad2(damping_turns2,ot_beta2,ot_alpha2,excitation_sizes2);

    //Poisson solver
    double kbb1=beam1_charge*beam2_charge*beam2_n_particle*beam1_r0/beam1_lorentz_factor;
    double kbb2=beam1_charge*beam2_charge*beam1_n_particle*beam2_r0/beam2_lorentz_factor;
    double klum1=beam1_n_particle*beam2_n_particle/beam1_n_macro;
    double klum2=beam1_n_particle*beam2_n_particle/beam2_n_macro;
    Poisson_Solver::soft_gaussian sg(kbb1,klum1,kbb2,klum2,gaussian_when_luminosity);


    // main loop
    vector<double> beam1_data_first=wb1.get_statistics();
    vector<double> beam2_data_first=wb2.get_statistics();

    /*
    for(unsigned ttt=0;ttt<100;++ttt){
        wb1.Pass(MX11,tccb1,MX21,lb);
        wb2.Pass(MX12,tccb2,MX22,lb);
        double luminosity=beam_beam(wb1,zslice1,res1,wb2,zslice2,res2,sg);
        wb1.Pass(rlb,MX31,tcca1,MX41,ot1);
        wb2.Pass(rlb,MX32,tcca2,MX42,ot2);
        if(rank==0){
            cout<<luminosity<<endl;
        }
    }
    */

    std::ofstream fout1,fout2,flumi;
    if(!lumi_out.empty() && rank==0){
        flumi.open(lumi_out,std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
        vector<string> fmt={"turn","luminosity"};
        std::ostringstream sout;
        std::copy(fmt.begin(),fmt.end(),std::ostream_iterator<string>(sout,","));
        string temp=sout.str();
        temp.pop_back();
        int tlen=0, len=temp.size();

        flumi.write((char*)&tlen,sizeof(int));
        flumi.write((char*) &len,sizeof(int));
        flumi.write(temp.c_str(),temp.size());
    }
    if(!mom_out1.empty() && !mom_out2.empty() && rank==0){
        fout1.open(mom_out1,std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
        fout2.open(mom_out2,std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
        vector<string> fmt={"turn",
                            "mean_x","std_x","mean_px","std_px","var_x_px","ex",
                            "mean_y","std_y","mean_py","std_py","var_y_py","ey",
                            "mean_z","std_z","mean_pz","std_pz","var_z_pz","ez",
                            "var_x_z","var_y_z"};
        std::ostringstream sout;
        std::copy(fmt.begin(),fmt.end(),std::ostream_iterator<string>(sout,","));
        string temp=sout.str();
        temp.pop_back();
        int tlen=1, len=temp.size();

        fout1.write((char*)&tlen,sizeof(int));
        fout1.write((char*) &len,sizeof(int));
        fout1.write(temp.c_str(),temp.size());
        fout2.write((char*)&tlen,sizeof(int));
        fout2.write((char*) &len,sizeof(int));
        fout2.write(temp.c_str(),temp.size());

        double n=0.0;
        fout1.write((char*)&n,sizeof(double));
        fout2.write((char*)&n,sizeof(double));
        fout1.write((char*)beam1_data_first.data(),sizeof(double)*beam1_data_first.size());
        fout2.write((char*)beam2_data_first.data(),sizeof(double)*beam2_data_first.size());
    }


    int current_turn=0;
    vector<double> turns(output_turns), luminosity(output_turns);
    vector<vector<double>> beam1_data(output_turns),beam2_data(output_turns);

    //auto vv=wb2.get_coordinate(4);
    while(current_turn<total_turns){
        for(int i=0;i<output_turns;++i){
            if(current_turn+i==beam1_replace_turn){
                wb1.generate().normalize(beam1_beta,beam1_alpha,beam1_sigma);
                wb1.add_offset(beam1_replace);
            }
            if(current_turn+i==beam2_replace_turn){
                wb2.generate().normalize(beam2_beta,beam2_alpha,beam2_sigma);
                wb2.add_offset(beam2_replace);
            }
            wb1.Pass(MX11,tccb1,MX21,lb);
            wb2.Pass(MX12,tccb2,MX22,lb);
            Beam::slice_type temp_ret1, temp_ret2;
            if(beam1_slice_method=="equal-area")
                temp_ret1=wb1.set_longitudinal_slice_equal_area(zslice1,res1,beam1_slice_center);
            else if(beam1_slice_method=="erfinv")
                temp_ret1=wb1.set_longitudinal_slice_specified(beam1_normalized_boundary,beam1_slice_center);
            else
                throw std::runtime_error("Unknown slice method.");
            if(beam2_slice_method=="equal-area")
                temp_ret2=wb2.set_longitudinal_slice_equal_area(zslice2,res2,beam2_slice_center);
            else if(beam2_slice_method=="erfinv")
                temp_ret2=wb2.set_longitudinal_slice_specified(beam2_normalized_boundary,beam2_slice_center);
            else
                throw std::runtime_error("Unknown slice method.");
            luminosity[i]=beam_beam(wb1,temp_ret1,wb2,temp_ret2,sg);
            wb1.Pass(rlb,MX31,tcca1,MX41,ot1,rad1);//rad1.do_for(wb1);
            wb2.Pass(rlb,MX32,tcca2,MX42,ot2,rad2);//rad2.do_for(wb2);
            turns[i]=current_turn+i+1;
            beam1_data[i]=wb1.get_statistics();
            beam2_data[i]=wb2.get_statistics();
            int temp_turn=current_turn+i;
            if(!beam1_output.empty()){
                if((temp_turn>=beam1_output_start) && (temp_turn<beam1_output_end) && ((temp_turn-beam1_output_start)%beam1_output_step==0))
                    wb1.write_to_file(beam1_output,beam1_output_npart);
            }
            if(!beam2_output.empty()){
                if((temp_turn>=beam2_output_start) && (temp_turn<beam2_output_end) && ((temp_turn-beam2_output_start)%beam2_output_step==0))
                    wb2.write_to_file(beam2_output,beam2_output_npart);
            }
        }
        current_turn+=output_turns;
        if(rank==0){
            int write_turn=current_turn+1;
            auto pos1=fout1.tellp(), pos2=fout2.tellp(), poslu=flumi.tellp();
            fout1.seekp(0);fout2.seekp(0);flumi.seekp(0);
            fout1.write((char*)&write_turn,sizeof(int));
            fout2.write((char*)&write_turn,sizeof(int));
            write_turn-=1;
            flumi.write((char*)&write_turn,sizeof(int));
            fout1.seekp(pos1);fout2.seekp(pos2);flumi.seekp(poslu);
            for(int i=0;i<output_turns;++i){
                flumi.write((char*)&turns[i],sizeof(double));
                fout1.write((char*)&turns[i],sizeof(double));
                fout2.write((char*)&turns[i],sizeof(double));
                flumi.write((char*)&luminosity[i],sizeof(double));
                fout1.write((char*)beam1_data[i].data(),beam1_data[i].size()*sizeof(double));
                fout2.write((char*)beam2_data[i].data(),beam2_data[i].size()*sizeof(double));
            }
        }
    }

    if(!mom_out1.empty() && !mom_out2.empty() && rank==0){
        fout1.close();fout2.close();
    }
    if(!lumi_out.empty() && rank==0){
        flumi.close();
    }
    MPI_Finalize();
    return 0;
}
