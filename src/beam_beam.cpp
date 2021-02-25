#include"beam.h"
#include<limits>
#include<iostream>
#include<stdexcept>
#include<algorithm>
#include<map>
#include<array>
#include<iostream>
#include<tuple>
#include<cassert>
#include<string>
#include<fstream>

/*Defined in poisson_solver.cpp*/
std::tuple<double,double> center_moments(std::vector<double>::const_iterator beg, std::vector<double>::const_iterator end, MPI_Comm comm);

/* 
 * find min and max value in specified dimension
 * dimension definition:
 * 0 ---> x
 * 1 ---> px
 * 2 ---> y
 * 3 ---> py
 * 4 ---> z
 * 5 ---> pz
 */
std::tuple<double,double> Beam::min_max(unsigned dimension) const{
    const std::vector<double> &v=*ptr_coord[dimension];
    double fmin=std::numeric_limits<double>::max();
    double fmax=std::numeric_limits<double>::min();

    for(unsigned i=0;i!=v.size();++i){
        if(v[i]<fmin)
            fmin=v[i];
        if(v[i]>fmax)
            fmax=v[i];
    }

    if(_comm!=MPI_COMM_NULL){
        MPI_Allreduce(MPI_IN_PLACE,&fmin,1,MPI_DOUBLE,MPI_MIN,_comm);
        MPI_Allreduce(MPI_IN_PLACE,&fmax,1,MPI_DOUBLE,MPI_MAX,_comm);
    }

    return std::make_tuple(fmin,fmax);
}

/*
 * Compute the histogram in specified dimension
 * return value:
 * (1) bin width
 * (2) bin centers
 * (3) bin frequency (normalized by the number of total macro particles)
 */
std::tuple<double,std::vector<double>,std::vector<double>> Beam::hist(unsigned dimension, unsigned bins) const{
    const auto &[fmin,fmax]=min_max(dimension);
    double width=(fmax-fmin)/bins;

    /*
    int rank=-1;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(rank==0){
        std::cout<<fmin<<"\t"<<fmax<<std::endl;
    }
    */

    std::vector<unsigned> counts(bins,0);
    const std::vector<double> &v=get_coordinate(dimension);
    for(unsigned i=0;i!=v.size();++i){
        unsigned which=(v[i]-fmin)/width;
        if(which==bins)
            --which;
        ++counts[which];
    }
    if(_comm!=MPI_COMM_NULL)
        MPI_Allreduce(MPI_IN_PLACE,counts.data(),bins,MPI_UNSIGNED,MPI_SUM,_comm);
    double sum=std::accumulate(counts.begin(),counts.end(),0.0);

    std::vector<double> freq(bins,0.0),centers(bins,0.0);
    for(unsigned i=0;i!=bins;++i){
        if(sum>0.0)
            freq[i]=counts[i]/sum;
        centers[i]=fmin+(i+0.5)*width;
    }
    return std::make_tuple(width,centers,freq);
}

/*
 * Slice the beam according the particles longitudinal position. There are nearly same number of particles in each slice.
 * Firstly, call function Beam::hist to perform hist statistics along the longitudinal direction.
 * Secondly, determine each slice boundaries by linear interpolating.
 * At last, find all particles in each slice. The slice center is the average of the longitudinal position of all particles in it.
 * parameters:
 * (1) zslice, the number of slices you wanted
 * (2) resolution, the number of bins in the first step equals to zslice*resolution
 * return value:
 * (1) the total particles, if there is no particle loss, it should be equal to Nmacro
 * (2) a vector of slice center, every slice longitudinal position
 * (3) a vector of slice weight, it equals to <particles in this slice>/<the total particles>
 * (4) a vector of slice boundary, the first and last values should be zmin and zmax
 * (5) a vector of particle index vectors
 */
Beam::slice_type Beam::set_longitudinal_slice_equal_area(unsigned zslice, unsigned resolution, const std::string &slice_center_pos) const{
    if(zslice==0)
        throw std::runtime_error("Error! Slice number should be greater than 0.");
    unsigned bins=zslice*resolution;
    const auto &[width,center,height]=hist(4,bins);

    std::vector<double> cumheight(bins);
    std::partial_sum(height.begin(),height.end(),cumheight.begin());
    cumheight.back()=1.0;

    std::vector<double> slice_boundary(zslice+1);
    slice_boundary.front()=center.front()-width/2.0;
    slice_boundary.back()=center.back()+width/2.0;

    auto current=cumheight.begin();
    for(unsigned i=1;i<zslice;++i){
        double value=static_cast<double>(i)/zslice;
        current=std::upper_bound(current,cumheight.end(),value);
        double y2=*current,y1=*(current-1);
        double x1,x2;
        auto pos=current-cumheight.begin();
        if(current==cumheight.begin()){
            x1=slice_boundary.front();
            x2=center[pos];
        }else if(current==cumheight.end()){
            x1=center[pos-1];
            x2=slice_boundary.back();
        }else{
            x2=center[pos];
            x1=center[pos-1];
        }
        slice_boundary[i]=x2*(value-y1)/(y2-y1)+x1*(value-y2)/(y1-y2);
    }

    std::vector<double> slice_center(zslice,0.0),slice_weight(zslice,0.0);
    std::vector<std::vector<unsigned>> particle_index_in_slice(zslice);

    for(unsigned i=0;i!=z.size();++i){
        auto which=std::upper_bound(slice_boundary.begin(),slice_boundary.end(),z[i])-slice_boundary.begin();
        if(which==slice_boundary.size()) // in case zmax overflows
            --which;
        if(which==0) //in case zmin overflows
            which=1;
        particle_index_in_slice[which-1].push_back(i);
    }

    for(unsigned i=0;i!=zslice;++i)
        slice_weight[i]=particle_index_in_slice[i].size();
    if(_comm!=MPI_COMM_NULL)
        MPI_Allreduce(MPI_IN_PLACE,slice_weight.data(),slice_weight.size(),MPI_DOUBLE,MPI_SUM,_comm);
    double sum=std::accumulate(slice_weight.begin(),slice_weight.end(),0.0);

    if(slice_center_pos=="centroid"){
        for(unsigned i=0;i!=zslice;++i){
            for(const auto & j : particle_index_in_slice[i])
                slice_center[i]+=z[j];
        }
        if(_comm!=MPI_COMM_NULL)
            MPI_Allreduce(MPI_IN_PLACE,slice_center.data(),slice_center.size(),MPI_DOUBLE,MPI_SUM,_comm);

        for(unsigned i=0;i!=zslice;++i){
            if(slice_weight[i]>0.0)
                slice_center[i]/=slice_weight[i];
        }
    }else if(slice_center_pos=="midpoint"){
        for(unsigned i=0;i!=zslice;++i)
            slice_center[i]=(slice_boundary[i]+slice_boundary[i+1])/2.0;
    }else{
        throw std::runtime_error("Unknown slice center style.");
    }

    for(unsigned i=0;i!=zslice;++i){
        if(sum>0.0)
            slice_weight[i]/=sum;
        else
            slice_weight[i]=0.0;
    }

    return std::make_tuple(sum,slice_center,slice_weight,slice_boundary,particle_index_in_slice);
}
Beam::slice_type Beam::set_longitudinal_slice_specified(const std::vector<double> &zpos, const std::string &slice_center_pos) const{
    const auto &[zmin,zmax]=min_max(4);
    const auto &[mean_z,std_z]=center_moments(z.cbegin(),z.cend(),_comm);
    const unsigned zslice=zpos.size()+1;
    std::vector<double> slice_boundary(zslice+1);

    slice_boundary.front()=zmin;
    slice_boundary.back()=zmax;

    for(unsigned i=0;i<zpos.size();++i){
        double t=zpos[i]*std_z+mean_z;
        if(t<zmin)
            slice_boundary[i+1]=zmin;
        else if(t>zmax)
            slice_boundary[i+1]=zmax;
        else
            slice_boundary[i+1]=t;
    }

    std::vector<double> slice_center(zslice,0.0),slice_weight(zslice,0.0);
    std::vector<std::vector<unsigned>> particle_index_in_slice(zslice);

    for(unsigned i=0;i!=z.size();++i){
        auto which=std::upper_bound(slice_boundary.begin(),slice_boundary.end(),z[i])-slice_boundary.begin();
        if(which==slice_boundary.size()) // in case zmax overflows
            --which;
        if(which==0) //in case zmin overflows
            which=1;
        particle_index_in_slice[which-1].push_back(i);
    }

    for(unsigned i=0;i!=zslice;++i)
        slice_weight[i]=particle_index_in_slice[i].size();
    if(_comm!=MPI_COMM_NULL)
        MPI_Allreduce(MPI_IN_PLACE,slice_weight.data(),slice_weight.size(),MPI_DOUBLE,MPI_SUM,_comm);
    double sum=std::accumulate(slice_weight.begin(),slice_weight.end(),0.0);

    if(slice_center_pos=="centroid"){
        for(unsigned i=0;i!=zslice;++i){
            for(const auto & j : particle_index_in_slice[i])
                slice_center[i]+=z[j];
        }
        if(_comm!=MPI_COMM_NULL)
            MPI_Allreduce(MPI_IN_PLACE,slice_center.data(),slice_center.size(),MPI_DOUBLE,MPI_SUM,_comm);

        for(unsigned i=0;i!=zslice;++i){
            if(slice_weight[i]>0.0)
                slice_center[i]/=slice_weight[i];
        }
    }else if(slice_center_pos=="midpoint"){
        for(unsigned i=0;i!=zslice;++i)
            slice_center[i]=(slice_boundary[i]+slice_boundary[i+1])/2.0;
    }else{
        throw std::runtime_error("Unknown slice center style.");
    }

    for(unsigned i=0;i!=zslice;++i){
        if(sum>0.0)
            slice_weight[i]/=sum;
        else
            slice_weight[i]=0.0;
    }

    return std::make_tuple(sum,slice_center,slice_weight,slice_boundary,particle_index_in_slice);
}

double beam_beam(Beam &beam1, const Beam::slice_type &slice_ret1, Beam &beam2, const Beam::slice_type &slice_ret2, const Poisson_Solver::solver_base &ps){
    double ret_lum=0.0;

    // slice 
    //const auto & [total1,center1,weight1,boundary1,index1]=beam1.set_longitudinal_slice_equal_area(ns1,res1);
    //const auto & [total2,center2,weight2,boundary2,index2]=beam2.set_longitudinal_slice_equal_area(ns2,res2);
    const auto & [total1,center1,weight1,boundary1,index1]=slice_ret1;
    const auto & [total2,center2,weight2,boundary2,index2]=slice_ret2;

    /*20200903*/
    /*
    int rank=-1,size=-1;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    */
    /*-------*/
    /*20200903*/
    /*
    if(rank==0){
        //std::cout<<center1.size()<<"\t";
        //std::cout<<weight1.size()<<"\t";
        //std::cout<<center2.size()<<"\t";
        //std::cout<<weight2.size()<<"\t";
        //std::cout<<std::endl;
        //for(const auto & i : weight1)
            //std::cout<<i<<"\t";
        //std::cout<<std::endl;
        //for(const auto & i : weight2)
            //std::cout<<i<<"\t";
        //std::cout<<"\n"<<std::endl;
        for(unsigned i=0;i<center1.size();++i)
            std::cout<<center1[i]<<"\t"<<weight1[i]<<"\n";
        std::cout<<std::endl;
    }
    */
    /*-------*/


    const unsigned ns1=center1.size(), ns2=center2.size();

    // collision time t = -(c1+c2)/2.0
    // where c1,c2 are two slices center
    std::map<double,std::pair<unsigned,unsigned>> collision_order;
    for(unsigned i=0;i<ns1;++i)
        for(unsigned j=0;j<ns2;++j)
            collision_order[-(center1[i]+center2[j])/2.0]=std::make_pair(i,j);

    // copy beam1 coordinates to coord1 and beam2 coordinates to coord2
    std::vector<std::vector<double>> coord1(ns1), coord2(ns2);
    for(unsigned k=0;k<ns1;++k){
        coord1[k].resize(index1[k].size()*6);
        for(unsigned i=0;i<index1[k].size();++i)
            for(unsigned j=0;j<6;++j)
                coord1[k][i+j*index1[k].size()]=beam1.get_coordinate(j,index1[k][i]);
    }
    for(unsigned k=0;k<ns2;++k){
        coord2[k].resize(index2[k].size()*6);
        for(unsigned i=0;i<index2[k].size();++i)
            for(unsigned j=0;j<6;++j)
                coord2[k][i+j*index2[k].size()]=beam2.get_coordinate(j,index2[k][i]);
    }

    for(const auto & each_pair : collision_order){
        const unsigned & s1=each_pair.second.first, s2=each_pair.second.second;

        std::array<double,4> param1{weight1[s1],boundary1[s1],center1[s1],boundary1[s1+1]};
        std::array<double,4> param2{weight2[s2],boundary2[s2],center2[s2],boundary2[s2+1]};

        /*20200903*/
        /*
        if(rank==0){
            std::ofstream out("log.txt",std::ofstream::app);
            //std::ostream &out=std::cout;
            out.flags(std::ios::scientific);
            out.precision(20);
            out<<s1<<"\t";
            for(const auto & i : param1)
                out<<i<<"\t";
            out<<std::endl;
            out<<s2<<"\t";
            for(const auto & i : param2)
                out<<i<<"\t";
            out<<std::endl;
            out.close();
        }
        MPI_File fh;
        MPI_File_open(MPI_COMM_WORLD,"2_before.txt",MPI_MODE_CREATE|MPI_MODE_WRONLY|MPI_MODE_APPEND,MPI_INFO_NULL,&fh);
        MPI_Offset cur;
        MPI_File_get_position(fh,&cur);
        unsigned t1pos=0,t2pos=coord2[s2].size();
        if(rank!=0){
            MPI_Recv(&t1pos,1,MPI_UNSIGNED,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            t2pos+=t1pos;
        }
        if(rank<size-1){
            MPI_Send(&t2pos,1,MPI_UNSIGNED,rank+1,0,MPI_COMM_WORLD);
        }
        unsigned ttlen=t2pos-t1pos;
        MPI_File_write_at_all(fh,cur+t1pos*sizeof(double)+rank*sizeof(unsigned),&ttlen,sizeof(unsigned),MPI_CHAR,MPI_STATUS_IGNORE);
        MPI_File_write_at_all(fh,cur+t1pos*sizeof(double)+(rank+1)*sizeof(unsigned),coord2[s2].data(),sizeof(double)*coord2[s2].size(),MPI_CHAR,MPI_STATUS_IGNORE);
        MPI_File_close(&fh);
        */
        /*-------*/
        ret_lum+=ps(coord1[s1],param1,beam1.get_comm(),coord2[s2],param2,beam2.get_comm());
        /*20200903*/
        /*
        MPI_File_open(MPI_COMM_WORLD,"2_after.txt",MPI_MODE_CREATE|MPI_MODE_WRONLY|MPI_MODE_APPEND,MPI_INFO_NULL,&fh);
        MPI_File_get_position(fh,&cur);
        MPI_File_write_at_all(fh,cur+t1pos*sizeof(double)+rank*sizeof(unsigned),&ttlen,sizeof(unsigned),MPI_CHAR,MPI_STATUS_IGNORE);
        MPI_File_write_at_all(fh,cur+t1pos*sizeof(double)+(rank+1)*sizeof(unsigned),coord2[s2].data(),sizeof(double)*coord2[s2].size(),MPI_CHAR,MPI_STATUS_IGNORE);
        MPI_File_close(&fh);
        */
        /*-------*/

    }

    //update beam1 and beam2 
    for(unsigned k=0;k<ns1;++k)
        for(unsigned i=0;i<index1[k].size();++i)
            for(unsigned j=0;j<6;++j)
                beam1.set_coordinate(j,index1[k][i],coord1[k][i+j*index1[k].size()]);
    for(unsigned k=0;k<ns2;++k)
        for(unsigned i=0;i<index2[k].size();++i)
            for(unsigned j=0;j<6;++j)
                beam2.set_coordinate(j,index2[k][i],coord2[k][i+j*index2[k].size()]);

    return ret_lum;
}
