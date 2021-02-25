#include"bb.h"
#include"calc.h"
#include<iostream>
#include<string>
#include<fstream>
#include<sstream>
#include<streambuf>
#include<map>
#include<vector>

using std::string;
using std::cin;
using std::cout;
using std::endl;
using std::vector;


int input(const string &file, const string &start){
    string filename;
    auto pos=file.find_last_of(".");
    if(pos!=string::npos)
        filename=file.substr(0,pos);
    else
        filename=file;

    std::ifstream in(file);
    string content=string(std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>());
    in.close();

    bbp::clear();
    bbp::parse(content.c_str());

    calculator::clear();

    if(!start.empty())
        calculator::parse(start.c_str());

    for(const string &str : bbp::udef_order){
        const string & temp=bbp::bbudef.at(str);
        if(temp.empty())
            continue;
        auto ret=calculator::parse(temp.c_str());
        if(ret.first)
            bbp::bbdbls[str]=ret.second.d;
        else
            bbp::bbints[str]=ret.second.i;
    }

    for(auto & each_map : bbp::bbstrs){
        each_map.second=fmt::parse(each_map.second.c_str(),filename);
    }

    return 0;

}

std::ostream& output(std::ostream &out){
    using bb_data=std::map<string,std::map<string,vector<string>>>;

    bb_data ret;
    for(const auto & each : bbp::bbints){
        string t;
        std::istringstream in(each.first);
        vector<string> v;
        while(std::getline(in,t,' '))
            v.push_back(t);
        int num=std::stoi(v[2]);
        if(num+1>ret[v[0]][v[1]].size())
            ret[v[0]][v[1]].resize(num+1);
        ret[v[0]][v[1]][num]=std::to_string(each.second);
    }
    for(const auto & each : bbp::bbdbls){
        string t;
        std::istringstream in(each.first);
        vector<string> v;
        while(std::getline(in,t,' '))
            v.push_back(t);
        int num=std::stoi(v[2]);
        if(num+1>ret[v[0]][v[1]].size())
            ret[v[0]][v[1]].resize(num+1);
        std::ostringstream temp_out;
        temp_out.copyfmt(out);
        temp_out<<each.second;
        ret[v[0]][v[1]][num]=temp_out.str();
    }
    for(const auto & each : bbp::bbstrs){
        string t;
        std::istringstream in(each.first);
        vector<string> v;
        while(std::getline(in,t,' '))
            v.push_back(t);
        int num=std::stoi(v[2]);
        if(num+1>ret[v[0]][v[1]].size())
            ret[v[0]][v[1]].resize(num+1);
        ret[v[0]][v[1]][num]="\""+each.second+"\"";
    }
    for(const auto & each : bbp::bbools){
        string t;
        std::istringstream in(each.first);
        vector<string> v;
        while(std::getline(in,t,' '))
            v.push_back(t);
        int num=std::stoi(v[2]);
        if(num+1>ret[v[0]][v[1]].size())
            ret[v[0]][v[1]].resize(num+1);
        if(each.second)
            ret[v[0]][v[1]][num]="true";
        else
            ret[v[0]][v[1]][num]="false";
    }

    for(const auto & i : ret){
        out<<"&"<<i.first<<"\n";
        for(const auto & j : i.second){
            out<<"\t"<<j.first<<" = ";
            for(const auto & k : j.second)
                out<<k<<",";
            out<<"\n";
        }
        out<<"&end\n\n";
    }

    return out;
}

/*
int main(int argc, char *argv[]){
    input(argv[1],argv[2]);
    std::ofstream out(argv[3]);
    out.flags(std::ios::scientific);
    output(out);
    return 0;
}
*/
