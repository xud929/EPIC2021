#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<unordered_map>
#include<vector>

using std::cin;using std::cout;using std::endl;using std::cerr;
using std::ifstream;
using std::ofstream;
using std::string;
using std::unordered_map;
using std::vector;

int main(int argc, char *argv[]){
    if(argc==1)
        return 0;
    ifstream in(argv[1], std::ifstream::in | std::ifstream::binary);

    int total_turns, description_size;
    in.read((char*)&total_turns,sizeof(int));
    in.read((char*)&description_size,sizeof(int));
    char *ptr_description=new char[description_size+1];
    in.read((char*)ptr_description,description_size);
    string description(ptr_description);
    unordered_map<string,int> a_map;
    std::istringstream sin(description);
    string a_temp;
    int index=0;
    while(std::getline(sin,a_temp,',')){
        a_map[a_temp]=index;
        ++index;
    }

    if(argc==2){
        cout<<total_turns<<endl;
        cout<<ptr_description<<endl;
        return 0;
    }

    if(argc%2!=0){
        cerr<<"Invalid number of arguments."<<endl;
        return 1;
    }

    string out_file,rows,cols;
    for(int i=2;i<argc;i+=2){
        if(string(argv[i])==string("-o"))
            out_file=argv[i+1];
        else if(string(argv[i])==string("-col"))
            cols=argv[i+1];
        else if(string(argv[i])==string("-row"))
            rows=argv[i+1];
        else{
            cerr<<"Invalid options."<<endl;
            return 1;
        }
    }

    std::ofstream fout;
    std::streambuf *backup=cout.rdbuf();

    if(!out_file.empty()){
        fout.open(out_file);
        cout.rdbuf(fout.rdbuf());
    }

    if(cols.empty())
        cols=*ptr_description;
    vector<int> col_idx;
    sin.str(cols);
    sin.clear();
    while(std::getline(sin,a_temp,',')){
        try{
            col_idx.push_back(a_map.at(a_temp));
        }catch(const std::exception &e){
            cerr<<"Invalid col option."<<endl;
            return 1;
        }
    }

    vector<int> row_setup={0,total_turns,1};
    if(!rows.empty()){
        sin.str(rows);
        sin.clear();
        string st;
        for(int i=0;i<3;++i){
            std::getline(sin,st,':');
            if(st.empty())
                continue;
            int n=std::stoi(st);
            if(n<0)
                n+=total_turns;
            row_setup[i]=n;
            if(sin.eof()){
                if(i==0)
                    row_setup[2]=total_turns;
                break;
            }
        }
    }
    int &beg=row_setup[0], &end=row_setup[1], &step=row_setup[2];
    cout.precision(16);
    cout.flags(std::ios::scientific);

    int pos=in.tellg();
    int each_row_bytes=a_map.size()*sizeof(double);
    for(int i=beg;i<end;i+=step){
        int row_pos=pos+each_row_bytes*i;
        for(const auto & j : col_idx){
            in.seekg(row_pos+j*sizeof(double));
            double temp;
            in.read((char*)&temp,sizeof(double));
            cout<<temp<<"\t";
        }
        cout<<"\n";
    }

    if(!out_file.empty()){
        fout.close();
    }
    cout.rdbuf(backup);
    delete []ptr_description;
    return 0;
}
