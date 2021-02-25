#include<string>
#include<iostream>
#include<unordered_map>
#include"../calc.h"

using std::cin;using std::cout;using std::endl;
using std::string;

int main(){
    string s;
    std::getline(cin,s);
    auto ret=calculator::parse(s.c_str());
    cout<<ret.first<<"\t";
    if(ret.first)
        cout<<ret.second.d<<endl;
    else
        cout<<ret.second.i<<endl;
    return 0;
}



