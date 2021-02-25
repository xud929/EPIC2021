#ifndef BB_H
#define BB_H
#include<unordered_map>
#include<vector>
#include<string>

namespace bbp{
    extern std::unordered_map<std::string, int> bbints;
    extern std::unordered_map<std::string, double> bbdbls;
    extern std::unordered_map<std::string, bool> bbools;
    extern std::unordered_map<std::string, std::string> bbstrs;
    extern std::unordered_map<std::string, std::string> bbudef;
    extern std::vector<std::string> udef_order;
    void clear();
    void parse(const char *);
    extern const std::string sep;

    template<typename T>
    int count_index(const std::string &, const std::string &);

    template<>
    inline int count_index<double>(const std::string & scope, const std::string &field){
        unsigned n=0;
        std::string key_base=scope+sep+field+sep;
        while(bbdbls.find(key_base+std::to_string(n))!=bbdbls.end())
            ++n;
        return n;
    }
    template<>
    inline int count_index<int>(const std::string & scope, const std::string &field){
        unsigned n=0;
        std::string key_base=scope+sep+field+sep;
        while(bbints.find(key_base+std::to_string(n))!=bbints.end())
            ++n;
        return n;
    }
    template<>
    inline int count_index<bool>(const std::string & scope, const std::string &field){
        unsigned n=0;
        std::string key_base=scope+sep+field+sep;
        while(bbools.find(key_base+std::to_string(n))!=bbools.end())
            ++n;
        return n;
    }
    template<>
    inline int count_index<std::string>(const std::string & scope, const std::string &field){
        unsigned n=0;
        std::string key_base=scope+sep+field+sep;
        while(bbstrs.find(key_base+std::to_string(n))!=bbstrs.end())
            ++n;
        return n;
    }

    template<typename T>
    T get(const std::string &, const std::string &, int);

    template<>
    inline int get<int>(const std::string &scope, const std::string &field, int i){
        return bbints.at(scope+sep+field+sep+std::to_string(i));
    }
    template<>
    inline double get<double>(const std::string &scope, const std::string &field, int i){
        return bbdbls.at(scope+sep+field+sep+std::to_string(i));
    }
    template<>
    inline std::string get<std::string>(const std::string &scope, const std::string &field, int i){
        return bbstrs.at(scope+sep+field+sep+std::to_string(i));
    }
    template<>
    inline bool get<bool>(const std::string &scope, const std::string &field, int i){
        return bbools.at(scope+sep+field+sep+std::to_string(i));
    }
}

namespace fmt{
    std::string parse(const char*, const std::string&);
}

#define FIELD(scope,variable,count) #scope" "#variable" "#count
#define INT0(scope,variable) bbp::bbints.at(FIELD(scope,variable,0))
#define INT1(scope,variable) bbp::bbints.at(FIELD(scope,variable,1))
#define INT2(scope,variable) bbp::bbints.at(FIELD(scope,variable,2))
#define INT3(scope,variable) bbp::bbints.at(FIELD(scope,variable,3))
#define DBL0(scope,variable) bbp::bbdbls.at(FIELD(scope,variable,0))
#define DBL1(scope,variable) bbp::bbdbls.at(FIELD(scope,variable,1))
#define DBL2(scope,variable) bbp::bbdbls.at(FIELD(scope,variable,2))
#define DBL3(scope,variable) bbp::bbdbls.at(FIELD(scope,variable,3))
#define BOL0(scope,variable) bbp::bbools.at(FIELD(scope,variable,0))
#define BOL1(scope,variable) bbp::bbools.at(FIELD(scope,variable,1))
#define BOL2(scope,variable) bbp::bbools.at(FIELD(scope,variable,2))
#define BOL3(scope,variable) bbp::bbools.at(FIELD(scope,variable,3))
#define STR0(scope,variable) bbp::bbstrs.at(FIELD(scope,variable,0))
#define STR1(scope,variable) bbp::bbstrs.at(FIELD(scope,variable,1))
#define STR2(scope,variable) bbp::bbstrs.at(FIELD(scope,variable,2))
#define STR3(scope,variable) bbp::bbstrs.at(FIELD(scope,variable,3))

#define IFININT0(x,scope,variable) if(bbp::bbints.find(FIELD(scope,variable,0))!=bbp::bbints.end()) x=INT0(scope,variable);
#define IFININT1(x,scope,variable) if(bbp::bbints.find(FIELD(scope,variable,1))!=bbp::bbints.end()) x=INT1(scope,variable);
#define IFININT2(x,scope,variable) if(bbp::bbints.find(FIELD(scope,variable,2))!=bbp::bbints.end()) x=INT2(scope,variable);
#define IFININT3(x,scope,variable) if(bbp::bbints.find(FIELD(scope,variable,3))!=bbp::bbints.end()) x=INT3(scope,variable);
#define IFINDBL0(x,scope,variable) if(bbp::bbdbls.find(FIELD(scope,variable,0))!=bbp::bbdbls.end()) x=DBL0(scope,variable);
#define IFINDBL1(x,scope,variable) if(bbp::bbdbls.find(FIELD(scope,variable,1))!=bbp::bbdbls.end()) x=DBL1(scope,variable);
#define IFINDBL2(x,scope,variable) if(bbp::bbdbls.find(FIELD(scope,variable,2))!=bbp::bbdbls.end()) x=DBL2(scope,variable);
#define IFINDBL3(x,scope,variable) if(bbp::bbdbls.find(FIELD(scope,variable,3))!=bbp::bbdbls.end()) x=DBL3(scope,variable);
#define IFINBOL0(x,scope,variable) if(bbp::bbools.find(FIELD(scope,variable,0))!=bbp::bbools.end()) x=BOL0(scope,variable);
#define IFINBOL1(x,scope,variable) if(bbp::bbools.find(FIELD(scope,variable,1))!=bbp::bbools.end()) x=BOL1(scope,variable);
#define IFINBOL2(x,scope,variable) if(bbp::bbools.find(FIELD(scope,variable,2))!=bbp::bbools.end()) x=BOL2(scope,variable);
#define IFINBOL3(x,scope,variable) if(bbp::bbools.find(FIELD(scope,variable,3))!=bbp::bbools.end()) x=BOL3(scope,variable);
#define IFINSTR0(x,scope,variable) if(bbp::bbstrs.find(FIELD(scope,variable,0))!=bbp::bbstrs.end()) x=STR0(scope,variable);
#define IFINSTR1(x,scope,variable) if(bbp::bbstrs.find(FIELD(scope,variable,1))!=bbp::bbstrs.end()) x=STR1(scope,variable);
#define IFINSTR2(x,scope,variable) if(bbp::bbstrs.find(FIELD(scope,variable,2))!=bbp::bbstrs.end()) x=STR2(scope,variable);
#define IFINSTR3(x,scope,variable) if(bbp::bbstrs.find(FIELD(scope,variable,3))!=bbp::bbstrs.end()) x=STR3(scope,variable);

#endif
