#ifndef CALC_H
#define CALC_H
#include<utility>
#include<unordered_map>

namespace calculator{

struct int_or_double{
    int_or_double()=default;
    int_or_double(int x):i(x){}
    int_or_double(double x):d(x){}
    union{
        int i;
        double d;
    };
};

void clear();
std::pair<int,int_or_double> parse(const char *);

extern std::unordered_map<std::string, int_or_double> stored_values;
extern std::unordered_map<std::string, int> stored_types;

}

#endif
