%{
#include<iostream>
#include<string>
#include<unordered_map>
#include<vector>
#include<cmath>
#include<stack>
#include<stdexcept>
#include<utility>
#include"calc.h"

extern "C" int yylex(void);
extern "C" void yyerror(const char *); 
extern "C" int zzwrap(void);

typedef struct zz_buffer_state * ZZ_BUFFER_STATE;
extern "C" ZZ_BUFFER_STATE zz_scan_string(const char * str);
extern "C" void zz_delete_buffer(ZZ_BUFFER_STATE buffer);

namespace calculator{

std::stack<int_or_double> ret;
static int reti;

std::unordered_map<std::string, int_or_double> stored_values;
std::unordered_map<std::string, int> stored_types;

int add(int, int);
int sub(int, int);
int mul(int, int);
int div(int, int);
int pow(int, int);
int uop(int, int);


}
%}

%union{
    int xi;
    double xd;
    char *xs;
}

%token <xi> INT UOP
%token <xd> DOUBLE
%token <xs> ID
%type  <xi> atom expr

%left  ','
%right '='
%left  '+' '-'
%left  '*' '/'
%left  POW
%right UOP
%nonassoc UMINUS


%%

expr    :   atom                                {$$=$1;calculator::reti=$$;}
        |   ID   '=' expr                       {$$=$3;calculator::stored_types[$1]=$3;calculator::stored_values[$1]=calculator::ret.top();free($1);calculator::reti=$$;}
        |   expr ',' expr                       {$$=$3;
                                                 calculator::int_or_double temp=calculator::ret.top();
                                                 calculator::ret.pop();
                                                 calculator::ret.pop();
                                                 calculator::ret.push(temp);
                                                 calculator::reti=$$;}
        ;

atom    :   INT                                 {$$=0;calculator::ret.push($1);}
        |   DOUBLE                              {$$=1;calculator::ret.push($1);}
        |   ID                                  {
                                                    try{
                                                        $$=calculator::stored_types.at($1);
                                                        calculator::ret.push(calculator::stored_values.at($1));
                                                        free($1);
                                                    }catch(std::exception e){
                                                        yyerror((std::string("Variable ")+std::string($1)+" undefined.").c_str());
                                                    }
                                                }
        |   atom '+' atom                       {$$=calculator::add($1,$3);}
        |   atom '-' atom                       {$$=calculator::sub($1,$3);}
        |   atom '*' atom                       {$$=calculator::mul($1,$3);}
        |   atom '/' atom                       {$$=calculator::div($1,$3);}
        |   atom POW atom                       {$$=calculator::pow($1,$3);}
        |   '('  atom  ')'                      {$$=$2;}
        |   UOP  atom                           {$$=calculator::uop($1,$2);}
        |   '-'  atom %prec UMINUS              {$$=calculator::uop('-',$2);}
        ;
%%

namespace calculator{

int add(int  t1, int t2){
    if(t1==0 && t2==0){
        int R=ret.top().i;ret.pop();
        int L=ret.top().i;ret.pop();
        int_or_double r(L+R);
        ret.push(r);
        return 0;
    }else if(t1==0 && t2==1){
        double R=ret.top().d;ret.pop();
        int L=ret.top().i;ret.pop();
        int_or_double r(L+R);
        ret.push(r);
        return 1;
    }else if(t1==1 && t2==0){
        int R=ret.top().i;ret.pop();
        double L=ret.top().d;ret.pop();
        int_or_double r(L+R);
        ret.push(r);
        return 1;
    }else if(t1==1 && t2==1){
        double R=ret.top().d;ret.pop();
        double L=ret.top().d;ret.pop();
        int_or_double r(L+R);
        ret.push(r);
        return 1;
    }else{
        throw std::runtime_error("Unknown types");
    }
}

int sub(int  t1, int t2){
    if(t1==0 && t2==0){
        int R=ret.top().i;ret.pop();
        int L=ret.top().i;ret.pop();
        int_or_double r(L-R);
        ret.push(r);
        return 0;
    }else if(t1==0 && t2==1){
        double R=ret.top().d;ret.pop();
        int L=ret.top().i;ret.pop();
        int_or_double r(L-R);
        ret.push(r);
        return 1;
    }else if(t1==1 && t2==0){
        int R=ret.top().i;ret.pop();
        double L=ret.top().d;ret.pop();
        int_or_double r(L-R);
        ret.push(r);
        return 1;
    }else if(t1==1 && t2==1){
        double R=ret.top().d;ret.pop();
        double L=ret.top().d;ret.pop();
        int_or_double r(L-R);
        ret.push(r);
        return 1;
    }else{
        throw std::runtime_error("Unknown types");
    }
}
int mul(int  t1, int t2){
    if(t1==0 && t2==0){
        int R=ret.top().i;ret.pop();
        int L=ret.top().i;ret.pop();
        int_or_double r(L*R);
        ret.push(r);
        return 0;
    }else if(t1==0 && t2==1){
        double R=ret.top().d;ret.pop();
        int L=ret.top().i;ret.pop();
        int_or_double r(L*R);
        ret.push(r);
        return 1;
    }else if(t1==1 && t2==0){
        int R=ret.top().i;ret.pop();
        double L=ret.top().d;ret.pop();
        int_or_double r(L*R);
        ret.push(r);
        return 1;
    }else if(t1==1 && t2==1){
        double R=ret.top().d;ret.pop();
        double L=ret.top().d;ret.pop();
        int_or_double r(L*R);
        ret.push(r);
        return 1;
    }else{
        throw std::runtime_error("Unknown types");
    }
}
int div(int  t1, int t2){
    if(t1==0 && t2==0){
        int R=ret.top().i;ret.pop();
        int L=ret.top().i;ret.pop();
        int_or_double r(L/R);
        ret.push(r);
        return 0;
    }else if(t1==0 && t2==1){
        double R=ret.top().d;ret.pop();
        int L=ret.top().i;ret.pop();
        int_or_double r(L/R);
        ret.push(r);
        return 1;
    }else if(t1==1 && t2==0){
        int R=ret.top().i;ret.pop();
        double L=ret.top().d;ret.pop();
        int_or_double r(L/R);
        ret.push(r);
        return 1;
    }else if(t1==1 && t2==1){
        double R=ret.top().d;ret.pop();
        double L=ret.top().d;ret.pop();
        int_or_double r(L/R);
        ret.push(r);
        return 1;
    }else{
        throw std::runtime_error("Unknown types");
        return -1;
    }
}
int pow(int  t1, int t2){
    if(t1==0 && t2==0){
        int R=ret.top().i;ret.pop();
        double L=ret.top().i;ret.pop();
        int_or_double r(std::pow(L,R));
        ret.push(r);
        return 1;
    }else if(t1==0 && t2==1){
        double R=ret.top().d;ret.pop();
        double L=ret.top().i;ret.pop();
        int_or_double r(std::pow(L,R));
        ret.push(r);
        return 1;
    }else if(t1==1 && t2==0){
        int R=ret.top().i;ret.pop();
        double L=ret.top().d;ret.pop();
        int_or_double r(std::pow(L,R));
        ret.push(r);
        return 1;
    }else if(t1==1 && t2==1){
        double R=ret.top().d;ret.pop();
        double L=ret.top().d;ret.pop();
        int_or_double r(std::pow(L,R));
        ret.push(r);
        return 1;
    }else{
        throw std::runtime_error("Unknown types");
        return -1;
    }
}

int uop(int op, int t){
    switch(op){
        case '-':
            if(t){
                int_or_double r(-ret.top().d);
                ret.pop();
                ret.push(r);
            }else{
                int_or_double r(-ret.top().i);
                ret.pop();
                ret.push(r);
            }
            return t;
        case 257:
            if(t){
                int_or_double r(std::exp(ret.top().d));
                ret.pop();
                ret.push(r);
            }else{
                int_or_double r(std::exp(ret.top().i));
                ret.pop();
                ret.push(r);
            }
            return 1;
        case 258:
            if(t){
                int_or_double r(std::log(ret.top().d));
                ret.pop();
                ret.push(r);
            }else{
                int_or_double r(std::log(ret.top().i));
                ret.pop();
                ret.push(r);
            }
            return 1;
        case 259:
            if(t){
                int_or_double r(std::sin(ret.top().d));
                ret.pop();
                ret.push(r);
            }else{
                int_or_double r(std::sin(ret.top().i));
                ret.pop();
                ret.push(r);
            }
            return 1;
        case 260:
            if(t){
                int_or_double r(std::cos(ret.top().d));
                ret.pop();
                ret.push(r);
            }else{
                int_or_double r(std::cos(ret.top().i));
                ret.pop();
                ret.push(r);
            }
            return 1;
        case 261:
            if(t){
                int_or_double r(std::tan(ret.top().d));
                ret.pop();
                ret.push(r);
            }else{
                int_or_double r(std::tan(ret.top().i));
                ret.pop();
                ret.push(r);
            }
            return 1;
        case 262:
            if(t){
                int_or_double r(1.0/std::tan(ret.top().d));
                ret.pop();
                ret.push(r);
            }else{
                int_or_double r(1.0/std::tan(ret.top().i));
                ret.pop();
                ret.push(r);
            }
            return 1;
        case 263:
            if(t){
                int_or_double r(std::sqrt(ret.top().d));
                ret.pop();
                ret.push(r);
            }else{
                int_or_double r(std::sqrt(ret.top().i));
                ret.pop();
                ret.push(r);
            }
            return 1;
        default:
            throw std::runtime_error("Unknown operator");
            break;
    }
}

std::pair<int,int_or_double> parse(const char *str){
    ZZ_BUFFER_STATE buffer=zz_scan_string(str);
    zzparse();
    zz_delete_buffer(buffer);
    int_or_double r=ret.top();
    ret.pop();
    if(!ret.empty())
        throw std::runtime_error("Something wrong happened during parsing.");
    return std::make_pair(reti,r);
}
void clear(){
    reti=-1;
    while(!ret.empty())
        ret.pop();
    stored_values.clear();
    stored_types.clear();
}

}

void yyerror(const char *str){
    std::cerr<<str<<std::endl;
}

int zzwrap(){
    return 1;
}
