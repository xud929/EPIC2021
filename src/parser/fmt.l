%{
//#include"fmt_yacc.h"
#include<iostream>
#include<sstream>
#include<string>
#include<cstring>
#include"calc.h"
namespace fmt{
    std::ostringstream out;
    std::string filename;
}
%}

%%
\%\%                                            fmt::out<<'%';
\%\{                                            fmt::out<<'{';
\%\}                                            fmt::out<<'}';
\%[sS]                                          fmt::out<<fmt::filename;
\%[1-9][0-9]*d\{[^\}]+\}                        {
                                                    int pos=0,len=std::strlen(yytext);
                                                    while(*(yytext+pos)!='d')
                                                        ++pos;
                                                    std::string temp(yytext,1,pos-1);
                                                    int width=std::stoi(temp);
                                                    temp=std::string(yytext,pos+2,len-pos-3);
                                                    auto x=calculator::parse(temp.c_str());
                                                    int xi=(x.first?x.second.d:x.second.i);
                                                    std::string ret=std::to_string(xi);
                                                    if(ret.size()<width)
                                                        ret=std::string(width-ret.size(),'0')+ret;
                                                    fmt::out<<ret;
                                                }
\%d\{[^\}]+\}                                   {
                                                    int len=std::strlen(yytext);
                                                    std::string temp=std::string(yytext,3,len-4);
                                                    auto x=calculator::parse(temp.c_str());
                                                    int xi;
                                                    if(x.first)
                                                        xi=x.second.d;
                                                    else
                                                        xi=x.second.i;
                                                    std::string ret=std::to_string(xi);
                                                    fmt::out<<ret;
                                                }
\%\.[1-9][0-9]*[ef]\{[^\}]+\}                   {
                                                    int pos=0,len=std::strlen(yytext);
                                                    while(*(yytext+pos)!='{')
                                                        ++pos;
                                                    std::string temp(yytext,pos+1,len-pos-2);
                                                    auto x=calculator::parse(temp.c_str());
                                                    double xd=(x.first?x.second.d:x.second.i);
                                                    temp=std::string(yytext,2,pos-3);
                                                    int prec=std::stoi(temp);
                                                    fmt::out.width(0);
                                                    fmt::out.precision(prec);
                                                    if(yytext[pos-1]=='e')
                                                        fmt::out.flags(std::ios::scientific);
                                                    else
                                                        fmt::out.flags(std::ios::fixed);
                                                    fmt::out<<xd;
                                                }
\%[1-9][0-9]*[ef]\{[^\}]+\}                     {
                                                    int pos=0,len=std::strlen(yytext);
                                                    while(*(yytext+pos)!='{')
                                                        ++pos;
                                                    std::string temp(yytext,pos+1,len-pos-2);
                                                    auto x=calculator::parse(temp.c_str());
                                                    double xd=(x.first?x.second.d:x.second.i);
                                                    temp=std::string(yytext,1,pos-2);
                                                    int width=std::stoi(temp);
                                                    fmt::out.width(width);
                                                    fmt::out.precision(6);
                                                    if(yytext[pos-1]=='e')
                                                        fmt::out.flags(std::ios::scientific);
                                                    else
                                                        fmt::out.flags(std::ios::fixed);
                                                    fmt::out.fill('0');
                                                    fmt::out<<xd;
                                                }
\%[1-9][0-9]*\.[1-9][0-9]*[ef]\{[^\}]+\}        {
                                                    int pos=0,len=std::strlen(yytext);
                                                    while(*(yytext+pos)!='.')
                                                        ++pos;
                                                    int pos_dot=pos;
                                                    while(*(yytext+pos)!='{')
                                                        ++pos;
                                                    std::string temp(yytext,pos+1,len-pos-2);
                                                    auto x=calculator::parse(temp.c_str());
                                                    double xd=(x.first?x.second.d:x.second.i);
                                                    temp=std::string(yytext,pos_dot+1,pos-pos_dot-2);
                                                    int prec=std::stoi(temp);
                                                    temp=std::string(yytext,1,pos_dot-1);
                                                    int width=std::stoi(temp);
                                                    fmt::out.precision(prec);
                                                    fmt::out.width(width);
                                                    if(yytext[pos-1]=='e')
                                                        fmt::out.flags(std::ios::scientific);
                                                    else
                                                        fmt::out.flags(std::ios::fixed);
                                                    fmt::out.fill('0');
                                                    fmt::out<<xd;
                                                }
\%f\{[^\}]+\}                                   {
                                                    int len=std::strlen(yytext);
                                                    std::string temp=std::string(yytext,3,len-4);
                                                    auto x=calculator::parse(temp.c_str());
                                                    double xd=(x.first?x.second.d:x.second.i);
                                                    fmt::out.precision(6);
                                                    fmt::out.width(0);
                                                    fmt::out<<xd;
                                                }
.                                               fmt::out<<*yytext;
%%

int yywrap(void) {
    return 1;
}

namespace fmt{
std::string parse(const char *str, const std::string &ps){
    out.str("");
    out.clear();
    filename=ps;
    YY_BUFFER_STATE buffer=ss_scan_string(str);
    sslex();
    ss_delete_buffer(buffer);
    return out.str();
}
}
