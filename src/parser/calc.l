%{
#include"calc_yacc.h"
#include<math.h>
#define yylval zzlval
%}

%option prefix="zz"

%%
[-+*/=(),]                                      return *yytext;
[0-9]+                                          {yylval.xi=atoi(yytext);return INT;}
[0-9]*\.?[0-9]*(([eE][-+]?)?[0-9]+)?            {yylval.xd=atof(yytext);return DOUBLE;}
[ \t\n]                                         ;
pi                                              {yylval.xd=M_PI;return DOUBLE;}
PI                                              {yylval.xd=M_PI;return DOUBLE;}
M_PI                                            {yylval.xd=M_PI;return DOUBLE;}
pmass                                           {yylval.xd=938.272081e6;return DOUBLE;}
emass                                           {yylval.xd=0.5109989461e6;return DOUBLE;}
re                                              {yylval.xd=2.8179403227e-15;return DOUBLE;}
clight                                          {yylval.xd=299792458.0;return DOUBLE;}
exp                                             {yylval.xi=257;return UOP;}
log                                             {yylval.xi=258;return UOP;}
sin                                             {yylval.xi=259;return UOP;}
cos                                             {yylval.xi=260;return UOP;}
tan                                             {yylval.xi=261;return UOP;}
cot                                             {yylval.xi=262;return UOP;}
sqrt                                            {yylval.xi=263;return UOP;}
\*\*                                            {return POW;}
[_a-zA-Z][_a-zA-Z0-9]*                          {yylval.xs=strdup(yytext);return ID;}
%%
