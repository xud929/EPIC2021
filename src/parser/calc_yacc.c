#ifndef lint
static const char yysccsid[] = "@(#)yaccpar	1.9 (Berkeley) 02/21/93";
#endif

#define YYBYACC 1
#define YYMAJOR 1
#define YYMINOR 9
#define YYPATCH 20130304

#define YYEMPTY        (-1)
#define yyclearin      (yychar = YYEMPTY)
#define yyerrok        (yyerrflag = 0)
#define YYRECOVERING() (yyerrflag != 0)


#ifndef yyparse
#define yyparse    zzparse
#endif /* yyparse */

#ifndef yylex
#define yylex      zzlex
#endif /* yylex */

#ifndef yyerror
#define yyerror    zzerror
#endif /* yyerror */

#ifndef yychar
#define yychar     zzchar
#endif /* yychar */

#ifndef yyval
#define yyval      zzval
#endif /* yyval */

#ifndef yylval
#define yylval     zzlval
#endif /* yylval */

#ifndef yydebug
#define yydebug    zzdebug
#endif /* yydebug */

#ifndef yynerrs
#define yynerrs    zznerrs
#endif /* yynerrs */

#ifndef yyerrflag
#define yyerrflag  zzerrflag
#endif /* yyerrflag */

#ifndef yylhs
#define yylhs      zzlhs
#endif /* yylhs */

#ifndef yylen
#define yylen      zzlen
#endif /* yylen */

#ifndef yydefred
#define yydefred   zzdefred
#endif /* yydefred */

#ifndef yydgoto
#define yydgoto    zzdgoto
#endif /* yydgoto */

#ifndef yysindex
#define yysindex   zzsindex
#endif /* yysindex */

#ifndef yyrindex
#define yyrindex   zzrindex
#endif /* yyrindex */

#ifndef yygindex
#define yygindex   zzgindex
#endif /* yygindex */

#ifndef yytable
#define yytable    zztable
#endif /* yytable */

#ifndef yycheck
#define yycheck    zzcheck
#endif /* yycheck */

#ifndef yyname
#define yyname     zzname
#endif /* yyname */

#ifndef yyrule
#define yyrule     zzrule
#endif /* yyrule */
#define YYPREFIX "zz"

#define YYPURE 0

#line 2 "calc.y"
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
#line 39 "calc.y"
#ifdef YYSTYPE
#undef  YYSTYPE_IS_DECLARED
#define YYSTYPE_IS_DECLARED 1
#endif
#ifndef YYSTYPE_IS_DECLARED
#define YYSTYPE_IS_DECLARED 1
typedef union{
    int xi;
    double xd;
    char *xs;
} YYSTYPE;
#endif /* !YYSTYPE_IS_DECLARED */
#line 148 "calc_yacc.c"

/* compatibility with bison */
#ifdef YYPARSE_PARAM
/* compatibility with FreeBSD */
# ifdef YYPARSE_PARAM_TYPE
#  define YYPARSE_DECL() yyparse(YYPARSE_PARAM_TYPE YYPARSE_PARAM)
# else
#  define YYPARSE_DECL() yyparse(void *YYPARSE_PARAM)
# endif
#else
# define YYPARSE_DECL() yyparse(void)
#endif

/* Parameters sent to lex. */
#ifdef YYLEX_PARAM
# define YYLEX_DECL() yylex(void *YYLEX_PARAM)
# define YYLEX yylex(YYLEX_PARAM)
#else
# define YYLEX_DECL() yylex(void)
# define YYLEX yylex()
#endif

/* Parameters sent to yyerror. */
#ifndef YYERROR_DECL
#define YYERROR_DECL() yyerror(const char *s)
#endif
#ifndef YYERROR_CALL
#define YYERROR_CALL(msg) yyerror(msg)
#endif

extern int YYPARSE_DECL();

#define INT 257
#define UOP 258
#define DOUBLE 259
#define ID 260
#define POW 261
#define UMINUS 262
#define YYERRCODE 256
static const short zzlhs[] = {                           -1,
    0,    0,    0,    1,    1,    1,    1,    1,    1,    1,
    1,    1,    1,    1,
};
static const short zzlen[] = {                            2,
    1,    3,    3,    1,    1,    1,    3,    3,    3,    3,
    3,    3,    2,    2,
};
static const short zzdefred[] = {                         0,
    4,    0,    5,    0,    0,    0,    0,    0,    6,   13,
    0,   14,    0,    0,    0,    0,    0,    0,    0,    2,
   12,    3,    0,    0,    0,    0,   11,
};
static const short zzdgoto[] = {                          7,
    8,
};
static const short zzsindex[] = {                       -40,
    0,  -36,    0,  -59,  -36,  -36,  -23,  -29,    0,    0,
  -40,    0,  -35,  -40,  -36,  -36,  -36,  -36,  -36,    0,
    0,    0,  -27,  -27, -244, -244,    0,
};
static const short zzrindex[] = {                         0,
    0,    0,    0,    1,    0,    0,    0,    3,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,   24,   29,   11,   19,    0,
};
static const short zzgindex[] = {                        16,
   20,
};
#define YYTABLESIZE 262
static const short zztable[] = {                          6,
    6,   11,    1,    6,    5,   21,   17,   15,    5,   16,
    9,   18,   17,   15,   17,   16,   19,   18,   10,   18,
   14,   10,    0,    7,   12,   13,   20,    0,    8,   22,
    0,    0,    0,    0,   23,   24,   25,   26,   27,    0,
    0,    0,    6,    6,    6,    6,    1,    6,    0,    0,
    0,    9,    9,    9,    9,    9,    0,    9,    0,   10,
   10,   10,   10,   10,    7,   10,    7,    7,    7,    8,
    0,    8,    8,    8,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    1,    2,    3,    4,
    1,    2,    3,    9,    0,   19,    0,    0,    0,    0,
    0,   19,    0,   19,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    6,
};
static const short zzcheck[] = {                         40,
    0,   61,    0,   40,   45,   41,   42,   43,   45,   45,
    0,   47,   42,   43,   42,   45,  261,   47,    0,   47,
   44,    2,   -1,    0,    5,    6,   11,   -1,    0,   14,
   -1,   -1,   -1,   -1,   15,   16,   17,   18,   19,   -1,
   -1,   -1,   42,   43,   44,   45,   44,   47,   -1,   -1,
   -1,   41,   42,   43,   44,   45,   -1,   47,   -1,   41,
   42,   43,   44,   45,   41,   47,   43,   44,   45,   41,
   -1,   43,   44,   45,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,  257,  258,  259,  260,
  257,  258,  259,  260,   -1,  261,   -1,   -1,   -1,   -1,
   -1,  261,   -1,  261,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,  261,
};
#define YYFINAL 7
#ifndef YYDEBUG
#define YYDEBUG 0
#endif
#define YYMAXTOKEN 262
#if YYDEBUG
static const char *yyname[] = {

"end-of-file",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,"'('","')'","'*'","'+'","','","'-'",0,"'/'",0,0,0,0,0,0,0,0,0,0,0,0,
0,"'='",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
"INT","UOP","DOUBLE","ID","POW","UMINUS",
};
static const char *yyrule[] = {
"$accept : expr",
"expr : atom",
"expr : ID '=' expr",
"expr : expr ',' expr",
"atom : INT",
"atom : DOUBLE",
"atom : ID",
"atom : atom '+' atom",
"atom : atom '-' atom",
"atom : atom '*' atom",
"atom : atom '/' atom",
"atom : atom POW atom",
"atom : '(' atom ')'",
"atom : UOP atom",
"atom : '-' atom",

};
#endif

int      yydebug;
int      yynerrs;

int      yyerrflag;
int      yychar;
YYSTYPE  yyval;
YYSTYPE  yylval;

/* define the initial stack-sizes */
#ifdef YYSTACKSIZE
#undef YYMAXDEPTH
#define YYMAXDEPTH  YYSTACKSIZE
#else
#ifdef YYMAXDEPTH
#define YYSTACKSIZE YYMAXDEPTH
#else
#define YYSTACKSIZE 10000
#define YYMAXDEPTH  500
#endif
#endif

#define YYINITSTACKSIZE 500

typedef struct {
    unsigned stacksize;
    short    *s_base;
    short    *s_mark;
    short    *s_last;
    YYSTYPE  *l_base;
    YYSTYPE  *l_mark;
} YYSTACKDATA;
/* variables for the parser stack */
static YYSTACKDATA yystack;
#line 92 "calc.y"

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
#line 622 "calc_yacc.c"

#if YYDEBUG
#include <stdio.h>		/* needed for printf */
#endif

#include <stdlib.h>	/* needed for malloc, etc */
#include <string.h>	/* needed for memset */

/* allocate initial stack or double stack size, up to YYMAXDEPTH */
static int yygrowstack(YYSTACKDATA *data)
{
    int i;
    unsigned newsize;
    short *newss;
    YYSTYPE *newvs;

    if ((newsize = data->stacksize) == 0)
        newsize = YYINITSTACKSIZE;
    else if (newsize >= YYMAXDEPTH)
        return -1;
    else if ((newsize *= 2) > YYMAXDEPTH)
        newsize = YYMAXDEPTH;

    i = (int) (data->s_mark - data->s_base);
    newss = (short *)realloc(data->s_base, newsize * sizeof(*newss));
    if (newss == 0)
        return -1;

    data->s_base = newss;
    data->s_mark = newss + i;

    newvs = (YYSTYPE *)realloc(data->l_base, newsize * sizeof(*newvs));
    if (newvs == 0)
        return -1;

    data->l_base = newvs;
    data->l_mark = newvs + i;

    data->stacksize = newsize;
    data->s_last = data->s_base + newsize - 1;
    return 0;
}

#if YYPURE || defined(YY_NO_LEAKS)
static void yyfreestack(YYSTACKDATA *data)
{
    free(data->s_base);
    free(data->l_base);
    memset(data, 0, sizeof(*data));
}
#else
#define yyfreestack(data) /* nothing */
#endif

#define YYABORT  goto yyabort
#define YYREJECT goto yyabort
#define YYACCEPT goto yyaccept
#define YYERROR  goto yyerrlab

int
YYPARSE_DECL()
{
    int yym, yyn, yystate;
#if YYDEBUG
    const char *yys;

    if ((yys = getenv("YYDEBUG")) != 0)
    {
        yyn = *yys;
        if (yyn >= '0' && yyn <= '9')
            yydebug = yyn - '0';
    }
#endif

    yynerrs = 0;
    yyerrflag = 0;
    yychar = YYEMPTY;
    yystate = 0;

#if YYPURE
    memset(&yystack, 0, sizeof(yystack));
#endif

    if (yystack.s_base == NULL && yygrowstack(&yystack)) goto yyoverflow;
    yystack.s_mark = yystack.s_base;
    yystack.l_mark = yystack.l_base;
    yystate = 0;
    *yystack.s_mark = 0;

yyloop:
    if ((yyn = yydefred[yystate]) != 0) goto yyreduce;
    if (yychar < 0)
    {
        if ((yychar = YYLEX) < 0) yychar = 0;
#if YYDEBUG
        if (yydebug)
        {
            yys = 0;
            if (yychar <= YYMAXTOKEN) yys = yyname[yychar];
            if (!yys) yys = "illegal-symbol";
            printf("%sdebug: state %d, reading %d (%s)\n",
                    YYPREFIX, yystate, yychar, yys);
        }
#endif
    }
    if ((yyn = yysindex[yystate]) && (yyn += yychar) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == yychar)
    {
#if YYDEBUG
        if (yydebug)
            printf("%sdebug: state %d, shifting to state %d\n",
                    YYPREFIX, yystate, yytable[yyn]);
#endif
        if (yystack.s_mark >= yystack.s_last && yygrowstack(&yystack))
        {
            goto yyoverflow;
        }
        yystate = yytable[yyn];
        *++yystack.s_mark = yytable[yyn];
        *++yystack.l_mark = yylval;
        yychar = YYEMPTY;
        if (yyerrflag > 0)  --yyerrflag;
        goto yyloop;
    }
    if ((yyn = yyrindex[yystate]) && (yyn += yychar) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == yychar)
    {
        yyn = yytable[yyn];
        goto yyreduce;
    }
    if (yyerrflag) goto yyinrecovery;

    yyerror("syntax error");

    goto yyerrlab;

yyerrlab:
    ++yynerrs;

yyinrecovery:
    if (yyerrflag < 3)
    {
        yyerrflag = 3;
        for (;;)
        {
            if ((yyn = yysindex[*yystack.s_mark]) && (yyn += YYERRCODE) >= 0 &&
                    yyn <= YYTABLESIZE && yycheck[yyn] == YYERRCODE)
            {
#if YYDEBUG
                if (yydebug)
                    printf("%sdebug: state %d, error recovery shifting\
 to state %d\n", YYPREFIX, *yystack.s_mark, yytable[yyn]);
#endif
                if (yystack.s_mark >= yystack.s_last && yygrowstack(&yystack))
                {
                    goto yyoverflow;
                }
                yystate = yytable[yyn];
                *++yystack.s_mark = yytable[yyn];
                *++yystack.l_mark = yylval;
                goto yyloop;
            }
            else
            {
#if YYDEBUG
                if (yydebug)
                    printf("%sdebug: error recovery discarding state %d\n",
                            YYPREFIX, *yystack.s_mark);
#endif
                if (yystack.s_mark <= yystack.s_base) goto yyabort;
                --yystack.s_mark;
                --yystack.l_mark;
            }
        }
    }
    else
    {
        if (yychar == 0) goto yyabort;
#if YYDEBUG
        if (yydebug)
        {
            yys = 0;
            if (yychar <= YYMAXTOKEN) yys = yyname[yychar];
            if (!yys) yys = "illegal-symbol";
            printf("%sdebug: state %d, error recovery discards token %d (%s)\n",
                    YYPREFIX, yystate, yychar, yys);
        }
#endif
        yychar = YYEMPTY;
        goto yyloop;
    }

yyreduce:
#if YYDEBUG
    if (yydebug)
        printf("%sdebug: state %d, reducing by rule %d (%s)\n",
                YYPREFIX, yystate, yyn, yyrule[yyn]);
#endif
    yym = yylen[yyn];
    if (yym)
        yyval = yystack.l_mark[1-yym];
    else
        memset(&yyval, 0, sizeof yyval);
    switch (yyn)
    {
case 1:
#line 61 "calc.y"
	{yyval.xi=yystack.l_mark[0].xi;calculator::reti=yyval.xi;}
break;
case 2:
#line 62 "calc.y"
	{yyval.xi=yystack.l_mark[0].xi;calculator::stored_types[yystack.l_mark[-2].xs]=yystack.l_mark[0].xi;calculator::stored_values[yystack.l_mark[-2].xs]=calculator::ret.top();free(yystack.l_mark[-2].xs);calculator::reti=yyval.xi;}
break;
case 3:
#line 63 "calc.y"
	{yyval.xi=yystack.l_mark[0].xi;
                                                 calculator::int_or_double temp=calculator::ret.top();
                                                 calculator::ret.pop();
                                                 calculator::ret.pop();
                                                 calculator::ret.push(temp);
                                                 calculator::reti=yyval.xi;}
break;
case 4:
#line 71 "calc.y"
	{yyval.xi=0;calculator::ret.push(yystack.l_mark[0].xi);}
break;
case 5:
#line 72 "calc.y"
	{yyval.xi=1;calculator::ret.push(yystack.l_mark[0].xd);}
break;
case 6:
#line 73 "calc.y"
	{
                                                    try{
                                                        yyval.xi=calculator::stored_types.at(yystack.l_mark[0].xs);
                                                        calculator::ret.push(calculator::stored_values.at(yystack.l_mark[0].xs));
                                                        free(yystack.l_mark[0].xs);
                                                    }catch(std::exception e){
                                                        yyerror((std::string("Variable ")+std::string(yystack.l_mark[0].xs)+" undefined.").c_str());
                                                    }
                                                }
break;
case 7:
#line 82 "calc.y"
	{yyval.xi=calculator::add(yystack.l_mark[-2].xi,yystack.l_mark[0].xi);}
break;
case 8:
#line 83 "calc.y"
	{yyval.xi=calculator::sub(yystack.l_mark[-2].xi,yystack.l_mark[0].xi);}
break;
case 9:
#line 84 "calc.y"
	{yyval.xi=calculator::mul(yystack.l_mark[-2].xi,yystack.l_mark[0].xi);}
break;
case 10:
#line 85 "calc.y"
	{yyval.xi=calculator::div(yystack.l_mark[-2].xi,yystack.l_mark[0].xi);}
break;
case 11:
#line 86 "calc.y"
	{yyval.xi=calculator::pow(yystack.l_mark[-2].xi,yystack.l_mark[0].xi);}
break;
case 12:
#line 87 "calc.y"
	{yyval.xi=yystack.l_mark[-1].xi;}
break;
case 13:
#line 88 "calc.y"
	{yyval.xi=calculator::uop(yystack.l_mark[-1].xi,yystack.l_mark[0].xi);}
break;
case 14:
#line 89 "calc.y"
	{yyval.xi=calculator::uop('-',yystack.l_mark[0].xi);}
break;
#line 897 "calc_yacc.c"
    }
    yystack.s_mark -= yym;
    yystate = *yystack.s_mark;
    yystack.l_mark -= yym;
    yym = yylhs[yyn];
    if (yystate == 0 && yym == 0)
    {
#if YYDEBUG
        if (yydebug)
            printf("%sdebug: after reduction, shifting from state 0 to\
 state %d\n", YYPREFIX, YYFINAL);
#endif
        yystate = YYFINAL;
        *++yystack.s_mark = YYFINAL;
        *++yystack.l_mark = yyval;
        if (yychar < 0)
        {
            if ((yychar = YYLEX) < 0) yychar = 0;
#if YYDEBUG
            if (yydebug)
            {
                yys = 0;
                if (yychar <= YYMAXTOKEN) yys = yyname[yychar];
                if (!yys) yys = "illegal-symbol";
                printf("%sdebug: state %d, reading %d (%s)\n",
                        YYPREFIX, YYFINAL, yychar, yys);
            }
#endif
        }
        if (yychar == 0) goto yyaccept;
        goto yyloop;
    }
    if ((yyn = yygindex[yym]) && (yyn += yystate) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == yystate)
        yystate = yytable[yyn];
    else
        yystate = yydgoto[yym];
#if YYDEBUG
    if (yydebug)
        printf("%sdebug: after reduction, shifting from state %d \
to state %d\n", YYPREFIX, *yystack.s_mark, yystate);
#endif
    if (yystack.s_mark >= yystack.s_last && yygrowstack(&yystack))
    {
        goto yyoverflow;
    }
    *++yystack.s_mark = (short) yystate;
    *++yystack.l_mark = yyval;
    goto yyloop;

yyoverflow:
    yyerror("yacc stack overflow");

yyabort:
    yyfreestack(&yystack);
    return (1);

yyaccept:
    yyfreestack(&yystack);
    return (0);
}
