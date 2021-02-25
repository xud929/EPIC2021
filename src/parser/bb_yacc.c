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

#define YYPREFIX "yy"

#define YYPURE 0

#line 2 "bb.y"
#include<iostream>
#include<unordered_map>
#include<string>
#include<cstring>
#include<vector>
#include"bb.h"

extern "C" int yylex(void);
extern "C" void yyerror(const char *);
extern "C" int yywrap(void);
typedef struct yy_buffer_state * YY_BUFFER_STATE;
extern "C" YY_BUFFER_STATE yy_scan_string(const char * str);
extern "C" void yy_delete_buffer(YY_BUFFER_STATE buffer);

namespace bbp{
    std::unordered_map<std::string, int> bbints;
    std::unordered_map<std::string, double> bbdbls;
    std::unordered_map<std::string, bool> bbools;
    std::unordered_map<std::string, std::string> bbstrs;
    std::unordered_map<std::string, std::string> bbudef;
    std::vector<std::string> udef_order;
    const std::string sep=" ";
}
static std::string current_scope;
static int count=0;

inline std::string join(const char *id, int count){
   return current_scope+bbp::sep+std::string(id)+bbp::sep+std::to_string(count);
}
#line 33 "bb.y"
#ifdef YYSTYPE
#undef  YYSTYPE_IS_DECLARED
#define YYSTYPE_IS_DECLARED 1
#endif
#ifndef YYSTYPE_IS_DECLARED
#define YYSTYPE_IS_DECLARED 1
typedef union{
	int xint;
	double xdouble;
	char *xstr;
} YYSTYPE;
#endif /* !YYSTYPE_IS_DECLARED */
#line 62 "bb_yacc.c"

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
#define BOOL 258
#define END 259
#define DOUBLE 260
#define BEG 261
#define ID 262
#define VAR 263
#define STR 264
#define SEPARATOR 265
#define YYERRCODE 256
static const short yylhs[] = {                           -1,
    0,    0,    0,    3,    2,    2,    2,    2,    2,    2,
    2,    2,    2,    2,    2,    2,    2,    2,    2,    2,
    1,    1,    4,    5,
};
static const short yylen[] = {                            2,
    1,    2,    3,    3,    3,    3,    3,    3,    3,    6,
    6,    6,    6,    6,    3,    3,    3,    3,    3,    3,
    1,    2,    1,    1,
};
static const short yydefred[] = {                         0,
   23,    0,    1,    0,    0,    2,    0,    0,    3,    0,
    0,   24,    0,    4,    5,    7,    6,    8,   21,    0,
    0,   15,   17,   16,   18,    0,   20,   22,    0,    0,
   10,   12,   11,   13,    0,
};
static const short yydgoto[] = {                          2,
   20,    8,    3,    4,   14,
};
static const short yysindex[] = {                      -257,
    0, -237,    0, -248, -257,    0,  -61, -253,    0, -247,
 -225,    0, -255,    0,    0,    0,    0,    0,    0, -229,
  -57,    0,    0,    0,    0, -229,    0,    0,  -24, -238,
    0,    0,    0,    0, -229,
};
static const short yyrindex[] = {                         0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0, -244,
    0,    0,    0,    0,    0, -236,    0,    0,    0,    0,
    0,    0,    0,    0, -232,
};
static const short yygindex[] = {                         0,
  -12,   25,   29,    0,    0,
};
#define YYTABLESIZE 38
static const short yytable[] = {                         10,
   26,   22,   23,    1,   24,   12,    7,   25,   19,   15,
   16,   13,   17,    7,    9,   18,   19,   35,   31,   32,
    9,   33,   19,    1,   34,   19,   14,    5,   19,   11,
    6,   21,   14,    9,   28,   29,   30,   27,
};
static const short yycheck[] = {                         61,
   13,  257,  258,  261,  260,  259,  262,  263,  264,  257,
  258,  265,  260,  262,  259,  263,  264,   30,  257,  258,
  265,  260,  259,  261,  263,  264,  259,  265,  265,   91,
    2,  257,  265,    5,  264,   93,   61,   13,
};
#define YYFINAL 2
#ifndef YYDEBUG
#define YYDEBUG 0
#endif
#define YYMAXTOKEN 265
#if YYDEBUG
static const char *yyname[] = {

"end-of-file",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"'='",0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"'['",0,"']'",0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"INT","BOOL","END",
"DOUBLE","BEG","ID","VAR","STR","SEPARATOR",
};
static const char *yyrule[] = {
"$accept : program",
"program : paragraph",
"program : program paragraph",
"program : program SEPARATOR paragraph",
"paragraph : beg statement end",
"statement : ID '=' INT",
"statement : ID '=' DOUBLE",
"statement : ID '=' BOOL",
"statement : ID '=' VAR",
"statement : ID '=' strings",
"statement : ID '[' INT ']' '=' INT",
"statement : ID '[' INT ']' '=' DOUBLE",
"statement : ID '[' INT ']' '=' BOOL",
"statement : ID '[' INT ']' '=' VAR",
"statement : ID '[' INT ']' '=' strings",
"statement : statement SEPARATOR INT",
"statement : statement SEPARATOR DOUBLE",
"statement : statement SEPARATOR BOOL",
"statement : statement SEPARATOR VAR",
"statement : statement SEPARATOR strings",
"statement : statement SEPARATOR statement",
"strings : STR",
"strings : strings STR",
"beg : BEG",
"end : END",

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
#line 90 "bb.y"

namespace bbp{
void clear(){
    bbints.clear();
    bbdbls.clear();
    bbools.clear();
    bbstrs.clear();
    bbudef.clear();
    udef_order.clear();
}
void parse(const char *str){
    YY_BUFFER_STATE buffer=yy_scan_string(str);
    yyparse();
    yy_delete_buffer(buffer);
}
}

void yyerror(const char *str){
	std::cerr<<str<<std::endl;
}
int yywrap(){
	return 1;
}

#line 257 "bb_yacc.c"

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
case 4:
#line 52 "bb.y"
	{free(yystack.l_mark[-1].xstr);}
break;
case 5:
#line 54 "bb.y"
	{count=0;bbp::bbints[join(yystack.l_mark[-2].xstr,count)]=yystack.l_mark[0].xint;yyval.xstr=yystack.l_mark[-2].xstr;}
break;
case 6:
#line 55 "bb.y"
	{count=0;bbp::bbdbls[join(yystack.l_mark[-2].xstr,count)]=yystack.l_mark[0].xdouble;yyval.xstr=yystack.l_mark[-2].xstr;}
break;
case 7:
#line 56 "bb.y"
	{count=0;bbp::bbools[join(yystack.l_mark[-2].xstr,count)]=yystack.l_mark[0].xint;yyval.xstr=yystack.l_mark[-2].xstr;}
break;
case 8:
#line 57 "bb.y"
	{count=0;
                                                 std::string temp=join(yystack.l_mark[-2].xstr,count);
                                                 bbp::udef_order.push_back(temp);
                                                 bbp::bbudef[temp]=std::string(yystack.l_mark[0].xstr);
                                                 yyval.xstr=yystack.l_mark[-2].xstr;free(yystack.l_mark[0].xstr);}
break;
case 9:
#line 62 "bb.y"
	{count=0;bbp::bbstrs[join(yystack.l_mark[-2].xstr,count)]=std::string(yystack.l_mark[0].xstr);yyval.xstr=yystack.l_mark[-2].xstr; delete [] yystack.l_mark[0].xstr;}
break;
case 10:
#line 63 "bb.y"
	{count=yystack.l_mark[-3].xint;bbp::bbints[join(yystack.l_mark[-5].xstr,count)]=yystack.l_mark[0].xint;yyval.xstr=yystack.l_mark[-5].xstr;}
break;
case 11:
#line 64 "bb.y"
	{count=yystack.l_mark[-3].xint;bbp::bbdbls[join(yystack.l_mark[-5].xstr,count)]=yystack.l_mark[0].xdouble;yyval.xstr=yystack.l_mark[-5].xstr;}
break;
case 12:
#line 65 "bb.y"
	{count=yystack.l_mark[-3].xint;bbp::bbools[join(yystack.l_mark[-5].xstr,count)]=yystack.l_mark[0].xint;yyval.xstr=yystack.l_mark[-5].xstr;}
break;
case 13:
#line 66 "bb.y"
	{count=yystack.l_mark[-3].xint;
                                                 std::string temp=join(yystack.l_mark[-5].xstr,count);
                                                 bbp::udef_order.push_back(temp);
                                                 bbp::bbudef[temp]=std::string(yystack.l_mark[0].xstr);
                                                 yyval.xstr=yystack.l_mark[-5].xstr;free(yystack.l_mark[0].xstr);}
break;
case 14:
#line 71 "bb.y"
	{count=yystack.l_mark[-3].xint;bbp::bbstrs[join(yystack.l_mark[-5].xstr,count)]=std::string(yystack.l_mark[0].xstr);yyval.xstr=yystack.l_mark[-5].xstr;delete [] yystack.l_mark[0].xstr;}
break;
case 15:
#line 72 "bb.y"
	{++count;bbp::bbints[join(yystack.l_mark[-2].xstr,count)]=yystack.l_mark[0].xint;yyval.xstr=yystack.l_mark[-2].xstr;}
break;
case 16:
#line 73 "bb.y"
	{++count;bbp::bbdbls[join(yystack.l_mark[-2].xstr,count)]=yystack.l_mark[0].xdouble;yyval.xstr=yystack.l_mark[-2].xstr;}
break;
case 17:
#line 74 "bb.y"
	{++count;bbp::bbools[join(yystack.l_mark[-2].xstr,count)]=yystack.l_mark[0].xint;yyval.xstr=yystack.l_mark[-2].xstr;}
break;
case 18:
#line 75 "bb.y"
	{++count;
                                                 std::string temp=join(yystack.l_mark[-2].xstr,count);
                                                 bbp::udef_order.push_back(temp);
                                                 bbp::bbudef[temp]=std::string(yystack.l_mark[0].xstr);
                                                 yyval.xstr=yystack.l_mark[-2].xstr;free(yystack.l_mark[0].xstr);}
break;
case 19:
#line 80 "bb.y"
	{++count;bbp::bbstrs[join(yystack.l_mark[-2].xstr,count)]=std::string(yystack.l_mark[0].xstr);yyval.xstr=yystack.l_mark[-2].xstr;delete [] yystack.l_mark[0].xstr;}
break;
case 20:
#line 81 "bb.y"
	{free(yystack.l_mark[-2].xstr);yyval.xstr=yystack.l_mark[0].xstr;}
break;
case 21:
#line 82 "bb.y"
	{yyval.xstr=new char[strlen(yystack.l_mark[0].xstr)+1];strcpy(yyval.xstr,yystack.l_mark[0].xstr);free(yystack.l_mark[0].xstr);}
break;
case 22:
#line 83 "bb.y"
	{yyval.xstr=new char[strlen(yystack.l_mark[-1].xstr)+strlen(yystack.l_mark[0].xstr)+1];strcpy(yyval.xstr,yystack.l_mark[-1].xstr);strcat(yyval.xstr,yystack.l_mark[0].xstr);free(yystack.l_mark[-1].xstr);free(yystack.l_mark[0].xstr);}
break;
case 23:
#line 85 "bb.y"
	{current_scope=std::string(yystack.l_mark[0].xstr);free(yystack.l_mark[0].xstr);}
break;
case 24:
#line 87 "bb.y"
	{current_scope.clear();}
break;
#line 559 "bb_yacc.c"
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
