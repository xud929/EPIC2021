#define INT 257
#define BOOL 258
#define END 259
#define DOUBLE 260
#define BEG 261
#define ID 262
#define VAR 263
#define STR 264
#define SEPARATOR 265
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
extern YYSTYPE yylval;
