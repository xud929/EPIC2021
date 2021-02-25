#define INT 257
#define UOP 258
#define DOUBLE 259
#define ID 260
#define POW 261
#define UMINUS 262
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
extern YYSTYPE zzlval;
