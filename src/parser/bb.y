%{
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
%}

%union{
	int xint;
	double xdouble;
	char *xstr;
};


%token <xint> INT BOOL END
%token <xdouble> DOUBLE
%token <xstr> BEG ID VAR STR
%type <xstr> strings statement

%left SEPARATOR

%%
program  : paragraph
         | program paragraph
         | program SEPARATOR paragraph
         ;
paragraph: beg statement end {free($2);}
		 ;
statement: ID '=' INT                           {count=0;bbp::bbints[join($1,count)]=$3;$$=$1;}
         | ID '=' DOUBLE                        {count=0;bbp::bbdbls[join($1,count)]=$3;$$=$1;}
         | ID '=' BOOL                          {count=0;bbp::bbools[join($1,count)]=$3;$$=$1;}
         | ID '=' VAR                           {count=0;
                                                 std::string temp=join($1,count);
                                                 bbp::udef_order.push_back(temp);
                                                 bbp::bbudef[temp]=std::string($3);
                                                 $$=$1;free($3);}
         | ID '=' strings                       {count=0;bbp::bbstrs[join($1,count)]=std::string($3);$$=$1; delete [] $3;}
         | ID '[' INT ']' '=' INT               {count=$3;bbp::bbints[join($1,count)]=$6;$$=$1;}
         | ID '[' INT ']' '=' DOUBLE            {count=$3;bbp::bbdbls[join($1,count)]=$6;$$=$1;}
         | ID '[' INT ']' '=' BOOL              {count=$3;bbp::bbools[join($1,count)]=$6;$$=$1;}
         | ID '[' INT ']' '=' VAR               {count=$3;
                                                 std::string temp=join($1,count);
                                                 bbp::udef_order.push_back(temp);
                                                 bbp::bbudef[temp]=std::string($6);
                                                 $$=$1;free($6);}
         | ID '[' INT ']' '=' strings           {count=$3;bbp::bbstrs[join($1,count)]=std::string($6);$$=$1;delete [] $6;}
         | statement SEPARATOR INT              {++count;bbp::bbints[join($1,count)]=$3;$$=$1;}
         | statement SEPARATOR DOUBLE           {++count;bbp::bbdbls[join($1,count)]=$3;$$=$1;}
         | statement SEPARATOR BOOL             {++count;bbp::bbools[join($1,count)]=$3;$$=$1;}
         | statement SEPARATOR VAR              {++count;
                                                 std::string temp=join($1,count);
                                                 bbp::udef_order.push_back(temp);
                                                 bbp::bbudef[temp]=std::string($3);
                                                 $$=$1;free($3);}
         | statement SEPARATOR strings          {++count;bbp::bbstrs[join($1,count)]=std::string($3);$$=$1;delete [] $3;}
		 | statement SEPARATOR statement        {free($1);$$=$3;}
strings  : STR                                  {$$=new char[strlen($1)+1];strcpy($$,$1);free($1);}
         | strings STR                          {$$=new char[strlen($1)+strlen($2)+1];strcpy($$,$1);strcat($$,$2);free($1);free($2);}
         ;
beg		 : BEG                                  {current_scope=std::string($1);free($1);}
	  	 ;
end		 : END                                  {current_scope.clear();}
	  	 ;
%%

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

