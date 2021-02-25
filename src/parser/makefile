CC=g++
cc=gcc
OO=bb_yacc.o bb_lex.o calc_yacc.o calc_lex.o fmt_lex.o input.o
.PHONY:target
target:${OO}
input.o:input.cpp bb.h calc.h
	g++ -O3 -c input.cpp -std=c++17
bb_lex.o:bb_lex.c
	${cc} -O3 -c bb_lex.c
bb_yacc.o:bb_yacc.c
	${CC} -O3 -c bb_yacc.c -std=c++17
bb_yacc.c:bb.l bb.y calc.l calc.y
	yacc -o bb_yacc.c -d bb.y
bb_lex.c:bb.l bb.y calc.l calc.y
	lex -o bb_lex.c bb.l
calc_lex.o:calc_lex.c
	${cc} -O3 -c calc_lex.c
calc_yacc.o:calc_yacc.c
	${CC} -c calc_yacc.c -std=c++17
calc_yacc.c:calc.l calc.y
	yacc -o calc_yacc.c -p zz -d calc.y
calc_lex.c:calc.l calc.y
	lex -o calc_lex.c -P zz calc.l
fmt_lex.c:fmt.l calc.l calc.y
	lex -o fmt_lex.c -P ss fmt.l
fmt_lex.o:fmt_lex.c
	${CC} -O3 -c fmt_lex.c -std=c++17
.PHONY:clean
clean:
	rm -f bb_lex.o bb_yacc.o bb_yacc.c bb_lex.c calc_lex.c calc_lex.o calc_yacc.c calc_yacc.o calc_yacc.h bb_yacc.h fmt_lex.c fmt_lex.o input.o

