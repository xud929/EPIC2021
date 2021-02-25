main:../calc.l ../calc.y main.cpp
	yacc -o calc_yacc.c -p zz -d ../calc.y 
	lex -o calc_lex.c -P zz ../calc.l 
	gcc -c calc_lex.c
	g++ -c calc_yacc.c -std=c++11
	g++ -c main.cpp -std=c++11
	g++ -o main main.o calc_lex.o calc_yacc.o
.PHONY:clean
clean:
	rm -f main main.o lex.zz.o y.tab.o lex.zz.c y.tab.c calc*.c calc*.o calc*.o y.tab.h
