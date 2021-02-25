CC=mpic++
oo=crab_cavity.o Lorentz_boost.o linear_map.o beam.o faddeeva.o erfinv.o beam_beam.o poisson_solver.o radiation.o
poo=../src/parser/bb_lex.o ../src/parser/bb_yacc.o ../src/parser/calc_lex.o ../src/parser/calc_yacc.o ../src/parser/fmt_lex.o ../src/parser/input.o
target=weak_strong
.PHONY:parser_compile clean
${target}:${target}.o ${oo} parser_compile
	${CC} -o ${target} ${target}.o ${oo} ${poo} -ltrng4
	mv ${target} ../bin/
tune_scan2:tune_scan2.o gwofz.o gtrack.o ${oo} parser_compile
	nvcc -dlink ${oo} ${poo} tune_scan.o gwofz.o gtrack.o -o link.o
	${CC} -o tune_scan2 ${oo} ${poo} tune_scan2.o gwofz.o gtrack.o link.o -ltrng4 -lcudart -lcudadevrt
	mv tune_scan2 ../bin/
gpu_weak_strong:gpu_weak_strong.o gwofz.o gtrack.o ${oo} parser_compile
	nvcc -dlink ${oo} ${poo} gpu_weak_strong.o gwofz.o gtrack.o -o link.o
	${CC} -o gpu_weak_strong ${oo} ${poo} gpu_weak_strong.o gwofz.o gtrack.o link.o -ltrng4 -lcudart -lcudadevrt
	mv gpu_weak_strong ../bin/
gpu_weak_strong.o:../src/gpu_weak_strong.cpp 
	${CC} -c -O3 ../src/gpu_weak_strong.cpp -std=c++17
tune_scan2.o:../src/tune_scan2.cpp 
	${CC} -c -O3 ../src/tune_scan2.cpp -std=c++17
gtrack.o:../src/gtrack.cu ../src/gtrack.h
	nvcc -c -O3 -dc ../src/gtrack.cu -std=c++14
${target}.o:../src/${target}.cpp ../src/crab_cavity.h ../src/constants.h ../src/beam.h
	${CC} -c -O3 ../src/${target}.cpp -std=c++17
crab_cavity.o:../src/crab_cavity.cpp ../src/crab_cavity.h ../src/acc_base.h ../src/constants.h
	${CC} -c -O3 ../src/crab_cavity.cpp -std=c++17
Lorentz_boost.o:../src/Lorentz_boost.cpp ../src/Lorentz_boost.h ../src/acc_base.h ../src/constants.h
	${CC} -c -O3 ../src/Lorentz_boost.cpp -std=c++17
linear_map.o:../src/linear_map.cpp ../src/linear_map.h ../src/acc_base.h ../src/constants.h
	${CC} -c -O3 ../src/linear_map.cpp -std=c++17
beam.o:../src/beam.cpp ../src/beam.h ../src/acc_base.h ../src/faddeeva.h
	${CC} -c -O3 ../src/beam.cpp -std=c++17
faddeeva.o:../src/faddeeva.cpp ../src/faddeeva.h
	${CC} -c -O3 ../src/faddeeva.cpp -std=c++17
gwofz.o:../src/gwofz.h ../src/gwofz.cu
	nvcc -c -O3 -dc ../src/gwofz.cu -std=c++14
erfinv.o:../src/erfinv.cpp ../src/constants.h
	${CC} -c -O3 ../src/erfinv.cpp -std=c++17
beam_beam.o:../src/beam_beam.cpp ../src/beam.h ../src/poisson_solver.h ../src/acc_base.h ../src/constants.h
	${CC} -c -O3 ../src/beam_beam.cpp -std=c++17
poisson_solver.o:../src/poisson_solver.h ../src/poisson_solver.cpp
	${CC} -c -O3 ../src/poisson_solver.cpp -std=c++17
radiation.o:../src/beam.h ../src/radiation.cpp ../src/radiation.h
	${CC} -c -O3 ../src/radiation.cpp -std=c++17
bbread:../src/bbread.cpp
	g++ -O3 -o bbread ../src/bbread.cpp -std=c++17
	#mv bbread ../bin/
parser_compile:
	cd ../src/parser/ && make
clean:
	rm *.o
	cd ../src/parser/ && make clean
	cd ../bin && rm -f
