CC=mpic++
oo=main.o test.o crab_cavity.o gtrack.o linear_map.o Lorentz_boost.o beam.o faddeeva.o erfinv.o gwofz.o
main:${oo}
	nvcc -dlink ${oo} -o link.o
	${CC} -o main ${oo} link.o -lcudart -lcudadevrt -ltrng4
main.o:main.cpp
	${CC} -c -O3 main.cpp -std=c++17
test.o:test.cu
	nvcc -c -O3 -dc test.cu -std=c++14
crab_cavity.o:crab_cavity.cpp crab_cavity.h acc_base.h constants.h
	${CC} -c -O3 crab_cavity.cpp -std=c++1z
linear_map.o:linear_map.cpp linear_map.h acc_base.h constants.h
	${CC} -c -O3 linear_map.cpp -std=c++1z
Lorentz_boost.o:Lorentz_boost.cpp Lorentz_boost.h acc_base.h constants.h
	${CC} -c -O3 Lorentz_boost.cpp -std=c++1z
gtrack.o:gtrack.cu gtrack.h
	nvcc -c -O3 -dc gtrack.cu -std=c++14
beam.o:beam.cpp beam.h acc_base.h faddeeva.h
	${CC} -c -O3 beam.cpp -std=c++1z
faddeeva.o:faddeeva.h faddeeva.cpp
	${CC} -c -O3 faddeeva.cpp -std=c++1z
erfinv.o:constants.h erfinv.cpp
	${CC} -c -O3 erfinv.cpp -std=c++1z
gwofz.o:gwofz.h gwofz.cu
	nvcc -c -O3 gwofz.cu -dc -std=c++14
.PHONY:clean
clean:
	rm *.o

