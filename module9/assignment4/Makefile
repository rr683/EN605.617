all: main

main: main.cu
	nvcc main.cu -lcurand -lcublas -o main

run: main
	./main

clean:
	rm -f main