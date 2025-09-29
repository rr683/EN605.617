all: assignment

assignment: assignment.cu
	nvcc assignment.cu -o assignment

run64: assignment
	./assignment 64 64

run128: assignment
	./assignment 128 128

run256: assignment
	./assignment 256 256

clean:
	rm -f assignment