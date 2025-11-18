all: distance_map

distance_map: distance_map.cpp
	g++ distance_map.cpp -lOpenCL -o distance_map

run: distance_map
	./distance_map 16 16

clean:
	rm -f distance_map