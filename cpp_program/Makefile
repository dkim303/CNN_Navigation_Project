CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -O2

program: main.o
	$(CXX) $(CXXFLAGS) main.o -o program

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

clean:
	rm -f *.o program

.PHONY: clean