#include <iostream>
#include <exception>
#include <string>

// Command Format: ./cpp_program/program models/<???>


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Invalid command line arguments"
        return 1;
    }

    std::string model_name(argv[1]);

    return 0;
}