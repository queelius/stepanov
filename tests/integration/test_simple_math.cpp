#include <iostream>
#include <stepanov/builtin_adaptors.hpp>
#include <stepanov/math.hpp>

using namespace generic_math;

int main() {
    std::cout << "Testing basic operations...\n";

    // Test product with small numbers
    std::cout << "product(5, 3) = " << product(5, 3) << std::endl;
    std::cout << "product(2, 4) = " << product(2, 4) << std::endl;

    // Test power
    std::cout << "power(2, 3) = " << power(2, 3) << std::endl;

    // Test sum
    std::cout << "sum(3, 4) = " << sum(3, 4) << std::endl;

    std::cout << "Tests completed!\n";
    return 0;
}