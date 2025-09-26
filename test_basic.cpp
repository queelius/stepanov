#include <iostream>
#include <stepanov/math.hpp>
#include <stepanov/gcd.hpp>

int main() {
    // Test power function
    std::cout << "Testing power: 2^10 = " << stepanov::power(2, 10) << std::endl;

    // Test GCD
    std::cout << "Testing gcd: gcd(48, 18) = " << stepanov::gcd(48, 18) << std::endl;

    return 0;
}