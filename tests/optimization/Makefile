CXX = g++
CXXFLAGS = -std=c++20 -O3 -Wall -Wextra -I../../include
DEBUG_FLAGS = -std=c++20 -g -O0 -Wall -Wextra -I../../include -fsanitize=address

# Test executables
TESTS = test_newton test_gradient_descent test_root_finding test_golden_section \
        test_simulated_annealing test_integration test_all

# Build all tests
all: $(TESTS)

# Individual test targets
test_newton: test_newton.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test_gradient_descent: test_gradient_descent.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test_root_finding: test_root_finding.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test_golden_section: test_golden_section.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test_simulated_annealing: test_simulated_annealing.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test_integration: test_integration.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test_all: test_all.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Debug builds
debug: CXXFLAGS = $(DEBUG_FLAGS)
debug: all

# Run all tests
test: all
	@echo "Running Newton's method tests..."
	@./test_newton
	@echo "\nRunning gradient descent tests..."
	@./test_gradient_descent
	@echo "\nRunning root finding tests..."
	@./test_root_finding
	@echo "\nRunning golden section tests..."
	@./test_golden_section
	@echo "\nRunning simulated annealing tests..."
	@./test_simulated_annealing
	@echo "\nRunning integration tests..."
	@./test_integration

# Run with valgrind for memory checking
memcheck: debug
	valgrind --leak-check=full --show-leak-kinds=all ./test_all

# Performance profiling
profile: CXXFLAGS += -pg
profile: all
	./test_integration
	gprof test_integration gmon.out > profile.txt
	@echo "Profile written to profile.txt"

# Clean build artifacts
clean:
	rm -f $(TESTS) *.o *.out profile.txt

# Check compilation of headers
check-headers:
	@echo "Checking header compilation..."
	@$(CXX) $(CXXFLAGS) -fsyntax-only ../../include/stepanov/optimization.hpp
	@$(CXX) $(CXXFLAGS) -fsyntax-only ../../include/stepanov/optimization/*.hpp
	@echo "All headers compile successfully!"

.PHONY: all test clean debug memcheck profile check-headers