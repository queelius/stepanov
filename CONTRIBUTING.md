# Contributing to Generic Math Library

Thank you for your interest in contributing to the Generic Math Library! We welcome contributions from the community and are grateful for any help you can provide.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## How to Contribute

### Reporting Issues

1. Check if the issue already exists in the [issue tracker](https://github.com/yourusername/generic_math/issues)
2. Create a new issue with a clear title and description
3. Include:
   - Steps to reproduce the problem
   - Expected behavior
   - Actual behavior
   - Compiler version and platform
   - Minimal code example demonstrating the issue

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit with clear messages (`git commit -m 'Add amazing feature'`)
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Guidelines

#### Code Style
- Follow the existing code style (snake_case for functions/variables, PascalCase for template parameters)
- Use meaningful variable and function names
- Add comments for complex algorithms
- Document mathematical requirements and assumptions

#### Testing
- Write unit tests for new functionality
- Ensure existing tests still pass
- Add benchmark tests for performance-critical code
- Test with multiple compiler versions when possible

#### Documentation
- Update relevant documentation
- Add examples for new features
- Document mathematical concepts and requirements
- Update the CHANGELOG.md for significant changes

### Building and Testing

```bash
# Configure with CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build

# Run tests
cd build && ctest
```

### Performance Considerations

When contributing performance improvements:
1. Provide benchmark results before and after
2. Test with different input sizes and structures
3. Consider memory usage alongside speed
4. Ensure optimizations don't break generic programming principles

## Areas for Contribution

### High Priority
- Additional matrix structure specializations
- More number theory algorithms
- Improved SIMD implementations
- Platform-specific optimizations

### Documentation
- Tutorial examples
- Mathematical proofs and explanations
- Performance tuning guides
- API documentation improvements

### Testing
- Edge case tests
- Property-based testing
- Cross-platform testing
- Compiler compatibility testing

## Questions?

Feel free to open an issue for any questions about contributing. We're here to help!