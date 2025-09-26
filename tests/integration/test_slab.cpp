#include <iostream>
#include <cassert>
#include <vector>
#include "stepanov/allocators.hpp"

using namespace stepanov;

int main() {
    std::cout << "Testing slab allocator...\n";

    struct test_object {
        int a;
        double b;
        char c[16];
    };

    slab_allocator<test_object, 32> slab(2);

    // Allocate one object
    test_object* obj1 = slab.allocate();
    assert(obj1 != nullptr);
    obj1->a = 42;

    std::cout << "  Allocated 1 object\n";

    // Allocate more objects
    std::vector<test_object*> objects;
    for (int i = 0; i < 10; ++i) {
        test_object* obj = slab.allocate();
        assert(obj != nullptr);
        obj->a = i;
        objects.push_back(obj);
    }

    std::cout << "  Allocated " << objects.size() << " more objects\n";
    std::cout << "  Slab count: " << slab.slab_count() << "\n";

    // Deallocate
    slab.deallocate(obj1);
    for (auto* obj : objects) {
        slab.deallocate(obj);
    }

    std::cout << "  All objects deallocated\n";

    std::cout << "PASSED\n";
    return 0;
}