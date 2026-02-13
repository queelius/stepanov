add_executable(test_peasant test_peasant.cpp)
target_link_libraries(test_peasant GTest::gtest_main)
target_include_directories(test_peasant PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_test(NAME test_peasant COMMAND test_peasant)

add_executable(test_examples test_examples.cpp)
target_link_libraries(test_examples GTest::gtest_main)
target_include_directories(test_examples PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_test(NAME test_examples COMMAND test_examples)
