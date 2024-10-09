#include <sharding_dense.h>

#include <string>

#include <doctest.h>

TEST_CASE("init") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    map.insert({std::string("hi"), 1});
    map.insert({"hello", 1});

    ankerl::unordered_dense::sharding_set<std::string> set2;
    set2.insert("hi");
    set2.insert(std::string{"hello"});
}
