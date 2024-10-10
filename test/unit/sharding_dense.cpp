#include <sharding_dense.h>

#include <string>

#include <doctest.h>

TEST_CASE("insert") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    const std::string hi{"hi"};
    map.insert({std::string("hi"), 1});
    map.insert({"hello", 1});
    map.insert(std::pair{"world", 1});
    map.insert(std::make_pair("!", 1));
    map.insert({hi, 3});

    ankerl::unordered_dense::sharding_set<std::string> set2;
    set2.insert("hi");
    set2.insert(std::string{"hello"});
}
