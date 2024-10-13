#include <sharding_dense.h>

#include <string>

#include <doctest.h>

TEST_CASE("insert") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    const std::string hi{"hi"};
    map.insert({std::string("hi"), 1});
    map.insert({"hello", 1});
    map.insert({"hello", 3});
    map.insert(std::pair{"world", 1});
    map.insert(std::make_pair("!", 1));
    map.insert({hi, 3});

    REQUIRE(map.size() == 4);
    REQUIRE(map.empty() == false);

    map.insert(map.cend(), {"end", 4});
    REQUIRE(map.size() == 5);

    ankerl::unordered_dense::sharding_set<std::string> set2;
    set2.insert("hi");
    set2.insert(std::string{"hello"});
    set2.insert("hello");
    REQUIRE(set2.size() == 2);
    REQUIRE(set2.empty() == false);

    set2.insert(set2.cend(), "end");
    REQUIRE(set2.size() == 3);
}

TEST_CASE("insert_more") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;

    auto to_insert = std::vector<std::pair<std::string, uint64_t>>{{"a", 1}, {"b", 2}, {"c", 3}};
    map.insert(to_insert.begin(), to_insert.end());
    REQUIRE(map.size() == 3);

    ankerl::unordered_dense::sharding_set<std::string> set2;
    std::vector<std::string> to_insert_set{"a", "b", "c"};
    set2.insert(to_insert_set.begin(), to_insert_set.end());
    REQUIRE(set2.size() == 3);
}


TEST_CASE("insert_or_assign") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    map.insert_or_assign("a", 1ULL);

    map.insert_or_assign("b", 2ULL);
    map.insert_or_assign("c", 3ULL);
    REQUIRE(map.size() == 3);
    map.insert_or_assign("a", 4ULL);
    REQUIRE(map.size() == 3);
    std::string a_val{"3"};
    map.insert_or_assign(a_val, 10ULL);
    REQUIRE(map.size() == 4);
    // REQUIRE(map["3"] == 10ULL);
}

TEST_CASE("emplace") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    map.emplace("a", 1ULL);
    REQUIRE(map.size() == 1);

    // try emplace with set
    ankerl::unordered_dense::sharding_set<std::string> set;
    set.emplace("a");
    REQUIRE(set.size() == 1);
    set.emplace(std::string{"b"});
    REQUIRE(set.size() == 2);
    std::string c_val{"c"};
    set.emplace(std::move(c_val));
    REQUIRE(set.size() == 3);
}

TEST_CASE("try_emplace") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    map.try_emplace("a", 1ULL);
    REQUIRE(map.size() == 1);
    map.try_emplace("b", 2ULL);
    REQUIRE(map.size() == 2);
    map.try_emplace("a", 3ULL);
    REQUIRE(map.size() == 2);
    map.try_emplace(map.cend(), std::string{"c"}, 4ULL);
    map.try_emplace("a", 3ULL);
}