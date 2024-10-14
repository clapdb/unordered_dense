#include <sharding_dense.h>

#include <string>
#include <fmt/core.h>

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
    CHECK_EQ(map.find("google"), map.cend());

    ankerl::unordered_dense::sharding_set<std::string> set2;
    set2.insert("hi");
    set2.insert(std::string{"hello"});
    set2.insert("hello");
    REQUIRE(set2.size() == 2);
    REQUIRE(set2.empty() == false);

    set2.insert(set2.cend(), "end");
    REQUIRE(set2.size() == 3);
    auto it2 = set2.find("google");
    REQUIRE(it2 == set2.cend());
}

TEST_CASE("insert_more") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;

    auto to_insert = std::vector<std::pair<std::string, uint64_t>>{{"a", 1}, {"b", 2}, {"c", 3}};
    map.insert(to_insert.begin(), to_insert.end());
    REQUIRE(map.size() == 3);
    auto a_it = map.find("a");
    REQUIRE(a_it != map.cend());
    CHECK_EQ(a_it->second, 1);
    auto b_it = map.find("b");
    REQUIRE(b_it != map.cend());
    CHECK_EQ(b_it->second, 2);
    auto c_it = map.find("c");
    REQUIRE(c_it != map.cend());
    CHECK_EQ(c_it->second, 3);

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
    auto [it1, success1] = map.emplace("a", 1ULL);
    REQUIRE(map.size() == 1);
    REQUIRE(success1);
    auto [it2, success2] = map.emplace("c", 1ULL);
    REQUIRE(success2);
    REQUIRE(map.size() == 2);
    auto [it3, success3] = map.emplace("a", 10ULL);
    REQUIRE(not success3);
    REQUIRE(map.size() == 2);
    CHECK_EQ(map["a"], 1ULL);

    // try emplace with set
    ankerl::unordered_dense::sharding_set<std::string> set;
    auto [it4, success4] = set.emplace("a");
    REQUIRE(set.size() == 1);
    REQUIRE(success4);
    auto [it5, success5] = set.emplace("b");
    REQUIRE(success5);
    REQUIRE(set.size() == 2);
    std::string c_val{"c"};
    auto [it6, success6] = set.emplace(std::move(c_val));
    REQUIRE(success6);
    REQUIRE(set.size() == 3);
    auto [it7, success7] = set.emplace("a");
    REQUIRE(set.size() == 3);
    REQUIRE(not success7);
}

TEST_CASE("try_emplace") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    auto [it1, success1] = map.try_emplace("a", 1ULL);
    REQUIRE(map.size() == 1);
    REQUIRE(success1);
    auto [it2, success2] = map.try_emplace("b", 2ULL);
    REQUIRE(success2);
    REQUIRE(map.size() == 2);
    auto [it3, success3] = map.try_emplace("a", 3ULL);
    REQUIRE(not success3);
    REQUIRE(map.size() == 2);
    auto _ = map.try_emplace(map.cend(), std::string{"c"}, 4ULL);
    REQUIRE(map.size() == 3);
    auto it = map.find("a");
    REQUIRE(it != map.cend());
    CHECK_EQ(it->second, 1ULL);
    REQUIRE(map["a"] == 1ULL);

    // set do not support try_emplace
}

TEST_CASE("at") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    map.insert({"a", 1});
    map.insert({"b", 2});
    map.insert({"c", 3});
    map.insert({"d", 4});
    map.insert({"e", 5});
    map.insert({"f", 6});
    map.insert({"g", 7});
    map.insert({"h", 8});
    map.insert({"i", 9});
    map.insert({"j", 10});
    // lookup
    CHECK_EQ(map.at("a"), 1);
    CHECK_EQ(map.at("b"), 2);
    CHECK_EQ(map.at("c"), 3);
    CHECK_EQ(map.at("d"), 4);
    CHECK_EQ(map.at("e"), 5);
    CHECK_EQ(map.at("f"), 6);
    CHECK_EQ(map.at("g"), 7);
    CHECK_EQ(map.at("h"), 8);
    CHECK_EQ(map.at("i"), 9);
    CHECK_EQ(map.at("j"), 10);
    CHECK_EQ(map["a"], 1);
    CHECK_EQ(map["b"], 2);
    CHECK_EQ(map["c"], 3);
    CHECK_EQ(map["d"], 4);
    CHECK_EQ(map["e"], 5);
    CHECK_EQ(map["f"], 6);
    CHECK_EQ(map["g"], 7);
    CHECK_EQ(map["h"], 8);
    CHECK_EQ(map["i"], 9);
    CHECK_EQ(map["j"], 10);
    std::string a{"a"};
    CHECK_EQ(map[a], 1);
}

TEST_CASE("operator[]") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    map.insert({"a", 1});
    map.insert({"b", 2});
    map.insert({"c", 3});
    map.insert({"d", 4});
    map["e"] = 5;
    CHECK_EQ(map["a"], 1);
    CHECK_EQ(map["b"], 2);
    CHECK_EQ(map["c"], 3);
    CHECK_EQ(map["d"], 4);
    CHECK_EQ(map["e"], 5);
    map["a"] = 11;
    std::string a{"a"};
    CHECK_EQ(map[a], 11);
}

TEST_CASE("count") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    map.insert({"a", 1});
    map.insert({"b", 2});
    map.insert({"c", 3});
    map.insert({"d", 4});
    map.insert({"e", 5});
    CHECK_EQ(map.count("a"), 1);
    CHECK_EQ(map.count("b"), 1);
    CHECK_EQ(map.count("c"), 1);
    CHECK_EQ(map.count("d"), 1);
    CHECK_EQ(map.count("e"), 1);
    CHECK_EQ(map.count("f"), 0);
    CHECK_EQ(map.count("g"), 0);
    CHECK_EQ(map.count("h"), 0);
    CHECK_EQ(map.count("i"), 0);
    CHECK_EQ(map.count("j"), 0);
}

TEST_CASE("find") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    map.insert({"a", 1});
    map.insert({"b", 2});
    map.insert({"c", 3});
    map.insert({"d", 4});
    map.insert({"e", 5});
    CHECK_EQ(map.find("a")->first, "a");
    CHECK_EQ(map.find("a")->second, 1);
    CHECK_EQ(map.find("b")->first, "b");
    CHECK_EQ(map.find("b")->second, 2);
    CHECK_EQ(map.find("c")->first, "c");
    CHECK_EQ(map.find("c")->second, 3);
    CHECK_EQ(map.find("d")->first, "d");
    CHECK_EQ(map.find("d")->second, 4);
    CHECK_EQ(map.find("e")->first, "e");
    CHECK_EQ(map.find("e")->second, 5);
    CHECK_EQ(map.find("f"), map.cend());
}


TEST_CASE("contains") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    map.insert({"a", 1});
    map.insert({"b", 2});
    map.insert({"c", 3});
    map.insert({"d", 4});
    map.insert({"e", 5});
    CHECK(map.contains("a"));
    CHECK(map.contains("b"));
    CHECK(map.contains("c"));
    CHECK(map.contains("d"));
    CHECK(map.contains("e"));
    CHECK(not map.contains("f"));
    CHECK(not map.contains("g"));
    CHECK(not map.contains("h"));
    CHECK(not map.contains("i"));
    CHECK(not map.contains("j"));
}

TEST_CASE("equal_range") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    map.insert({"a", 1});
    map.insert({"b", 2});
    map.insert({"c", 3});
    map.insert({"d", 4});
    map.insert({"e", 5});
    
    auto range = map.equal_range("a");
    REQUIRE(range.first != map.cend());
    CHECK_EQ(range.first->first, "a");
    CHECK_EQ(range.first->second, 1);
    auto range2 = map.equal_range("z");
    CHECK_EQ(range2.first, map.cend());
    CHECK_EQ(range2.second, map.cend());
}

TEST_CASE("bucket_count") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    map.insert({"a", 1});
    map.insert({"b", 2});
    map.insert({"c", 3});
    map.insert({"d", 4});
    map.insert({"e", 5});
    // the default bucket count of a internal table is 4
    CHECK_EQ(map.bucket_count(), 32);
    for (uint32_t i = 0; i < 8UL; ++i) {
        CHECK_EQ(map.bucket_count(i), 4);
    }
}   

TEST_CASE("load_factor") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    map.insert({"a", 1});
    map.insert({"b", 2});
    map.insert({"c", 3});
    map.insert({"d", 4});
    map.insert({"e", 5});
    map.insert({"f", 5});
    map.insert({"g", 5});
    map.insert({"h", 5});
    CHECK_EQ(map.load_factor(), 0.25);
}

TEST_CASE("reserve") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    map.reserve(128);
    CHECK_EQ(map.size(), 0);
    CHECK_EQ(map.bucket_count(), 256);
}

TEST_CASE("erase") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    map.insert({"a", 1});
    map.insert({"b", 2});
    map.insert({"c", 3});
    map.insert({"d", 4});
    map.insert({"e", 5});
    
    map.erase(map.find("a"));
    CHECK_EQ(map.size(), 4);
    CHECK_EQ(map.find("a"), map.cend());
    
    map.erase("b");
    CHECK_EQ(map.size(), 3);
    CHECK_EQ(map.find("b"), map.cend());

    map.erase(map.find("c"));
    CHECK_EQ(map.size(), 2);
    CHECK_EQ(map.find("c"), map.cend());
}

TEST_CASE("erase_range") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    map.insert({"a", 1});
    map.insert({"b", 2});
    map.insert({"c", 3});
    map.insert({"d", 4});
    map.insert({"e", 5});

    map.erase(map.cbegin(), map.cend());
    CHECK_EQ(map.size(), 0);
}

TEST_CASE("clear") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    map.insert({"a", 1});
    map.insert({"b", 2});
    map.insert({"c", 3});
    map.insert({"d", 4});
    map.insert({"e", 5});
    map.clear();
    CHECK_EQ(map.size(), 0);
} 

TEST_CASE("swap") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map1;
    map1.insert({"a", 1});
    map1.insert({"b", 2});
    map1.insert({"c", 3});
    
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map2;
    map2.insert({"f", 6});
    map2.insert({"g", 7});
    map2.insert({"h", 8});
    map2.insert({"i", 9});
    map2.insert({"j", 10});

    map1.swap(map2);
    CHECK_EQ(map1.size(), 5);
    CHECK_EQ(map2.size(), 3);
}

TEST_CASE("extract") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    map.insert({"a", 1});
    map.insert({"b", 2});
    map.insert({"c", 3});
    map.insert({"d", 4});
    map.insert({"e", 5});

    auto it = map.find("e");
    auto node = map.extract(it);
    CHECK(node.first == "e");
    CHECK(node.second == 5);
    CHECK_EQ(map.size(), 4);
    CHECK_EQ(map.find("e"), map.cend());
}

TEST_CASE("for-loop-all") {
    ankerl::unordered_dense::sharding_map<std::string, uint64_t> map;
    map.insert({"a", 1});
    map.insert({"b", 2});
    map.insert({"c", 3});
    map.insert({"d", 4});
    map.insert({"e", 5});
    map.insert({"f", 6});
    map.insert({"g", 7});
    map.insert({"h", 8});
    map.insert({"i", 9});
    map.insert({"j", 10});
    map.insert({"k", 11});
    map.insert({"l", 12});
    map.insert({"m", 13});
    map.insert({"n", 14});
    map.insert({"o", 15});
    map.insert({"p", 16});
    map.insert({"q", 17});
    map.insert({"r", 18});
    map.insert({"s", 19});

    for (auto& [key, value] : map) {
        CHECK_EQ(map[key], value);
    }
}
