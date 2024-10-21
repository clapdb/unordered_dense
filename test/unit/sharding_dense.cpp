#include <cmath>
#include <sharding_dense.h>

#include <string>
#include <fmt/core.h>
#include <utility>

#include <doctest.h>

template<typename Map>
void map_test_insert() {
    Map map;
    const std::string hi{"hi"};
    map.insert({std::string{"hi"}, 1});
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
}

template<typename Set>
void set_test_insert() {
    Set set;
    set.insert("hi");
    set.insert(std::string{"hello"});
    set.insert("hello");
    REQUIRE(set.size() == 2);
    REQUIRE(set.empty() == false);
}


TEST_CASE("insert") {
    map_test_insert<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_insert<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
    set_test_insert<ankerl::unordered_dense::sharding_set<std::string>>();  
    set_test_insert<ankerl::unordered_dense::segmented_sharding_set<std::string>>();
}


template<typename Map>
void map_test_insert_more() {
    Map map;
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
}

template<typename Set>
void set_test_insert_more() {
    Set set;
    auto to_insert = std::vector<std::string>{"a", "b", "c"};
    set.insert(to_insert.begin(), to_insert.end());
    REQUIRE(set.size() == 3);
    CHECK_NE(set.find("a"), set.cend());
    CHECK_NE(set.find("b"), set.cend());
    CHECK_NE(set.find("c"), set.cend());
}

TEST_CASE("insert_more") {
    map_test_insert_more<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_insert_more<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
    set_test_insert_more<ankerl::unordered_dense::sharding_set<std::string>>();
    set_test_insert_more<ankerl::unordered_dense::segmented_sharding_set<std::string>>();
}


template<typename Map>
void map_test_insert_or_assign() {
    Map map;
    map.insert_or_assign("a", 1ULL);
    REQUIRE(map.size() == 1);
    CHECK_EQ(map["a"], 1ULL);
    map.insert_or_assign("b", 2ULL);
    REQUIRE(map.size() == 2);
    map.insert_or_assign("a", 4ULL);
    CHECK_EQ(map["a"], 4ULL);
    REQUIRE(map.size() == 2);
    std::string a_val{"3"};
    map.insert_or_assign(a_val, 10ULL);
    REQUIRE(map.size() == 3);
    CHECK_EQ(map["3"], 10ULL);
}

TEST_CASE("insert_or_assign") {
    map_test_insert_or_assign<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_insert_or_assign<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
}

template<typename Map>
void map_test_emplace() {
    Map map;
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
}

template<typename Set>
void set_test_emplace() {
    Set set;
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

TEST_CASE("emplace") {
    map_test_emplace<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_emplace<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
    set_test_emplace<ankerl::unordered_dense::sharding_set<std::string>>();
    set_test_emplace<ankerl::unordered_dense::segmented_sharding_set<std::string>>();
}

template<typename Map>
void map_test_try_emplace() {
    Map map;
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
}


TEST_CASE("try_emplace") {
    map_test_try_emplace<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_try_emplace<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
    // set do not support try_emplace
}

template<typename Map>
void map_test_at() {
    Map map;

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


TEST_CASE("at") {
    map_test_at<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_at<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
    // set do not support at
}


template<typename Map>
void map_test_operator_square_brackets() {
    Map map;
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

TEST_CASE("operator[]") {
    map_test_operator_square_brackets<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_operator_square_brackets<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
}

template<typename Map>
void map_test_count() {
    Map map;
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
}

template<typename Set>
void set_test_count() {
    Set set;
    set.insert("a");
    set.insert("b");
    set.insert("c");
    set.insert("d");
    set.insert("e");
    CHECK_EQ(set.count("a"), 1);
    CHECK_EQ(set.count("b"), 1);
    CHECK_EQ(set.count("c"), 1);
    CHECK_EQ(set.count("d"), 1);
    CHECK_EQ(set.count("e"), 1);
    CHECK_EQ(set.count("f"), 0);
    CHECK_EQ(set.count("g"), 0);
    CHECK_EQ(set.count("h"), 0);
    CHECK_EQ(set.count("i"), 0);
}

TEST_CASE("count") {
    map_test_count<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_count<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
    set_test_count<ankerl::unordered_dense::sharding_set<std::string>>();
    set_test_count<ankerl::unordered_dense::segmented_sharding_set<std::string>>();
}

template<typename Map>
void map_test_find() {
    Map map;
    map.insert({"a", 1});
    map.insert({"b", 2});
    map.insert({"c", 3});
    map.insert({"d", 4});
    map.insert({"e", 5});
    CHECK_EQ(map.find("a")->first, "a"  );
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

TEST_CASE("find") {
    map_test_find<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_find<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
}

template<typename Map>
void map_test_contains() {
    Map map;
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

template<typename Set>
void set_test_contains() {
    Set set;
    set.insert("a");
    set.insert("b");
    set.insert("c");
    set.insert("d");
    set.insert("e");
    CHECK(set.contains("a"));
    CHECK(set.contains("b"));
    CHECK(set.contains("c"));
    CHECK(set.contains("d"));
    CHECK(set.contains("e"));
    CHECK(not set.contains("f"));
    CHECK(not set.contains("g"));
    CHECK(not set.contains("h"));
    CHECK(not set.contains("i"));
    CHECK(not set.contains("j"));
}   

TEST_CASE("contains") {
    map_test_contains<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_contains<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
    set_test_contains<ankerl::unordered_dense::sharding_set<std::string>>();
    set_test_contains<ankerl::unordered_dense::segmented_sharding_set<std::string>>();
}

template<typename Map>
void map_test_equal_range() {
    Map map;
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

template<typename Set>
void set_test_equal_range() {
    Set set;
    set.insert("a");
    set.insert("b");
    set.insert("c");
    set.insert("d");
    set.insert("e");
    auto range = set.equal_range("a");
    REQUIRE(range.first != set.cend());
    REQUIRE(range.second != set.cend());

    auto range2 = set.equal_range("z");
    REQUIRE(range2.first == set.cend());
    REQUIRE(range2.second == set.cend());
}

TEST_CASE("equal_range") {
    map_test_equal_range<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_equal_range<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
    set_test_equal_range<ankerl::unordered_dense::sharding_set<std::string>>();
    set_test_equal_range<ankerl::unordered_dense::segmented_sharding_set<std::string>>();
}

template<typename Map>
void map_test_bucket_count() {
    Map map;
    map.insert({"a", 1});
    map.insert({"b", 2});
    map.insert({"c", 3});
    map.insert({"d", 4});
    map.insert({"e", 5});
    CHECK_EQ(map.bucket_count(), 32);
    for (uint32_t i = 0; i < 8UL; ++i) {
        CHECK_EQ(map.bucket_count(i), 4);
    }
}

template<typename Set>
void set_test_bucket_count() {
    Set set;
    set.insert("a");
    set.insert("b");
    set.insert("c");
    set.insert("d");
    set.insert("e");
    CHECK_EQ(set.bucket_count(), 32);
    for (uint32_t i = 0; i < 8UL; ++i) {
        CHECK_EQ(set.bucket_count(i), 4);
    }
}

TEST_CASE("bucket_count") {
    map_test_bucket_count<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_bucket_count<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
    set_test_bucket_count<ankerl::unordered_dense::sharding_set<std::string>>();
    set_test_bucket_count<ankerl::unordered_dense::segmented_sharding_set<std::string>>();
}   

template<typename Map>
void map_test_load_factor() {
    Map map;
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

template<typename Set>
void set_test_load_factor() {
    Set set;
    set.insert("a");
    set.insert("b");
    set.insert("c");
    set.insert("d");
    set.insert("e");
    set.insert("f");
    set.insert("g");
    set.insert("h");
    CHECK_EQ(set.load_factor(), 0.25);
}

TEST_CASE("load_factor") {
    map_test_load_factor<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_load_factor<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
    set_test_load_factor<ankerl::unordered_dense::sharding_set<std::string>>();
    set_test_load_factor<ankerl::unordered_dense::segmented_sharding_set<std::string>>();
}

template<typename Map>
void map_test_reserve() {
    Map map;
    map.reserve(128);
    CHECK_EQ(map.size(), 0);
    CHECK_EQ(map.bucket_count(), 256);
}

template<typename Set>
void set_test_reserve() {
    Set set;
    set.reserve(128);
    CHECK_EQ(set.size(), 0);
    CHECK_EQ(set.bucket_count(), 256);
}

TEST_CASE("reserve") {
    map_test_reserve<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_reserve<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
    set_test_reserve<ankerl::unordered_dense::sharding_set<std::string>>();
    set_test_reserve<ankerl::unordered_dense::segmented_sharding_set<std::string>>();
}

template<typename Map>
void map_test_erase() {
    Map map;
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

    map.erase("z");
    CHECK_EQ(map.size(), 2);
    CHECK_EQ(map.find("z"), map.cend());
}

template<typename Set>
void set_test_erase() {
    Set set;
    set.insert("a");
    set.insert("b");
    set.insert("c");
    set.insert("d");
    set.insert("e");

    set.erase(set.find("a"));
    CHECK_EQ(set.size(), 4);
    CHECK_EQ(set.find("a"), set.cend());
    
    set.erase("b");
    CHECK_EQ(set.size(), 3);
    CHECK_EQ(set.find("b"), set.cend());

    set.erase(set.find("c"));
    CHECK_EQ(set.size(), 2);
    CHECK_EQ(set.find("c"), set.cend());

    set.erase("z");
    CHECK_EQ(set.size(), 2);
    CHECK_EQ(set.find("z"), set.cend());
}

TEST_CASE("erase") {
    map_test_erase<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_erase<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
    set_test_erase<ankerl::unordered_dense::sharding_set<std::string>>();
    set_test_erase<ankerl::unordered_dense::segmented_sharding_set<std::string>>();
}

template<typename Map>
void map_test_erase_range() {
    Map map;
    map.insert({"a", 1});
    map.insert({"b", 2});
    map.insert({"c", 3});
    map.insert({"d", 4});
    map.insert({"e", 5});

    map.erase(map.cbegin(), map.cend());
    CHECK_EQ(map.size(), 0);
}

template<typename Set>
void set_test_erase_range() {
    Set set;
    set.insert("a");
    set.insert("b");
    set.insert("c");
    set.insert("d");
    set.insert("e");

    set.erase(set.cbegin(), set.cend());
    CHECK_EQ(set.size(), 0);
}

TEST_CASE("erase_range") {
    map_test_erase_range<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_erase_range<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
    set_test_erase_range<ankerl::unordered_dense::sharding_set<std::string>>();
    set_test_erase_range<ankerl::unordered_dense::segmented_sharding_set<std::string>>();
}

template<typename Map>
void map_test_clear() {
    Map map;
    map.insert({"a", 1});
    map.insert({"b", 2});
    map.insert({"c", 3});
    map.insert({"d", 4});
    map.insert({"e", 5});
    map.clear();
    CHECK_EQ(map.size(), 0);
}

template<typename Set>
void set_test_clear() {
    Set set;
    set.insert("a");
    set.insert("b");
    set.insert("c");
    set.insert("d");
    set.insert("e");
    set.clear();
    CHECK_EQ(set.size(), 0);
}

TEST_CASE("clear") {
    map_test_clear<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_clear<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
    set_test_clear<ankerl::unordered_dense::sharding_set<std::string>>();
    set_test_clear<ankerl::unordered_dense::segmented_sharding_set<std::string>>();
} 

template<typename Map>
void map_test_swap() {
    Map map1;
    map1.insert({"a", 1});
    map1.insert({"b", 2});
    map1.insert({"c", 3});
    
    Map map2;
    map2.insert({"f", 6});
    map2.insert({"g", 7});
    map2.insert({"h", 8});
    map2.insert({"i", 9});
    map2.insert({"j", 10});

    map1.swap(map2);
    CHECK_EQ(map1.size(), 5);
    CHECK_EQ(map2.size(), 3);
}

template<typename Set>
void set_test_swap() {
    Set set1;
    set1.insert("a");
    set1.insert("b");
    set1.insert("c");

    Set set2;
    set2.insert("f");
    set2.insert("g");
    set2.insert("h");
    set2.insert("i");
    set2.insert("j");

    set1.swap(set2);
    CHECK_EQ(set1.size(), 5);
    CHECK_EQ(set2.size(), 3);
}

TEST_CASE("swap") {
    map_test_swap<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_swap<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
    set_test_swap<ankerl::unordered_dense::sharding_set<std::string>>();
    set_test_swap<ankerl::unordered_dense::segmented_sharding_set<std::string>>();
}


template<typename Map>
void map_test_extract() {
    Map map;
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

template<typename Set>
void set_test_extract() {
    Set set;
    set.insert("a");
    set.insert("b");
    set.insert("c");
    set.insert("d");
    set.insert("e");

    auto it = set.find("e");
    auto node = set.extract(it);
    CHECK(node == "e");
    CHECK_EQ(set.size(), 4);
    CHECK_EQ(set.find("e"), set.cend());
}

TEST_CASE("extract") {
    map_test_extract<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_extract<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
    set_test_extract<ankerl::unordered_dense::sharding_set<std::string>>();
    set_test_extract<ankerl::unordered_dense::segmented_sharding_set<std::string>>();
}

template<typename Map>
void map_test_for_loop_all() {
    Map map;
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
    REQUIRE(map.size() == 19);
    for (auto& [key, value] : map) {
        CHECK_EQ(map[key], value);
    }
}

template<typename Set>
void set_test_for_loop_all() {
    Set set;
    set.insert("a");
    set.insert("b");
    set.insert("c");
    set.insert("d");
    set.insert("e");
    REQUIRE(set.size() == 5);
    for (auto it = set.begin(); it != set.end(); ++it) {
        CHECK(it != set.cend());
        CHECK_EQ(*it, *set.find(*it));
    }
}

TEST_CASE("for-loop-all") {
    map_test_for_loop_all<ankerl::unordered_dense::sharding_map<std::string, uint64_t>>();
    map_test_for_loop_all<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t>>();
    set_test_for_loop_all<ankerl::unordered_dense::sharding_set<std::string>>();
    set_test_for_loop_all<ankerl::unordered_dense::segmented_sharding_set<std::string>>();
}

struct vector_hash {
    auto operator()(const std::vector<char>& vec) const noexcept -> uint64_t {
        return ankerl::unordered_dense::detail::wyhash::hash(vec.data(), vec.size());
    }
};

TEST_CASE("vector as key") {
    ankerl::unordered_dense::sharding_map<std::vector<char>, uint64_t, 16, vector_hash> map;
    map.emplace({'a', 'b', 'c'}, 1);
    map.emplace({'d', 'e', 'f'}, 2);
    map.emplace({'g', 'h', 'i'}, 3);
    REQUIRE(map.size() == 3);

    for (auto& [key, value] : map) {
        CHECK_EQ(map[key], value);
        fmt::print("{}: {}\n", key.data(), value);
    }
}


template<typename Map>
void map_test_begin() {
    Map map;
    CHECK_EQ(map.begin(), map.end());
    map.emplace(std::string("x"), 1);
    const auto& internal_map = map.internals();
    CHECK(internal_map[0].empty());
    CHECK_NE(map.begin(), map.end());
    CHECK_EQ(map.begin()->first, "x");
    map.clear();
    CHECK_EQ(map.begin(), map.end());
    map.emplace(std::string("y"), 2);
    const auto& internal_map2 = map.internals();
    CHECK(internal_map2[0].empty());
    CHECK_NE(map.begin(), map.end());
    CHECK_EQ(map.begin()->first, "y");
    CHECK_EQ(map.begin()->second, 2);
}

TEST_CASE("begin") {
    map_test_begin<ankerl::unordered_dense::sharding_map<std::string, uint64_t, 256>>();
    map_test_begin<ankerl::unordered_dense::segmented_sharding_map<std::string, uint64_t, 256>>();
}
