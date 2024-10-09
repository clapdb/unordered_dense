
#pragma once

#include "ankerl/unordered_dense.h"
#include <array>
#include <algorithm>
#include <numeric>

namespace ankerl::unordered_dense {

namespace detail {

template <class Key,
          class T,
          class Hash,
          class keyEqual,
          class AllocatorOrContainer,
          class Bucket,
          class BucketContainer,
          bool IsSegmented,
          uint32_t Shards,
          typename Dispatcher,
          template <typename, typename> class ValueContainer = std::vector>
class horizontal_sharded_table {
    using value_container_type = ValueContainer<T, AllocatorOrContainer>;
    using internal_table = table<Key, T, Hash, keyEqual, AllocatorOrContainer, Bucket, BucketContainer, IsSegmented, ValueContainer>;
    // using iterator = typename internal_table::iterator;
    // using const_iterator = typename internal_table::const_iterator;
    using value_type = typename internal_table::value_type;
    /** 
     * wrap internal_table::iterator to add shard index
     * if meet the end of the shard, move to the next shard
     * if meet the end of the last shard, return end iterator
     */
    template<bool IsConst>
    class iterator {
        uint32_t _shard;
        using internal_iterator = std::conditional_t<IsConst, typename internal_table::const_iterator, typename internal_table::iterator>;
        internal_iterator _it;

        
        
    };
    class const_iterator : public internal_table::const_iterator {
        using internal_table::const_iterator::const_iterator;
    };

private:
    std::array<internal_table, Shards> _maps{};
    Dispatcher _dispatcher{};

private:
    struct dispatch_result {
        uint64_t hash;
        uint32_t shard;
        };
        auto dispatch(const Key& key) const -> dispatch_result {
            auto hash = internal_table::mixed_hash(key);
            return _dispatcher(hash);
        }
        
    public:
    [[nodiscard]] auto empty() const -> bool {
        return std::all_of(_maps.begin(), _maps.end(), [](const auto& map) { return map.empty(); });
    }
    [[nodiscard]] auto size() const -> size_t {
        return std::accumulate(_maps.begin(), _maps.end(), size_t{0}, [](size_t sum, const auto& map) { return sum + map.size(); });
    }
    [[nodiscard]] auto max_size() const -> size_t {
        return _maps[0].max_size();
    }

    auto clear() -> void {
        for (auto& map : _maps) {
            map.clear();
        }
    }

    auto insert(value_type const& value) -> std::pair<iterator, bool> {
        auto dispatch_result = dispatch(value);
        return _maps[dispatch_result.shard].emplace_with_hash(dispatch_result.hash, value);
    }

    auto insert(value_type&& value) -> std::pair<iterator, bool> {
        auto dispatch_result = dispatch(value);
        return _maps[dispatch_result.shard].emplace_with_hash(dispatch_result.hash, std::move(value));
    }

    template <class P, std::enable_if_t<std::is_constructible_v<value_type, P&&>, bool> = true>
    auto insert(P&& value) -> std::pair<iterator, bool> {
        auto real_value = value_type{std::forward<P>(value)};
        return insert(std::move(real_value));
    }

    auto insert(const_iterator /*hint*/, value_type const& value) -> iterator {
        auto dispatch_result = dispatch(value);
        return _maps[dispatch_result.shard].emplace_with_hash(dispatch_result.hash, value);
    }

    auto insert(const_iterator /*hint*/, value_type&& value) -> iterator {
        auto dispatch_result = dispatch(value);
        return _maps[dispatch_result.shard].emplace_with_hash(dispatch_result.hash, std::move(value));
    }

    template <class P, std::enable_if_t<std::is_constructible_v<value_type, P&&>, bool> = true>
    auto insert(const_iterator /*hint*/, P&& value) -> iterator {
        auto dispatch_result = dispatch(value);
        return _maps[dispatch_result.shard].emplace_with_hash(dispatch_result.hash, std::forward<P>(value));
    }

    template <class InputIt>
    void insert(InputIt first, InputIt last) {
        for (auto it = first; it != last; ++it) {
            insert(*it);
        }
    }

    void insert(std::initializer_list<value_type> ilist) {
        insert(ilist.begin(), ilist.end());
    }

    // nonstandard API: *this is emptied.
    // Also see "A Standard flat_map" https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p0429r9.pdf
    auto extract() && -> value_container_type {
        value_container_type values{};
        for (auto& map : _maps) {
            auto extracted = map.extract();
            values.insert(values.end(), extracted.begin(), extracted.end());
        }
        return values;
    }

    template <class M, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto insert_or_assign(Key const& key, M&& mapped) -> std::pair<iterator, bool> {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].insert_or_assign_with_hash(dispatch_result.hash, key, std::forward<M>(mapped));
    }

    template <class M, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto insert_or_assign(Key&& key, M&& mapped) -> std::pair<iterator, bool> {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].insert_or_assign_with_hash(dispatch_result.hash, std::move(key), std::forward<M>(mapped));
    }

    template<typename K,
             typename M,
             typename Q = T,
             typename H = Hash,
             typename KE = keyEqual,
             std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto insert_or_assign(K&& key, M&& mapped) -> std::pair<iterator, bool> {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].insert_or_assign_with_hash(dispatch_result.hash, std::forward<K>(key), std::forward<M>(mapped));
    }

    template<class M, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto insert_or_assign(const_iterator /*hint*/, Key const& key, M&& mapped) -> iterator {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].insert_or_assign_with_hash(dispatch_result.hash, key, std::forward<M>(mapped)).first;
    }

    template<class M, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto insert_or_assign(const_iterator /*hint*/, Key&& key, M&& mapped) -> iterator {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].insert_or_assign_with_hash(dispatch_result.hash, std::move(key), std::forward<M>(mapped)).first;
    }
    
    template<typename K,
             typename M,
             typename Q = T,
             typename H = Hash,
             typename KE = keyEqual,
             std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto insert_or_assign(const_iterator /*hint*/, K&& key, M&& mapped) -> iterator {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].insert_or_assign_with_hash(dispatch_result.hash, std::forward<K>(key), std::forward<M>(mapped)).first;
    }

    template <class K,
              typename Q = T,
              typename H = Hash,
              typename KE = keyEqual,
              std::enable_if_t<!is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto emplace(K&& key) -> std::pair<iterator, bool> {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].emplace_with_hash(dispatch_result.hash, std::forward<K>(key));
    }

    template<class... Args>
    auto emplace(Key&& key, Args&&... args) -> std::pair<iterator, bool> {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].emplace_with_hash(
            dispatch_result.hash, std::forward<Key>(key), std::forward<Args>(args)...);
    }

    template<class... Args>
    auto emplace_hint(const_iterator /*hint*/, Key&& key, Args&&... args) -> iterator {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].emplace_with_hash(
            dispatch_result.hash, std::forward<Key>(key), std::forward<Args>(args)...);
    }

    template<class... Args, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto try_emplace(Key const& key, Args&&... args) -> std::pair<iterator, bool> {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].do_try_emplace_with_hash(
            dispatch_result.hash, key, std::forward<Args>(args)...);
    }

    template<class... Args, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto try_emplace(Key&& key, Args&&... args) -> std::pair<iterator, bool> {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].do_try_emplace_with_hash(
            dispatch_result.hash, std::forward<Key>(key), std::forward<Args>(args)...);
    }

    template<class... Args, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto try_emplace(const_iterator /*hint*/, Key const& key, Args&&... args) -> iterator {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].do_try_emplace_with_hash(
            dispatch_result.hash, key, std::forward<Args>(args)...);
    }

    template<class... Args, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto try_emplace(const_iterator /*hint*/, Key&& key, Args&&... args) -> iterator {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].do_try_emplace_with_hash(
            dispatch_result.hash, std::move<Key>(key), std::forward<Args>(args)...);
    }

    template <
        typename K,
        typename... Args,
        typename Q = T,
        typename H = Hash,
        typename KE = keyEqual,
        std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE> && is_neither_convertible_v<K&&, iterator, const_iterator>,
                         bool> = true>
    auto try_emplace(K&& key, Args&&... args) -> std::pair<iterator, bool> {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].do_try_emplace_with_hash(
            dispatch_result.hash, std::forward<K>(key), std::forward<Args>(args)...);
    }

    template <
        typename K,
        typename... Args,
        typename Q = T,
        typename H = Hash,
        typename KE = keyEqual,
        std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE> && is_neither_convertible_v<K&&, iterator, const_iterator>,
                         bool> = true>
    auto try_emplace(const_iterator /*hint*/, K&& key, Args&&... args) -> iterator {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].do_try_emplace_with_hash(
            dispatch_result.hash, std::forward<K>(key), std::forward<Args>(args)...);
    }

            

    

}; // class horizontal_sharded_table

}  // namespace detail

ANKERL_UNORDERED_DENSE_EXPORT template <class Key,
                                        class T,
                                        class Hash = hash<Key>,
                                        class KeyEqual = std::equal_to<Key>,
                                        class AllocatorOrContainer = std::allocator<std::pair<Key, T>>,
                                        class Bucket = bucket_type::standard,
                                        class BucketContainer = detail::default_container_t>
using sharding_map = detail::horizontal_sharded_table<Key, T, Hash, KeyEqual, AllocatorOrContainer, Bucket, BucketContainer, false, 8, detail::shard_dispatcher<8>>;


ANKERL_UNORDERED_DENSE_EXPORT template <class Key,
                                        class T = void,
                                        class Hash = hash<Key>,
                                        class KeyEqual = std::equal_to<Key>,
                                        class AllocatorOrContainer = std::allocator<Key>,
                                        class Bucket = bucket_type::standard,
                                        class BucketContainer = detail::default_container_t>
using sharding_set = detail::horizontal_sharded_table<Key, T, Hash, KeyEqual, AllocatorOrContainer, Bucket, BucketContainer, false, 8, detail::shard_dispatcher<8>>;
}  // namespace ankerl::unordered_dense
