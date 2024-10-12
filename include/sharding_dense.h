
#pragma once

#include "ankerl/unordered_dense.h"
#include <array>
#include <algorithm>
#include <numeric>
#include <type_traits>

namespace ankerl::unordered_dense {

namespace detail {

template<uint32_t Shards>
class shard_dispatcher {
public:
    // TODO(leo): implement a better hash dispatcher to make shards more balanced working with internal_table's hash policy
    // FIXME
    auto operator()(uint64_t hash) const -> uint32_t {
        return hash % Shards;
    }
};

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
    using reference = typename internal_table::reference;
    using const_reference = typename internal_table::const_reference;
    using pointer = typename internal_table::pointer;

    /**
     * wrap internal_table::iterator to add shard index
     * if meet the end of the shard, move to the next shard
     * if meet the end of the last shard, return end iterator
     *
     * the iterator is wrapper, so we can not control the the internal_iterator's implementation.
     * because we need compare two iterators from different shards, so we can not use internal_iterator's comparation directly.
     *
     */
    template<bool IsConst>
    class iteratorT {
        friend class horizontal_sharded_table;
        using internal_iterator = std::conditional_t<IsConst, typename internal_table::const_iterator, typename internal_table::iterator>;
        using table_type = std::conditional_t<IsConst, const horizontal_sharded_table, horizontal_sharded_table>;
        using difference_type = typename internal_table::value_container_type::difference_type;
        table_type * _table = nullptr;
        uint32_t _shard = 0;
        internal_iterator _it = {};

    public:
        iteratorT(table_type * table, uint32_t shard, internal_iterator it) : _table(table), _shard(shard), _it(std::move(it)) {}
        // no default constructor
        iteratorT() = delete;
        // copy constructor
        iteratorT(const iteratorT& other) noexcept : _table(other._table), _shard(other._shard), _it(other._it) {}

        // copy a const iterator from iterator
        template <bool OtherIsConst, typename = typename std::enable_if_t<IsConst && !OtherIsConst>::type>
        iteratorT(const iteratorT<OtherIsConst>& other) noexcept
            : _table(other._table)
            , _shard(other._shard)
            , _it(other._it) {}

        // move constructor
        iteratorT(iteratorT&& other) noexcept : _shard(other._shard), _it(other._it) {}
        // copy assignment
        auto operator=(const iteratorT& other) -> iteratorT& {
            _table = other._table;
            _shard = other._shard;
            _it = other._it;
            return *this;
        }

        // assign a const iterator from a iterator
        template<bool OtherIsConst, typename = typename std::enable_if_t<IsConst && !OtherIsConst>::type>
        auto operator=(const iteratorT<OtherIsConst>& other) -> iteratorT& {
            _table = other._table;
            _shard = other._shard;
            _it = other._it;
            return *this;
        }

        constexpr auto operator++() noexcept -> iteratorT& {
            if (++_it == _table->_maps[_shard].end()) {
                ++_shard;
                if (_shard < Shards) {
                    _it = _table->_maps[_shard].begin();
                }
            }
            return *this;
        }

        constexpr auto operator++(int) noexcept -> iteratorT {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
        
        constexpr auto operator+(difference_type n) const noexcept -> iteratorT {
            difference_type distance_to_end_in_current_shard = _table->_maps[_shard].end() - _it;
            if (distance_to_end_in_current_shard > n) { // in the same shard
                return iteratorT(_table, _shard, _it + n);
            } else if (_shard == Shards - 1) { // already in the last shard
                // out of range or end, just return end
                return iteratorT(_table, _shard, std::move(_table->_maps[_shard].end()));
            } else { // in next shard
                // use recursive call to move to the next shard
                return iteratorT(_table, _shard + 1, _table->_maps[_shard + 1].begin()) + (n - distance_to_end_in_current_shard);
            }
        }

        constexpr auto operator-(difference_type n) const noexcept -> iteratorT {
            difference_type distance_to_begin_in_current_shard = _it - _table->_maps[_shard].begin();
            if (distance_to_begin_in_current_shard > n) {
                return iteratorT(_table, _shard, _it - n);
            } else if (_shard == 0) {
                // out of range or begin, just return begin
                return {_table, 0, _table->_map[0]->begin()};
            } else { // in previous shard
                // use recursive call to move to the previous shard
                return iterator(_table, _shard - 1, _table->_maps[_shard - 1].end()) - (n - distance_to_begin_in_current_shard);
            }
        }

        template<bool OtherIsConst>
        constexpr auto operator-(iteratorT<OtherIsConst> const& other) const noexcept -> difference_type {
            if (_shard == other._shard) {
                return _it - other._it;
            } else if (_shard > other._shard) {
                // other is in previous shard
                difference_type distance = other.end() - other._it;
                for (uint32_t i = other._shard + 1; i < _shard; ++i) {
                    distance += _table->_maps[i].size();
                }
                distance += _it - _table->_maps[_shard].begin();
                return distance;
            }
                // other is in next shard
            difference_type distance = _table->_maps[_shard].end() - _it;
            for (uint32_t i = _shard + 1; i < other._shard; ++i) {
                distance += _table->_maps[i].size();
            }
            distance += other._it - other._maps[other._shard].begin();
            return -distance;
        }

        constexpr auto operator*() const noexcept -> std::conditional_t<IsConst, const_reference, reference> {
            return *_it;
        }

        constexpr auto operator->() const noexcept -> pointer {
            return _it.operator->();
        }

        template<bool OtherIsConst>
        constexpr auto operator==(iteratorT<OtherIsConst> const& other) const noexcept -> bool {
            // no need to check shard, internal::iterator is a pointer?
            // the segment_vector's iterator is not a pointer, just compare the index, it sucks
            // compare same shard's iterator
            if (_shard == other._shard) {
                return _it == other._it;
            }
            // if both of them are end, they are equal, else they are not
            return (_it == _table->_maps[_shard].end() && other._it == _table->_maps[other._shard].end());
        }

        template<bool OtherIsConst>
        constexpr auto operator!=(iteratorT<OtherIsConst> const& other) const noexcept -> bool {
            return !(*this == other);
        }
    }; // class iteratorT
    using iterator = iteratorT<false>;
    using const_iterator = iteratorT<true>;

private:
    std::array<internal_table, Shards> _maps{};
    Dispatcher _dispatcher{};

private:
    struct dispatch_result_t {
        uint64_t hash;
        uint32_t shard;
    };

    auto dispatch(const Key& key) const -> dispatch_result_t {
        auto hash = _maps[0].mixed_hash(key);
        return {hash, _dispatcher(hash)};
    }

    public:
    // iterator member functions
    auto begin() -> iterator {
        return iterator(this, 0, _maps[0].begin());
    }

    auto begin() const -> const_iterator {
        return const_iterator(this, 0, _maps[0].begin());
    }

    auto cbegin() const -> const_iterator {
        return const_iterator(this, 0, _maps[0].begin());
    }

    auto end() -> iterator {
        return iterator(this, Shards, _maps[Shards - 1].end());
    }

    auto end() const -> const_iterator {
        return const_iterator(this, Shards, _maps[Shards - 1].end());
    }

    auto cend() const -> const_iterator {
        return const_iterator(this, Shards, _maps[Shards - 1].end());
    }

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
        if constexpr (is_map_v<T>) {
            auto dispatch_result = dispatch(value.first);
            return _maps[dispatch_result.shard].emplace_with_hash(dispatch_result.hash, value);
        }
        static_assert(std::is_constructible_v<Key, decltype(value)> && not is_map_v<T>);
        auto dispatch_result = dispatch(value);
        return _maps[dispatch_result.shard].emplace_with_hash(dispatch_result.hash, value);
    }

    auto insert(value_type&& value) -> std::pair<iterator, bool> {
        // check if value_pair is std::pair
        if constexpr (is_map_v<T>) {
            Key key{value.first};
            auto dispatch_result = dispatch(key);
            auto [internal_iter, success] = _maps[dispatch_result.shard].emplace_with_hash(dispatch_result.hash, std::move(value.first), std::move(value.second));
            return {iterator(this, dispatch_result.shard, internal_iter), success};
        } else {
            static_assert(std::is_constructible_v<Key, decltype(value)> && not is_map_v<T>);
            Key key{value};
            auto dispatch_result = dispatch(key);
            auto [internal_iter, success] =
                _maps[dispatch_result.shard].emplace_with_hash(dispatch_result.hash, std::move(value));
            return {iterator(this, dispatch_result.shard, internal_iter), success};
        }
    }

    template <class P, std::enable_if_t<std::is_constructible_v<value_type, P&&>, bool> = true>
    auto insert(P&& value) -> std::pair<iterator, bool> {
        auto real_value = value_type{std::forward<P>(value)};
        return insert(std::move(real_value));
    }

    auto insert(const_iterator /*hint*/, value_type const& value) -> iterator {
        return insert(value).first;
    }

    auto insert(const_iterator /*hint*/, value_type&& value) -> iterator {
        return insert(std::move(value)).first;
    }

    template <class P, std::enable_if_t<std::is_constructible_v<value_type, P&&>, bool> = true>
    auto insert(const_iterator /*hint*/, P&& value) -> iterator {
        auto real_value = value_type{std::forward<P>(value)};
        return insert(std::move(real_value)).first;
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
        return {
            this,
            dispatch_result.shard,
            _maps[dispatch_result.shard].insert_or_assign_with_hash(dispatch_result.hash, key, std::forward<M>(mapped)).first};
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
