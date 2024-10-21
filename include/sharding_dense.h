
#pragma once

#include "ankerl/unordered_dense.h"
#include <array>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <cassert>
#include <iostream>

namespace ankerl::unordered_dense {

namespace detail {

template <uint32_t Shards>
class shard_dispatcher {
public:
    // TODO(leo): implement a better hash dispatcher to make shards more balanced working with internal_table's hash policy
    // FIXME
    auto operator()(uint64_t hash) const -> uint32_t {
        static_assert(Shards > 0 && (Shards & (Shards - 1)) == 0, "Shards must be power of 2");
        return (hash >> 8UL) & (Shards - 1);
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
    using const_pointer = typename internal_table::const_pointer;

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
        using iter_reference = std::conditional_t<IsConst || not is_map_v<T>, const_reference, reference>;
        using iter_pointer = std::conditional_t<IsConst || not is_map_v<T>, const_pointer, pointer>;
        table_type * _table = nullptr;
        uint32_t _shard = 0;
        internal_iterator _it = {};
    private:
        [[nodiscard]] inline bool is_current_shard_end() const {
            return _it == _table->_maps[_shard].end();
        }
        [[nodiscard]] inline uint32_t next_available_shard() const {
            uint32_t next_shard = _shard + 1;
            // seek to the next non-empty shard, if no next available shard, return Shards
            while(next_shard < Shards && _table->_maps[next_shard].empty()) {
                ++next_shard;
            }
            return next_shard;
        }

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
        iteratorT(iteratorT&& other) noexcept : _table(other._table), _shard(other._shard), _it(other._it) { }
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
            // not check current is shard's end, just move
            ++_it;
            if (ANKERL_UNORDERED_DENSE_UNLIKELY(is_current_shard_end())) {
                auto next_shard = next_available_shard();
                if (next_shard == Shards) {
                    // out of range, just return the end iterator
                    _shard = Shards - 1;
                    _it = _table->_maps[_shard].end();
                } else {
                    _shard = next_shard;
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
        
        constexpr auto operator+(difference_type n) noexcept -> iteratorT {
            if (_table->_maps[_shard].end() - _it > n) {
                return iteratorT(_table, _shard, _it + n);
            }
            // else move to next shards
            n -= (_table->_maps[_shard].end() - _it);
            auto current_shard = _shard + 1;
            // for-loop current shard to last shard, to find the nth after element from current shard
            while (n >= 0 and current_shard < Shards) {
                auto shard_size = _table->_maps[current_shard].size();
                if (ANKERL_UNORDERED_DENSE_LIKELY(shard_size > 0)) {
                    if (n < static_cast<difference_type>(shard_size)) {
                        break;
                    }
                    n -= shard_size;
                }
                ++current_shard;
            }
            if (ANKERL_UNORDERED_DENSE_UNLIKELY(current_shard == Shards)) {
                // out of range, just return the end iterator
                return _table->end();
            }
            return iteratorT(_table, current_shard, _table->_maps[current_shard].begin() + n);
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

        constexpr auto operator*() noexcept -> iter_reference {
            return _it.operator*();
        }

        constexpr auto operator*() const noexcept -> const_reference {
            return _it.operator*();
        }

        constexpr auto operator->() noexcept -> iter_pointer {
            return _it.operator->();
        }

        constexpr auto operator->() const noexcept -> const_pointer {
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
            // return (_it == _table->_maps[_shard].end() && other._it == _table->_maps[other._shard].end());
            auto current_end = _table->_maps[_shard].end();
            bool this_is_end = (_it == current_end);
            bool other_is_end = (other._it == other._table->_maps[other._shard].end());
            return this_is_end && other_is_end;
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
        // find the first non-empty shard
        for (uint32_t i = 0; i < Shards; ++i) {
            if (not _maps[i].empty()) {
                return iterator(this, i, _maps[i].begin());
            }
        }
        return end();
    }

    auto begin() const -> const_iterator {
        // find the first non-empty shard
        for (uint32_t i = 0; i < Shards; ++i) {
            if (not _maps[i].empty()) {
                return const_iterator(this, i, _maps[i].begin());
            }
        }
        return end();
    }

    auto cbegin() const -> const_iterator {
        return begin();
    }

    auto end() -> iterator {
        return iterator(this, Shards - 1, _maps[Shards - 1].end());
    }

    auto end() const -> const_iterator {
        return const_iterator(this, Shards - 1, _maps[Shards - 1].end());
    }

    auto cend() const -> const_iterator {
        return const_iterator(this, Shards - 1, _maps[Shards - 1].end());
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
        auto [internal_iter, success] = _maps[dispatch_result.shard].do_insert_or_assign_with_hash(dispatch_result.hash, key, std::forward<M>(mapped));
        return {iterator(this, dispatch_result.shard, internal_iter), success};
    }

    template <class M, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto insert_or_assign(Key&& key, M&& mapped) -> std::pair<iterator, bool> {
        auto dispatch_result = dispatch(key);
        auto [internal_iter, success] = _maps[dispatch_result.shard].do_insert_or_assign_with_hash(dispatch_result.hash, std::forward<Key>(key), std::forward<M>(mapped));
        return {iterator(this, dispatch_result.shard, internal_iter), success};
    }

    template<typename K,
             typename M,
             typename Q = T,
             typename H = Hash,
             typename KE = keyEqual,
             std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto insert_or_assign(K&& key, M&& mapped) -> std::pair<iterator, bool> {
        auto dispatch_result = dispatch(key);
        auto [internal_iter, success] = _maps[dispatch_result.shard].do_insert_or_assign_with_hash(dispatch_result.hash, std::forward<K>(key), std::forward<M>(mapped));
        return {iterator(this, dispatch_result.shard, internal_iter), success};
    }

    template<class M, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto insert_or_assign(const_iterator /*hint*/, Key const& key, M&& mapped) -> iterator {
        auto dispatch_result = dispatch(key);
        return {
            this,
            dispatch_result.shard,
            _maps[dispatch_result.shard].do_insert_or_assign_with_hash(dispatch_result.hash, key, std::forward<M>(mapped)).first};
    }

    template<class M, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto insert_or_assign(const_iterator /*hint*/, Key&& key, M&& mapped) -> iterator {
        auto dispatch_result = dispatch(key);
        auto [internal_iter, success] = _maps[dispatch_result.shard].do_insert_or_assign_with_hash(
            dispatch_result.hash, std::move(key), std::forward<M>(mapped));
        return {this, dispatch_result.shard, internal_iter};
    }
    
    template<typename K,
             typename M,
             typename Q = T,
             typename H = Hash,
             typename KE = keyEqual,
             std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto insert_or_assign(const_iterator /*hint*/, K&& key, M&& mapped) -> iterator {
        auto dispatch_result = dispatch(key);
        auto [internal_iter, success] = _maps[dispatch_result.shard].do_insert_or_assign_with_hash(dispatch_result.hash, std::forward<K>(key), std::forward<M>(mapped));
        return {this, dispatch_result.shard, internal_iter};
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
        auto [internal_iter, success] = _maps[dispatch_result.shard].emplace_with_hash(
            dispatch_result.hash, std::forward<Key>(key), std::forward<Args>(args)...);
        return {iterator(this, dispatch_result.shard, internal_iter), success};
    }

    template<class... Args>
    auto emplace_hint(const_iterator /*hint*/, Key&& key, Args&&... args) -> iterator {
        auto dispatch_result = dispatch(key);
        auto [internal_iter, success] = _maps[dispatch_result.shard].emplace_with_hash(
            dispatch_result.hash, std::forward<Key>(key), std::forward<Args>(args)...);
        return {this, dispatch_result.shard, internal_iter};
    }

    template<class... Args, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto try_emplace(Key const& key, Args&&... args) -> std::pair<iterator, bool> {
        auto dispatch_result = dispatch(key);
        auto [internal_iter, success] = _maps[dispatch_result.shard].do_try_emplace_with_hash(
            dispatch_result.hash, key, std::forward<Args>(args)...);
        return {iterator(this, dispatch_result.shard, internal_iter), success};
    }

    template<class... Args, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto try_emplace(Key&& key, Args&&... args) -> std::pair<iterator, bool> {
        auto dispatch_result = dispatch(key);
        auto [internal_iter, success] = _maps[dispatch_result.shard].do_try_emplace_with_hash(
            dispatch_result.hash, std::forward<Key>(key), std::forward<Args>(args)...);
        return {iterator(this, dispatch_result.shard, internal_iter), success};
    }

    template<class... Args, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto try_emplace(const_iterator /*hint*/, Key const& key, Args&&... args) -> iterator {
        auto dispatch_result = dispatch(key);
        auto [internal_iter, success] = _maps[dispatch_result.shard].do_try_emplace_with_hash(
            dispatch_result.hash, key, std::forward<Args>(args)...);
        return {this, dispatch_result.shard, internal_iter};
    }

    template<class... Args, typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto try_emplace(const_iterator /*hint*/, Key&& key, Args&&... args) -> iterator {
        auto dispatch_result = dispatch(key);
        auto [internal_iter, success] = _maps[dispatch_result.shard].do_try_emplace_with_hash(
            dispatch_result.hash, std::forward<Key>(key), std::forward<Args>(args)...);
        return {this, dispatch_result.shard, internal_iter};
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
        auto [internal_iter, success] = _maps[dispatch_result.shard].do_try_emplace_with_hash(
            dispatch_result.hash, std::forward<K>(key), std::forward<Args>(args)...);
        return {iterator(this, dispatch_result.shard, internal_iter), success};
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
        auto [internal_iter, success] = _maps[dispatch_result.shard].do_try_emplace_with_hash(
            dispatch_result.hash, std::forward<K>(key), std::forward<Args>(args)...);
        return {this, dispatch_result.shard, internal_iter};
    }


    // erase a single iterator
    auto erase(iterator it) -> iterator {
        assert(it._table == this);
        return iterator(this, it._shard, _maps[it._shard].erase(it._it));
    }
            
    auto extract(iterator it) -> value_type {
        assert(it._table == this);
        return _maps[it._shard].extract(it._it);
    }

    // erase a range of elements
    auto erase(const_iterator first, const_iterator last) -> iterator {
        assert(first._table == this);
        assert(last._table == this);
        assert(last._shard >= first._shard);
        if (first._shard == last._shard) {
            return iterator(this, first._shard, _maps[first._shard].erase(first._it, last._it));
        }
        // erase last._shard
        auto ret = _maps[last._shard].erase(_maps[last._shard].begin(), last._it);
        // erase first._shard
        ret = _maps[first._shard].erase(first._it, _maps[first._shard].end());
        // clear the shards in between
        for (auto idx = first._shard + 1; idx < last._shard; ++idx) {
            _maps[idx].clear();
        }
        return iterator(this, first._shard, ret);
    }

    auto erase(const Key& key) -> size_t {
        auto dispatch_result = dispatch(key);
        // every low probability function, just dispatch, no need to avoid duplicated hashing call.
        return _maps[dispatch_result.shard].erase(key);
    }

    auto extract(const Key& key) -> std::optional<value_type> {
        auto dispatch_result = dispatch(key);
        // every low probability function, just dispatch, no need to avoid duplicated hashing call.
        return _maps[dispatch_result.shard].extract(key);
    }

    template <class K,
              typename H = Hash,
              typename KE = keyEqual,
              std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto erase(K&& key) -> size_t {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].erase(std::forward<K>(key));
    }

    template <class K,
              typename H = Hash,
              typename KE = keyEqual,
              std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto extract(K&& key) -> std::optional<value_type> {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].extract(std::forward<K>(key));
    }

    void swap(horizontal_sharded_table& other) noexcept(noexcept(std::is_nothrow_swappable_v<value_container_type> &&
                                                      std::is_nothrow_swappable_v<Hash> && std::is_nothrow_swappable_v<keyEqual>)) {
        using std::swap;
        swap(other, *this);
    }   

    // lookup
    template <typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto at(const Key& key) -> Q& {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].do_at_with_hash(dispatch_result.hash, key);
    }

    template<typename K,
             typename Q = T,
             typename H = Hash,
             typename KE = keyEqual,
             std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto at(K const& key) -> Q& {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].do_at_with_hash(dispatch_result.hash, key);
    }

    template <typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto at(const Key& key) const -> Q const& {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].do_at_with_hash(dispatch_result.hash, key);
    }

    template<typename K,
             typename Q = T,
             typename H = Hash,
             typename KE = keyEqual,
             std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto at(K const& key) const -> Q const& {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].do_at_with_hash(dispatch_result.hash, key);
    }   

    template <typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto operator[](Key const& key) -> Q& {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].do_try_emplace_with_hash(dispatch_result.hash, key).first->second;
    }

    template <typename Q = T, std::enable_if_t<is_map_v<Q>, bool> = true>
    auto operator[](Key&& key) -> Q& {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].do_try_emplace_with_hash(dispatch_result.hash, std::move(key)).first->second;
    }

    template<typename K,
             typename Q = T,
             typename H = Hash,
             typename KE = keyEqual,
             std::enable_if_t<is_map_v<Q> && is_transparent_v<H, KE>, bool> = true>
    auto operator[](K&& key) -> Q& {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].do_try_emplace_with_hash(dispatch_result.hash, std::forward<K>(key)).first->second;
    }

    auto count(Key const& key) const -> size_t {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].do_find_with_hash(dispatch_result.hash, key) == _maps[dispatch_result.shard].end()
                   ? 0
                   : 1;
    }

    template <class K,
              typename H = Hash,
              typename KE = keyEqual,
              std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto count(K const& key) const -> size_t {
        auto dispatch_result = dispatch(key);
        return _maps[dispatch_result.shard].do_find_with_hash(dispatch_result.hash, key) == _maps[dispatch_result.shard].end()
                   ? 0
                   : 1;
    }

    auto find(Key const& key) -> iterator {
        auto dispatch_result = dispatch(key);
        return iterator(
            this, dispatch_result.shard, _maps[dispatch_result.shard].do_find_with_hash(dispatch_result.hash, key));
    }

    auto find(Key const& key) const -> const_iterator {
        auto dispatch_result = dispatch(key);
        return const_iterator(
            this, dispatch_result.shard, _maps[dispatch_result.shard].do_find_with_hash(dispatch_result.hash, key));
    }

    template <class K,
              typename H = Hash,
              typename KE = keyEqual,
              std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto find(K const& key) -> iterator {
        auto dispatch_result = dispatch(key);
        return iterator(this, dispatch_result.shard, _maps[dispatch_result.shard].do_find_with_hash(dispatch_result.hash, key));
    }

    template <class K,
              typename H = Hash,
              typename KE = keyEqual,
              std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto find(K const& key) const -> const_iterator {
        auto dispatch_result = dispatch(key);
        return const_iterator(this, dispatch_result.shard, _maps[dispatch_result.shard].do_find_with_hash(dispatch_result.hash, key));
    }

    auto contains(Key const& key) const -> bool {
        return find(key) != end();
    }

    template <class K,
              typename H = Hash,
              typename KE = keyEqual,
              std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto contains(K const& key) const -> bool {
        return find(key) != end();
    }

    auto equal_range(Key const& key) -> std::pair<iterator, iterator> {
        auto it = find(key);
        return {it, it == end() ? end() : it + 1};
    }

    auto equal_range(const Key& key) const -> std::pair<const_iterator, const_iterator> {
        auto it = find(key);
        return {it, it == end() ? end() : it + 1};
    }

    template <class K,
              typename H = Hash,
              typename KE = keyEqual,
              std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto equal_range(K const& key) -> std::pair<iterator, iterator> {
        auto it = find(key);
        return {it, it == end() ? end() : it + 1};
    }

    template <class K,
              typename H = Hash,
              typename KE = keyEqual,
              std::enable_if_t<is_transparent_v<H, KE>, bool> = true>
    auto equal_range(K const& key) const -> std::pair<const_iterator, const_iterator> {
        auto it = find(key);
        return {it, it == end() ? end() : it + 1};
    }


    // bucket interface ////////
    // because the bucket interface is from std::unordered_map
    auto bucket_count() const noexcept -> size_t {
        // get accumulated size of all shards
    return std::accumulate(_maps.begin(), _maps.end(), size_t(0),
                           [](size_t acc, const auto& map) { return acc + map.bucket_count(); });
    }

    auto bucket_count(uint32_t shard) const noexcept -> size_t {
        return _maps[shard].bucket_count();
    }

    static constexpr auto max_bucket_count() noexcept -> size_t {
        return internal_table::max_size();
    }

    // hash policy ////////
    [[nodiscard]] auto load_factor() const -> float {
        return bucket_count() ? static_cast<float>(size()) / static_cast<float>(bucket_count()) : 0.0F;
    }

    [[nodiscard]] auto load_factor(uint32_t shard) const -> float {
        return _maps[shard].load_factor();
    }

    [[nodiscard]] auto max_load_factor() const -> std::array<float, Shards> {
        // get max_load_factor from all shards
        std::array<float, Shards> ret;
        std::transform(_maps.begin(), _maps.end(), ret.begin(), [](const auto& map) {
            return map.max_load_factor();
        });
        return ret;
    }

    [[nodiscard]] auto max_load_factor(uint32_t shard) const -> float {
        return _maps[shard].max_load_factor();
    }

    void max_load_factor(uint32_t shard, float ml) {
        _maps[shard].max_load_factor(ml);
    }

    void rehash(uint32_t shard, size_t count) {
        _maps[shard].rehash(count);
    }

    void reserve(uint32_t size) {
        uint32_t size_per_shard = size / Shards;
        for (auto& map : _maps) {
            map.reserve(size_per_shard);
        }
    }

    // observers ////////
    auto hash_function() const -> Hash {
        return _maps[0].hash_function();
    }

    auto key_eq() const -> keyEqual {
        return _maps[0].key_eq();
    }
    
    // nonstandard API: expose the underlying values container
    [[nodiscard]] auto values(uint32_t shard) const noexcept -> value_container_type const& {
        return _maps[shard].values();
    }

    friend auto operator==(horizontal_sharded_table const& a, horizontal_sharded_table const& b) -> bool {
        for (uint32_t shard = 0; shard < Shards; ++shard) {
            if (a._maps[shard] != b._maps[shard]) {
                return false;
            }
        }
        return true;
    }

    friend auto operator!=(horizontal_sharded_table const& a, horizontal_sharded_table const& b) -> bool {
        return a._maps != b._maps;
    }
}; // class horizontal_sharded_table

}  // namespace detail

ANKERL_UNORDERED_DENSE_EXPORT template <class Key,
                                        class T,
                                        uint32_t Shard = 8,
                                        class Hash = hash<Key>,
                                        class KeyEqual = std::equal_to<Key>,
                                        class AllocatorOrContainer = std::allocator<std::pair<Key, T>>,
                                        class Bucket = bucket_type::standard,
                                        class BucketContainer = detail::default_container_t,
                                        template <typename, typename> class ValueContainer = std::vector>
using sharding_map = detail::horizontal_sharded_table<Key,
                                                      T,
                                                      Hash,
                                                      KeyEqual,
                                                      AllocatorOrContainer,
                                                      Bucket,
                                                      BucketContainer,
                                                      false,
                                                      Shard,
                                                      detail::shard_dispatcher<Shard>,
                                                      ValueContainer>;

ANKERL_UNORDERED_DENSE_EXPORT template <class Key,
                                        uint32_t Shard = 8,
                                        class Hash = hash<Key>,
                                        class KeyEqual = std::equal_to<Key>,
                                        class AllocatorOrContainer = std::allocator<Key>,
                                        class Bucket = bucket_type::standard,
                                        class BucketContainer = detail::default_container_t,
                                        template <typename, typename> class ValueContainer = std::vector>
using sharding_set = detail::horizontal_sharded_table<Key,
                                                      void,
                                                      Hash,
                                                      KeyEqual,
                                                      AllocatorOrContainer,
                                                      Bucket,
                                                      BucketContainer,
                                                      false,
                                                      Shard,
                                                      detail::shard_dispatcher<Shard>,
                                                      ValueContainer>;

template <class Key,
          class T,
          uint32_t Shard = 8,
          class Hash = hash<Key>,
          class KeyEqual = std::equal_to<Key>,
          class AllocatorOrContainer = std::allocator<std::pair<Key, T>>,
          class Bucket = bucket_type::standard,
          class BucketContainer = detail::default_container_t,
          template <typename, typename> class ValueContainer = std::vector>
using segmented_sharding_map = detail::horizontal_sharded_table<Key,
                                                              T,
                                                              Hash,
                                                              KeyEqual,
                                                              AllocatorOrContainer,
                                                              Bucket,
                                                              BucketContainer,
                                                              true,
                                                              Shard,
                                                              detail::shard_dispatcher<Shard>,
                                                              ValueContainer>;

template <class Key,
          uint32_t Shard = 8,
          class Hash = hash<Key>,
          class KeyEqual = std::equal_to<Key>,
          class AllocatorOrContainer = std::allocator<Key>,
          class Bucket = bucket_type::standard,
          class BucketContainer = detail::default_container_t,
          template <typename, typename> class ValueContainer = std::vector>
using segmented_sharding_set = detail::horizontal_sharded_table<Key,
                                                                void,
                                                                Hash,
                                                                KeyEqual,
                                                                AllocatorOrContainer,
                                                                Bucket,
                                                                BucketContainer,
                                                                true,
                                                                Shard,
                                                                detail::shard_dispatcher<Shard>,
                                                                ValueContainer>;

}  // namespace ankerl::unordered_dense
