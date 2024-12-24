#ifndef TSL_ROBIN_MAP_H
#define TSL_ROBIN_MAP_H

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>

#include "robin_hash.h"

namespace tsl {

template <class Key, class T, class Hash = std::hash<Key>,
          class KeyEqual = std::equal_to<Key>,
          class Allocator = std::allocator<std::pair<Key, T>>,
          bool StoreHash = false,
          class GrowthPolicy = tsl::rh::power_of_two_growth_policy<2>>
class robin_map {
 private:
  template <typename U>
  using has_is_transparent = tsl::detail_robin_hash::has_is_transparent<U>;

  class KeySelect {
   public:
    using key_type = Key;

    const key_type& operator()(
        const std::pair<Key, T>& key_value) const noexcept {
      return key_value.first;
    }

    key_type& operator()(std::pair<Key, T>& key_value) noexcept {
      return key_value.first;
    }
  };

  class ValueSelect {
   public:
    using value_type = T;

    const value_type& operator()(
        const std::pair<Key, T>& key_value) const noexcept {
      return key_value.second;
    }

    value_type& operator()(std::pair<Key, T>& key_value) noexcept {
      return key_value.second;
    }
  };

  using ht = detail_robin_hash::robin_hash<std::pair<Key, T>, KeySelect,
                                           ValueSelect, Hash, KeyEqual,
                                           Allocator, StoreHash, GrowthPolicy>;

 public:
  using key_type = typename ht::key_type;
  using mapped_type = T;
  using value_type = typename ht::value_type;
  using size_type = typename ht::size_type;
  using difference_type = typename ht::difference_type;
  using hasher = typename ht::hasher;
  using key_equal = typename ht::key_equal;
  using allocator_type = typename ht::allocator_type;
  using reference = typename ht::reference;
  using const_reference = typename ht::const_reference;
  using pointer = typename ht::pointer;
  using const_pointer = typename ht::const_pointer;
  using iterator = typename ht::iterator;
  using const_iterator = typename ht::const_iterator;

 public:
  robin_map() : robin_map(ht::DEFAULT_INIT_BUCKETS_SIZE) {}

  explicit robin_map(size_type bucket_count, const Hash& hash = Hash(),
                     const KeyEqual& equal = KeyEqual(),
                     const Allocator& alloc = Allocator())
      : m_ht(bucket_count, hash, equal, alloc) {}

  robin_map(size_type bucket_count, const Allocator& alloc)
      : robin_map(bucket_count, Hash(), KeyEqual(), alloc) {}

  robin_map(size_type bucket_count, const Hash& hash, const Allocator& alloc)
      : robin_map(bucket_count, hash, KeyEqual(), alloc) {}

  explicit robin_map(const Allocator& alloc)
      : robin_map(ht::DEFAULT_INIT_BUCKETS_SIZE, alloc) {}

  template <class InputIt>
  robin_map(InputIt first, InputIt last,
            size_type bucket_count = ht::DEFAULT_INIT_BUCKETS_SIZE,
            const Hash& hash = Hash(), const KeyEqual& equal = KeyEqual(),
            const Allocator& alloc = Allocator())
      : robin_map(bucket_count, hash, equal, alloc) {
    insert(first, last);
  }

  template <class InputIt>
  robin_map(InputIt first, InputIt last, size_type bucket_count,
            const Allocator& alloc)
      : robin_map(first, last, bucket_count, Hash(), KeyEqual(), alloc) {}

  template <class InputIt>
  robin_map(InputIt first, InputIt last, size_type bucket_count,
            const Hash& hash, const Allocator& alloc)
      : robin_map(first, last, bucket_count, hash, KeyEqual(), alloc) {}

  robin_map(std::initializer_list<value_type> init,
            size_type bucket_count = ht::DEFAULT_INIT_BUCKETS_SIZE,
            const Hash& hash = Hash(), const KeyEqual& equal = KeyEqual(),
            const Allocator& alloc = Allocator())
      : robin_map(init.begin(), init.end(), bucket_count, hash, equal, alloc) {}

  robin_map(std::initializer_list<value_type> init, size_type bucket_count,
            const Allocator& alloc)
      : robin_map(init.begin(), init.end(), bucket_count, Hash(), KeyEqual(),
                  alloc) {}

  robin_map(std::initializer_list<value_type> init, size_type bucket_count,
            const Hash& hash, const Allocator& alloc)
      : robin_map(init.begin(), init.end(), bucket_count, hash, KeyEqual(),
                  alloc) {}

  robin_map& operator=(std::initializer_list<value_type> ilist) {
    m_ht.clear();

    m_ht.reserve(ilist.size());
    m_ht.insert(ilist.begin(), ilist.end());

    return *this;
  }

  allocator_type get_allocator() const { return m_ht.get_allocator(); }

  iterator begin() noexcept { return m_ht.begin(); }
  const_iterator begin() const noexcept { return m_ht.begin(); }
  const_iterator cbegin() const noexcept { return m_ht.cbegin(); }

  iterator end() noexcept { return m_ht.end(); }
  const_iterator end() const noexcept { return m_ht.end(); }
  const_iterator cend() const noexcept { return m_ht.cend(); }

  bool empty() const noexcept { return m_ht.empty(); }
  size_type size() const noexcept { return m_ht.size(); }
  size_type max_size() const noexcept { return m_ht.max_size(); }

  void clear() noexcept { m_ht.clear(); }

  std::pair<iterator, bool> insert(const value_type& value) {
    return m_ht.insert(value);
  }

  template <class P, typename std::enable_if<std::is_constructible<
                         value_type, P&&>::value>::type* = nullptr>
  std::pair<iterator, bool> insert(P&& value) {
    return m_ht.emplace(std::forward<P>(value));
  }

  std::pair<iterator, bool> insert(value_type&& value) {
    return m_ht.insert(std::move(value));
  }

  iterator insert(const_iterator hint, const value_type& value) {
    return m_ht.insert_hint(hint, value);
  }

  template <class P, typename std::enable_if<std::is_constructible<
                         value_type, P&&>::value>::type* = nullptr>
  iterator insert(const_iterator hint, P&& value) {
    return m_ht.emplace_hint(hint, std::forward<P>(value));
  }

  iterator insert(const_iterator hint, value_type&& value) {
    return m_ht.insert_hint(hint, std::move(value));
  }

  template <class InputIt>
  void insert(InputIt first, InputIt last) {
    m_ht.insert(first, last);
  }

  void insert(std::initializer_list<value_type> ilist) {
    m_ht.insert(ilist.begin(), ilist.end());
  }

  template <class M>
  std::pair<iterator, bool> insert_or_assign(const key_type& k, M&& obj) {
    return m_ht.insert_or_assign(k, std::forward<M>(obj));
  }

  template <class M>
  std::pair<iterator, bool> insert_or_assign(key_type&& k, M&& obj) {
    return m_ht.insert_or_assign(std::move(k), std::forward<M>(obj));
  }

  template <class M>
  iterator insert_or_assign(const_iterator hint, const key_type& k, M&& obj) {
    return m_ht.insert_or_assign(hint, k, std::forward<M>(obj));
  }

  template <class M>
  iterator insert_or_assign(const_iterator hint, key_type&& k, M&& obj) {
    return m_ht.insert_or_assign(hint, std::move(k), std::forward<M>(obj));
  }

  template <class... Args>
  std::pair<iterator, bool> emplace(Args&&... args) {
    return m_ht.emplace(std::forward<Args>(args)...);
  }

  template <class... Args>
  iterator emplace_hint(const_iterator hint, Args&&... args) {
    return m_ht.emplace_hint(hint, std::forward<Args>(args)...);
  }

  template <class... Args>
  std::pair<iterator, bool> try_emplace(const key_type& k, Args&&... args) {
    return m_ht.try_emplace(k, std::forward<Args>(args)...);
  }

  template <class... Args>
  std::pair<iterator, bool> try_emplace(key_type&& k, Args&&... args) {
    return m_ht.try_emplace(std::move(k), std::forward<Args>(args)...);
  }

  template <class... Args>
  iterator try_emplace(const_iterator hint, const key_type& k, Args&&... args) {
    return m_ht.try_emplace_hint(hint, k, std::forward<Args>(args)...);
  }

  template <class... Args>
  iterator try_emplace(const_iterator hint, key_type&& k, Args&&... args) {
    return m_ht.try_emplace_hint(hint, std::move(k),
                                 std::forward<Args>(args)...);
  }

  iterator erase(iterator pos) { return m_ht.erase(pos); }
  iterator erase(const_iterator pos) { return m_ht.erase(pos); }
  iterator erase(const_iterator first, const_iterator last) {
    return m_ht.erase(first, last);
  }
  size_type erase(const key_type& key) { return m_ht.erase(key); }

  void erase_fast(iterator pos) { return m_ht.erase_fast(pos); }

  size_type erase(const key_type& key, std::size_t precalculated_hash) {
    return m_ht.erase(key, precalculated_hash);
  }

  template <
      class K, class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  size_type erase(const K& key) {
    return m_ht.erase(key);
  }

  template <
      class K, class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  size_type erase(const K& key, std::size_t precalculated_hash) {
    return m_ht.erase(key, precalculated_hash);
  }

  void swap(robin_map& other) { other.m_ht.swap(m_ht); }

  T& at(const Key& key) { return m_ht.at(key); }

  T& at(const Key& key, std::size_t precalculated_hash) {
    return m_ht.at(key, precalculated_hash);
  }

  const T& at(const Key& key) const { return m_ht.at(key); }

  const T& at(const Key& key, std::size_t precalculated_hash) const {
    return m_ht.at(key, precalculated_hash);
  }

  template <
      class K, class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  T& at(const K& key) {
    return m_ht.at(key);
  }

  template <
      class K, class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  T& at(const K& key, std::size_t precalculated_hash) {
    return m_ht.at(key, precalculated_hash);
  }

  template <
      class K, class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  const T& at(const K& key) const {
    return m_ht.at(key);
  }

  template <
      class K, class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  const T& at(const K& key, std::size_t precalculated_hash) const {
    return m_ht.at(key, precalculated_hash);
  }

  T& operator[](const Key& key) { return m_ht[key]; }
  T& operator[](Key&& key) { return m_ht[std::move(key)]; }

  size_type count(const Key& key) const { return m_ht.count(key); }

  size_type count(const Key& key, std::size_t precalculated_hash) const {
    return m_ht.count(key, precalculated_hash);
  }

  template <
      class K, class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  size_type count(const K& key) const {
    return m_ht.count(key);
  }

  template <
      class K, class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  size_type count(const K& key, std::size_t precalculated_hash) const {
    return m_ht.count(key, precalculated_hash);
  }

  iterator find(const Key& key) { return m_ht.find(key); }

  iterator find(const Key& key, std::size_t precalculated_hash) {
    return m_ht.find(key, precalculated_hash);
  }

  const_iterator find(const Key& key) const { return m_ht.find(key); }

  const_iterator find(const Key& key, std::size_t precalculated_hash) const {
    return m_ht.find(key, precalculated_hash);
  }

  template <
      class K, class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  iterator find(const K& key) {
    return m_ht.find(key);
  }

  template <
      class K, class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  iterator find(const K& key, std::size_t precalculated_hash) {
    return m_ht.find(key, precalculated_hash);
  }

  template <
      class K, class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  const_iterator find(const K& key) const {
    return m_ht.find(key);
  }

  template <
      class K, class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  const_iterator find(const K& key, std::size_t precalculated_hash) const {
    return m_ht.find(key, precalculated_hash);
  }

  bool contains(const Key& key) const { return m_ht.contains(key); }

  bool contains(const Key& key, std::size_t precalculated_hash) const {
    return m_ht.contains(key, precalculated_hash);
  }

  template <
      class K, class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  bool contains(const K& key) const {
    return m_ht.contains(key);
  }

  template <
      class K, class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  bool contains(const K& key, std::size_t precalculated_hash) const {
    return m_ht.contains(key, precalculated_hash);
  }

  std::pair<iterator, iterator> equal_range(const Key& key) {
    return m_ht.equal_range(key);
  }

  std::pair<iterator, iterator> equal_range(const Key& key,
                                            std::size_t precalculated_hash) {
    return m_ht.equal_range(key, precalculated_hash);
  }

  std::pair<const_iterator, const_iterator> equal_range(const Key& key) const {
    return m_ht.equal_range(key);
  }

  std::pair<const_iterator, const_iterator> equal_range(
      const Key& key, std::size_t precalculated_hash) const {
    return m_ht.equal_range(key, precalculated_hash);
  }

  template <
      class K, class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  std::pair<iterator, iterator> equal_range(const K& key) {
    return m_ht.equal_range(key);
  }

  template <
      class K, class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  std::pair<iterator, iterator> equal_range(const K& key,
                                            std::size_t precalculated_hash) {
    return m_ht.equal_range(key, precalculated_hash);
  }

  template <
      class K, class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  std::pair<const_iterator, const_iterator> equal_range(const K& key) const {
    return m_ht.equal_range(key);
  }

  template <
      class K, class KE = KeyEqual,
      typename std::enable_if<has_is_transparent<KE>::value>::type* = nullptr>
  std::pair<const_iterator, const_iterator> equal_range(
      const K& key, std::size_t precalculated_hash) const {
    return m_ht.equal_range(key, precalculated_hash);
  }

  size_type bucket_count() const { return m_ht.bucket_count(); }
  size_type max_bucket_count() const { return m_ht.max_bucket_count(); }

  float load_factor() const { return m_ht.load_factor(); }

  float min_load_factor() const { return m_ht.min_load_factor(); }
  float max_load_factor() const { return m_ht.max_load_factor(); }

  void min_load_factor(float ml) { m_ht.min_load_factor(ml); }
  void max_load_factor(float ml) { m_ht.max_load_factor(ml); }

  void rehash(size_type count_) { m_ht.rehash(count_); }
  void reserve(size_type count_) { m_ht.reserve(count_); }

  hasher hash_function() const { return m_ht.hash_function(); }
  key_equal key_eq() const { return m_ht.key_eq(); }

  iterator mutable_iterator(const_iterator pos) {
    return m_ht.mutable_iterator(pos);
  }

  template <class Serializer>
  void serialize(Serializer& serializer) const {
    m_ht.serialize(serializer);
  }

  template <class Deserializer>
  static robin_map deserialize(Deserializer& deserializer,
                               bool hash_compatible = false) {
    robin_map map(0);
    map.m_ht.deserialize(deserializer, hash_compatible);

    return map;
  }

  friend bool operator==(const robin_map& lhs, const robin_map& rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }

    for (const auto& element_lhs : lhs) {
      const auto it_element_rhs = rhs.find(element_lhs.first);
      if (it_element_rhs == rhs.cend() ||
          element_lhs.second != it_element_rhs->second) {
        return false;
      }
    }

    return true;
  }

  friend bool operator!=(const robin_map& lhs, const robin_map& rhs) {
    return !operator==(lhs, rhs);
  }

  friend void swap(robin_map& lhs, robin_map& rhs) { lhs.swap(rhs); }

 private:
  ht m_ht;
};

template <class Key, class T, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>, class Allocator = std::allocator<std::pair<Key, T>>, bool StoreHash = false>
using robin_pg_map = robin_map<Key, T, Hash, KeyEqual, Allocator, StoreHash,
                               tsl::rh::prime_growth_policy>;

}

#endif
