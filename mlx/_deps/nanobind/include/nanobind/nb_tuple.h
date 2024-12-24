NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename... Ts> struct tuple;
template <> struct tuple<> {
    template <size_t> using type = void;
};

template <typename T, typename... Ts> struct tuple<T, Ts...> : tuple<Ts...> {
    using Base = tuple<Ts...>;

    tuple() = default;
    tuple(const tuple &) = default;
    tuple(tuple &&) = default;
    tuple& operator=(tuple &&) = default;
    tuple& operator=(const tuple &) = default;

    template <typename A, typename... As>
    NB_INLINE tuple(A &&a, As &&...as)
        : Base((detail::forward_t<As>) as...), value((detail::forward_t<A>) a) {}

    template <size_t I> NB_INLINE auto& get() {
        if constexpr (I == 0)
            return value;
        else
            return Base::template get<I - 1>();
    }

    template <size_t I> NB_INLINE const auto& get() const {
        if constexpr (I == 0)
            return value;
        else
            return Base::template get<I - 1>();
    }

    template <size_t I>
    using type = std::conditional_t<I == 0, T, typename Base::template type<I - 1>>;

private:
    T value;
};

template <typename... Ts> tuple(Ts &&...) -> tuple<std::decay_t<Ts>...>;

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

template <typename... Ts>
struct std::tuple_size<nanobind::detail::tuple<Ts...>>
    : std::integral_constant<size_t, sizeof...(Ts)> { };

template <size_t I, typename... Ts>
struct std::tuple_element<I, nanobind::detail::tuple<Ts...>> {
    using type = typename nanobind::detail::tuple<Ts...>::template type<I>;
};
