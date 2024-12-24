NAMESPACE_BEGIN(NB_NAMESPACE)
using ssize_t = std::make_signed_t<size_t>;

NAMESPACE_BEGIN(detail)

struct void_type { };

template <bool... Bs> struct index_1;
template <bool... Bs> struct index_n;

template <> struct index_1<> { constexpr static size_t value = 0; };
template <> struct index_n<> { constexpr static size_t value = 0; };

template <bool B, bool... Bs> struct index_1<B, Bs...> {
    constexpr static size_t value_rec = index_1<Bs...>::value;
    constexpr static size_t value = B ? 0 : (value_rec + 1);
};

template <bool B, bool... Bs> struct index_n<B, Bs...> {
    constexpr static size_t value_rec = index_n<Bs...>::value;
    constexpr static size_t value = (value_rec < sizeof...(Bs) || !B) ? (value_rec + 1) : 0;
};

template <bool... Bs> constexpr size_t index_1_v = index_1<Bs...>::value;
template <bool... Bs> constexpr size_t index_n_v = index_n<Bs...>::value;

template <typename T> struct intrinsic_type                       { using type = T; };
template <typename T> struct intrinsic_type<const T>              { using type = typename intrinsic_type<T>::type; };
template <typename T> struct intrinsic_type<T*>                   { using type = typename intrinsic_type<T>::type; };
template <typename T> struct intrinsic_type<T&>                   { using type = typename intrinsic_type<T>::type; };
template <typename T> struct intrinsic_type<T&&>                  { using type = typename intrinsic_type<T>::type; };
template <typename T, size_t N> struct intrinsic_type<const T[N]> { using type = typename intrinsic_type<T>::type; };
template <typename T, size_t N> struct intrinsic_type<T[N]>       { using type = typename intrinsic_type<T>::type; };
template <typename T> using intrinsic_t = typename intrinsic_type<T>::type;

template <typename T>
constexpr bool is_pointer_v = std::is_pointer_v<std::remove_reference_t<T>>;

template <typename T, typename U>
using forwarded_type = std::conditional_t<std::is_lvalue_reference_v<T>,
                                          std::remove_reference_t<U> &,
                                          std::remove_reference_t<U> &&>;

template <typename T, typename U> NB_INLINE forwarded_type<T, U> forward_like_(U &&u) {
    return (forwarded_type<T, U>) u;
}

template <typename T>
constexpr bool is_std_char_v =
    std::is_same_v<T, char>
#if defined(NB_HAS_U8STRING)
    || std::is_same_v<T, char8_t> /* std::u8string */
#endif
    || std::is_same_v<T, char16_t> ||
    std::is_same_v<T, char32_t> || std::is_same_v<T, wchar_t>;

template <bool V> using enable_if_t = std::enable_if_t<V, int>;

template <typename T>
constexpr bool is_lambda_v = !std::is_function_v<T> && !std::is_pointer_v<T> &&
                             !std::is_member_pointer_v<T>;

template <typename T> struct analyze_method { };
template <typename Cls, typename Ret, typename... Args>
struct analyze_method<Ret (Cls::*)(Args...)> {
    using func = Ret(Args...);
    static constexpr size_t argc = sizeof...(Args);
};

template <typename Cls, typename Ret, typename... Args>
struct analyze_method<Ret (Cls::*)(Args...) noexcept> {
    using func = Ret(Args...);
    static constexpr size_t argc = sizeof...(Args);
};

template <typename Cls, typename Ret, typename... Args>
struct analyze_method<Ret (Cls::*)(Args...) const> {
    using func = Ret(Args...);
    static constexpr size_t argc = sizeof...(Args);
};

template <typename Cls, typename Ret, typename... Args>
struct analyze_method<Ret (Cls::*)(Args...) const noexcept> {
    using func = Ret(Args...);
    static constexpr size_t argc = sizeof...(Args);
};

template <typename F>
struct strip_function_object {
    using type = typename analyze_method<decltype(&F::operator())>::func;
};

template <typename Function, typename F = std::remove_reference_t<Function>>
using function_signature_t = std::conditional_t<
    std::is_function_v<F>, F,
    typename std::conditional_t<
        std::is_pointer_v<F> || std::is_member_pointer_v<F>,
        std::remove_pointer<F>,
        strip_function_object<F>>::type>;

template <typename T>
using forward_t = std::conditional_t<std::is_lvalue_reference_v<T>, T, T &&>;

template <typename...> inline constexpr bool false_v = false;

template <typename... Args> struct overload_cast_impl {
    template <typename Return>
    constexpr auto operator()(Return (*pf)(Args...)) const noexcept
                              -> decltype(pf) { return pf; }

    template <typename Return, typename Class>
    constexpr auto operator()(Return (Class::*pmf)(Args...), std::false_type = {}) const noexcept
                              -> decltype(pmf) { return pmf; }

    template <typename Return, typename Class>
    constexpr auto operator()(Return (Class::*pmf)(Args...) const, std::true_type) const noexcept
                              -> decltype(pmf) { return pmf; }
};

template <typename SFINAE, template <typename> typename Op, typename Arg>
struct detector : std::false_type { };

template <template <typename> typename Op, typename Arg>
struct detector<std::void_t<Op<Arg>>, Op, Arg>
    : std::true_type { };

/* This template is used for docstring generation and specialized in
   ``stl/{variant,optional.h}`` to strip away std::optional and
   ``std::variant<std::monostate>`` in top-level argument types and
   avoid redundancy when combined with nb::arg(...).none(). */
template <typename T> struct remove_opt_mono { using type = T; };

template <typename T>
auto has_shared_from_this_impl(T *ptr) ->
    decltype(ptr->weak_from_this().lock().get(), std::true_type{});
std::false_type has_shared_from_this_impl(...);

template <typename T>
constexpr bool has_shared_from_this_v = decltype(has_shared_from_this_impl((T *) nullptr))::value;

struct type_caster_base_tag {
    static constexpr bool IsClass = true;
};

template <typename Caster>
constexpr bool is_base_caster_v = std::is_base_of_v<type_caster_base_tag, Caster>;

template <typename T> using is_class_caster_test = std::enable_if_t<T::IsClass>;

template <typename Caster>
constexpr bool is_class_caster_v = detail::detector<void, is_class_caster_test, Caster>::value;

template<typename T, typename = int>
struct is_complex : std::false_type {};

template<typename T>
struct is_complex<T, enable_if_t<std::is_same_v<
                                     decltype(std::declval<T>().real()),
                                     typename T::value_type>
                              && std::is_same_v<
                                     decltype(std::declval<T>().imag()),
                                     typename T::value_type>
                              && (sizeof(T) ==
                                  2 * sizeof(typename T::value_type))>>
    : std::true_type {};

template<typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

NAMESPACE_END(detail)

template <typename... Args>
static constexpr detail::overload_cast_impl<Args...> overload_cast = {};
static constexpr auto const_ = std::true_type{};

template <template<typename> class Op, typename Arg>
constexpr bool is_detected_v = detail::detector<void, Op, Arg>::value;

template <typename T>
using remove_opt_mono_t = typename detail::remove_opt_mono<T>::type;

template <template <typename> typename Base, typename T>
std::true_type is_base_of_template(const Base<T>*);

template <template <typename> typename Base>
std::false_type is_base_of_template(...);

template <typename T, template <typename> typename Base>
constexpr bool is_base_of_template_v = decltype(is_base_of_template<Base>(std::declval<T *>()))::value;

NAMESPACE_END(NB_NAMESPACE)
