NAMESPACE_BEGIN(NB_NAMESPACE)

enum class rv_policy {
    automatic,
    automatic_reference,
    take_ownership,
    copy,
    move,
    reference,
    reference_internal,
    none
    /* Note to self: nb_func.h assumes that this value fits into 3 bits,
       hence no further policies can be added. */
};

NAMESPACE_END(NB_NAMESPACE)
