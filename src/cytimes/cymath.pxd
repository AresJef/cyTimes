# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Absolute
cdef int abs(int num) except -1
cdef long abs_l(long num) except -1
cdef long long abs_ll(long long num) except -1
cdef double abs_f(double num) except -1
cdef long double abs_lf(long double num) except -1
# Ceil
cdef long ceil(double num) noexcept
cdef long long ceil_l(long double num) noexcept
# Floor
cdef long floor(double num) noexcept
cdef long long floor_l(long double num) noexcept
# Round
cdef long round(double num) noexcept
cdef long long round_l(long double num) noexcept
# Maximum
cdef double max_f(double num1, double num2) noexcept
cdef long double max_lf(long double num1, long double num2) noexcept
# Minimum
cdef double min_f(double num1, double num2) noexcept
cdef long double min_lf(long double num1, long double num2) noexcept
# Clipping
cdef long long clip(long long num, long long min_val, long long max_val) noexcept
cdef double clip_f(double num, double min_val, double max_val) noexcept
cdef long double clip_lf(long double num, long double min_val, long double max_val) noexcept
# Sign
cdef double copysign(double num, double sign) noexcept
cdef long double copysign_l(long double num, long double sign) noexcept
cdef int signfactor(double num) noexcept
cdef int signfactor_l(long double num) noexcept
# Validation
cdef bint is_inf(long double num) except -1
cdef bint is_finite(long double num) except -1
cdef bint is_nan(long double num) except -1
cdef bint is_normal(long double num) except -1
