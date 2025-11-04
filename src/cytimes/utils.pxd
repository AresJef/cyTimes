# cython: language_level=3

# Cython imports
cimport cython
cimport numpy as np
from libc.limits cimport LLONG_MIN
from libc cimport math, stdlib, time
from cpython cimport datetime
from cpython.time cimport time as unix_time
from cpython.unicode cimport (
    PyUnicode_AsUTF8,
    PyUnicode_DecodeUTF8,
    PyUnicode_ReadChar as str_read,
    PyUnicode_GET_LENGTH as str_len,
    PyUnicode_FromOrdinal as str_chr,
)

# Constants --------------------------------------------------------------------------------------------
cdef:
    # . calendar
    int DAYS_BR_MONTH[13]
    int DAYS_IN_MONTH[13]
    int DAYS_BR_QUARTER[5]
    int DAYS_IN_QUARTER[5]
    # . date
    int ORDINAL_MAX
    # . datetime
    #: EPOCH (1970-01-01)
    datetime.datetime EPOCH_DT
    long long EPOCH_YEAR
    long long EPOCH_MONTH
    long long EPOCH_DAY
    long long EPOCH_HOUR
    long long EPOCH_MINUTE
    long long EPOCH_SECOND
    long long EPOCH_MILLISECOND
    long long EPOCH_MICROSECOND
    #: EPOCH pre-compute
    long long EPOCH_C4
    long long EPOCH_C100
    long long EPOCH_C400
    long long EPOCH_CBASE
    # . timezone
    datetime.tzinfo UTC
    int NULL_TZOFFSET
    object _LOCAL_TIMEZONE
    set _UTC_ALIASES
    # . conversion for seconds
    long long SS_MINUTE
    long long SS_HOUR
    long long SS_DAY
    # . conversion for milliseconds
    long long MS_SECOND
    long long MS_MINUTE
    long long MS_HOUR
    long long MS_DAY
    # . conversion for microseconds
    long long US_MILLISECOND
    long long US_SECOND
    long long US_MINUTE
    long long US_HOUR
    long long US_DAY
    # . conversion for nanoseconds
    long long NS_MICROSECOND
    long long NS_MILLISECOND
    long long NS_SECOND
    long long NS_MINUTE
    long long NS_HOUR
    long long NS_DAY
    # . conversion for timedelta64
    double TD64_YY_DAY
    long long TD64_YY_SECOND
    long long TD64_YY_MILLISECOND
    long long TD64_YY_MICROSECOND
    long long TD64_YY_NANOSECOND
    double TD64_MM_DAY
    long long TD64_MM_SECOND
    long long TD64_MM_MILLISECOND
    long long TD64_MM_MICROSECOND
    long long TD64_MM_NANOSECOND
    # . datetime64 range
    #: Minimum datetime64 in nanoseconds (1677-09-21 00:12:43.145224193)
    long long DT64_NS_YY_MIN
    long long DT64_NS_MM_MIN
    long long DT64_NS_WW_MIN
    long long DT64_NS_DD_MIN
    long long DT64_NS_HH_MIN
    long long DT64_NS_MI_MIN
    long long DT64_NS_SS_MIN
    long long DT64_NS_MS_MIN
    long long DT64_NS_US_MIN
    long long DT64_NS_NS_MIN
    #: Maximum datetime64 in nanoseconds (2262-04-11 23:47:16.854775807)
    long long DT64_NS_YY_MAX
    long long DT64_NS_MM_MAX
    long long DT64_NS_WW_MAX
    long long DT64_NS_DD_MAX
    long long DT64_NS_HH_MAX
    long long DT64_NS_MI_MAX
    long long DT64_NS_SS_MAX
    long long DT64_NS_MS_MAX
    long long DT64_NS_US_MAX
    long long DT64_NS_NS_MAX
    # . datetime64 dtype
    object DT64_DTYPE_YY
    object DT64_DTYPE_MM
    object DT64_DTYPE_WW
    object DT64_DTYPE_DD
    object DT64_DTYPE_HH
    object DT64_DTYPE_MI
    object DT64_DTYPE_SS
    object DT64_DTYPE_MS
    object DT64_DTYPE_US
    object DT64_DTYPE_NS
    object DT64_DTYPE_PS
    object DT64_DTYPE_FS
    object DT64_DTYPE_AS
    # . numpy datetime units
    int DT_NPY_UNIT_YY
    int DT_NPY_UNIT_MM
    int DT_NPY_UNIT_WW
    int DT_NPY_UNIT_DD
    int DT_NPY_UNIT_HH
    int DT_NPY_UNIT_MI
    int DT_NPY_UNIT_SS
    int DT_NPY_UNIT_MS
    int DT_NPY_UNIT_US
    int DT_NPY_UNIT_NS
    int DT_NPY_UNIT_PS
    int DT_NPY_UNIT_FS
    int DT_NPY_UNIT_AS


# Struct -----------------------------------------------------------------------------------------------
ctypedef struct ymd:
    int year
    int month
    int day

ctypedef struct hmsf:
    int hour
    int minute
    int second
    int microsecond

ctypedef struct sf:
    int second
    int microsecond

ctypedef struct dtm:
    int year
    int month
    int day
    int hour
    int minute
    int second
    int microsecond

ctypedef struct iso:
    int year
    int week
    int weekday

cdef extern from "<time.h>" nogil:
    cdef struct tm:
        int  tm_sec
        int  tm_min
        int  tm_hour
        int  tm_mday
        int  tm_mon
        int  tm_year
        int  tm_wday
        int  tm_yday
        int  tm_isdst

# Math -------------------------------------------------------------------------------------------------
cdef inline long long math_mod(long long num, long long div, long long offset=0) except *:
    """Compute module with Python semantics `<'int'>`.

    :param num `<'int'>`: Dividend.
    :param div `<'int'>`: Divisor (non-zero).
    :param offset `<'int'>`: Optional value to add the result. Defaults to `0`
    :returns `<'int'>`: The modulo result.
    :raises `<'ZeroDivisionError'>`: When `div` is zero.
    
    ## Equivalent
    >>> (num % div) + offset
    """
    if div == 0:
        raise ZeroDivisionError("math_mod: division by zero.")
    if div == 1 or div == -1:
        return offset

    cdef long long r
    with cython.cdivision(True):
        r = num % div
    if r == 0:
        return offset
    if (r ^ div) < 0:  # different signs → adjust
        r += div
    return r + offset

cdef inline long long math_div_even(long long num, long long div, long long offset=0) except *:
    """Divide then round to nearest, ties-to-even (bankers' rounding) `<'int'>`.

    :param num `<'int'>`: Dividend.
    :param div `<'int'>`: Divisor (non-zero).
    :param offset `<'int'>`: Optional value to add the result. Defaults to `0`
    :returns `<'int'>`: The division result.
    :raises `<'ZeroDivisionError'>`: When `div` is zero.
    :raises `<'OverflowError'>`: When the result does not fit in signed 64-bit.

    ## Equivalent
    >>> round(num / div, 0) + offset  
    """
    if div == 0:
        raise ZeroDivisionError("math_div_even: division by zero.")
    if div == -1 and num == LLONG_MIN:
        raise OverflowError("math_div_even: result does not fit in signed 64-bit")

    cdef long long q, r
    with cython.cdivision(True):
        q = num / div; r = num % div
    if r == 0:
        return q + offset          # exact division

    cdef: 
        long long abs_f = -div if div < 0 else div
        long long abs_r = -r if r < 0 else r
        long long half  = abs_f - abs_r
    if abs_r > half or (abs_r == half and (q & 1) != 0):
        if (num ^ div) < 0:  
            q -= 1  # different signs → bump down 
        else:
            q += 1  # same sign → bump up
    return q + offset

cdef inline long long math_div_up(long long num, long long div, long long offset=0) except *:
    """Divide then round half away from zero (round-half-up) `<'int'>`.

    :param num `<'int'>`: Dividend.
    :param div `<'int'>`: Divisor (non-zero).
    :param offset `<'int'>`: Optional value to add the result. Defaults to `0`
    :returns `<'int'>`: The division result.
    :raises `<'ZeroDivisionError'>`: When `div` is zero.
    :raises `<'OverflowError'>`: When the result does not fit in signed 64-bit.

    ## Equivalent
    >>> (Decimal(num) / Decimal(div)).to_integral_value(rounding=ROUND_HALF_UP) + offset
    """
    if div == 0:
        raise ZeroDivisionError("math_div_up: division by zero.")
    if div == -1 and num == LLONG_MIN:
        raise OverflowError("math_div_up: result does not fit in signed 64-bit")

    cdef long long q, r
    with cython.cdivision(True):
        q = num / div; r = num % div
    if r == 0:
        return q + offset        # exact division

    cdef: 
        long long abs_f = -div if div < 0 else div
        long long abs_r = -r if r < 0 else r
        long long half  = abs_f - abs_r
    if abs_r >= half:
        if (num ^ div) < 0:  
            q -= 1  # different signs → bump down 
        else:
            q += 1  # same sign → bump up
    return q + offset

cdef inline long long math_div_down(long long num, long long div, long long offset=0) except *:
    """Divide then round half toward zero (round-half-down) `<'int'>`.

    :param num `<'int'>`: Dividend.
    :param div `<'int'>`: Divisor (non-zero).
    :param offset `<'int'>`: Optional value to add the result. Defaults to `0`
    :returns `<'int'>`: The division result.
    :raises `<'ZeroDivisionError'>`: When `div` is zero.
    :raises `<'OverflowError'>`: When the result does not fit in signed 64-bit.

    ## Equivalent
    >>> (Decimal(num) / Decimal(div)).to_integral_value(rounding=ROUND_HALF_DOWN) + offset
    """
    if div == 0:
        raise ZeroDivisionError("math_div_down: division by zero.")
    if div == -1 and num == LLONG_MIN:
        raise OverflowError("math_div_down: result does not fit in signed 64-bit")

    cdef long long q, r
    with cython.cdivision(True):
        q = num / div; r = num % div
    if r == 0:
        return q + offset        # exact division

    cdef:
        long long abs_f = -div if div < 0 else div
        long long abs_r = -r if r < 0 else r
        long long half  = abs_f - abs_r
    if abs_r > half:
        if (num ^ div) < 0:
            q -= 1  # different signs → bump down
        else:
            q += 1  # same sign → bump up
    return q + offset

cdef inline long long math_div_ceil(long long num, long long div, long long offset=0) except *:
    """Divide then take the mathematical ceiling `<'int'>`.

    :param num `<'int'>`: Dividend.
    :param div `<'int'>`: Divisor (non-zero).
    :param offset `<'int'>`: Optional value to add the result. Defaults to `0`
    :returns `<'int'>`: The division result.
    :raises `<'ZeroDivisionError'>`: When `div` is zero.
    :raises `<'OverflowError'>`: When the result does not fit in signed 64-bit.

    ## Equivalent
    >>> math.ceil(num / div) + offset
    """
    if div == 0:
        raise ZeroDivisionError("math_div_ceil: division by zero.")
    if div == -1 and num == LLONG_MIN:
        raise OverflowError("math_div_ceil: result does not fit in signed 64-bit")

    cdef long long q, r
    with cython.cdivision(True):
        q = num / div; r = num % div
    if r == 0:
        return q + offset         # exact division
        
    if (num ^ div) >= 0:  # same sign → bump up
        q += 1
    return q + offset

cdef inline long long math_div_floor(long long num, long long div, long long offset=0) except *:
    """Divide then take the mathematical floor `<'int'>`.

    :param num `<'int'>`: Dividend.
    :param div `<'int'>`: Divisor (non-zero).
    :param offset `<'int'>`: Optional value to add the result. Defaults to `0`
    :returns `<'int'>`: The division result.
    :raises `<'ZeroDivisionError'>`: When `div` is zero.
    :raises `<'OverflowError'>`: When the result does not fit in signed 64-bit.

    ## Equivalent
    >>> math.floor(num / div) + offset
    """
    if div == 0:
        raise ZeroDivisionError("math_div_floor: division by zero.")
    if div == -1 and num == LLONG_MIN:
        raise OverflowError("math_div_floor: result does not fit in signed 64-bit")

    cdef long long q, r
    with cython.cdivision(True):
        q = num / div; r = num % div
    if r == 0:
        return q + offset         # exact division

    if (num ^ div) < 0:        # different sign → bump down
        q -= 1
    return q + offset

cdef inline long long math_div_trunc(long long num, long long div, long long offset=0) except *:
    """Divide and truncate toward zero (C/CPython style) `<'int'>`.

    :param num `<'int'>`: Dividend.
    :param div `<'int'>`: Divisor (non-zero).
    :param offset `<'int'>`: Optional value to add the result. Defaults to `0`
    :returns `<'int'>`: The division result.
    :raises `<'ZeroDivisionError'>`: When `div` is zero.
    :raises `<'OverflowError'>`: When the result does not fit in signed 64-bit.
    """
    if div == 0:
        raise ZeroDivisionError("math_div_trunc: division by zero.")
    if div == -1 and num == LLONG_MIN:
        raise OverflowError("math_div_trunc: result does not fit in signed 64-bit")

    cdef long long q
    with cython.cdivision(True):
        q = num / div          # truncates toward zero
    return q + offset

cdef inline unsigned long long abs_diff_ull(long long a, long long b) noexcept nogil:
    """Return |a - b| as uint64 using unsigned arithmetic `<'int'>`.

    Safe for all long long pairs (avoids signed overflow/UB near LLONG_MIN/LLONG_MAX).
    """
    cdef unsigned long long ua = <unsigned long long> a
    cdef unsigned long long ub = <unsigned long long> b
    return ua - ub if a >= b else ub - ua

# Parser -----------------------------------------------------------------------------------------------
# . check: character
cdef inline bint is_dot(Py_UCS4 ch) noexcept nogil:
    """Check whether `ch` is a dot `'.'` `<'bool'>`.
    
    - Dot: `'.'` (46)
    """
    return ch == 46

cdef inline bint is_comma(Py_UCS4 ch) noexcept nogil:
    """Check whether `ch` is a comma `','` `<'bool'>`.
    
    - Comma: `','` (44)
    """
    return ch == 44

cdef inline bint is_plus_sign(Py_UCS4 ch) noexcept nogil:
    """Check whether `ch` is a plus sign `'+'` `<'bool'>`.
    
    - Plus sign: `'+'` (43)
    """
    return ch == 43

cdef inline bint is_minus_sign(Py_UCS4 ch) noexcept nogil:
    """Check whether `ch` is a minus sign `'-'` `<'bool'>`.
    
    - Minus sign: `'-'` (45)
    """
    return ch == 45

cdef inline bint is_plus_or_minus_sign(Py_UCS4 ch) noexcept nogil:
    """Check whether `ch` is a plus or minus sign `'-'` `<'bool'>`.
    
    - Plus sign : `'+'` (43)
    - Minus sign: `'-'` (45)
    """
    return is_plus_sign(ch) or is_minus_sign(ch)

cdef inline bint is_iso_sep(Py_UCS4 ch) noexcept nogil:
    """Check whether `ch` is an ISO 8601 date-time separator `<'bool'>`.

    - Date / Time separators: `' '` (32) or `'T'` (case-insensitive: 84 & 116).
    """
    return ch == 32 or ch == 84 or ch == 116

cdef inline bint is_date_sep(Py_UCS4 ch) noexcept nogil:
    """Check whether `ch` is a date-field seperator `<'bool'>`.

    - Date-field separators: `'-'` (45), `'.'` (46) or `'/'` (47)
    """
    return 45 <= ch <= 47

cdef inline bint is_time_sep(Py_UCS4 ch) noexcept nogil:
    """Check whether `ch` is the time-field separator `<'bool'>`.

    - Time-field seperator: `':'` (58)
    """
    return ch == 58

cdef inline bint is_isoweek_sep(Py_UCS4 ch) noexcept nogil:
    """Check whether `ch` is the ISO week designator `<'bool'>`.

    - ISO week designator: `'W'` (case-insensitive: 87 & 119).
    """
    return ch == 87 or ch == 119

cdef inline bint is_ascii_ctl(Py_UCS4 ch) noexcept nogil:
    """Check whether `ch` is a control charactor `<'bool'>`.

    - ASCII control characters (0-31) and (127)
    """
    return ch <= 31 or ch == 127

cdef inline bint is_ascii_ctl_or_space(Py_UCS4 ch) noexcept nogil:
    """Check whether `ch` is a control or space charactor `<'bool'>`.
    
    - ASCII control characters (0-31) and (127)
    - ASCII space character: (32)
    """
    return ch <= 32 or ch == 127

cdef inline bint is_ascii_digit(Py_UCS4 ch) noexcept nogil:
    """Check whether `ch` is an ASCII digit `<'bool'>`.

    - ASSCI digits: `'0'` (48) ... `'9'` (57)
    """
    return 48 <= ch <= 57
    
cdef inline bint is_ascii_letter_upper(Py_UCS4 ch) noexcept nogil:
    """Check whether `ch` is an uppercase ASCII letter `<'bool'>`.

    - Uppercase ASCII letters: `'A'` (65) ... `'Z'` (90)
    """
    return 65 <= ch <= 90

cdef inline bint is_ascii_letter_lower(Py_UCS4 ch) noexcept nogil:
    """Check whether `ch` is a lowercase ASCII letter `<'bool'>`.

    - Lowercase ASCII letters: `'a'` (97) ... `'z'` (122)
    """
    return 97 <= ch <= 122

cdef inline bint is_ascii_letter(Py_UCS4 ch) noexcept nogil:
    """Check whether `ch` is an ASCII letter (case-insensitive) `<'bool'>`.

    - Uppercase ASCII letters: `'A'` (65) ... `'Z'` (90)
    - Lowercase ASCII letters: `'a'` (97) ... `'z'` (122)
    """
    return is_ascii_letter_lower(ch) or is_ascii_letter_upper(ch)

cdef inline bint is_alpha(Py_UCS4 ch) noexcept:
    """Check whether `ch` is an alphabetic character `<'bool'>`.

    - Uses Unicode definition of alphabetic characters.
    """
    return True if is_ascii_letter(ch) else ch.isalpha()

# . check: string
cdef inline bint is_str_dot(str token, Py_ssize_t token_len=-1) except -1:
    """Check whether `token` is a single-character dot `'.'` `<'bool'>`.

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length. 
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` is a single-character dot `'.'`; False otherwise.
    """
    if token is None:
        return False
    if token_len <= 0:
        token_len = str_len(token)
    if token_len != 1:
        return False
    return is_dot(str_read(token, 0))

cdef inline bint is_str_comma(str token, Py_ssize_t token_len=-1) except -1:
    """Check whether `token` is a single-character comma `','` `<'bool'>`.

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length. 
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` is a single-character comma `'.,`; False otherwise.
    """
    if token is None:
        return False
    if token_len <= 0:
        token_len = str_len(token)
    if token_len != 1:
        return False
    return is_comma(str_read(token, 0))

cdef inline bint is_str_iso_sep(str token, Py_ssize_t token_len=-1) except -1:
    """Check whether `token` is a single-character ISO 8601 date-time separator `<'bool'>`.

    - Date / Time separators: `' '` or `'T'` (case-insensitive).

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length. 
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` is a single-character ISO date-time separator; False otherwise.
    """
    if token is None:
        return False
    if token_len <= 0:
        token_len = str_len(token)
    if token_len != 1:
        return False
    return is_iso_sep(str_read(token, 0))

cdef inline bint is_str_date_sep(str token, Py_ssize_t token_len=-1) except -1:
    """Check whether `token` is a single-character date-field seperator `<'bool'>`.

    - Date-field separators: `'-'`, `'.'` or `'/'`

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length. 
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` is a single-character date-field separator; False otherwise.
    """
    if token is None:
        return False
    if token_len <= 0:
        token_len = str_len(token)
    if token_len != 1:
        return False
    return is_date_sep(str_read(token, 0))

cdef inline bint is_str_time_sep(str token, Py_ssize_t token_len=-1) except -1:
    """Check whether `token` is a single-character time-field seperator `<'bool'>`.

    - Time-field seperator: `':'`

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length. 
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` is a single-character time-field separator; False otherwise.
    """
    if token is None:
        return False
    if token_len <= 0:
        token_len = str_len(token)
    if token_len != 1:
        return False
    return is_time_sep(str_read(token, 0))

cdef inline bint is_str_isoweek_sep(str token, Py_ssize_t token_len=-1) except -1:
    """Check whether `token` is a single-character ISO week designator `<'bool'>`.

    - ISO week designator: `'W'` (case-insensitive).

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length. 
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` is a single-character ISO week designator; False otherwise.
    """
    if token is None:
        return False
    if token_len <= 0:
        token_len = str_len(token)
    if token_len != 1:
        return False
    return is_isoweek_sep(str_read(token, 0))

cdef inline bint is_str_ascii_ctl(str token, Py_ssize_t token_len=-1) except -1:
    """Check whether `token` is a single control charactor `<'bool'>`.

    - ASCII control characters (0-31) and (127)

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length. 
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` is a single control character; False otherwise.
    """
    if token is None:
        return False
    if token_len <= 0:
        token_len = str_len(token)
    if token_len != 1:
        return False
    return is_ascii_ctl(str_read(token, 0))

cdef inline bint is_str_ascii_ctl_or_space(str token, Py_ssize_t token_len=-1) except -1:
    """Check whether `token` is a single control or space charactor `<'bool'>`.

    - ASCII control characters (0-31) and (127)
    - ASCII space character: (32)

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length. 
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` is a single control or space character; False otherwise.
    """
    if token is None:
        return False
    if token_len <= 0:
        token_len = str_len(token)
    if token_len != 1:
        return False
    return is_ascii_ctl_or_space(str_read(token, 0))

cdef inline bint is_str_ascii_digits(str token, Py_ssize_t token_len=-1) except -1:
    """Check whether `token` only contains ASCII digits `<'bool'>`.

    - ASSCI digits: `'0'` ... `'9'`

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` only contains ASCII digits; False otherwise.
    """
    if token is None:
        return False
    if token_len <= 0:
        token_len = str_len(token)
    if token_len == 0:
        return False
    cdef Py_ssize_t i
    for i in range(token_len):
        if not is_ascii_digit(str_read(token, i)):
            return False
    return True

cdef inline bint is_str_ascii_letters_upper(str token, Py_ssize_t token_len=-1) except -1:
    """Check whether `token` only contains uppercase ASCII letters `<'bool'>`.

    - Uppercase ASCII letters: `'A'` ... `'Z'`

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` only contains uppercase ASCII letters; False otherwise.
    """
    if token is None:
        return False
    if token_len <= 0:
        token_len = str_len(token)
    if token_len == 0:
        return False
    cdef Py_ssize_t i
    for i in range(token_len):
        if not is_ascii_letter_upper(str_read(token, i)):
            return False
    return True

cdef inline bint is_str_ascii_letters_lower(str token, Py_ssize_t token_len=-1) except -1:
    """Check whether `token` only contains lowercase ASCII letters `<'bool'>`.

    - Lowercase ASCII letters: `'a'` ... `'z'`

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` only contains lowercase ASCII letters; False otherwise.
    """
    if token is None:
        return False
    if token_len <= 0:
        token_len = str_len(token)
    if token_len == 0:
        return False
    cdef Py_ssize_t i
    for i in range(token_len):
        if not is_ascii_letter_lower(str_read(token, i)):
            return False
    return True

cdef inline bint is_str_ascii_letters(str token, Py_ssize_t token_len=-1) except -1:
    """Check whether `token` only contains letters (case-insensitive) `<'bool'>`.

    - ASCII letters: `'a'` ... `'z'` and `'A'` ... `'Z'`

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` only contains letters (case-insensitive); False otherwise.
    """
    if token is None:
        return False
    if token_len <= 0:
        token_len = str_len(token)
    if token_len == 0:
        return False
    cdef Py_ssize_t i
    for i in range(token_len):
        if not is_ascii_letter(str_read(token, i)):
            return False
    return True

cdef inline bint is_str_alphas(str token, Py_ssize_t token_len=-1) except -1:
    """Check whether `token` only contains alphabetic characters `<'bool'>`.

    - Uses Unicode definition of alphabetic characters.

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` only contains alphabetic characters; False otherwise.
    """
    if token is None:
        return False
    if token_len <= 0:
        token_len = str_len(token)
    if token_len == 0:
        return False
    cdef Py_ssize_t i
    for i in range(token_len):
        if not is_alpha(str_read(token, i)):
            return False
    return True

# . parse
cdef inline int parse_numeric_kind(str token, Py_ssize_t token_len=-1) except -1:
    """Classify `token` string as certain numeric kind `<'int'>`.

    :param token `<'str'>`: The input token string to classify.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: Classification result:

        - `1` = integer (ASCII digits only)
        - `2` = decimal (ASCII digits with a single `'.'`; more than one `'.'` is invalid)
        - `0` = not numeric (any other case, including empty string, `'.'` alone and prefixing `'+/-'` signs)
    """
    # Validate
    if token is None:
        return 0
    if token_len <= 0:
        token_len = str_len(token)
    if token_len == 0:
        return 0

    cdef:
        bint has_num = False
        bint has_dot = False
        Py_ssize_t i
        Py_UCS4 ch

    for i in range(token_len):
        ch = str_read(token, i)
        if is_ascii_digit(ch):
            has_num = True
        elif ch == 46:  # '.'
            if has_dot:
                return 0
            has_dot = True
        else:
            return 0

    if not has_num:
        return 0
    return 2 if has_dot else 1

cdef inline int parse_isoyear(str token, Py_ssize_t pos, Py_ssize_t token_len=-1) except -2:
    """Parse ISO format year component (YYYY) from a string,
    returns `-1` for invalid ISO years `<'int'>`.

    This function extracts and parses the year component from an ISO date string.
    It reads four characters starting at the specified position and converts them
    into an integer representing the year. The function ensures that the parsed
    year is valid (i.e., between '0000' and '9999').

    :param token `<'str'>`: The input token string containing the ISO year to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO year.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO year [1..9999], or `-1` if invalid.
    """
    # Validate
    if token is None:
        return -1  # exit: invalid
    if token_len <= 0:
        token_len = str_len(token)
    if pos < 0 or token_len - pos < 4:
        return -1  # exit: invalid

    # Parse value
    cdef Py_UCS4 c0 = str_read(token, pos)
    if not is_ascii_digit(c0):
        return -1  # exit: invalid
    cdef Py_UCS4 c1 = str_read(token, pos + 1)
    if not is_ascii_digit(c1):
        return -1  # exit: invalid
    cdef Py_UCS4 c2 = str_read(token, pos + 2)
    if not is_ascii_digit(c2):
        return -1  # exit: invalid
    cdef Py_UCS4 c3 = str_read(token, pos + 3)
    if not is_ascii_digit(c3):
        return -1  # exit: invalid

    # Convert to integer
    cdef int out = (
        (ord(c0) - 48) * 1000 +
        (ord(c1) - 48) *  100 +
        (ord(c2) - 48) *   10 +
        (ord(c3) - 48)
    )
    return out if out > 0 else -1

cdef inline int parse_isomonth(str token, Py_ssize_t pos, Py_ssize_t token_len=-1)  except -2:
    """Parse ISO format month component (MM) from a string,
    returns `-1` for invalid ISO months `<'int'>`.

    This function extracts and parses the month component from an ISO date string.
    It reads two characters starting at the specified position and converts them
    into an integer representing the month. The function ensures that the parsed
    month is valid (i.e., between '01' and '12').

    :param token `<'str'>`: The input token string containing the ISO month to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO month.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO month `[1..12]`, or `-1` if invalid.
    """
    # Validate
    if token is None:
        return -1  # exit: invalid
    if token_len <= 0:
        token_len = str_len(token)
    if pos < 0 or token_len - pos < 2:
        return -1  # exit: invalid

    # Parse value
    cdef Py_UCS4 c0 = str_read(token, pos)
    if not is_ascii_digit(c0):
        return -1  # exit: invalid
    cdef Py_UCS4 c1 = str_read(token, pos + 1)
    if not is_ascii_digit(c1):
        return -1  # exit: invalid

    # Convert to integer
    cdef int out = (
        (ord(c0) - 48) * 10 + 
        (ord(c1) - 48)
    )
    return out if 1 <= out <= 12 else -1

cdef inline int parse_isoday(str token, Py_ssize_t pos, Py_ssize_t token_len=-1) except -2:
    """Parse ISO format day component (DD) from a string,
    returns `-1` for invalid ISO days `<'int'>`.

    This function extracts and parses the day component from an ISO date string.
    It reads two characters starting at the specified position and converts them
    into an integer representing the day. The function ensures that the parsed day
    is valid (i.e., between '01' and '31').

    :param token `<'str'>`: The input token string containing the ISO day to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO day.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO day `[1..31]`, or `-1` if invalid.
    """
    # Validate
    if token is None:
        return -1  # exit: invalid
    if token_len <= 0:
        token_len = str_len(token)
    if pos < 0 or token_len - pos < 2:
        return -1  # exit: invalid

    # Parse value
    cdef Py_UCS4 c0 = str_read(token, pos)
    if not is_ascii_digit(c0):
        return -1  # exit: invalid
    cdef Py_UCS4 c1 = str_read(token, pos + 1)
    if not is_ascii_digit(c1):
        return -1  # exit: invalid

    # Convert to integer
    cdef int out = (
        (ord(c0) - 48) * 10 + 
        (ord(c1) - 48)
    )
    return out if 1 <= out <= 31 else -1

cdef inline int parse_isoweek(str token, Py_ssize_t pos, Py_ssize_t token_len=-1) except -2:
    """Parse an ISO format week number component (WW) from a string,
    returns `-1` for invalid ISO week number `<'int'>`.

    This function extracts and parses the week number from an ISO date string.
    It reads two characters starting at the specified position and converts them
    into an integer representing the week number. The function ensures that the
    parsed week number is valid (i.e., between '01' and '53').

    :param token `<'str'>`: The input token string containing the ISO week number to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO week number.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO week number `[1..53]`, or `-1` if invalid.
    """
    # Validate
    if token is None:
        return -1  # exit: invalid
    if token_len <= 0:
        token_len = str_len(token)
    if pos < 0 or token_len - pos < 2:
        return -1  # exit: invalid

    # Parse value
    cdef Py_UCS4 c0 = str_read(token, pos)
    if not is_ascii_digit(c0):
        return -1  # exit: invalid
    cdef Py_UCS4 c1 = str_read(token, pos + 1)
    if not is_ascii_digit(c1):
        return -1  # exit: invalid

    # Convert to integer
    cdef int out = (
        (ord(c0) - 48) * 10 + 
        (ord(c1) - 48)
    )
    return out if 1 <= out <= 53 else -1

cdef inline int parse_isoweekday(str token, Py_ssize_t pos, Py_ssize_t token_len=-1) except -2:
    """Parse an ISO format weekday component (D) from a string,
    returns `-1` for invalid ISO weekdays `<'int'>`.

    This function extracts and parses the weekday component from an ISO date string.
    It reads a single character at the specified position and converts it into an
    integer representing the ISO weekday, where Monday is 1 and Sunday is 7.

    :param token `<'str'>`: The input token string containing the ISO weekday to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO weekday.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO weekday `[1..7]`, or `-1` if invalid.
    """
    # Validate
    if token is None:
        return -1  # exit: invalid
    if token_len <= 0:
        token_len = str_len(token)
    if pos < 0 or token_len - pos < 1:
        return -1  # exit: invalid

    # Parse value
    cdef Py_UCS4 ch = str_read(token, pos)

    # Convert to integer
    cdef int out = ord(ch) - 48
    return out if 1 <= out <= 7 else -1

cdef inline int parse_isodoy(str token, Py_ssize_t pos, Py_ssize_t token_len=-1) except -2:
    """Parse an ISO format day-of-year component (DDD) from a string,
    returns `-1` for invalid ISO day of the year `<'int'>`.

    This function extracts and parses the day of the year from an ISO date string.
    It reads three characters starting at the specified position and converts them
    into an integer representing the day of the year. The function ensures that the
    parsed days are valid (i.e., between '001' and '366').

    :param token `<'str'>`: The input token string containing the ISO day of the year to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO day of the year.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO day of the year `[1..366]`, or `-1` if invalid.
    """
    # Validate
    if token is None:
        return -1  # exit: invalid
    if token_len <= 0:
        token_len = str_len(token)
    if pos < 0 or token_len - pos < 3:
        return -1  # exit: invalid

    # Parse value
    cdef Py_UCS4 c0 = str_read(token, pos)
    if not is_ascii_digit(c0):
        return -1  # exit: invalid
    cdef Py_UCS4 c1 = str_read(token, pos + 1)
    if not is_ascii_digit(c1):
        return -1  # exit: invalid
    cdef Py_UCS4 c2 = str_read(token, pos + 2)
    if not is_ascii_digit(c2):
        return -1  # exit: invalid

    # Convert to integer
    cdef int out = (
        (ord(c0) - 48) * 100 +
        (ord(c1) - 48) * 10  +
        (ord(c2) - 48)
    )
    return out if 1 <= out <= 366 else -1    

cdef inline int parse_isohour(str token, Py_ssize_t pos, Py_ssize_t token_len=-1) except -2:
    """Parse an ISO format hour (HH) component from a string,
    returns `-1` for invalid ISO hours `<'int'>`.

    This function extracts and parses the hour component from a time string in ISO format.
    It reads two characters starting at the specified position and converts them into an
    integer representing the hours. The function ensures that the parsed hours are valid
    (i.e., between '00' and '23').

    :param token `<'str'>`: The input token string containing the ISO hour to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO hour.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO hour `[0..23]`, or `-1` if invalid.
    """
    # Validate
    if token is None:
        return -1  # exit: invalid
    if token_len <= 0:
        token_len = str_len(token)
    if pos < 0 or token_len - pos < 2:
        return -1  # exit: invalid

    # Parse value
    cdef Py_UCS4 c0 = str_read(token, pos)
    if not is_ascii_digit(c0):
        return -1  # exit: invalid
    cdef Py_UCS4 c1 = str_read(token, pos + 1)
    if not is_ascii_digit(c1):
        return -1  # exit: invalid

    # Convert to integer
    cdef int out = (
        (ord(c0) - 48) * 10 + 
        (ord(c1) - 48)
    )
    return out if 0 <= out <= 23 else -1

cdef inline int parse_isominute(str token, Py_ssize_t pos, Py_ssize_t token_len=-1) except -2:
    """Parse an ISO format minute (MM) component from a string,
    returns `-1` for invalid ISO minutes `<'int'>`.

    This function extracts and parses the minute component from a time string in ISO format.
    It reads two characters starting at the specified position and converts them into an
    integer representing the minutes. The function ensures that the parsed minutes are valid
    (i.e., between '00' and '59').

    :param token `<'str'>`: The input token string containing the ISO minute to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO minute.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO minute `[0..59]`, or `-1` if invalid.
    """
    # Validate
    if token is None:
        return -1  # exit: invalid
    if token_len <= 0:
        token_len = str_len(token)
    if pos < 0 or token_len - pos < 2:
        return -1  # exit: invalid

    # Parse value
    cdef Py_UCS4 c0 = str_read(token, pos)
    if not is_ascii_digit(c0):
        return -1  # exit: invalid
    cdef Py_UCS4 c1 = str_read(token, pos + 1)
    if not is_ascii_digit(c1):
        return -1  # exit: invalid

    # Convert to integer
    cdef int out = (
        (ord(c0) - 48) * 10 + 
        (ord(c1) - 48)
    )
    return out if 0 <= out <= 59 else -1

cdef inline int parse_isosecond(str token, Py_ssize_t pos, Py_ssize_t token_len=-1) except -2:
    """Parse an ISO format second (SS) component from a string,
    returns `-1` for invalid ISO seconds `<'int'>`.

    This function extracts and parses the second component from a time string in ISO format.
    It reads two characters starting at the specified position and converts them into an
    integer representing the seconds. The function ensures that the parsed seconds are valid
    (i.e., between '00' and '59').

    :param token `<'str'>`: The input token string containing the ISO second to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO second.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO second `[0..59]`, or `-1` if invalid.
    """
    return parse_isominute(token, pos, token_len)

cdef inline int parse_isofraction(str token, Py_ssize_t pos, Py_ssize_t token_len=-1) except -2:
    """Parse an ISO fractional time component (fractions of a second) from a string,
    returns `-1` for invalid ISO fraction `<'int'>`.

    This function extracts and parses a fractional time component in ISO format (e.g.,
    the fractional seconds in "2023-11-25T14:30:15.123456Z"). It reads up to six digits
    after the starting position, padding with zeros if necessary to ensure a six-digit
    integer representation.

    :param token `<'str'>`: The input token string containing the ISO fraction to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO fraction.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO fraction `[0..999,999]`, or `-1` if invalid.
    """
    # Validate
    if token is None:
        return -1  # exit: invalid
    if token_len <= 0:
        token_len = str_len(token)
    if pos < 0 or token_len - pos < 1:
        return -1  # exit: invalid

    # Parse value
    cdef:
        int f_size = 0
        int out = 0
        Py_UCS4 ch
    while pos < token_len and f_size < 6:
        ch = str_read(token, pos)
        if not is_ascii_digit(ch):
            break
        out = out * 10 + (ord(ch) - 48)
        pos += 1; f_size += 1

    # Compensate missing digits (pad with zeros)
    return scale_fraction_to_us(out, f_size)

cdef inline sf parse_second_and_fraction(str token, Py_ssize_t pos, Py_ssize_t token_len=-1) except *:
    """Parse a `seconds` token with an optional fractional part into (second, microsecond) `<'struct:sf'>`.

    This reads a decimal token starting at `pos` for whole `seconds`, optionally
    followed by a dot `.` and up to 6 fractional digits which are scaled to
    microseconds (truncated if more digits appear beyond 6). The function is
    intentionally strict and returns an invalid result for out-of-domain inputs.

    :param token `<'str'>`: The input token string containing `seconds` information.
    :param pos `<'int'>`: The starting position in the `token` of the `seconds`.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'struct:sf'>`: A small struct with fields:

        - `second`: parsed whole seconds (0..59) or -1 if invalid
        - `microsecond`: parsed fractional part in microseconds (0..999_999),
            or -1 if no fraction / value is invalid,

    ## Rules
    - Input format: `'SS'` or `'SS.f'` where `S` and `f` are ASCII digits.
    - Decimal separator: dot only `'.'`; comma is NOT accepted.
    - Fraction length: up to 6 digits kept; extra digits (if any) stop parsing
      early and are `ignored` (no rounding).
    - Seconds domain: 0..59 only. Leap-second `60` is not accepted.
    - Whitespace and other characters terminate parsing.
    - On any invalid input (e.g., no digits, seconds > 59, out-of-bounds indexes),
      returns `sf(second=-1, microsecond=-1)`.
    """
    # Validate
    cdef sf out
    if token is None:
        out.second = out.microsecond = -1
        return out  # exit: invalid
    if token_len <= 0:
        token_len = str_len(token)
    if pos < 0 or token_len - pos < 1:
        out.second = out.microsecond = -1
        return out  # exit: invalid

    # Parse value
    cdef:
        unsigned long long i_part = 0
        unsigned long long f_part = 0
        int f_size = 0
        bint has_dot = False
        bint has_digit = False
        Py_UCS4 ch
    
    while pos < token_len and f_size < 6:
        ch = str_read(token, pos)
        if is_ascii_digit(ch):
            if not has_dot:
                i_part = i_part * 10 + (ord(ch) - 48)
            else:
                f_part = f_part * 10 + (ord(ch) - 48)
                f_size += 1
            has_digit = True
            pos += 1
        elif ch == 46:  # '.'
            if has_dot:
                break
            has_dot = True
            pos += 1
        else:
            break

    # No digits / second out of range: invalid
    if not has_digit or i_part > 59:
        out.second = out.microsecond = -1
    # Pure integer: second only
    elif not has_dot:
        out.second = i_part
        out.microsecond = -1
    # Float: second & microsecond
    else:
        out.second = i_part
        out.microsecond = scale_fraction_to_us(f_part, f_size)
    return out

cdef inline int scale_fraction_to_us(int fraction, int fraction_size) noexcept:
    """Scale a fractional time component to microseconds based on its size `<'int'>`.

    This function takes a fractional time component (e.g., milliseconds, microseconds)
    and scales it to microseconds based on the number of digits provided. It supports
    fractions with sizes ranging from 1 to 6 digits.

    :param fraction `<'int'>`: The fractional time component to scale.
    :param fraction_size `<'int'>`: The number of digits in the fractional component.
        Must be in the range [1..6].
    :returns `<'int'>`: The scaled fractional time in microseconds.
        Returns `-1` if `fraction_size` is out of range.

    ## Example
    >>> scale_fraction_to_us(123, 3)  
        # 123 milliseconds → 123000 microseconds
    """
    if fraction_size == 6:
        return fraction
    elif fraction_size == 5:
        return fraction * 10
    elif fraction_size == 4:
        return fraction * 100
    elif fraction_size == 3:
        return fraction * 1_000
    elif fraction_size == 2:
        return fraction * 10_000
    elif fraction_size == 1:
        return fraction * 100_000
    else:
        return -1

# . slice and convert
cdef inline unsigned long long slice_to_uint(str token, Py_ssize_t start, Py_ssize_t size, Py_ssize_t token_len=-1) except -1:
    """Slice a substring from a string and convert to an unsigned integer `<'int'>`.

    This function slices a portion of the input `token` string starting
    at 'start' and spanning 'size' of characters. The sliced substring is
    validated to ensure it contains only ASCII digits, before converting
    to unsigned integer.

    :param token `<'str'>`: The input token string to slice and convert.
    :param start `<'int'>`: The starting index for slicing the `token`.
    :param size `<'int'>`: The size of the slice.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The converted unsigned integer.
    :raises `<'ValueError'>`: When the slice is out of range, or contains non-digit characters.

    ## Notice
    - Only ASCII digits ('0' to '9') are accepted.

    ## Equivalent
    >>> int(data[start:start+size])  # without '+/-' signs
    """
    # Validate
    if token is None:
        raise ValueError("slice_to_uint: 'data' cannot be None")
    if start < 0:
        raise ValueError("slice_to_uint: 'start' must be >= 0, instead got %d" % start)
    if size <= 0:
        raise ValueError("slice_to_uint: 'size' must be > 0, instead got %d" % size)
    if token_len <= 0:
        token_len = str_len(token)
    cdef Py_ssize_t end = start + size
    if end > token_len:
        raise ValueError("slice_to_uint: out of range (start=%d size=%d token_len=%d)" % (start, size, token_len))

    # Slice and convert
    cdef: 
        unsigned long long out = 0
        Py_ssize_t i
        Py_UCS4 ch

    for i in range(start, end):
        ch = str_read(token, i)
        if not is_ascii_digit(ch):
            raise ValueError(
                "Cannot convert '%s' to unsigned integer. Contains invalid character '%s' "
                "(non-ASCII digit)." % (token[start: end], str_chr(ch))
            )
        out = out * 10 + (ord(ch) - 48)
    return out

cdef inline double slice_to_ufloat(str token, Py_ssize_t start, Py_ssize_t size, Py_ssize_t token_len=-1) except *:
    """Slice a substring from a string and convert to a non-negative double-precision float `<'float'>`.

    This function slices a portion of the input `token` string starting
    at 'start' and spanning 'size' of characters. The sliced substring is
    then converted to a non-negative double-precision floating-point number.

    :param token `<'str'>`: The input token string to slice and convert.
    :param start `<'int'>`: The starting index for slicing the `token`.
    :param size `<'int'>`: The size of the slice.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'float'>`: The converted non-negative double-precision float.
    :raises `<'ValueError'>`: When the slice is out of range, or cannot be converted to float.

    ## Notice
    - Only ASCII digits ('0' to '9') are accepted.
    - At most one decimal point ('.') is allowed.
    
    ## Equivalent
    >>> float(data[start:start+size])  # without '+/-' signs
    """
    # Validate
    if token is None:
        raise ValueError("slice_to_ufloat: 'data' cannot be None")
    if start < 0:
        raise ValueError("slice_to_ufloat: 'start' must be >= 0, instead got %d" % start)
    if size <= 0:
        raise ValueError("slice_to_ufloat: 'size' must be > 0, instead got %d" % size)
    if token_len <= 0:
        token_len = str_len(token)
    cdef Py_ssize_t end = start + size
    if end > token_len:
        raise ValueError("slice_to_ufloat: out of range (start=%d size=%d token_len=%d)" % (start, size, token_len))

    # Slice and convert
    cdef:
        unsigned long long i_part = 0
        unsigned long long f_part = 0
        double f_scale = 1.0
        bint has_dot = False
        bint has_digit = False
        Py_ssize_t i
        Py_UCS4 ch

    for i in range(start, end):
        ch = str_read(token, i)
        if is_ascii_digit(ch):
            if not has_dot:
                i_part = i_part * 10 + (ord(ch) - 48)
            else:
                f_part = f_part * 10 + (ord(ch) - 48)
                f_scale *= 0.1
            has_digit = True
        elif ch == 46:  # '.'
            if has_dot:
                raise ValueError(
                    "Cannot convert '%s' to non-negative float: "
                    "contains more than one decimal point." % (token[start: end])
                )
            has_dot = True
        else:
            raise ValueError(
                "Cannot convert '%s' to non-negative float: contains invalid character '%s' "
                "(non-ASCII digit)." % (token[start: end], str_chr(ch))
            )

    if not has_digit:
        raise ValueError(
            "Cannot convert '%s' to non-negative float: "
            "no digits found." % (token[start: end])
        )
    elif not has_dot:
        return (<double> i_part)
    else:
        return (<double> i_part) + (<double> f_part) * f_scale

# Time -------------------------------------------------------------------------------------------------
# . gmtime
cdef inline tm tm_gmtime(double ts) except *:
    """Convert a Unix timestamp to UTC calendar time `<'struct:tm'>`.

    :param ts `<'float'>`: Unix timestamp (seconds since the Unix Epoch).
        Fractional seconds are floored.
    :returns `<'struct:tm'>`: UTC calendar time.

        - tm.tm_sec   [0..59]
        - tm.tm_min   [0..59]
        - tm.tm_hour  [0..23]
        - tm.tm_mday  [1..31]
        - tm.tm_mon   [1..12]
        - tm.tm_year  [Gregorian year number]
        - tm.tm_wday  [0..6, 0=Monday]
        - tm.tm_yday  [1..366]
        - tm.tm_isdst [-1..1]

    ## Equivalent
    >>> time.gmtime(ts)
    """
    cdef:
        time.time_t tic = <time.time_t> math.floor(ts)
        tm* t = time.gmtime(&tic)
    if t is NULL:
        raise RuntimeError("Fail to convert timestamp '%s' to UTC calendar time." % ts)
    
    # Fix 0-based date values (and the 1900-based year).
    # See tmtotuple() in https://github.com/python/cpython/blob/master/Modules/timemodule.c
    cdef tm out = t[0]                   # copy struct       
    out.tm_year += 1900                  # years since 1900 → Gregorian absolute year
    out.tm_mon += 1                      # 0..11 (0-based month) → 1..12
    if out.tm_sec > 59:                  # clamp leap seconds
        out.tm_sec = 59
    out.tm_wday = (out.tm_wday + 6) % 7  # 0..6 (Sun..Sat) → 0..6 (Mon..Sun)
    out.tm_yday += 1                     # 0..365 (0-based day of year) → 1..366
    return out

cdef inline long long ts_gmtime(double ts) except *:
    """Convert a timestamp to UTC seconds ticks since the Unix Epoch `<'int'>`.

    :param ts `<'float'>`: Unix timestamp (seconds since the Unix Epoch).
        Fractional seconds are floored.
    :returns `<'int'>`: Integer in seconds since the Unix Epoch, representing the UTC time.
    """
    cdef: 
        tm t = tm_gmtime(ts)
        long long ord = ymd_to_ord(t.tm_year, t.tm_mon, t.tm_mday)
        long long hh = t.tm_hour
        long long mi = t.tm_min
        long long ss = t.tm_sec

    return (ord - EPOCH_DAY) * SS_DAY + hh * SS_HOUR + mi * SS_MINUTE + ss

# . localtime
cdef inline tm tm_localtime(double ts) except *:
    """Convert a Unix timestamp to local calendar time `<'struct:tm'>`.

    :param ts `<'float'>`: Unix timestamp (seconds since the Unix Epoch).
        Fractional seconds are floored.
    :returns `<'struct:tm'>`: Local calendar time.

        - tm.tm_sec   [0..59]
        - tm.tm_min   [0..59]
        - tm.tm_hour  [0..23]
        - tm.tm_mday  [1..31]
        - tm.tm_mon   [1..12]
        - tm.tm_year  [Gregorian year number]
        - tm.tm_wday  [0..6, 0=Monday]
        - tm.tm_yday  [1..366]
        - tm.tm_isdst [-1..1]

    ## Equivalent
    >>> time.localtime(ts)
    """
    cdef:
        time.time_t tic = <time.time_t> math.floor(ts)
        tm* t = time.localtime(&tic)
    if t is NULL:
        raise RuntimeError("Fail to convert timestamp '%s' to local calendar time." % ts)

    # Fix 0-based date values (and the 1900-based year).
    # See tmtotuple() in https://github.com/python/cpython/blob/master/Modules/timemodule.c
    cdef tm out = t[0]                   # copy struct       
    out.tm_year += 1900                  # years since 1900 → Gregorian absolute year
    out.tm_mon += 1                      # 0..11 (0-based month) → 1..12
    if out.tm_sec > 59:                  # clamp leap seconds
        out.tm_sec = 59
    out.tm_wday = (out.tm_wday + 6) % 7  # 0..6 (Sun..Sat) → 0..6 (Mon..Sun)
    out.tm_yday += 1                     # 0..365 (0-based day of year) → 1..366
    return out

cdef inline long long ts_localtime(double ts) except *:
    """Convert a timestamp to local seconds ticks since the Unix Epoch `<'int'>`.

    :param ts `<'float'>`: Unix timestamp (seconds since the Unix Epoch).
        Fractional seconds are floored.
    :returns `<'int'>`: Integer in seconds since the Unix Epoch, representing the local time.
    """
    cdef: 
        tm t = tm_localtime(ts)
        long long ord = ymd_to_ord(t.tm_year, t.tm_mon, t.tm_mday)
        long long hh = t.tm_hour
        long long mi = t.tm_min
        long long ss = t.tm_sec
        
    return (ord - EPOCH_DAY) * SS_DAY + hh * SS_HOUR + mi * SS_MINUTE + ss

# . conversion
cdef inline long long sec_to_us(double value) except *:
    """Convert seconds (float) to microseconds (int) `<'int'>`.

    Decimal **truncate-towards-zero** at microsecond precision:
    the result is built from the fixed-point string with exactly 
    6 fractional digits, ignoring any further digits.

    :param value `<'float'>`: Seconds.
    :returns `<'int'>`: Microseconds.

    ## Examples
    ```
    1.000001    -> 1,000,001  (6 fractional digits)
    1.23456789  -> 1,234,567  (digits beyond 6 are discarded)
    -0.0000005  -> 0          (truncate toward zero)
    ```
    """
    if not math.isfinite(value):
        raise OverflowError("Seconds value must be finite, got '%s'" % value)

    cdef: 
        str s = "%.7f" % value
        bint neg = str_read(s, 0) == "-"
        Py_ssize_t start = 1 if neg else 0
        Py_ssize_t dot = str_len(s) - 8
        long long sec = slice_to_uint(s, start, dot - start)
        long long us = slice_to_uint(s, dot + 1, 6)

    us = sec * US_SECOND + us
    return -us if neg else us

cdef inline str tm_strformat(tm t, str fmt):
    """Format a calendar time `tm` using C `strftime` 
    and return a Unicode string `<'str'>`.

    :param t `<'struct:tm'>`: Calendar time.

        - Expects a `tm` using the following conventions:

            - tm_year = absolute Gregorian year (e.g., 2025)
            - tm_mon  = 1..12
            - tm_wday = 0..6 (Mon..Sun)
            - tm_yday = 1..365/366

        - The fields are normalized internally back to POSIX conventions:

            - tm_year → years since 1900
            - tm_mon  → 0..11
            - tm_wday → 0..6 (Sun..Sat)
            - tm_yday → 0..365

    :param fmt `<'str'>`: strftime-compatible format string.
    :returns `<'str'>`: Formatted text.

    ## Notice
    - Output decoding assumes a UTF-8 C locale. If the process locale is not UTF-8,
      expansions of `%a`, `%A`, `%b`, `%B`, etc. may fail to decode.
    """
    # Validate format
    if fmt is None:
        raise ValueError("tm_strformat: 'fmt' cannot be None")
    if str_len(fmt) == 0:
        return fmt

    # Normalize fields back to POSIX conventions
    t.tm_year -= 1900
    t.tm_mon -= 1
    t.tm_wday = (t.tm_wday + 1) % 7
    t.tm_yday -= 1

    # Perform strftime
    cdef: 
        const char* cfmt = PyUnicode_AsUTF8(fmt)
        size_t cap = <size_t> 256            # Initial buffer size
        size_t cap_max = <size_t> (1 << 15)  # Maximum buffer size: 32 KiB
        char* buf
        size_t n

    while True:
        # Allocate buffer
        buf = <char*> stdlib.malloc(cap)
        if buf == NULL:
            raise MemoryError("tm_strformat: failed to allocate buffer")

        # Returns 0 if buffer too small (or other failure)
        n = time.strftime(buf, cap, cfmt, &t)
        if n > 0:
            # Decode as UTF-8 (assumes UTF-8 C locale)
            try:
                return PyUnicode_DecodeUTF8(buf, n, NULL)
            finally:
                stdlib.free(buf)
        
        # Retry with larger buffer
        stdlib.free(buf)
        if cap >= cap_max:
            raise OverflowError("tm_strformat: expanded output exceeds buffer limit")
        cap <<= 1

cdef inline tm tm_fr_us(long long value) noexcept nogil:
    """Convert microseconds since the Unix epoch to 'struct:tm' `<'struct:tm'>`.

    :param value `<'int'>`: The microsecond ticks since epoch.
    :returns `<'struct:tm'>`: The corresponding 'struct:tm' representation.

        - tm.tm_sec   [0..59]
        - tm.tm_min   [0..59]
        - tm.tm_hour  [0..23]
        - tm.tm_mday  [1..31]
        - tm.tm_mon   [1..12]
        - tm.tm_year  [Gregorian year number]
        - tm.tm_wday  [0..6, 0=Monday]
        - tm.tm_yday  [1..366]
        - tm.tm_isdst [-1]
    """
    cdef:
        long long q, r, ord
        int yy, mm, dd
        ymd _ymd
        tm out

    with cython.cdivision(True):
        # Y/M/D from ordinal
        q = value / US_DAY; r = value % US_DAY
        if r < 0:
            q -= 1; r += US_DAY
        ord = q + EPOCH_DAY
        _ymd = ymd_fr_ord(ord)
        yy, mm, dd = _ymd.year, _ymd.month, _ymd.day

        # Time-of-day
        out.tm_hour = r / US_HOUR;   r %= US_HOUR
        out.tm_min  = r / US_MINUTE; r %= US_MINUTE
        out.tm_sec  = r / US_SECOND

        # Weekday
        r = (ord - 1) % 7  # 0=Monday
        out.tm_wday = r + 7 if r < 0 else r

    # Remaining tm fields
    out.tm_year = yy
    out.tm_mon  = mm
    out.tm_mday = dd
    out.tm_yday = days_bf_month(yy, mm) + dd
    out.tm_isdst = -1  # unknown
    return out

cdef inline tm tm_fr_sec(double value) noexcept nogil:
    """Convert from seconds (float) epoch to 'struct:tm' `<'struct:tm'>`.

    :param value `<'float'>`: Seconds since Unix epoch.
    :returns `<'struct:tm'>`: The corresponding 'struct:tm' representation.

        - tm.tm_sec   [0..59]
        - tm.tm_min   [0..59]
        - tm.tm_hour  [0..23]
        - tm.tm_mday  [1..31]
        - tm.tm_mon   [1..12]
        - tm.tm_year  [Gregorian year number]
        - tm.tm_wday  [0..6, 0=Monday]
        - tm.tm_yday  [1..366]
        - tm.tm_isdst [-1]
    """
    cdef: 
        long long sec = <long long> math.floor(value)
        long long q, r, ord
        int yy, mm, dd
        ymd _ymd
        tm out

    with cython.cdivision(True):
        # Y/M/D from ordinal
        q = sec / SS_DAY; r = sec % SS_DAY
        if r < 0:
            q -= 1
            r += SS_DAY
        ord = q + EPOCH_DAY
        _ymd = ymd_fr_ord(ord)
        yy, mm, dd = _ymd.year, _ymd.month, _ymd.day

        # Time-of-day
        out.tm_hour = r / SS_HOUR;   r %= SS_HOUR
        out.tm_min  = r / SS_MINUTE; r %= SS_MINUTE
        out.tm_sec  = r

        # Weekday
        r = (ord - 1) % 7  # 0=Monday
        out.tm_wday = r + 7 if r < 0 else r

    # Remaining tm fields
    out.tm_year = yy
    out.tm_mon  = mm
    out.tm_mday = dd
    out.tm_yday = days_bf_month(yy, mm) + dd
    out.tm_isdst = -1  # unknown
    return out

cdef inline hmsf hmsf_fr_us(long long value) noexcept nogil:
    """Extract time-of-day from microsecond ticks since epoch `<'struct:hmsf'>`.

    :param value `<'int'>`: Microseconds since Unix epoch.
    :returns `<'struct:hmsf'>`: The extracted time-of-day components.
    
        - hmsf.hour        [0..23]
        - hmsf.minute      [0..59]
        - hmsf.second      [0..59]
        - hmsf.microsecond [0..999_999].
    """
    cdef:
        long long r
        hmsf out

    with cython.cdivision(True):
        # Hour
        r = value % US_DAY
        if r < 0:
            r += US_DAY
        out.hour   = r / US_HOUR;   r %= US_HOUR
        # Minute
        out.minute = r / US_MINUTE; r %= US_MINUTE
        # Second
        out.second = r / US_SECOND; r %= US_SECOND
        # Microsecond
        out.microsecond = r

    return out

cdef inline hmsf hmsf_fr_sec(double value) except *:
    """Extract time-of-day from seconds (float) since epoch `<'struct:hmsf'>`.

    :param value `<'float'>`: Seconds since Unix epoch.
    :returns `<'struct:hmsf'>`: The extracted time-of-day components.
        
        - hmsf.hour [0..23]
        - hmsf.minute [0..59]
        - hmsf.second [0..59]
        - hmsf.microsecond [0..999_999].
    """
    return hmsf_fr_us(sec_to_us(value))

# Calendar ---------------------------------------------------------------------------------------------
# . year
cdef inline bint is_leap_year(long long year) noexcept nogil:
    """Determine whether `year` is a leap year under 
    the proleptic Gregorian rules `<'bool'>`.

    :param year `<'int'>`: Gregorian year number.
    :returns `<'bool'>`: True if `year` is a leap year, else False.
    """
    with cython.cdivision(True):
        if (year % 4) != 0:
            return False
        if (year % 100) != 0:
            return True
        return (year % 400) == 0

cdef inline bint is_long_year(long long year) noexcept nogil:
    """Determine whether `year` is a “long year” (has ISO week 53) 
    under the proleptic Gregorian rules `<'bool'>`.

    A year has ISO week 53 **if** January 1 is a Thursday, or January 1 is a
    Wednesday **and** the year is a leap year (ISO-8601 week rules).

    :param year `<'int'>`: Gregorian year number. 
    :returns `<'bool'>`: True if `year` is a long year, else False.
    """
    cdef int wkd = ymd_weekday(year, 1, 1)
    return (wkd == 3) or (wkd == 2 and is_leap_year(year))

cdef inline long long leap_years(long long year, bint inclusive) noexcept nogil:
    """Count leap years between `year` and `0001-01-01` under
    the proleptic Gregorian rules `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param inclusive `<'bool'>`: If True, include `year` itself when leap.

        - `False` strictly before Jan 1 of `year` (… < y < year)
        - `True`  up to and including `year`      (… < y ≤ year)

    :returns `<'int'>`: Count of leap years up to the boundary
        (negative when the boundary is earlier than 0001-01-01).
    """
    cdef: 
        long long yy = year if inclusive else year - 1
        long long q4, q100, q400, r
        long long out

    with cython.cdivision(True):
        # Positive
        if yy >= 0:
            out = (yy // 4) - (yy // 100) + (yy // 400)

        # Negative
        else:
            # floor(yy / 4)
            q4, r = yy / 4, yy % 4
            if r != 0:
                q4 -= 1
            # floor(yy / 100)
            q100, r = yy / 100, yy % 100
            if r != 0:
                q100 -= 1
            # floor(yy / 400)
            q400, r = yy / 400, yy % 400
            if r != 0:
                q400 -= 1
            out = q4 - q100 + q400

    return out

cdef inline long long leaps_bt_years(long long year1, long long year2) noexcept nogil:
    """Compute the total number of Gregorian leap years between `year1` and `year2`
    under the proleptic Gregorian rules `<'int'>`.

    :param year1 `<'int'>`: First gregorian year.
    :param year2 `<'int'>`: Second gregorian year.
    :returns `<'int'>`: Number of leap years between the two years.
    """
    if year1 > year2:
        return leap_years(year1, False) - leap_years(year2, False)
    else:
        return leap_years(year2, False) - leap_years(year1, False)

cdef inline int days_in_year(long long year) noexcept nogil:
    """Determine the number of days in `year` under the 
    proleptic Gregorian rules `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :returns `<'int'>`: 366 for leap years, else 365.
    """
    return 366 if is_leap_year(year) else 365

cdef inline long long days_bf_year(long long year) noexcept nogil:
    """Compute the number of days strictly before `January 1` of `year`
    under the proleptic Gregorian rules `<'int'>`.

    This is the signed offset (in days) from `0001-01-01` to `year-01-01`:

        - days_bf_year(1) == 0
        - days_bf_year(0) == -366  (year 0 ≡ 1 BCE, leap)
        - Negative values indicate dates before '0001-01-01'.

    :param year `<'int'>`: Gregorian year number.
    :returns `<'int'>`: Signed count of days before Jan-1 of `year`.
    """
    cdef long long y0 = year - 1  # 0-based year
    return 365 * y0 + leap_years(y0, True)

cdef inline int day_of_year(long long year, int month, int day) noexcept nogil:
    """Compute the `1-based` ordinal day-of-year for the given Y/M/D
    under the proleptic Gregorian rules `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :param day `<'int'>`: Day of month.
        Automatically clamps to the valid range for the (clamped) month and year.
    :returns `<'int'>`: `1..365/366` — ordinal day of year (Jan-01 → 1).
    """
    if day > 28:
        day = min(day, days_in_month(year, month))
    elif day < 1:
        day = 1
    return days_bf_month(year, month) + day

# . quarter
cdef inline int quarter_of_month(int month) noexcept nogil:
    """Return the calendar quarter index (1..4) for `month`
    under the Gregorian calendar `<'int'>`.

    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :returns `<'int'>`: Quarter number in 1..4
        (Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec).
    """
    # Jan - Mar & out-of-range: Q1
    if month < 4:
        return 1
    # Apr - Jun: Q2
    elif month < 7:
        return 2
    # Jul - Sep: Q3
    elif month < 10:
        return 3
    # Oct - Dec & out-of-range: Q4
    else:
        return 4

cdef inline int days_in_quarter(long long year, int month) noexcept nogil:
    """Return the number of days in calendar quarter containing 
    `month` in `year` under the proleptic Gregorian rules `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :returns `<'int'>`: Number of days in the quarter.

        - Non-leap [Q1=90, Q2=91, Q3=92, Q4=92]
        - Leap     [Q1=91, Q2=91, Q3=92, Q4=92]
    """
    if month < 4: # Q1
        return 91 if is_leap_year(year) else 90
    elif month < 7:
        return 91  # Q2
    else:
        return 92  # Q3 & Q4

cdef inline int days_bf_quarter(long long year, int month) noexcept nogil:
    """Return the number of days strictly before the first day of the
    calendar quarter containing `month` in `year` under the proleptic
    Gregorian rules `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :returns `<'int'>`: Days before the quarter start.

        - Non-leap [Q1=0, Q2=90, Q3=181, Q4=273]
        - Leap     [Q1=0, Q2=91, Q3=182, Q4=274]
    """
    if month < 4: 
        return 0  # Q1
    cdef int leap = <int> is_leap_year(year)
    if month < 7:
        return 90 + leap  # Q2
    elif month < 10:
        return 181 + leap # Q3
    else:
        return 273 + leap # Q4

cdef inline int day_of_quarter(long long year, int month, int day) noexcept nogil:
    """Compute the number of days between the 1st day of the quarter and the
    given Y/M/D under the proleptic Gregorian rules `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :param day `<'int'>`: Day of month.
        Automatically clamps to the valid range for the (clamped) month/year.
    :returns `<'int'>`: 0-based days since the quarter start.
    """
    return day_of_year(year, month, day) - days_bf_quarter(year, month)

# . month
cdef inline int days_in_month(long long year, int month) noexcept nogil:
    """Return the number of days in `month` of `year` under 
    the proleptic Gregorian rules `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12. 
        Automatically clamps out-of-range values to [1..12].
    :returns `<'int'>`: Number of days in the month (28-31).
    """
    # Month Jan(1) or out-of-range: 31 days
    if month <= 1:
        return 31
    # Month Feb(2): 28 or 29 days (leap)
    elif month == 2:
        return 29 if is_leap_year(year) else 28
    # Month Apr(4), Jun(6), Sep(9), Nov(11): 30 days
    elif month == 4 or month == 6 or month == 9 or month == 11:
        return 30
    # Other months: 31 days
    else:
        return 31

cdef inline int days_bf_month(long long year, int month) noexcept nogil:
    """Return the number of days strictly before the first day of `month`
    in `year` under the proleptic Gregorian rules `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :returns `<'int'>`: Days before `year-month-01` (Jan→0, Feb→31, Mar→59/60, ...).
    """
    # Jan(1) or out-of-range: 0 days
    if month <= 1:
        return 0
    # Feb(2): 31 days
    if month == 2:
        return 31
    # Mar(3)..Dec(12)
    cdef int days = DAYS_BR_MONTH[month - 1] if month < 12 else 334
    return days + 1 if is_leap_year(year) else days

# . week
cdef inline int ymd_weekday(long long year, int month, int day) noexcept nogil:
    """Compute the weekday for Y/M/D (`0=Mon … 6=Sun`) under 
    the proleptic Gregorian rules `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :param day `<'int'>`: Day of month.
        Automatically clamps to the valid range for the (clamped) month/year.
    :returns `<'int'>`: Weekday number in 0..6 (Mon=0).
    """
    cdef long long ord0 = ymd_to_ord(year, month, day) - 1  # 0-based, Mon=0 at 0001-01-01
    cdef long long r
    with cython.cdivision(True):
        r = ord0 % 7
    return <int> (r + 7 if r < 0 else r)

# . iso
cdef inline iso ymd_isocalendar(long long year, int month, int day) noexcept nogil:
    """Compute the ISO calendar from the Gregorian date Y/M/D `<'struct:iso'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :param day `<'int'>`: Day of month.
        Automatically clamps to the valid range for the (clamped) month/year.
    :returns `<'struct:iso'>`: The ISO calendar components.

        - iso.year    [ISO calendar year]
        - iso.week    [1..52/53]
        - iso.weekday [1..7 (Mon..Sun)]
    """
    # Find ISO year by comparing against ISO week-1 Monday boundaries
    cdef:
        long long ord    = ymd_to_ord(year, month, day)
        long long ord_w1 = iso_week1_mon_ord(year)
        long long iso_year, iso_week, iso_wday, w1, w1_next, r
        iso out

    # ISO year
    if ord < ord_w1:
        iso_year = year - 1
        w1 = iso_week1_mon_ord(iso_year)
    else:
        w1_next = iso_week1_mon_ord(year + 1)
        if ord >= w1_next:
            iso_year = year + 1
            w1 = w1_next
        else:
            iso_year = year
            w1 = ord_w1

    # ISO week & weekday
    with cython.cdivision(True):
        iso_week = (ord - w1) // 7 + 1
        r = (ord - 1) % 7
        if r < 0:
            r += 7
        iso_wday = r + 1

    out.year = iso_year
    out.week = iso_week
    out.weekday = iso_wday
    return out

cdef inline long long ymd_isoyear(long long year, int month, int day) noexcept nogil:
    """Compute the ISO calendar `year` from the Gregorian date Y/M/D `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :param day `<'int'>`: Day of month.
        Automatically clamps to the valid range for the (clamped) month/year.
    :returns `<'int'>`: The ISO calendar year.
    """
    cdef: 
        long long ord    = ymd_to_ord(year, month, day)
        long long ord_w1 = iso_week1_mon_ord(year)
        long long w1_next

    if ord < ord_w1:
        return year - 1
    else:
        w1_next = iso_week1_mon_ord(year + 1)
        return year if ord < w1_next else year + 1

cdef inline int ymd_isoweek(long long year, int month, int day) noexcept nogil:
    """Compute the ISO calendar `week` from the Gregorian date Y/M/D `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :param day `<'int'>`: Day of month.
        Automatically clamps to the valid range for the (clamped) month/year.
    :returns `<'int'>`: The ISO calendar week (1..52/53).
    """
    cdef: 
        long long ord    = ymd_to_ord(year, month, day)
        long long ord_w1 = iso_week1_mon_ord(year)
        long long w1, w1_next

    if ord < ord_w1:
        w1 = iso_week1_mon_ord(year - 1)
    else:
        w1_next = iso_week1_mon_ord(year + 1)
        w1 = ord_w1 if ord < w1_next else w1_next

    with cython.cdivision(True):
        return (ord - w1) // 7 + 1

cdef inline int ymd_isoweekday(long long year, int month, int day) noexcept nogil:
    """Compute the ISO calendar `weekday` from the Gregorian date Y/M/D `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :param day `<'int'>`: Day of month.
        Automatically clamps to the valid range for the (clamped) month/year.
    :returns `<'int'>`: The ISO calendar weekday (1=Mon...7=Sun).
    """
    return ymd_weekday(year, month, day) + 1

# . Y/M/D
cdef inline long long ymd_to_ord(long long year, int month, int day) noexcept nogil:
    """Convert a Gregorian Y/M/D to a **1-based** ordinal day under 
    the proleptic Gregorian rules `<'int'>`.

    - Astronomical year numbering (year 0 == 1 BCE).
    - Result is **1-based** with `0001-01-01 -> 1`.
      Dates before `0001-01-01` yield `<= 0`.

    :param year  `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :param day `<'int'>`: Day of month.
        Automatically clamps to the valid range for the (clamped) month/year.
    :returns `<'int'>`: Ordinal day count (1-based).
    """
    if day > 28:
        day = min(day, days_in_month(year, month))
    elif day < 1:
        day = 1
    return days_bf_year(year) + days_bf_month(year, month) + day

cdef inline ymd ymd_fr_ord(long long value) noexcept nogil:
    """Convert Gregorian ordinal day to Y/M/D `<'struct:ymd'>`.

    :param value `<'int'>`: Gregorian ordinal day.
    :returns `<'struct:ymd'>`: Converted Y/M/D components.
    
        - ymd.year  [Gregorian year number]
        - ymd.month [1..12]
        - ymd.day   [1..31]
    """
    # Convert to 0-based offset from 0001-01-01
    cdef: 
        long long n, r, n400, n100, n4, n1, yy, mm, days_bf
        ymd out
    n = value - 1
    
    with cython.cdivision(True):
        # Number of complete 400-year cycles
        n400 = n / 146_097; r = n % 146_097
        if r < 0:
            n400 -= 1; r += 146_097
        n = r
        # Number of complete 100-year cycles within the 400-year cycle
        n100 = n / 36_524;  n %= 36_524
        # Number of complete 4-year cycles within the 100-year cycle
        n4   = n / 1_461;   n %= 1_461
        # Number of complete years within the 4-year cycle
        n1   = n / 365;     n %= 365

    # We now know the year and the offset from January 1st.  Leap years are
    # tricky, because they can be century years.  The basic rule is that a
    # leap year is a year divisible by 4, unless it's a century year --
    # unless it's divisible by 400.  So the first thing to determine is
    # whether year is divisible by 4.  If not, then we're done -- the answer
    # is December 31 at the end of the year.
    yy = n400 * 400 + n100 * 100 + n4 * 4 + n1 + 1
    if n100 == 4 or n1 == 4:
        out.year, out.month, out.day = yy - 1, 12, 31

    # Now the year is correct, and n is the offset from January 1.  We find
    # the month via an estimate that's either exact or one too large.
    else:
        mm = (n + 50) >> 5
        days_bf = days_bf_month(yy, mm)
        if days_bf > n:
            mm -= 1
            days_bf = days_bf_month(yy, mm)
        out.year, out.month, out.day = yy, mm, n - days_bf + 1
            
    return out

cdef inline ymd ymd_fr_us(long long value) noexcept nogil:
    """Convert microsecond ticks since the Unix epoch to Y/M/D `<'struct:ymd'>`.

    :param value `<'int'>`: The microsecond ticks since epoch.
    :returns `<'struct:ymd'>`: Converted Y/M/D components.

        - ymd.year  [Gregorian year number]
        - ymd.month [1..12]
        - ymd.day   [1..31]
    """
    # Convert to ordinal
    cdef long long q, r
    with cython.cdivision(True):
        q = value / US_DAY; r = value % US_DAY
        if r < 0:
            q -= 1
    return ymd_fr_ord(q + EPOCH_DAY)

cdef inline ymd ymd_fr_sec(double value) noexcept nogil:
    """Convert seconds (float) since the Unix epoch to Y/M/D `<'struct:ymd'>`.

    :param value `<'float'>`: Seconds since Unix epoch.
    :returns `<'struct:ymd'>`: Converted Y/M/D components.

        - ymd.year  [Gregorian year number]
        - ymd.month [1..12]
        - ymd.day   [1..31]
    """
    # Convert to ordinal
    cdef: 
        long long sec = <long long> math.floor(value)
        long long q, r
    with cython.cdivision(True):
        q = sec / SS_DAY; r = sec % SS_DAY
        if r < 0:
            q -= 1
    return ymd_fr_ord(q + EPOCH_DAY)

cdef inline ymd ymd_fr_isocalendar(long long year, int week, int weekday) noexcept nogil:
    """Create `struct:ymd` from ISO calendar values 
    (ISO year, ISO week, ISO weekday) `<'struct:ymd'>`.

    :param year `<'int'>`: ISO year number.
    :param week `<'int'>`: ISO week index. Automatically clamped to [1..53].
    :param weekday`<'int'>`: ISO weekday. Automatically clamped to [1..7] (Mon..Sun).
    :returns `<'struct:ymd'>`: Gregorian Y/M/D in the proleptic Gregorian calendar.

        - ymd.year  [Gregorian year number]
        - ymd.month [1..12]
        - ymd.day   [1..31]
    """
    # Clamp weekday to [1..7]
    if weekday < 1:
        weekday = 1
    elif weekday > 7:
        weekday = 7

    # Clamp week to [1..53]
    if week < 1:
        week = 1
    elif week > 53:
        week = 53

    # Handle week 53 in a non-long ISO year by rolling into the next ISO year.
    if week == 53 and not is_long_year(year):
        year += 1
        week = 1

    # Monday of ISO week 1 for the (possibly adjusted) ISO year
    cdef long long ord_w1 = iso_week1_mon_ord(year)

    # Target ordinal: Monday of week 1 + (week-1)*7 + (weekday-1)
    cdef long long days = (week - 1) * 7 + (weekday - 1)

    # Convert from ordinal to Y/M/D
    return ymd_fr_ord(ord_w1 + days)

cdef inline ymd ymd_fr_day_of_year(long long year, int doy) noexcept nogil:
    """Create `struct:ymd` from Gregorian year and day-of-year `<'struct:ymd'>`.

    :param year  `<'int'>`: Gregorian year number.
    :param doy `<'int'>`: The day-of-year.
        Automatically clamped to [1..365/366] (depends on whether year leaps).
    :returns `<'struct:ymd'>`: Gregorian Y/M/D in the proleptic Gregorian calendar.

        - ymd.year  [Gregorian year number]
        - ymd.month [1..12]
        - ymd.day   [1..31]
    """
    cdef:
        int leap = is_leap_year(year) 
        int max_d = 365 + leap
        ymd out

    # Clamp days 
    if doy < 1:
        doy = 1
    elif doy > max_d:
        doy = max_d

    # Fast-path: Jan
    if doy <= 31:
        out.year, out.month, out.day = year, 1, doy
        return out

    # Fast-path: Feb
    cdef int d_before_mar = 59 + leap  # days before March 1
    if doy <= d_before_mar:
        out.year, out.month, out.day = year, 2, doy - 31
        return out

    # March-based conversion using 153-day month blocks
    # Convert to "days since March 1" (1-based -> 0-based)
    cdef int d, m_from_mar, mon, dom
    d = doy - d_before_mar - 1  # 1 => # 0..306 (0=Mar-1)

    with cython.cdivision(True):
        # Month index from March: 0..9
        m_from_mar = (5 * d + 2) // 153
        # Day (1-based)
        dom = d - (153 * m_from_mar + 2) // 5 + 1
        # month (3..12)
        mon = m_from_mar + 3

    out.year, out.month, out.day = year, mon, dom
    return out

cdef inline long long iso_week1_mon_ord(long long year) noexcept nogil:
    """Return the ordinal (1-based) of the Monday starting ISO week 1 
    of `year` under the proleptic Gregorian rules `<'int'>`.

    ISO week 1 is the week that contains January 4 (equivalently, the
    week whose Thursday lies in `year`). This returns the Gregorian
    ordinal day number (0001-01-01 = 1) of that week's Monday.

    :param year `<'int'>`: Gregorian year number.
    :returns `<'int'>`: Ordinal of the Monday starting ISO week 1.
    """
    cdef long long jan4 = ymd_to_ord(year, 1, 4)  # ordinal of Jan 4
    cdef long long wkd  = (jan4 - 1) % 7
    if wkd < 0:
        wkd += 7
    return jan4 - wkd

# . date & time
cdef inline dtm dtm_fr_us(long long value) noexcept nogil:
    """Convert microsecond ticks since the Unix epoch to `struct:dtm` (date + time).

    :param value `<'int'>`: The microsecond ticks since epoch.
    :returns `<'struct:dtm'>`: Converted date/time components.

        - dtm.year          [Gregorian year number]
        - dtm.month         [1..12]
        - dtm.day           [1..31]
        - dtm.hour          [0..23]
        - dtm.minute        [0..59]
        - dtm.second        [0..59]
        - dtm.microsecond   [0..999999]
    """
    cdef:
        long long q, r
        ymd _ymd
        dtm out

    with cython.cdivision(True):
        # Split into days and remainder (mathematical floor)
        q = value / US_DAY; r = value % US_DAY
        if r < 0:
            q -= 1; r += US_DAY

        # Date
        _ymd = ymd_fr_ord(q + EPOCH_DAY)
        out.year  = _ymd.year
        out.month = _ymd.month
        out.day   = _ymd.day

        # Time-of-day
        out.hour   = r / US_HOUR;   r %= US_HOUR
        out.minute = r / US_MINUTE; r %= US_MINUTE
        out.second = r / US_SECOND
        out.microsecond = r % US_SECOND

    return out

cdef inline dtm dtm_fr_sec(double value) except *:
    """Convert seconds (float) since the Unix epoch to `struct:dtm` (date + time).

    :param value `<'float'>`: Seconds since Unix epoch.
    :returns `<'struct:dtm'>`: Converted date/time components.

        - dtm.year          [Gregorian year number]
        - dtm.month         [1..12]
        - dtm.day           [1..31]
        - dtm.hour          [0..23]
        - dtm.minute        [0..59]
        - dtm.second        [0..59]
        - dtm.microsecond   [0..999999]
    """
    return dtm_fr_us(sec_to_us(value))

# . fractions
cdef inline int combine_absolute_ms_us(int ms, int us) noexcept nogil:
    """Combine milliseconds and microseconds into total microseconds (replacement semantics) `<'int'>`.

    :param ms `<'int'>`: Absolute milliseconds (field replacement). Negative means `no change`.
    :param us `<'int'>`: Absolute microseconds (field replacement). Negative means `no change`.
    :returns `<'int'>`: Absolute microseconds in [0, 999_999], or `-1` if both inputs are negative.

    ## Rules
    - Negative input means `no change` for that field.
    - If BOTH ms and us are negative: return `-1` (caller should keep existing value).
    - If ms >= 0:
        * ms sets the millisecond field (clamped to 0..999).
        * us, if >= 0, sets the sub-millisecond remainder (ignored above 999 via `% 1000`).
        * Thousands in `us` are ignored (ms has higher priority).
          => return ms * 1000 + (us % 1000) in [0, 999_999]
    - Else (ms < 0 and us >= 0):
        * us directly sets absolute microseconds (clamped to 0..999_999).
    """
    # Valid millisecond 
    if ms >= 0:
        # clamp 'ms' to 0..999
        ms = min(ms, 999) * 1_000
        # 'us' acts only as sub-ms remainder
        us = ms + (us % 1_000) if us > 0 else ms

    # Valid microsecond
    elif us >= 0:
        # clamp 'us' to 0..999,999
        us = min(us, 999_999)

    # Both invalid
    else:
        us = -1
        
    return us

# datetime.date ----------------------------------------------------------------------------------------
# . generate
cdef inline datetime.date date_new(int year=1, int month=1, int day=1, object dclass=None):
    """Create a new date `<'datetime.date'>`.

    :param year  `<'int'>`: Gregorian year number. Defaults to `1`.
    :param month `<'int'>`: Month [1..12]. Defaults to `1`.
    :param day   `<'int'>`: Day [1..31]. Defaults to `1`.
    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.
    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified).
    """
    # Construct date
    if dclass is not None and dclass is not datetime.date:
        try:
            return dclass(year=year, month=month, day=day)
        except Exception as err:
            raise TypeError("Cannot create date using custom 'dclass' %s, %s" % (dclass, err)) from err
    return datetime.date_new(year, month, day)

cdef inline datetime.date date_now(object tzinfo=None, object dclass=None):
    """Get today's date `<'datetime.date'>`.

    :param tzinfo `<'tzinfo/None'>`: Optional timezone. Defaults to `None`.

        - If specified, return the current date in that timezone.
        - Otherwise, return the current local date (equivalent to `datetime.date.today()`).

    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.

    :returns `<'datetime.date'>`: Today's date (or subclass if `dclass` is specified).
    """
    return date_fr_dt(dt_now(tzinfo, None), dclass)

# . type check
cdef inline bint is_date(object obj) except -1:
    """Check if an object is an instance or subclass of `datetime.date` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.
    
    ## Equivalent
    >>> isinstance(obj, datetime.date)
    """
    return datetime.PyDate_Check(obj)

cdef inline bint is_date_exact(object obj) except -1:
    """Check if an object is an exact `datetime.date` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.
    
    ## Equivalent
    >>> type(obj) is datetime.date
    """
    return datetime.PyDate_CheckExact(obj)

# . conversion: to
cdef inline str date_strformat(datetime.date date, str fmt):
    """Format a `datetime.date` with a strftime-style format `<'str'>`.

    :param date `<'datetime.date'>`: Date to format.
    :param fmt  `<'str'>`: Strftime-compatible format string.
    :returns `<'str'>`: Formatted text.

    ## Notice
    - Output decoding assumes a UTF-8 C locale.

    ## Equivalent
    >>> date.strftime(fmt)
    """
    return tm_strformat(date_to_tm(date), fmt)

cdef inline str date_isoformat(datetime.date date):
    """Format `datetime.date` to the ISO-8601 calendar string `YYYY-MM-DD` `<'str'>`.

    :param date `<'datetime.date'>`: Date to format.
    :returns `<'str'>`: ISO string in the form `YYYY-MM-DD`.
    """
    return "%04d-%02d-%02d" % (date.year, date.month, date.day)

cdef inline tm date_to_tm(datetime.date date) noexcept:
    """Convert a `datetime.date` to `<'struct:tm'>`.

    :param date `<'datetime.date'>`: Date to convert.
    :returns `<'struct:tm'>`: The corresponding 'struct:tm' representation.

        - tm.tm_sec   [0]
        - tm.tm_min   [0]
        - tm.tm_hour  [0]
        - tm.tm_mday  [1..31]
        - tm.tm_mon   [1..12]
        - tm.tm_year  [Gregorian year number]
        - tm.tm_wday  [0..6, 0=Monday]
        - tm.tm_yday  [1..366]
        - tm.tm_isdst [-1]
    """
    cdef:
        int yy = date.year
        int mm = date.month
        int dd = date.day
        tm out

    out.tm_sec = 0
    out.tm_min = 0
    out.tm_hour = 0
    out.tm_mday = dd
    out.tm_mon = mm
    out.tm_year = yy
    out.tm_wday = ymd_weekday(yy, mm, dd)
    out.tm_yday = days_bf_month(yy, mm) + dd
    out.tm_isdst = -1
    return out

cdef inline long long date_to_us(datetime.date date) noexcept:
    """Convert a `datetime.date` to microseconds since the 
    Unix epoch (UTC midnight) `<'int'>`.

    :param date `<'datetime.date'>`: Date to convert.
    :returns `<'int'>`: Microseconds from epoch to the start of `date`.
        Negative for dates before 1970-01-01.
    """
    return (date_to_ord(date) - EPOCH_DAY) * US_DAY

cdef inline double date_to_sec(datetime.date date) noexcept:
    """Convert a `datetime.date` to seconds since the 
    Unix epoch (UTC midnight) `<'float'>`.

    :param date `<'datetime.date'>`: Date to convert.
    :returns `<'float'>`: Seconds from epoch to the start of `date`.
        Negative for dates before 1970-01-01.
    """
    return <double> ((date_to_ord(date) - EPOCH_DAY) * SS_DAY)

cdef inline long long date_to_ord(datetime.date date) noexcept:
    """Convert a `datetime.date` to the Gregorian ordinal (0001-01-01 = 1) `<'int'>`.

    :param date `<'datetime.date'>`: Date to convert.
    :returns `<'int'>`: Ordinal day number in the proleptic Gregorian calendar.
    """
    return ymd_to_ord(date.year, date.month, date.day)

# . conversion: from
cdef inline datetime.date date_fr_us(long long value, object dclass=None):
    """Create date from microseconds since the Unix epoch `<'datetime.date'>`.

    :param value `<'int'>`: Microseconds since epoch.
    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.
    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified).
    """
    cdef ymd _ymd = ymd_fr_us(value)
    return date_new(_ymd.year, _ymd.month, _ymd.day, dclass)

cdef inline datetime.date date_fr_sec(double value, object dclass=None):
    """Create date from seconds since the Unix epoch `<'datetime.date'>`.

    :param value `<'float'>`: Seconds since epoch.
    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.
    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified).
    """
    cdef ymd _ymd = ymd_fr_sec(value)
    return date_new(_ymd.year, _ymd.month, _ymd.day, dclass)

cdef inline datetime.date date_fr_ord(int value, object dclass=None):
    """Create date from a Gregorian ordinal (0001-01-01 = 1) `<'datetime.date'>`.

    :param value `<'int'>`: Gregorian ordinal day.
    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.
    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified).
    """
    cdef ymd _ymd = ymd_fr_ord(value)
    return date_new(_ymd.year, _ymd.month, _ymd.day, dclass)

cdef inline datetime.date date_fr_ts(double value, object dclass=None):
    """Create date from a POSIX timestamp in **local** time `<'datetime.date'>`.

    :param value `<'float'>`: POSIX timestamp in **local** time.
    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.
    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified) 
        in the system **local** time zone.
    """
    cdef datetime.date date = datetime.date_from_timestamp(value)
    return date if dclass is None else date_fr_date(date, dclass)

cdef inline datetime.date date_fr_date(datetime.date date, object dclass=None):
    """Create date from another date (or subclass) `<'datetime.date'>`.
    
    :param date `<'datetime.date'>`: The source date (including subclasses).
    :param dclass `<'type[datetime.date]/None'>`: Target date class. Defaults to `None`.
        If `None` set to python's built-in `datetime.date`.
        If `date` is already of type `dclass`, returns `date` directly.
    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified) 
        with the same date fields.
    """
    if dclass is None:
        dclass = datetime.date
    if dclass is type(date):
        return date
    return date_new(date.year, date.month, date.day, dclass)

cdef inline datetime.date date_fr_dt(datetime.datetime dt, object dclass=None):
    """Create date from a datetime (include subclass) `<'datetime.date'>`.
    
    :param dt `<'datetime.datetime'>`: Datetime to extract the date from (including subclasses).
    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.
    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified) 
        with the same date fields.
    """
    return date_new(dt.year, dt.month, dt.day, dclass)

# . manipulation
cdef inline datetime.date date_add_delta(datetime.date date,
    int years=0, int quarters=0, int months=0, int weeks=0, long long days=0, long long hours=0,
    long long minutes=0, long long seconds=0, long long milliseconds=0, long long microseconds=0,
    int year=-1, int month=-1, int day=-1, int weekday=-1, int hour=-1, int minute=-1,
    int second=-1, int millisecond=-1, int microsecond=-1, object dclass=None
):
    """Add relative and absolute deltas to date `<'datetime.date'>`.

    ## Absolute Deltas (Replace specified fields)

    :param year `<'int'>`: Absolute year. Defaults to `-1` (no change).
    :param month `<'int'>`: Absolute month. Defaults to `-1` (no change).
    :param day `<'int'>`: Absolute day. Defaults to `-1` (no change).
    :param weekday `<'int'>`: Absolute weekday (0=Mon...6=Sun). Defaults to `-1` (no change).
    :param hour `<'int'>`: Absolute hour. Defaults to `-1` (no change).
    :param minute `<'int'>`: Absolute minute. Defaults to `-1` (no change).
    :param second `<'int'>`: Absolute second. Defaults to `-1` (no change).
    :param millisecond `<'int'>`: Absolute millisecond. Defaults to `-1` (no change).
        Overrides `microsecond` milliseoncd part if both are provided.
    :param microsecond `<'int'>`: Absolute microsecond. Defaults to `-1` (no change).

    ## Relative Deltas (Add to specified fields)

    :param years `<'int'>`: Relative years. Defaults to `0`.
    :param quarters `<'int'>`: Relative quarters (3 months). Defaults to `0`.
    :param months `<'int'>`: Relative months. Defaults to `0`.
    :param weeks `<'int'>`: Relative weeks (7 days). Defaults to `0`.
    :param days `<'int'>`: Relative days. Defaults to `0`.
    :param hours `<'int'>`: Relative hours. Defaults to `0`.
    :param minutes `<'int'>`: Relative minutes. Defaults to `0`.
    :param seconds `<'int'>`: Relative seconds. Defaults to `0`.
    :param milliseconds `<'int'>`: Relative milliseconds (1,000 us). Defaults to `0`.
    :param microseconds `<'int'>`: Relative microseconds. Defaults to `0`.

    ## Date Class
    
    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.

    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified)
        after applying the specified deltas.
    """
    # Date fields
    # ----------------------------------------------------
    cdef:
        int o_yy = date.year
        int yy = (year if year > 0 else o_yy) + years
        int o_mm = date.month
        int mm = (month if month > 0 else o_mm) + months + (quarters * 3)
        int o_dd = date.day
        int dd = (day if day > 0 else o_dd)
        long long q, r
    # . normalize month
    if mm != o_mm:
        mm -= 1
        with cython.cdivision(True):
            q = mm / 12; r = mm % 12
        if r < 0:
            q -= 1; r += 12
        yy += q; mm = r + 1
    # . relative days
    cdef long long rel_days = days + (weeks * 7)

    # Time fields
    # ----------------------------------------------------
    cdef long long us = combine_absolute_ms_us(millisecond, microsecond)
    us = (
        (us if us >= 0 else 0) + milliseconds * US_MILLISECOND + microseconds +
        ((second if second >= 0 else 0) + seconds) * US_SECOND +
        ((minute if minute >= 0 else 0) + minutes) * US_MINUTE +
        ((hour if hour >= 0 else 0) + hours)       * US_HOUR
    )
    # . normalize time to days
    if us != 0:
        with cython.cdivision(True):
            q = us / US_DAY; r = us % US_DAY
        if r < 0:
            q -= 1
        rel_days += q

    # Handle day delta
    # ----------------------------------------------------
    cdef long long old_ord, new_ord
    cdef ymd _ymd
    if rel_days != 0 or weekday >= 0:
        old_ord = ymd_to_ord(yy, mm, dd)
        new_ord = old_ord + rel_days
        # . adjust to weekday if needed
        if weekday >= 0:
            with cython.cdivision(True):
                r = (new_ord + 6) % 7
                if r < 0:
                    r += 7
            new_ord += min(weekday, 6) - r  # weekday clamped to Sun=6
        if old_ord != new_ord:
            _ymd = ymd_fr_ord(new_ord)
            yy, mm, dd = _ymd.year, _ymd.month, _ymd.day

    # Compare new dates
    if yy == o_yy and mm == o_mm and dd == o_dd:
        return date_fr_date(date, dclass)

    # Clamp day to valid days in month
    if dd > 28:
        dd = min(dd, days_in_month(yy, mm))

    # New date
    return date_new(yy, mm, dd, dclass)

# datetime.datetime ------------------------------------------------------------------------------------
# . generate
cdef inline datetime.datetime dt_new(
    int year=1, int month=1, int day=1, int hour=0, int minute=0, int second=0, 
    int microsecond=0, object tzinfo=None, int fold=0, object dtclass=None,
):
    """Create a new datetime `<'datetime.datetime'>`.

    :param year  `<'int'>`: Gregorian year number. Defaults to `1`.
    :param month `<'int'>`: Month [1..12]. Defaults to `1`.
    :param day   `<'int'>`: Day [1..31]. Defaults to `1`.
    :param hour `<'int'>`: Hour [0..23]. Defaults to `0`.
    :param minute `<'int'>`: Minute [0..59]. Defaults to `0`.
    :param second `<'int'>`: Second [0..59]. Defaults to `0`.
    :param microsecond `<'int'>`: Microsecond [0..999999]. Defaults to `0`.
    :param tzinfo `<'tzinfo/None'>`: Optional timezone. Defaults to `None`.
    :param fold `<'int'>`: Optional fold flag for ambiguous times (0 or 1). Defaults to `0`.
        Only relevant if `tzinfo` is not `None`, and values other than `1` are treated as `0`.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).
    """
    # Normalize fold (0/1 only)
    fold = 0 if tzinfo is None else fold == 1

    # Construct datetime
    if dtclass is not None and dtclass is not datetime.datetime:
        try:
            return dtclass(
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=second,
                microsecond=microsecond,
                tzinfo=tzinfo,
                fold=fold,
            )
        except Exception as err:
            raise TypeError("Cannot create datetime using custom 'dtclass' %s, %s" % (dtclass, err)) from err
    return datetime.datetime_new(year, month, day, hour, minute, second, microsecond, tzinfo, fold)

cdef inline datetime.datetime dt_now(object tzinfo=None, object dtclass=None):
    """Get the current datetime `<'datetime.datetime'>`.

    :param tzinfo `<'tzinfo/None'>`: Optional timezone. Defaults to `None`.

        - If specified, return an aware datetime in that timezone.
        - Otherwise, return a naive local-time datetime.

    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.

    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).

    ## Equivalent
    >>> datetime.datetime.now(tzinfo)
    """
    cdef double tic = unix_time()
    cdef datetime.datetime dt = datetime.datetime_from_timestamp(tic, tzinfo)
    return dt if dtclass is None else dt_fr_dt(dt, dtclass)
    
# . type check
cdef inline bint is_dt(object obj) except -1:
    """Check if an object is an instance or subclass of `datetime.datetime` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> isinstance(obj, datetime.datetime)
    """
    return datetime.PyDateTime_Check(obj)

cdef inline bint is_dt_exact(object obj) except -1:
    """Check if an object is an exact `datetime.datetime` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> type(obj) is datetime.datetime
    """
    return datetime.PyDateTime_CheckExact(obj)

# . conversion: to
cdef inline long long dt_local_mktime(datetime.datetime dt) except *:
    """Interpret a naive `datetime.datetime` as local POSIX timestamp `<'int'>`.

    Mirrors CPython's naive branch (uses the **system** local timezone database),
    handling ambiguous/nonexistent local times via `dt.fold`. Any `tzinfo` attached
    to `dt` is ignored, because the function is designed for naive datetime only.

    :param dt `<'datetime.datetime'>`: Timezone-naive datetime.
    :returns `<'int'>`: Local POSIX timestamp in seconds, without fractions.
    """
    cdef:
        long long ord, t, hh, mi, ss
        long long adj1, adj2, u1, u2, t1, t2

    # Seconds since epoch from local wall fields (no timezone adjustment here)
    ord = dt_to_ord(dt, False)            # <-- make sure this is the right helper
    hh, mi, ss = dt.hour, dt.minute, dt.second
    t = (ord - EPOCH_DAY) * SS_DAY + hh * SS_HOUR + mi * SS_MINUTE + ss

    # local(u) := ts_localtime(u)
    adj1 = ts_localtime(t) - t
    u1 = t - adj1
    t1 = ts_localtime(u1)

    if t1 == t:
        # We found one solution, but it may not be the one we need.
        # Look for an earlier solution (if `fold` is 0), or a later
        # one (if `fold` is 1).
        u2  = u1 - SS_DAY if dt.fold == 0 else u1 + SS_DAY
        adj2 = ts_localtime(u2) - u2
        if adj1 == adj2:
            return u1
    else:
        adj2 = t1 - u1

    # Final attempt with the other offset
    u2 = t - adj2
    t2 = ts_localtime(u2)
    if t2 == t:
        return u2
    if t1 == t:
        return u1

    # We have found both offsets adj1 and adj2,
    # but neither t - adj1 nor t - adj2 is the
    # solution. This means t is in the gap.
    return max(u1, u2) if dt.fold == 0 else min(u1, u2)

cdef inline str dt_strformat(datetime.datetime dt, str fmt):
    """Format a `datetime.datetime` with a strftime-style format `<'str'>`.

    :param dt `<'datetime.datetime'>`: Datetime to format.
    :param fmt `<'str'>`: Strftime-compatible format string.
    :returns `<'str'>`: Formatted text.

    ## Equivalent
    >>> dt.strftime(fmt)
    """
    if fmt is None:
        raise ValueError("dt_strformat: 'fmt' cannot be None")
    cdef:
        list parts = []
        Py_ssize_t size = str_len(fmt)
        Py_ssize_t i = 0
        str frepl = None
        str zrepl = None
        str Zrepl = None
        str s_tmp = None
        Py_UCS4 ch

    # Scan once, expand %f/%z/%Z, pass through others
    while i < size:
        ch = str_read(fmt, i)
        i += 1

        # Format identifier: '%'
        if ch == "%":
            if i < size:
                ch = str_read(fmt, i)
                i += 1
                # . fraction
                if ch == "f":
                    if frepl is None:
                        frepl = "%06d" % dt.microsecond
                    parts.append(frepl)
                # . utc offset
                elif ch == "z":
                    if zrepl is None:
                        s_tmp = tz_utcformat(dt.tzinfo, dt)
                        zrepl = "" if s_tmp is None else s_tmp
                    parts.append(zrepl)
                # . timezone name
                elif ch == "Z":
                    if Zrepl is None:
                        s_tmp = dt_tzname(dt)
                        Zrepl = "" if s_tmp is None else s_tmp
                    parts.append(Zrepl)
                # . Pass through unknown/regular specifiers and %%
                else:
                    parts.append("%")
                    parts.append(str_chr(ch))

            else:
                # Trailing '%': treat as literal '%'
                parts.append("%")

        # Normal character
        else:
            parts.append(str_chr(ch))

    # Format to string
    return tm_strformat(dt_to_tm(dt, False), "".join(parts))

cdef inline str dt_isoformat(datetime.datetime dt, str sep="T", bint utc=False):
    """Convert a `datetime.datetime` to ISO string `<'str'>`.

    :param dt `<'datetime.datetime'>`: The datetime to convert.
    :param sep `<'str'>`: Date/time separator. Defaults to `"T"`.
    :param utc <'bool'>: Whether to append a UTC offset. Defaults to False.
        
        - If False or `dt` is timezone-naive, UTC offset is ignored.
        - If True and `dt` is timezone-aware, append UTC offset (e.g., +0530).

    :returns `<'str'>`: Formatted text.
    """
    if sep is None:
        raise ValueError("dt_isoformat: 'sep' cannot be None.")
    cdef:
        int us = dt.microsecond
        str s_utc = tz_utcformat(dt.tzinfo, dt) if utc else None

    if us == 0:
        if s_utc is None:
            return "%04d-%02d-%02d%s%02d:%02d:%02d" % (
                dt.year, dt.month, dt.day, sep,
                dt.hour, dt.minute, dt.second,
            )
        return "%04d-%02d-%02d%s%02d:%02d:%02d%s" % (
            dt.year, dt.month, dt.day, sep,
            dt.hour, dt.minute, dt.second, s_utc,
        )
    else:
        if s_utc is None:
            return "%04d-%02d-%02d%s%02d:%02d:%02d.%06d" % (
                dt.year, dt.month, dt.day, sep,
                dt.hour, dt.minute, dt.second, us,
            )
        return "%04d-%02d-%02d%s%02d:%02d:%02d.%06d%s" % (
            dt.year, dt.month, dt.day, sep,
            dt.hour, dt.minute, dt.second, us, s_utc,
        ) 

cdef inline tm dt_to_tm(datetime.datetime dt, bint utc=False) except *:
    """Convert a `datetime.datetime` to 'struct:tm' `<'struct:tm'>`.

    :param dt `<'datetime.datetime'>`: Datetime to convert.
    :param utc `<'bool'>`: Whether to subtract UTC offset before conversion. Defaults to `False`.

        - If True and `dt` is timezone-aware, subtract `utcoffset(dt)`
          (convert to UTC) and set `tm_isdst = 0`.
        - If True and `dt` is naive, treat it as UTC (no offset), with
          `tm_isdst = 0`.
        - If False, interpret fields as local wall time; for aware 
          datetimes set `tm_isdst > 0` iff `dst(dt)` is non-zero, 
          else 0; for naive set -1.

    :returns `<'struct:tm'>`: Fields in project conventions:

        - tm.tm_sec   [0..59]
        - tm.tm_min   [0..59]
        - tm.tm_hour  [0..23]
        - tm.tm_mday  [1..31]
        - tm.tm_mon   [1..12]
        - tm.tm_year  [Gregorian year number]
        - tm.tm_wday  [0..6, 0=Monday]
        - tm.tm_yday  [1..366]
        - tm.tm_isdst [-1..1]
    """
    cdef: 
        int yy = dt.year
        int mm = dt.month
        int dd = dt.day
        int hh = dt.hour
        int mi = dt.minute
        int ss = dt.second
        object tzinfo = dt.tzinfo
        int isdst
        datetime.timedelta off
        tm out

    # Naive: utc=True -> treat as UTC
    if tzinfo is None:
        isdst = 0 if utc else -1

    # Aware -> UTC: subtract utcoffset
    elif utc:
        off = tz_utcoffset(tzinfo, dt)
        if off is not None:
            dt = dt_add(dt, -off.day, -off.second, -off.microsecond, None)
            yy, mm, dd = dt.year, dt.month, dt.day
            hh, mi, ss = dt.hour, dt.minute, dt.second
        isdst = 0
    
    # Aware, local wall time: set DST flag from tzinfo.dst(dt)
    else:
        off = tz_dst(tzinfo, dt)
        if off is not None:
            isdst = 1 if td_to_us(off) else 0
        else:
            isdst = -1

    out.tm_sec, out.tm_min, out.tm_hour = ss, mi, hh
    out.tm_mday, out.tm_mon, out.tm_year = dd, mm, yy
    out.tm_wday = ymd_weekday(yy, mm, dd)
    out.tm_yday = days_bf_month(yy, mm) + dd
    out.tm_isdst = isdst
    return out

cdef inline str dt_to_ctime(datetime.datetime dt):
    """Convert a datetime to a C `ctime-style` string `<'str'>`.

    :param dt `<'datetime.datetime'>`: Datetime to convert.
    :returns `<'str'>`: C time string.

    ## Example
    >>> dt_to_ctime(datetime.datetime(2024, 10, 1, 8, 19, 5))
    >>> 'Tue Oct  1 08:19:05 2024'
    """
    cdef:
        int yy = dt.year
        int mm = dt.month
        int dd = dt.day
        int hh = dt.hour
        int mi = dt.minute
        int ss = dt.second
        int wkd = ymd_weekday(yy, mm, dd)
        str s_wkd, s_mm
    
    # Weekday
    if wkd == 0:
        s_wkd = "Mon"
    elif wkd == 1:
        s_wkd = "Tue"
    elif wkd == 2:
        s_wkd = "Wed"
    elif wkd == 3:
        s_wkd = "Thu"
    elif wkd == 4:
        s_wkd = "Fri"
    elif wkd == 5:
        s_wkd = "Sat"
    else:
        s_wkd = "Sun"

    # Month
    if mm == 1:
        s_mm = "Jan"
    elif mm == 2:
        s_mm = "Feb"
    elif mm == 3:
        s_mm = "Mar"
    elif mm == 4:
        s_mm = "Apr"
    elif mm == 5:
        s_mm = "May"
    elif mm == 6:
        s_mm = "Jun"
    elif mm == 7:
        s_mm = "Jul"
    elif mm == 8:
        s_mm = "Aug"
    elif mm == 9:
        s_mm = "Sep"
    elif mm == 10:
        s_mm = "Oct"
    elif mm == 11:
        s_mm = "Nov"
    else:
        s_mm = "Dec"

    # Fromat
    return "%s %s %2d %02d:%02d:%02d %04d" % (s_wkd, s_mm, dd, hh, mi, ss, yy)
   
cdef inline long long dt_to_us(datetime.datetime dt, bint utc=False) except *:
    """Convert a `datetime.datetime` to microseconds since the Unix epoch `<'int'>`.

    :param dt `<'datetime.datetime'>`: Datetime to convert (naive or aware).
    :param utc `<'bool'>`: Whether to subtract UTC offset before conversion. Defaults to `False`.

        - If False or `dt` is naive, return the total microseconds
          without adjustment (UTC offset ignored).
        - If True and `dt` is timezone-aware, subtract its UTC offset
          (i.e., convert to UTC) from the total microseconds.

    :returns `<'int'>`: Microseconds since the Unix epoch.
    """
    # Compute microseconds
    cdef:
        long long ord = dt_to_ord(dt, False)
        long long us  = (
            (ord - EPOCH_DAY)           * US_DAY +
            (<long long> dt.hour)       * US_HOUR +
            (<long long> dt.minute)     * US_MINUTE +
            (<long long> dt.second)     * US_SECOND +
            (<long long> dt.microsecond)
        )
    if not utc:
        return us

    # Aware-only: get UTC offset; None means treat as naive
    cdef datetime.timedelta off = dt_utcoffset(dt)
    return us if off is None else us - td_to_us(off)

cdef inline double dt_to_sec(datetime.datetime dt, bint utc=False) except *:
    """Convert a `datetime.datetime` to seconds since the Unix epoch `<'float'>`.

    :param dt `<'datetime.datetime'>`: Datetime to convert (naive or aware).
    :param utc `<'bool'>`: Whether to subtract UTC offset before conversion. Defaults to `False`.

        - If False or `dt` is naive, return the total seconds
          without adjustment (UTC offset ignored).
        - If True and `dt` is timezone-aware, subtract its UTC offset
          (i.e., convert to UTC) from the total seconds.

    :returns `<'float'>`: Seconds since the Unix epoch.
    """
    cdef long long us = dt_to_us(dt, utc)
    return <double> us * 1e-6

cdef inline long long dt_to_ord(datetime.datetime dt, bint utc=False) except -1:
    """Convert a `datetime.datetime` to its Gregorian ordinal day `<'int'>`.

    :param dt `<'datetime.datetime'>`: Datetime to convert (naive or aware).
    :param utc `<'bool'>`: Whether to subtract UTC offset before conversion. Defaults to `False`.

        - If False or `dt` is naive, return the ordinal without adjustment
          (UTC offset ignored).
        - If True and `dt` is timezone-aware, adjust by the UTC offset and
          decrement/increment the ordinal by 1 when the UTC wall clock falls
          before 00:00 or on/after 24:00.

    :returns `<'int'>`: Gregorian ordinal day (0001-01-01 == 1).
    """
    # Compute ordinal from Y/M/D
    cdef long long ord = ymd_to_ord(dt.year, dt.month, dt.day)
    if not utc:
        return ord

    # Aware-only: get UTC offset; None means treat as naive
    cdef datetime.timedelta off = dt_utcoffset(dt)
    if off is None:
        return ord

    # Compare at microsecond precision: local TOD vs offset
    # us in [0, 86_400_000_000) 
    cdef long long us = (
        (<long long>dt.hour)            * US_HOUR +
        (<long long>dt.minute)          * US_MINUTE +
        (<long long>dt.second)          * US_SECOND +
        (<long long>dt.microsecond)
    )

    # us_off in (-86_400_000_000, 86_400_000_000)
    cdef long long us_off = (
        (<long long> off.day)        * US_DAY +
        (<long long> off.second)     * US_SECOND +
        (<long long> off.microsecond)
    )

    # Adjust ordinal according to the offset
    cdef long long adj = us - us_off
    if adj < 0:
        ord -= 1          # UTC is "earlier" calendar day
    elif adj >= US_DAY:
        ord += 1          # UTC is "later" calendar day
    return ord

cdef inline double dt_to_ts(datetime.datetime dt) except *:
    """Convert a `datetime.datetime` to POSIX timestamp (seconds since the Unix epoch) `<'float'>`.

    :param dt `<'datetime.datetime'>`: Datetime to convert (naive or aware).
    
        - Naive: interpret as local time (like CPython: mktime-style + microseconds)
        - Aware: subtract utcoffset at that wall time.

    :returns `<'float'>`: POSIX timestamp in seconds.

    ## Equivalent
    >>> dt.timestamp()
    """
    # Timezone-aware
    if dt.tzinfo is not None:
        return dt_to_sec(dt, True)

    # Timezone-naive: interpret as local time
    cdef double t = <double> dt_local_mktime(dt)
    return t + (<double> dt.microsecond) * 1e-6

cdef inline long long dt_as_epoch(datetime.datetime dt, str as_unit, bint utc=False) except *:
    """Convert a `datetime.datetime` to an integer count since the Unix epoch `<'int'>`,

    :param dt `<'datetime.datetime'>`: Datetime to convert (naive or aware).
    :param as_unit `<'str'>`: Output unit, supports:

        - 'Y'  (years since 1970)         — calendar-based index
        - 'Q'  (quarters since 1970Q1)    — calendar-based index
        - 'M'  (months since 1970-01)     — calendar-based index
        - 'W'  (weeks since 1970-01-01)   — floor division by 7 days
        - 'D'  (days since 1970-01-01)
        - 'h'  (hours since epoch)
        - 'm'  (minutes since epoch)
        - 's'  (seconds since epoch)
        - 'ms' (milliseconds since epoch)
        - 'us' (microseconds since epoch)

    :param utc `<'bool'>`: Whether to subtract UTC offset before conversion. Defaults to `False`.

        - If False or `dt` is naive, interpret the fields without 
          adjustment (UTC offset ignored).
        - If True and `dt` is timezone-aware, subtract its UTC offset
          (i.e., convert to UTC) before converting.

    :returns `<'int'>`: Epoch count in the requested unit.

    ## Semantics
    - Calendar units ('Y','Q','M') are computed as simple calendar indexes relative
      to 1970-01-01 (no leap/length normalization).
    - 'W' uses mathematical floor division of days by 7 (weeks start aligned to the epoch).
    - Subsecond units:

        * 'ms' truncates microseconds toward zero (i.e., floor for non-negative datetimes).
        * 'us' reproduces the exact microsecond field.
    """
    if as_unit is None:
        _raise_invalid_nptime_str_unit_error(as_unit)
    cdef:
        datetime.timedelta off
        Py_ssize_t size = str_len(as_unit)
        long long n, q, r, hh, mi, ss, us
        Py_UCS4 ch

    # Optional UTC adjustment
    if utc:
        off = tz_utcoffset(dt.tzinfo, dt)
        if off is not None:
            dt = dt_add(dt, -off.day, -off.second, -off.microsecond, None)

    # Unit: 's', 'm', 'h', 'D', 'W', 'M', 'Q', 'Y'
    if size == 1:
        ch = str_read(as_unit, 0)
        # . year
        if ch == "Y":
            return dt.year - EPOCH_YEAR
        # . quarter
        if ch == "Q":
            return (dt.year - EPOCH_YEAR) * 4 + quarter_of_month(dt.month) - 1
        # . month
        if ch == "M":
            return (dt.year - EPOCH_YEAR) * 12 + dt.month - 1
        # . week
        n = dt_to_ord(dt, False) - EPOCH_DAY
        if ch == "W":
            with cython.cdivision(True):
                q = n / 7; r = n % 7
                if r < 0:
                    q -= 1
            return q
        # . day
        if ch == "D":
            return n
        # . hour
        hh = dt.hour
        if ch == "h":
            return n * 24 + hh
        # . minute
        mi = dt.minute
        if ch == "m":
            return n * 1_440 + hh * 60 + mi
        # . second
        ss = dt.second
        if ch == "s":
            return n * SS_DAY + hh * SS_HOUR + mi * SS_MINUTE + ss
        
    # Unit: 'ns', 'us', 'ms'
    elif size == 2 and str_read(as_unit, 1) == "s":
        ch = str_read(as_unit, 0)
        # . millisecond
        n = dt_to_ord(dt, False) - EPOCH_DAY
        hh, mi, ss, us = dt.hour, dt.minute, dt.second, dt.microsecond
        if ch == "m":
            with cython.cdivision(True):
                us = us / 1_000
            return n * MS_DAY + hh * MS_HOUR + mi * MS_MINUTE + ss * MS_SECOND + us
        # . microsecond
        if ch == "u":
            return n * US_DAY + hh * US_HOUR + mi * US_MINUTE + ss * US_SECOND + us

    # Unit: 'min' for pandas compatibility
    elif size == 3 and as_unit == "min":
        n = dt_to_ord(dt, False) - EPOCH_DAY
        hh, mi = dt.hour, dt.minute
        return n * 1_440 + hh * 60 + mi

    # Unsupported unit
    _raise_invalid_nptime_str_unit_error(as_unit)

cdef inline long long dt_as_epoch_W_iso(datetime.datetime dt, int weekday, bint utc=False) except *:
    """Convert a `datetime.datetime` to an integer count of whole 
    weeks since the Unix epoch, where weeks are aligned to a chosen 
    ISO weekday (1=Mon..7=Sun) `<'int'>`.

    :param dt `<'datetime.datetime'>`: Datetime to convert (naive or aware).
    :param weekday `<'int'>`: ISO weekday that defines the *start* of each week bucket
        (1=Mon .. 7=Sun). Values outside 1..7 are clamped.
    :param utc `<'bool'>`: Whether to subtract UTC offset before conversion. Defaults to `False`.

        - If False or `dt` is naive, interpret the fields without 
          adjustment (UTC offset ignored).
        - If True and `dt` is timezone-aware, subtract its UTC offset
          (i.e., convert to UTC) before converting.
    
    :returns `<'int'>`: Number of completed weeks since 1970-01-01 with the given alignment.

    ## Notice
    - 1970-01-01 is Thursday (ISO=4). When `weekday=4`, this matches the default 
      Thursday alignment. When `weekday=1`, weeks are Monday-aligned, etc.
    - Uses mathematical floor division so dates before the epoch are handled correctly.
    """
    cdef long long adj = 4 - min(max(weekday, 1), 7)
    cdef long long n = dt_to_ord(dt, utc) - EPOCH_DAY + adj
    cdef long long q, r
    with cython.cdivision(True):
        q = n / 7; r = n % 7
        if r < 0:
            q -= 1
    return q

# . conversion: from
cdef inline datetime.datetime dt_fr_us(long long value, object tzinfo=None, object dtclass=None):
    """Create datetime from microseconds since the Unix epoch `<'datetime.datetime'>`.

    :param value `<'int'>`: Microseconds since epoch.
    :param tzinfo `<'tzinfo/None'>`: Optional timezone to **attach**. Defaults to `None`.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).
    """
    cdef dtm _dtm = dtm_fr_us(value)
    return dt_new(
        _dtm.year, _dtm.month, _dtm.day, _dtm.hour, _dtm.minute,
        _dtm.second, _dtm.microsecond, tzinfo, 0, dtclass,
    )

cdef inline datetime.datetime dt_fr_sec(double value, object tzinfo=None, object dtclass=None):
    """Create datetime from seconds since the Unix epoch `<'datetime.datetime'>`.

    :param value `<'float'>`: Seconds since epoch.
    :param tzinfo `<'tzinfo/None'>`: Optional timezone to **attach**. Defaults to `None`.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).
    """
    return dt_fr_us(sec_to_us(value), tzinfo, dtclass)

cdef inline datetime.datetime dt_fr_ord(int value, object tzinfo=None, object dtclass=None):
    """Create datetime from a Gregorian ordinal (0001-01-01 = 1) `<'datetime.datetime'>`.

    :param value `<'int'>`: Gregorian ordinal day.
    :param tzinfo `<'tzinfo/None'>`: Optional timezone to **attach**. Defaults to `None`.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).
    """
    cdef ymd _ymd = ymd_fr_ord(value)
    return dt_new(_ymd.year, _ymd.month, _ymd.day, 0, 0, 0, 0, tzinfo, 0, dtclass)

cdef inline datetime.datetime dt_fr_ts(double value, object tzinfo=None, object dtclass=None):
    """Create datetime from a POSIX timestamp with optional timezone `<'datetime.datetime'>`.

    :param value `<'float'>`: POSIX timestamp.
    :param tzinfo `<'tzinfo/None'>`: Optional timezone. Defaults to `None`.

        - If specified, return an aware datetime in that timezone.
        - Otherwise, return a naive local-time datetime.

    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).
    
    ## Equivalent
    >>> datetime.datetime.fromtimestamp(value, tzinfo)
    """
    cdef datetime.datetime dt = datetime.datetime_from_timestamp(value, tzinfo)
    return dt if dtclass is None else dt_fr_dt(dt, dtclass)

cdef inline datetime.datetime dt_fr_date(datetime.date date, object tzinfo=None, object dtclass=None):
    """Create datetime from date (include subclass) `<'datetime.datetime'>`.

    :param date `<'datetime.date'>`: The source date (including subclasses).
    :param tzinfo `<'tzinfo/None'>`: Optional timezone to **attach**. Defaults to `None`.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified)
        with the same date fields (time fields are set to 0).
    """
    return dt_new(date.year, date.month, date.day, 0, 0, 0, 0, tzinfo, 0, dtclass)

cdef inline datetime.datetime dt_fr_dt(datetime.datetime dt, object dtclass=None):
    """Create datetime from another datetime (include subclass) `<'datetime.datetime'>`.

    :param dt `<'datetime.datetime'>`: The source datetime (including subclasses).
    :param dtclass `<'type[datetime.datetime]/None'>`: Target datetime class. Defaults to `None`.
        If `None` set to python's built-in `datetime.datetime`.
        If `dt` is already of type `dtclass`, returns `dt` directly.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified)
        with the same fields and tzinfo.
    """
    if dtclass is None:
        dtclass = datetime.datetime
    if dtclass is type(dt):
        return dt
    return dt_new(
        dt.year, dt.month, dt.day, dt.hour, dt.minute,
        dt.second, dt.microsecond, dt.tzinfo, dt.fold, dtclass,
    )

cdef inline datetime.datetime dt_fr_time(datetime.time time, object dtclass=None):
    """Create datetime from time (include subclass) `<'datetime.datetime'>`.

    :param time `<'datetime.time'>`: The source time (including subclasses).
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified)
        with the same time fields and tzinfo (date fields are set to 1970-01-01).
    """
    return dt_new(
        1970, 1, 1, time.hour, time.minute, time.second, 
        time.microsecond, time.tzinfo, time.fold, dtclass,
    )

cdef inline datetime.datetime dt_combine(datetime.date date=None, datetime.time time=None, object tzinfo=None, object dtclass=None):
    """Create a `datetime.datetime` by combining a date and a time `<'datetime.datetime'>`.

    :param date `<'datetime.date/None'>`: The source date (including subclasses). Defaults to `None`.
        If None, uses today's **local** date.
    :param time `<'datetime.time/None'>`: The source time (including subclasses). Defaults to `None`.
        If None, uses 00:00:00.000000.
    :param tzinfo `<'tzinfo/None'>`: Optional timezone to **attach**. Defaults to `None`.
        If specifed, overrides `time.tzinfo` (if any).
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).
    """
    cdef int yy, mm, dd, hh, mi, ss, us, fold
    # Date
    if date is None:
        date = date_now(None, None)
    yy, mm, dd = date.year, date.month, date.day

    # Time
    if time is None:
        hh = mi = ss = us = fold = 0
    else:
        hh, mi, ss = time.hour, time.minute, time.second
        us, fold   = time.microsecond, time.fold
        if tzinfo is None:
            tzinfo = time.tzinfo

    # New datetime
    return dt_new(yy, mm, dd, hh, mi, ss, us, tzinfo, fold, dtclass)

# . manipulation
cdef inline datetime.datetime dt_add(datetime.datetime dt, int days=0, int seconds=0, int microseconds=0, object dtclass=None):
    """Add a day/second/microsecond delta to a `datetime.datetime` `<'datetime.datetime'>`.

    This performs **wall-clock arithmetic** like `dt + datetime.timedelta(...)`:
    - No timezone normalization is applied; `tzinfo` and `fold` are preserved.
    - For aware datetimes, the local clock fields are shifted by the delta (even
    across DST boundaries) without converting to UTC.

    :param dt `<'datetime.datetime'>`: Base datetime (naive or aware).
    :param days `<'int'>`: Days to add (can be negative). Defaults to 0.
    :param seconds `<'int'>`: Seconds to add (can be negative). Defaults to 0.
    :param microseconds `<'int'>`: Microseconds to add (can be negative). Defaults to 0.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).

    ## Equivalent
    >>> dt + datetime.timedelta(days, seconds, microseconds)
    """
    # Fast-path: days only — preserve time fields and bump ordinal
    cdef ymd _ymd
    if seconds == 0 and microseconds == 0:
        # Early exit
        if days == 0:
            return dt_fr_dt(dt, dtclass)

        # Compute new ordinal, then back to Y/M/D
        _ymd = ymd_fr_ord(dt_to_ord(dt, False) + days)

        # New datetime
        return dt_new(
            _ymd.year, _ymd.month, _ymd.day, dt.hour, dt.minute,
            dt.second, dt.microsecond, dt.tzinfo, dt.fold, dtclass,
        )

    # General path: add delta in microseconds
    cdef long long us = dt_to_us(dt, False)
    us += (
        (<long long> days)          * US_DAY +
        (<long long> seconds)       * US_SECOND +
        (<long long> microseconds)
    )

    # Compute new date & time
    cdef dtm _dtm = dtm_fr_us(us)

    # New datetime
    return dt_new(
        _dtm.year, _dtm.month, _dtm.day, _dtm.hour, _dtm.minute,
        _dtm.second, _dtm.microsecond, dt.tzinfo, dt.fold, dtclass,
    )
    
cdef inline datetime.datetime dt_add_delta(datetime.datetime dt,
    int years=0, int quarters=0, int months=0, int weeks=0, long long days=0, long long hours=0,
    long long minutes=0, long long seconds=0, long long milliseconds=0, long long microseconds=0,
    int year=-1, int month=-1, int day=-1, int weekday=-1, int hour=-1, int minute=-1,
    int second=-1, int millisecond=-1, int microsecond=-1, object dtclass=None
):
    """Add relative and absolute deltas to datetime `<'datetime.datetime'>`.

    ## Absolute Deltas (Replace specified fields)

    :param year `<'int'>`: Absolute year. Defaults to `-1` (no change).
    :param month `<'int'>`: Absolute month. Defaults to `-1` (no change).
    :param day `<'int'>`: Absolute day. Defaults to `-1` (no change).
    :param weekday `<'int'>`: Absolute weekday (0=Mon...6=Sun). Defaults to `-1` (no change).
    :param hour `<'int'>`: Absolute hour. Defaults to `-1` (no change).
    :param minute `<'int'>`: Absolute minute. Defaults to `-1` (no change).
    :param second `<'int'>`: Absolute second. Defaults to `-1` (no change).
    :param millisecond `<'int'>`: Absolute millisecond. Defaults to `-1` (no change).
        Overrides `microsecond` milliseoncd part if both are provided.
    :param microsecond `<'int'>`: Absolute microsecond. Defaults to `-1` (no change).

    ## Relative Deltas (Add to specified fields)

    :param years `<'int'>`: Relative years. Defaults to `0`.
    :param quarters `<'int'>`: Relative quarters (3 months). Defaults to `0`.
    :param months `<'int'>`: Relative months. Defaults to `0`.
    :param weeks `<'int'>`: Relative weeks (7 days). Defaults to `0`.
    :param days `<'int'>`: Relative days. Defaults to `0`.
    :param hours `<'int'>`: Relative hours. Defaults to `0`.
    :param minutes `<'int'>`: Relative minutes. Defaults to `0`.
    :param seconds `<'int'>`: Relative seconds. Defaults to `0`.
    :param milliseconds `<'int'>`: Relative milliseconds (1,000 us). Defaults to `0`.
    :param microseconds `<'int'>`: Relative microseconds. Defaults to `0`.

    ## Datetime Class
    
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
        
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).
        after applying the specified deltas.
    """
    # Date fields
    # ----------------------------------------------------
    cdef:
        int o_yy = dt.year
        int yy = (year if year > 0 else o_yy) + years
        int o_mm = dt.month
        int mm = (month if month > 0 else o_mm) + months + (quarters * 3)
        int o_dd = dt.day
        int dd = (day if day > 0 else o_dd)
        long long q, r
    # . normalize month
    if not 0 < mm <= 12:
        mm -= 1
        with cython.cdivision(True):
            q = mm / 12; r = mm % 12
        if r < 0:
            q -= 1; r += 12
        yy += q; mm = r + 1
    # . relative days
    cdef long long rel_days = days + (weeks * 7)

    # Time fields
    # ----------------------------------------------------
    cdef:
        long long o_hh = dt.hour
        long long hh = (hour if hour >= 0 else o_hh) + hours
        long long o_mi = dt.minute
        long long mi = (minute if minute >= 0 else o_mi) + minutes
        long long o_ss = dt.second
        long long ss = (second if second >= 0 else o_ss) + seconds
        long long o_us = dt.microsecond
        long long us = combine_absolute_ms_us(millisecond, microsecond)
    us = (us if us >= 0 else o_us) + milliseconds * US_MILLISECOND + microseconds
    # . normalize microseconds
    if not 0 <= us < US_SECOND:
        with cython.cdivision(True):
            q = us / US_SECOND; r = us % US_SECOND
        if r < 0:
            q -= 1; r += US_SECOND
        ss += q; us = r
    # . normalize seconds
    if not 0 <= ss < 60:
        with cython.cdivision(True):
            q = ss / 60; r = ss % 60
        if r < 0:
            q -= 1; r += 60
        mi += q; ss = r
    # . normalize minutes
    if not 0 <= mi < 60:
        with cython.cdivision(True):
            q = mi / 60; r = mi % 60
        if r < 0:
            q -= 1; r += 60
        hh += q; mi = r
    # . normalize hours to days
    if not 0 <= hh < 24:
        with cython.cdivision(True):
            q = hh / 24; r = hh % 24
        if r < 0:
            q -= 1; r += 24
        rel_days += q; hh = r

    # Handle day delta
    # ----------------------------------------------------
    cdef long long old_ord, new_ord
    cdef ymd _ymd
    if rel_days != 0 or weekday >= 0:
        old_ord = ymd_to_ord(yy, mm, dd)
        new_ord = old_ord + rel_days
        # . adjust to weekday if needed
        if weekday >= 0:
            with cython.cdivision(True):
                r = (new_ord + 6) % 7
                if r < 0:
                    r += 7
            new_ord += min(weekday, 6) - r  # weekday clamped to Sun=6
        if old_ord != new_ord:
            _ymd = ymd_fr_ord(new_ord)
            yy, mm, dd = _ymd.year, _ymd.month, _ymd.day

    # Compare new date/time
    if (yy == o_yy and mm == o_mm and dd == o_dd and
        hh == o_hh and mi == o_mi and ss == o_ss and us == o_us):
        return dt_fr_dt(dt, dtclass)

    # Clamp day to valid days in month
    if dd > 28:
        dd = min(dd, days_in_month(yy, mm))

    # New datetime
    return dt_new(yy, mm, dd, hh, mi, ss, us, dt.tzinfo, dt.fold, dtclass)

cdef inline datetime.datetime dt_replace_tz(datetime.datetime dt, object tzinfo, object dtclass=None):
    """Create a copy of `dt` with `tzinfo` replaced `<'datetime.datetime'>`.

    :param dt `<'datetime.datetime'>`: Source datetime (naive or aware).
    :param tzinfo `<'tzinfo/None'>`: The target tzinfo to attach.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified) 
        with the same fields except `tzinfo`.

    ## Equivalent
    >>> dt.replace(tzinfo=tzinfo)
    """
    # Same tzinfo
    if tzinfo is dt.tzinfo:
        return dt_fr_dt(dt, dtclass)

    # New datetime
    return dt_new(
        dt.year, dt.month, dt.day, dt.hour, dt.minute,
        dt.second, dt.microsecond, tzinfo, dt.fold, dtclass,
    )

cdef inline datetime.datetime dt_replace_fold(datetime.datetime dt, int fold, object dtclass=None):
    """Create a copy of `dt` with `fold` set to 0 or 1 `<'datetime.datetime'>`.

    :param dt `<'datetime.datetime'>`: Source datetime (naive or aware).
    :param fold `<'int'>`: Must be 0 or 1; otherwise `ValueError` is raised.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified) 
        with the same fields except `fold`.

    ## Equivalent
    >>> dt.replace(fold=fold)
    """
    # Validate
    if fold != 0 and fold != 1:
        raise ValueError("fold must be either 0 or 1, got %d" % (fold))

    # Fast-path: same fold
    if fold == dt.fold:
        return dt_fr_dt(dt, dtclass)

    # New datetime
    return dt_new(
        dt.year, dt.month, dt.day, dt.hour, dt.minute,
        dt.second, dt.microsecond, dt.tzinfo, fold, dtclass,
    )

# . tzinfo
cdef inline str dt_tzname(datetime.datetime dt):
    """Return the display name of a datetime's timezone `<'str/None'>`.

    :param dt `<'datetime.datetime'>`: The source datetime (naive or aware).
    :returns `<'str/None'>`: The timezone name, or `None` if `dt` 
        is naive or it's `tzinfo` is unavailable. 
    
    ## Equivalent
    >>> dt.tzname()
    """
    return tz_name(dt.tzinfo, dt)

cdef inline datetime.timedelta dt_dst(datetime.datetime dt):
    """Return the DST offset of a datetime's timezone `<'timedelta/None'>`.

    :param dt `<'datetime.datetime'>`: The source datetime (naive or aware).
    :returns `<'datetime.timedelta/None'>`: The DST offset, or `None` if `dt` 
        is naive or it's `tzinfo` is unavailable.
    
    ## Equivalent
    >>> dt.dst()
    """
    return tz_dst(dt.tzinfo, dt)

cdef inline datetime.timedelta dt_utcoffset(datetime.datetime dt):
    """Return the UTC offset of a datetime's timezone `<'timedelta/None'>`.

    :param dt `<'datetime.datetime'>`: The source datetime (naive or aware).
    :returns `<'datetime.timedelta/None'>`: The UTC offset, or `None` if `dt` 
        is naive or it's `tzinfo` is unavailable.
    
    ## Equivalent
    >>> dt.utcoffset()
    """
    return tz_utcoffset(dt.tzinfo, dt)

cdef inline datetime.datetime dt_astimezone(datetime.datetime dt, object tzinfo=None, object dtclass=None):
    """Convert a `datetime.datetime` to another time zone `<'datetime.datetime'>`.

    :param dt `<'datetime.datetime'>`: Datetime to convert (naive or aware).
    :param tzinfo `<'tzinfo/None'>`: Target time zone. Defaults to `None`.

        - If `None`, the system **local** time zone is used.
        - Must be a `tzinfo-compatible` object when provided.

    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified) 
        representing the **same instant** expressed in the target time zone. For naive 
        inputs + `tzinfo is None`, this *localizes* the datetime to the system local zone.

    ## Semantics
    - **Aware → Aware:** Convert by subtracting the source `utcoffset(dt)`,
      attach the target tzinfo, then normalize via `tzinfo.fromutc(...)`.
    - **Naive + tzinfo is None:** Treat `dt` as local wall time and simply attach
      the local tzinfo (no clock change).
    - **Naive + tzinfo provided:** Treat `dt` as local wall time, convert the
      instant to UTC, attach `tzinfo`, then normalize via `tzinfo.fromutc(...)`.
    - **Identity fast path:** If `dt.tzinfo is tzinfo`, return `dt` unchanged.

    ## DST / fold
    - Ambiguous/nonexistent local times are resolved by delegating 
      to `tzinfo.fromutc(...)`, which honors `dt.fold` per PEP 495.

    ## Equivalent
    >>> dt.astimezone(tzinfo)
    """
    cdef: 
        object my_tz = datetime.datetime_tzinfo(dt)
        object to_tz = tzinfo
        datetime.timedelta my_off

    # Resolve target tzinfo
    if to_tz is None:
        to_tz = tz_local()
        # Fast-exit: naive + local -> localize
        if my_tz is None:
            return dt_replace_tz(dt, to_tz, dtclass)
    elif not is_tz(to_tz):
        raise TypeError(
            "Expects an instance of 'datetime.tzinfo', "
            "instead got %s." % type(to_tz)
        )

    # Fast-exit: exact same tzinfo
    if my_tz is to_tz:
        return dt_fr_dt(dt, dtclass)

    # Naive datetime cases
    if my_tz is None:
        my_tz = tz_local()
        # Fast-exit: naive + local -> localize
        if my_tz is to_tz:
            return dt_replace_tz(dt, to_tz, dtclass)
        # Access local UTC offset (my_off is local)
        my_off = tz_utcoffset(my_tz, dt)

    # Aware datetime case: handle no-offset tzinfo
    # by using local tzinfo fallback (CPython pattern)
    else:
        my_off = tz_utcoffset(my_tz, dt)
        if my_off is None:
            my_tz = tz_local()
            # Fast-exit: exact same tzinfo
            if my_tz is to_tz:
                return dt_fr_dt(dt, dtclass)
            # Access local UTC offset (my_off is local)
            my_off = tz_utcoffset(my_tz, dt)

    # Convert to UTC microseconds, then attach tzinfo and normalize via fromutc
    cdef long long us = dt_to_us(dt, False) - td_to_us(my_off)
    return to_tz.fromutc(dt_fr_us(us, to_tz, dtclass))

cdef inline datetime.datetime dt_normalize_tz(datetime.datetime dt, object dtclass=None):
    """Normalize an aware datetime against its own tzinfo `<'datetime.datetime'>`.

    :param dt `<'datetime.datetime'>`: Datetime to normalize.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The normalized datetime (or subclass if `dtclass` is specified).

    ## Behavior:
      - If `dt` is timezone-niave → returns `dt` as-is.
      - Compute the timezone offsets at the same wall-clock time
        with `fold=0` and `fold=1`.
      - If offset with `fold=1` is greater than with `fold=0` (spring-forward gap), 
        shift wall time by the offset delta `Δ`:

          - `fold == 0` → move backward by Δ (to the last valid wall time)
          - `fold == 1` → move forward by Δ (to the next valid wall time)

      - Otherwise (ambiguous or no change), returns `dt` as-is.
      - If either offset is unavailable, returns `dt` as-is.
    """
    # Timezone-naive: no-op
    cdef object tzinfo = dt.tzinfo
    if tzinfo is None:
        return dt_fr_dt(dt, dtclass)

    # Probe offsets at both folds around the same wall time
    cdef int fold = dt.fold
    cdef long long off_f0, off_f1
    if fold == 1:
        off_f0 = tz_utcoffset_sec(tzinfo, dt_replace_fold(dt, 0, None))
        off_f1 = tz_utcoffset_sec(tzinfo, dt)
    else:
        off_f0 = tz_utcoffset_sec(tzinfo, dt)
        off_f1 = tz_utcoffset_sec(tzinfo, dt_replace_fold(dt, 1, None))

    # There is no positive delta or either offset is 
    # unknown (sentinel = -100,000); return as-is.
    cdef long long delta = off_f1 - off_f0
    if delta <= 0 or off_f0 == NULL_TZOFFSET or off_f1 == NULL_TZOFFSET:
        return dt_fr_dt(dt, dtclass)

    # Gap: apply Δ depending on which side of the gap 
    # the input fold denotes
    cdef long long us = dt_to_us(dt, False)
    if fold == 1:
        us += delta * US_SECOND
    else:
        us -= delta * US_SECOND
    return dt_fr_us(us, tzinfo, dtclass)

# datetime.time ----------------------------------------------------------------------------------------
# . generate
cdef inline datetime.time time_new(
    int hour=0, int minute=0, int second=0, int microsecond=0, 
    object tzinfo=None, int fold=0, object tclass=None,
):
    """Create a new time `<'datetime.time'>`.

    :param hour `<'int'>`: Hour [0..23]. Defaults to `0`.
    :param minute `<'int'>`: Minute [0..59]. Defaults to `0`.
    :param second `<'int'>`: Second [0..59]. Defaults to `0`.
    :param microsecond `<'int'>`: Microsecond [0..999999]. Defaults to `0`.
    :param tzinfo `<'tzinfo/None'>`: Optional timezone. Defaults to `None`.
    :param fold `<'int'>`: Optional fold flag for ambiguous times (0 or 1). Defaults to `0`.
        Only relevant if 'tzinfo' is not `None`, and values other than `1` are treated as `0`.
    :param tclass `<'type[datetime.time]/None'>`: Optional custom time class. Defaults to `None`.
        if `None` uses python's built-in `datetime.time` as the constructor.
    :returns `<'datetime.time'>`: The resulting time (or subclass if `tclass` is specified).
    """
    # Normalize fold (0/1 only)
    fold = 0 if tzinfo is None else fold == 1

    # Construct time
    if tclass is not None and tclass is not datetime.time:
        try:
            return tclass(
                hour=hour, 
                minute=minute, 
                second=second, 
                microsecond=microsecond, 
                tzinfo=tzinfo, 
                fold=fold,
            )
        except Exception as err:
            raise TypeError("Cannot create time using custom 'tclass' %s, %s" % (tclass, err)) from err
    return datetime.time_new(hour, minute, second, microsecond, tzinfo, fold)

cdef inline datetime.time time_now(object tzinfo=None, object tclass=None):
    """Get the current time `<'datetime.time'>`.

    :param tzinfo `<'tzinfo/None'>`: Optional timezone. Defaults to `None`.

        - If specified, return an aware time in that timezone.
        - Otherwise, return a naive local time.

    :param tclass `<'type[datetime.time]/None'>`: Optional custom time class. Defaults to `None`.
        if `None` uses python's built-in `datetime.time` as the constructor.

    :returns `<'datetime.time'>`: The current time (or subclass if `tclass` is specified).
    
    ## Equivalent
    >>> datetime.datetime.now(tzinfo).time()
    """
    return time_fr_dt(dt_now(tzinfo, None), tclass)
    
# . type check
cdef inline bint is_time(object obj) except -1:
    """Check if an object is an instance or sublcass of `datetime.time` `<'bool'>`.
    
    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> isinstance(obj, datetime.time)
    """
    return datetime.PyTime_Check(obj)

cdef inline bint is_time_exact(object obj) except -1:
    """Check if an object is an exact `datetime.time` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> type(obj) is datetime.time
    """
    return datetime.PyTime_CheckExact(obj)

# . conversion: to
cdef inline str time_isoformat(datetime.time time, bint utc=False):
    """Convert a `datetime.time` to ISO string `<'str'>`.

    :param time `<'datetime.time'>`: The time to convert.
    :param utc <'bool'>: Whether to append a UTC offset. Defaults to False.
        
        - If False or `time` is timezone-naive, UTC offset is ignored.
        - If True and `time` is timezone-aware, append UTC offset (e.g., +0530).

    :returns `<'str'>`: Formatted text.
    """
    cdef:
        int us = time.microsecond
        str s_utc = tz_utcformat(time.tzinfo, None) if utc else None

    if us == 0:
        if s_utc is None:
            return "%02d:%02d:%02d" % (time.hour, time.minute, time.second)
        return "%02d:%02d:%02d%s" % (time.hour, time.minute, time.second, s_utc)
    else:
        if s_utc is None:
            return "%02d:%02d:%02d.%06d" % (time.hour, time.minute, time.second, us)
        return "%02d:%02d:%02d.%06d%s" % (time.hour, time.minute, time.second, us, s_utc)

cdef inline long long time_to_us(datetime.time time) noexcept:
    """Convert a `datetime.time` to total microseconds `<'int'>`.

    :param time `<'datetime.time'>`: Time to convert.
    :returns `<'int'>`: Microseconds since midnight in [0, 86_400_000_000).
    """
    return (
        (<long long> time.hour)       * US_HOUR +
        (<long long> time.minute)     * US_MINUTE +
        (<long long> time.second)     * US_SECOND +
        (<long long> time.microsecond)
    )
    
cdef inline double time_to_sec(datetime.time time) noexcept:
    """Convert a `datetime.time` to total seconds `<'float'>`.

    :param time `<'datetime.time'>`: Time to convert.
    :returns `<'float'>`: Seconds since midnight in [0.0, 86400.0).
    """
    cdef long long us = time_to_us(time)
    return <double> us * 1e-6

# . conversion: from
cdef inline datetime.time time_fr_us(long long value, object tzinfo=None, object tclass=None):
    """Create time from microseconds since the Unix epoch `<'datetime.time'>`.

    :param value `<'int'>`: Microseconds since epoch.
    :param tzinfo `<'tzinfo/None'>`: Optional timezone to **attach**. Defaults to `None`.
    :param tclass `<'type[datetime.time]/None'>`: Optional custom time class. Defaults to `None`.
        if `None` uses python's built-in `datetime.time` as the constructor.
    :returns `<'datetime.time'>`: The resulting time (or subclass if `tclass` is specified).
    """
    cdef hmsf _hmsf = hmsf_fr_us(value)
    return time_new(
        _hmsf.hour, _hmsf.minute, _hmsf.second, 
        _hmsf.microsecond, tzinfo, 0, tclass
    )

cdef inline datetime.time time_fr_sec(double value, object tzinfo=None, object tclass=None):
    """Create time from seconds since the Unix epoch `<'datetime.time'>`.

    :param value `<'int'>`: Seconds since epoch.
    :param tzinfo `<'tzinfo/None'>`: Optional timezone to **attach**. Defaults to `None`.
    :param tclass `<'type[datetime.time]/None'>`: Optional custom time class. Defaults to `None`.
        if `None` uses python's built-in `datetime.time` as the constructor.
    :returns `<'datetime.time'>`: The resulting time (or subclass if `tclass` is specified).
    """
    return time_fr_us(sec_to_us(value), tzinfo, tclass)

cdef inline datetime.time time_fr_time(datetime.time time, object tclass=None):
    """Create time from another time (include subclass) `<'datetime.time'>`.

    :param time `<'datetime.time'>`: The source time (including subclasses).
    :param tclass `<'type[datetime.time]/None'>`: Target time class. Defaults to `None`.
        If `None` set to python's built-in `datetime.time`.
        If `time` is already of type `tclass`, returns `time` directly.
    :returns `<'datetime.time'>`: The resulting time (or subclass if `tclass` is specified)
        with the same time fields and tzinfo.
    """
    if tclass is None:
        tclass = datetime.time
    if tclass is type(time):
        return time
    return time_new(
        time.hour, time.minute, time.second, 
        time.microsecond, time.tzinfo, time.fold, tclass
    )

cdef inline datetime.time time_fr_dt(datetime.datetime dt, object tclass=None):
    """Create time from datetime (include subclass) `<'datetime.time'>`.

    :param dt `<'datetime.datetime'>`: The source datetime (including subclasses).
    :param tclass `<'type[datetime.time]/None'>`: Optional custom time class. Defaults to `None`.
        if `None` uses python's built-in `datetime.time` as the constructor.
    :returns `<'datetime.time'>`: The resulting time (or subclass if `tclass` is specified)
        with the same time fields and tzinfo.
    """
    return time_new(
        dt.hour, dt.minute, dt.second, 
        dt.microsecond, dt.tzinfo, dt.fold, tclass
    )

# datetime.timedelta -----------------------------------------------------------------------------------
# . generate
cdef inline datetime.timedelta td_new(int days=0, int seconds=0, int microseconds=0, object tdclass=None):
    """Create a new timedelta `<'datetime.timedelta'>`.
    
    :param days `<'int'>`: Days (can be negative). Defaults to 0.
    :param seconds `<'int'>`: Seconds (can be negative). Defaults to 0.
    :param microseconds `<'int'>`: Microseconds (can be negative). Defaults to 0.
    :param tdclass `<'type[datetime.timedelta]/None'>`: Optional custom timedelta class. Defaults to `None`.
        if `None` uses python's built-in `datetime.timedelta` as the constructor.
    :returns `<'datetime.timedelta'>`: The resulting timedelta (or subclass if `tdclass` is specified).
    
    ## Equivalent
    >>> datetime.timedelta(days, seconds, microseconds)
    """
    # Construct timedelta
    if tdclass is not None and tdclass is not datetime.timedelta:
        try:
            return tdclass(days=days, seconds=seconds, microseconds=microseconds)
        except Exception as err:
            raise TypeError("Cannot create timedelta using custom 'tdclass' %s, %s" % (tdclass, err)) from err
    return datetime.timedelta_new(days, seconds, microseconds)

# . type check
cdef inline bint is_td(object obj) except -1:
    """Check if an object is an instance or subclass of `datetime.timedelta` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> isinstance(obj, datetime.timedelta)
    """
    return datetime.PyDelta_Check(obj)

cdef inline bint is_td_exact(object obj) except -1:
    """Check if an object is an exact `datetime.timedelta` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> type(obj) is datetime.timedelta    
    """
    return datetime.PyDelta_CheckExact(obj)

# . conversion: to
cdef inline str td_isoformat(datetime.timedelta td):
    """Convert timedelta to string in ISO-like string (±HH:MM:SS[.f]) `<'str'>`.
    
    :param td `<'datetime.timedelta'>`: The timedelta to convert.
    :returns `<'str'>`: The ISO-like string in the format of ±HH:MM:SS[.f]
    """
    cdef:
        long long days = td.day
        long long sec  = td.second
        long long us   = td.microsecond
        long long hh, mi
        bint neg
    sec = days * SS_DAY + sec

    # Determine sign and normalize to absolute value without multiplying by 1e6.
    if sec < 0:
        neg = True
        sec = -sec
        # Borrow 1 second so that we can complement microseconds.
        if us != 0:
            sec -= 1                # |-X sec, +us|
            us = 1_000_000 - us     # (|X|-1) seconds => (1_000_000 - us)
    else:
        neg = False

    # Split absolute seconds into H:M:S
    with cython.cdivision(True):
        hh = sec / SS_HOUR;     sec %= SS_HOUR
        mi = sec / SS_MINUTE;   sec %= SS_MINUTE

    # Emit
    if us == 0:
        return ("-%02d:%02d:%02d" if neg else "%02d:%02d:%02d") % (hh, mi, sec)
    else:
        return ("-%02d:%02d:%02d.%06d" if neg else "%02d:%02d:%02d.%06d") % (hh, mi, sec, us)

cdef inline str td_utcformat(datetime.timedelta td):
    """Convert a `datetime.timedelta` to a UTC offset string (±HHMM) `<'str'>`.

    :param td `<'datetime.timedelta'>`: The timedelta to convert.
    :returns `<'str'>`: The UTC offset string in the format of ±HHMM
    """
    cdef:
        long long days = td.day
        long long sec  = td.second
        long long hh, mi
        bint neg
    sec = days * SS_DAY + sec

    # Check sign
    if sec < 0:
        sec = -sec
        neg = True
    else:
        neg = False

    # Split absolute seconds into H:M
    with cython.cdivision(True):
        sec %= SS_DAY
        hh = sec / SS_HOUR;     sec %= SS_HOUR
        mi = sec / SS_MINUTE

    # Format to string
    return ("-%02d%02d" if neg else "+%02d%02d") % (hh, mi)

cdef inline long long td_to_us(datetime.timedelta td) noexcept:
    """Convert a `datetime.timedelta` to total microseconds `<'int'>`.
    
    :param td `<'datetime.timedelta'>`: The timedelta to convert.
    :returns `<'int'>`: Total microseconds of the timedelta
    """
    return (
        (<long long> td.day)        * US_DAY +
        (<long long> td.second)     * US_SECOND +
        (<long long> td.microsecond)
    )

cdef inline double td_to_sec(datetime.timedelta td) noexcept:
    """Convert a `datetime.timedelta` to total seconds `<'float'>`.
    
    :param td `<'datetime.timedelta'>`: The timedelta to convert.
    :returns `<'seconds'>`: Total seconds of the timedelta
    """
    cdef:
        long long days = td.day
        long long sec = td.second
        double fsec = <double> (days * SS_DAY + sec)
        double frac = (<double> td.microsecond) * 1e-6
    return fsec + frac

# . conversion: from
cdef inline datetime.timedelta td_fr_us(long long value, object tdclass=None):
    """Create timedelta from microseconds `<'datetime.timedelta'>`.

    :param value `<'int'>`: Delta in microseconds.
    :param tdclass `<'type[datetime.timedelta]/None'>`: Optional custom timedelta class. Defaults to `None`.
        if `None` uses python's built-in `datetime.timedelta` as the constructor.
    :returns `<'datetime.timedelta'>`: The resulting timedelta (or subclass if `tdclass` is specified).
    """
    cdef long long q, r
    cdef int dd, ss, us
    with cython.cdivision(True):
        q = value / US_DAY; r = value % US_DAY
        if r < 0:
            q -= 1; r += US_DAY
        dd = <int> q
        ss = <int> (r / US_SECOND)
        us = <int> (r % US_SECOND)
    return td_new(dd, ss, us, tdclass)

cdef inline datetime.timedelta td_fr_sec(double value, object tdclass=None):
    """Create timedelta from seconds `<'datetime.timedelta'>`.

    :param value `<'float'>`: Delta in seconds.
    :param tdclass `<'type[datetime.timedelta]/None'>`: Optional custom timedelta class. Defaults to `None`.
        if `None` uses python's built-in `datetime.timedelta` as the constructor.
    :returns `<'datetime.timedelta'>`: The resulting timedelta (or subclass if `tdclass` is specified).
    """
    return td_fr_us(sec_to_us(value), tdclass)

cdef inline datetime.timedelta td_fr_td(datetime.timedelta td, object tdclass=None):
    """Create timedelta from another timedelta (or subclass) `<'datetime.timedelta'>`.
    
    :param td `<'datetime.timedelta'>`: The source timedelta (including subclasses).
    :param tdclass `<'type[datetime.timedelta]/None'>`: Target timedelta class. Defaults to `None`.
        If `None` set to python's built-in `datetime.timedelta`.
        If `td` is already of type `tdclass`, returns `td` directly.
    :returns `<'datetime.timedelta'>`: The resulting timedelta (or subclass if `tdclass` is specified)
        with the same days, seconds and microseconds.
    """
    if tdclass is None:
        tdclass = datetime.timedelta
    if tdclass is type(td):
        return td
    return td_new(td.day, td.second, td.microsecond, tdclass)

# datetime.tzinfo --------------------------------------------------------------------------------------
# . generate
cdef inline object tz_new(int hours=0, int minites=0, int seconds=0):
    """Create a new fixed-offset timezone `<'datetime.timezone'>`.

    :param hours `<'int'>`: Hour component of the fixed offset. Defaults to 0.
    :param minutes `<'int'>`: Minute component of the fixed offset. Defaults to 0.
    :param seconds `<'int'>`: Second component of the fixed offset. Defaults to 0.
    :returns `<'datetime.timezone'>`: The corresponding fixed-offset timezone.

    Equivalent:
    >>> datetime.timezone(datetime.timedelta(hours=hours, minutes=minutes))
    """
    # Range: inclusive ±24:00 per CPython (datetime.timezone)
    cdef long long sec = hours * 3_600 + minites * 60 + seconds
    if not -86400 < sec < 86400:
        raise ValueError(
            "Timezone offset %d seconds from (hours=%d, minutes=%d, seconds=%d) out of range. "
            "Must be strictly between [-86400..86400] (exclusive)" % (hours, minites, seconds, sec)
        )

    # New timezone
    return datetime.timezone_new(datetime.timedelta_new(0, sec, 0), None)

cdef inline object tz_local(datetime.datetime dt=None):
    """Get the process-local timezone `<'zoneinfo.ZoneInfo/datetime.timezone'>`.

    :param dt `<'datetime.datetime/None'>`: Ignored, reserved for future use. Defaults to `None`.
    :returns `<'zoneinfo.ZoneInfo/datetime.timezone'>`: The local timezone.

        - Returns a concrete IANA zone (`zoneinfo.ZoneInfo`) when possible 
          using [babel](https://github.com/python-babel/babel) `LOCALTZ` name. 
        - If that fails, falls back to a fixed-offset `datetime.timezone` using 
          the **current** local UTC offset (but no DST rules).

    ## Notice
    - The local timezone is resolved and cached `once at module import`
      and reused on subsequent calls. It does not track historical or 
      future transitions.
    """
    return _LOCAL_TIMEZONE

cdef inline int tz_local_sec(datetime.datetime dt=None) except -200_000:
    """Return the local UTC offset (in whole seconds) at the given instant `<'int'>`.

    :param dt `<'datetime.datetime/None'>`: The datetime to evaluate. Defaults to `None`.

        - If `dt` is aware, its UTC offset is computed for that wall time.
        - If `dt` is naive, it is interpreted as *local time* (mktime-style) and
          DST ambiguity is resolved using `fold`. 
        - If `dt` is None, use current time.

    :returns `<'int'>`: The local UTC offset in seconds (positive east of UTC, negative west).
    """
    cdef:
        long long ts1, ts2
        double ts
        tm loc, gmt
        long long ord_loc, ord_gmt

    # Decide the timestamp (seconds since epoch)
    if dt is None:
        # current time
        ts = unix_time()
    else:
        # naive: interpret as local wall time
        if dt.tzinfo is None:
            ts1 = dt_local_mktime(dt)
            # probe the other fold to handle gaps/ambiguous times
            ts2 = dt_local_mktime(dt_replace_fold(dt, 1 - dt.fold, None))
            if ts2 != ts1 and ((ts2 > ts1) == dt.fold):
                ts = <double> ts2
            else:
                ts = <double> ts1
        # aware: use UTC instant
        else:
            ts = dt_to_sec(dt, True)

    # Compute offset via ordinals + time-of-day
    loc = tm_localtime(ts)
    ord_loc = ymd_to_ord(loc.tm_year, loc.tm_mon, loc.tm_mday)
    gmt = tm_gmtime(ts)
    ord_gmt = ymd_to_ord(gmt.tm_year, gmt.tm_mon, gmt.tm_mday)
    return (
        (ord_loc - ord_gmt)         * SS_DAY    +
        (loc.tm_hour - gmt.tm_hour) * SS_HOUR   +
        (loc.tm_min - gmt.tm_min)   * SS_MINUTE +
        (loc.tm_sec - gmt.tm_sec)
    )

cdef object tz_parse(object tz)

# . type check
cdef inline bint is_tz(object obj) except -1:
    """Check if an object is an instance or sublcass of `datetime.tzinfo` `<'bool'>`.
    
    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> isinstance(obj, datetime.tzinfo)
    """
    return datetime.PyTZInfo_Check(obj)

cdef inline bint is_tz_exact(object obj) except -1:
    """Check if an object is an exact `datetime.tzinfo` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> type(obj) is datetime.date
    """
    return datetime.PyTZInfo_CheckExact(obj)

# . access
cdef inline str tz_name(object tz, datetime.datetime dt=None):
    """Return the display name from a tzinfo `<'str/None'>`.

    :param tz `<'datetime.tzinfo/None'>`: Time zone object. If `None`, return as-is.
    :param dt `<'datetime.datetime/None'>`: Optional datetime to evaluate
        (some tzinfos vary by date). Defaults to `None`.
    :returns `<'str/None'>`: A zone name like 'UTC', 'EST', 'CET', or `None` if unavailable.

    Equivalent to:
    >>> tz.tzname(dt)
    """
    if tz is None:
        return None
    try:
        return tz.tzname(dt)
    except Exception as err:
        if not is_tz(tz):
            raise TypeError(
                "Expects an instance of 'datetime.tzinfo', "
                "instead got %s." % type(tz)
            ) from err
        raise err

cdef inline datetime.timedelta tz_dst(object tz, datetime.datetime dt=None):
    """Return the DST offset of a tzinfo `<'timedelta/None'>`.

    :param tz `<'datetime.tzinfo/None'>`: Time zone object. If `None`, return as-is.
    :param dt `<'datetime.datetime/None'>`: Optional datetime to evaluate
        (some tzinfos vary by date). Defaults to `None`.
    :returns `<'datetime.timedelta/None'>`: A DST offset, or `None` if not applicable.

    ## Equivalent
    >>> tz.dst(dt)
    """
    if tz is None:
        return None
    try:
        return tz.dst(dt)
    except Exception as err:
        if not is_tz(tz):
            raise TypeError(
                "Expects an instance of 'datetime.tzinfo', "
                "instead got %s." % type(tz)
            ) from err
        raise err

cdef inline datetime.timedelta tz_utcoffset(object tz, datetime.datetime dt=None):
    """Return the UTC offset from a tzinfo `<'timedelta/None'>`.

    :param tz `<'datetime.tzinfo/None'>`: Time zone object. If `None`, return as-is.
    :param dt `<'datetime.datetime/None'>`: Optional datetime to evaluate
        (some tzinfos vary by date). Defaults to `None`.
    :returns `<'datetime.timedelta/None'>`: A UTC offset, `None` if not applicable.

    ## Equivalent
    >>> tz.utcoffset(dt)
    """
    if tz is None:
        return None
    try:
        return tz.utcoffset(dt)
    except Exception as err:
        if not is_tz(tz):
            raise TypeError(
                "Expects an instance of 'datetime.tzinfo', "
                "instead got %s." % type(tz)
            ) from err
        raise err

cdef inline int tz_utcoffset_sec(object tz, datetime.datetime dt=None) except -200_000:
    """Return the UTC offset in total seconds from a tzinfo `<'int'>`.

    :param tz `<'datetime.tzinfo/None'>`: Time zone object.
    :param dt `<'datetime.datetime/None'>`: Optional datetime to evaluate
        (some tzinfos vary by date). Defaults to `None`.
    :returns `<'int'>`: Total whole seconds of the UTC offset,
        or sentinel (-100,000) when the offset is unavailable.
    """
    cdef datetime.timedelta off = tz_utcoffset(tz, dt)
    if off is None:
        return NULL_TZOFFSET

    cdef int sec = (off.day * (<int> SS_DAY) + off.second)
    if not -86400 < sec < 86400:
        raise ValueError(
            "Timezone offset %d seconds from (%s, %s) out of range. "
            "Must be strictly between [-86400..86400] (exclusive)." % (sec, tz, type(tz))
        )
    return sec

cdef inline str tz_utcformat(object tz, datetime.datetime dt=None):
    """Return the UTC offset in ISO format string (±HHMM) from a tzinfo `<'str/None'>`.

    :param tz `<'datetime.tzinfo/None'>`: Time zone object.
    :param dt `<'datetime.datetime/None'>`: Optional datetime to evaluate
        (some tzinfos vary by date). Defaults to `None`.
    :returns `<'str/None'>`: A string like '+0530' or '-0700', 
        or `None` when the offset is unavailable.
    """
    cdef datetime.timedelta off = tz_utcoffset(tz, dt)
    return None if off is None else td_utcformat(off)

# NumPy: time unit -------------------------------------------------------------------------------------
cdef inline str nptime_unit_int2str(int unit):
    """Map numpy time unit to its corresponding string form `<'str'>`.
    
    :param unit `<'int'>`: An NPY_DATETIMEUNIT enum value.
    :returns `<'str'>`: The corresponding string form:
        'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'Y', etc.
    """
    # Common units
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return "ns"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return "us"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return "ms"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return "s"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return "m"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return "h"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return "D"

    # Uncommon units
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:
        return "Y"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:
        return "M"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:
        return "W"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return "ps"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return "fs"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return "as"

    # Unsupported unit
    _raise_invalid_nptime_int_unit_error(unit)

cdef inline object nptime_unit_int2dt64(int unit):
    """Maps numpy time unit to its corresponding datetime dtype `<'np.dtype'>`.
    
    :param unit `<'int'>`: An NPY_DATETIMEUNIT enum value.
    :returns `<'np.dtype'>`: The corresponding datetime dtype:
        np.dtype('datetime64[ns]'), np.dtype('datetime64[Y]'), etc.
    """
    # Common units
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return DT64_DTYPE_NS
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return DT64_DTYPE_US
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return DT64_DTYPE_MS
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return DT64_DTYPE_SS
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return DT64_DTYPE_MI
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return DT64_DTYPE_HH
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return DT64_DTYPE_DD

    # Uncommon units
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:
        return DT64_DTYPE_YY
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:
        return DT64_DTYPE_MM
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:
        return DT64_DTYPE_WW
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return DT64_DTYPE_PS
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return DT64_DTYPE_FS
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return DT64_DTYPE_AS

    # Unsupported unit
    _raise_invalid_nptime_int_unit_error(unit)

cdef inline int nptime_unit_str2int(str unit) except -1:
    """Maps numpy time unit string form to its corresponding
    NPY_DATETIMEUNIT enum value `<'int'>`.

    :param unit `<'str'>`: Time unit in its string form:
        'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'Y', etc.
    :returns `<'int'>`: The corresponding NPY_DATETIMEUNIT enum value.
    """
    if unit is None:
        _raise_invalid_nptime_str_unit_error(unit)
    cdef:
        Py_ssize_t size = str_len(unit)
        Py_UCS4 ch

    # Unit: 's', 'm', 'h', 'D', 'Y', 'M', 'W', 'B'
    if size == 1:
        # Common units
        ch = str_read(unit, 0)
        if ch == "s":
            return np.NPY_DATETIMEUNIT.NPY_FR_s
        if ch == "m":
            return np.NPY_DATETIMEUNIT.NPY_FR_m
        if ch == "h":
            return np.NPY_DATETIMEUNIT.NPY_FR_h
        if ch == "D":
            return np.NPY_DATETIMEUNIT.NPY_FR_D
        # Uncommon units
        if ch == "Y":
            return np.NPY_DATETIMEUNIT.NPY_FR_Y
        if ch == "M":
            return np.NPY_DATETIMEUNIT.NPY_FR_M
        if ch == "W":
            return np.NPY_DATETIMEUNIT.NPY_FR_W
        # if ch == "B":
        #     return np.NPY_DATETIMEUNIT.NPY_FR_B

    # Unit: 'ns', 'us', 'ms', 'ps', 'fs', 'as'
    elif size == 2 and str_read(unit, 1) == "s":
        # Common units
        ch = str_read(unit, 0)
        if ch == "n":
            return np.NPY_DATETIMEUNIT.NPY_FR_ns
        if ch == "u":
            return np.NPY_DATETIMEUNIT.NPY_FR_us
        if ch == "m":
            return np.NPY_DATETIMEUNIT.NPY_FR_ms
        # Uncommon units
        if ch == "p":
            return np.NPY_DATETIMEUNIT.NPY_FR_ps
        if ch == "f":
            return np.NPY_DATETIMEUNIT.NPY_FR_fs
        if ch == "a":
            return np.NPY_DATETIMEUNIT.NPY_FR_as

    # Unit: 'min' for pandas compatibility
    elif size == 3 and unit == "min":
        return np.NPY_DATETIMEUNIT.NPY_FR_m

    # Unsupported unit
    _raise_invalid_nptime_str_unit_error(unit)

cdef inline object nptime_unit_str2dt64(str unit):
    """Maps numpy time unit string form to its corresponding
    datetime dtype `<'np.dtype'>`.

    :param unit `<'str'>`: Time unit in its string form:
        'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'Y', etc.
    :returns `<'np.dtype'>`: The corresponding datetime dtype:
        np.dtype('datetime64[ns]'), np.dtype('datetime64[Y]'), etc.
    """
    if unit is None:
        _raise_invalid_nptime_str_unit_error(unit)
    cdef:
        Py_ssize_t size = str_len(unit)
        Py_UCS4 ch

    # Unit: 's', 'm', 'h', 'D', 'Y', 'M', 'W', 'B'
    if size == 1:
        # Common units
        ch = str_read(unit, 0)
        if ch == "s":
            return DT64_DTYPE_SS
        if ch == "m":
            return DT64_DTYPE_MI
        if ch == "h":
            return DT64_DTYPE_HH
        if ch == "D":
            return DT64_DTYPE_DD
        # Uncommon units
        if ch == "Y":
            return DT64_DTYPE_YY
        if ch == "M":
            return DT64_DTYPE_MM
        if ch == "W":
            return DT64_DTYPE_WW

    # Unit: 'ns', 'us', 'ms', 'ps', 'fs', 'as'
    elif size == 2 and str_read(unit, 1) == "s":
        # Common units
        ch = str_read(unit, 0)
        if ch == "n":
            return DT64_DTYPE_NS
        if ch == "u":
            return DT64_DTYPE_US
        if ch == "m":
            return DT64_DTYPE_MS
        # Uncommon units
        if ch == "p":
            return DT64_DTYPE_PS
        if ch == "f":
            return DT64_DTYPE_FS
        if ch == "a":
            return DT64_DTYPE_AS

    # Unit: 'min' for pandas compatibility
    elif size == 3 and unit == "min":
        return DT64_DTYPE_MI

    # Unsupported unit
    _raise_invalid_nptime_str_unit_error(unit)

cdef inline int get_arr_nptime_unit(np.ndarray arr) except -1:
    """Get the time unit (NPY_DATETIMEUNIT enum) of a datetime-like array `<'int'>`.

    :param arr `<'np.ndarray'>`: An array with dtype `datetime64[*]` or `timedelta64[*]`.
    :returns `<'int'>`: The time unit as an integer form 
        its corresponding NPY_DATETIMEUNIT enum value
    """
    cdef: 
        int dtype = np.PyArray_TYPE(arr)
        np.npy_int64 zero = 0
        object scalar
    
    # datetime-array
    if dtype == np.NPY_TYPES.NPY_DATETIME:
        scalar = np.PyArray_ToScalar(&zero, arr)
        return <int> (<np.PyDatetimeScalarObject*>scalar).obmeta.base

    # timedelta-array  
    if dtype == np.NPY_TYPES.NPY_TIMEDELTA:
        scalar = np.PyArray_ToScalar(&zero, arr)
        return <int> (<np.PyTimedeltaScalarObject*>scalar).obmeta.base

    # Unsupported
    _raise_invalid_nptime_dtype_error(arr)

# . errors
cdef inline bint _raise_invalid_nptime_int_unit_error(int unit) except -1:
    """(internal) Raise error for an invalid numpy time unit value."""
    raise ValueError(
        "Unsupported numpy time unit (NPY_DATETIMEUNIT enum) '%d'.\n"
        "Accepts: %d→'Y', %d→'M', %d→'W', %d→'D', %d→'h', %d→'m', "
        "%d→'s', %d→'ms', %d→'us', %d→'ns'" % (
            unit, 
            DT_NPY_UNIT_YY,
            DT_NPY_UNIT_MM,
            DT_NPY_UNIT_WW,
            DT_NPY_UNIT_DD,
            DT_NPY_UNIT_HH,
            DT_NPY_UNIT_MI,
            DT_NPY_UNIT_SS,
            DT_NPY_UNIT_MS,
            DT_NPY_UNIT_US,
            DT_NPY_UNIT_NS,
        )
    )

cdef inline bint _raise_invalid_nptime_str_unit_error(object unit) except -1:
    """(internal) Raise error for an invalid numpy time unit string form."""
    raise ValueError(
        "Unsupported time unit %r.\n"
        "Accepts: 'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'" % (unit,)
    )

cdef inline bint _raise_invalid_nptime_dtype_error(np.ndarray arr) except -1:
    """(internal) Raise error for an array that is not dtype `datetime64[*]` or `timedelta64[*]`."""
    raise TypeError(
        "Expects 'ndarray[datetime64[*]]' or 'ndarray[timedelta64[*]]', "
        "instead got 'ndarray[%s]'" % arr.dtype
    )

# NumPy: datetime64 ------------------------------------------------------------------------------------
# . type check
cdef inline bint is_dt64(object obj) except -1:
    """Check if an object is an instance or subclass of `np.datetime64` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> isinstance(obj, np.datetime64)
    """
    return np.is_datetime64_object(obj)

cdef inline bint assure_dt64(object obj) except -1:
    """Assure an object is an instance of np.datetime64."""
    if not np.is_datetime64_object(obj):
        raise TypeError(
            "Expects an instance of 'np.datetime64', "
            "instead got %s." % type(obj)
        )
    return True

# . conversion
cdef inline np.npy_int64 dt64_as_int64_us(object dt64, np.npy_int64 offset=0):
    """Convert a np.datetime64 to int64 microsecond ('us') ticks `<'int'>`.

    :param dt64 `<'np.datetime64'>`: The datetime64 to convert.
    :param offset `<'int'>`: An optional offset added after conversion. Defaults to `0`.
    :returns `<'int'>`: The int64 value representing the datetime64 in microseconds.
    
    ## Equivalent
    >>> dt64.astype("datetime64[us]").astype("int64") + offset
    """
    # Get unit & value
    assure_dt64(dt64)
    cdef: 
        np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(dt64)
        np.npy_int64 value = np.get_datetime64_value(dt64)
        np.npy_int64 q, r

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        with cython.cdivision(True):
            q = value / 1_000; r = value % 1_000
            if r < 0:
                q -= 1
        return q + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return value + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return value * US_MILLISECOND + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return value * US_SECOND + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return value * US_MINUTE + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return value * US_HOUR + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return value * US_DAY + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return value * US_DAY * 7 + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _dt64_M_as_int64_D(value, US_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _dt64_Y_as_int64_D(value, US_DAY, offset)

    # Unsupported unit
    _raise_dt64_to_reso_error(dt64, "us")

cdef inline np.npy_int64 _dt64_Y_as_int64_D(np.npy_int64 value, np.npy_int64 factor=1, np.npy_int64 offset=0) noexcept nogil:
    """(internal) Convert np.datetime64[Y] to int64 day ticks (D), then
    scale by `factor` and add `offset` `<'int'>`.
    
    :param value `<'int'>`: The year value of the np.datetime64[Y].
    :param factor `<'int'>`: Post-conversion scale applied to day counts. Defaults to `1`.
    :param offset `<'int'>`: Optional value added after scaling. Defaults to `0`.
    :returns `<'int'>`: The int64 value after conversion.
    """
    # Absolute proleptic Gregorian year
    cdef np.npy_int64 yy = value + EPOCH_YEAR
    # Leap-years between 1970-01-01 and 31-Dec-(year-1) inclusive
    cdef np.npy_int64 leaps = leap_years(yy, False) - EPOCH_CBASE
    # Total days since 1970-01-01 at day resolution, then scale+shift
    return (value * 365 + leaps) * factor + offset

cdef inline np.npy_int64 _dt64_M_as_int64_D(np.npy_int64 value, np.npy_int64 factor=1, np.npy_int64 offset=0) noexcept nogil:
    """(internal) Convert np.datetime64[M] to int64 day ticks (D), then
    scale by `factor` and add `offset` `<'int'>`.
    
    :param value `<'int'>`: The month value of the np.datetime64[M].
    :param factor `<'int'>`: Post-conversion scale applied to day counts. Defaults to `1`.
    :param offset `<'int'>`: Optional value added after scaling. Defaults to `0`.
    :returns `<'int'>`: The int64 value after conversion.
    """
    # Absolute proleptic Gregorian year & month
    cdef np.npy_int64 yy_ep, yy, mm, leaps
    with cython.cdivision(True):
        yy_ep = value / 12; mm = value % 12
    if mm < 0:
        yy_ep -= 1; mm += 12
    yy = yy_ep + EPOCH_YEAR     # absolute year
    mm += 1                     # 1-based month
    # Leap-years between 1970-01-01 and 31-Dec-(year-1) inclusive
    leaps = leap_years(yy, False) - EPOCH_CBASE
    # Total days since 1970-01-01 at day resolution, then scale+shift
    return (yy_ep * 365 + leaps + days_bf_month(yy, mm)) * factor + offset

cdef inline datetime.datetime dt64_to_dt(object dt64, object tzinfo=None, object dtclass=None):
    """Convert np.datetime64 to datetime `<'datetime.datetime'>`.
    
    :param dt64 `<'np.datetime64'>`: The datetime64 to convert.
    :param tzinfo `<'datetime.tzinfo/None'>`: An optional timezone to attach 
        to the resulting datetime. Defaults to `None`.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).
    """
    return dt_fr_us(dt64_as_int64_us(dt64, 0), tzinfo, dtclass)

# . errors
cdef inline bint _raise_dt64_to_reso_error(object dt64, str to_unit) except -1:
    """(internal) Raise error for unsupported conversion unit for datetime64/timedelta64.
    
    :param dt64 `<'np.datetime64/np.timedelta64'>`: The numpy datetime-like object.
    :param to_unit `<'str'>`: The target conversion numpy unit.
    """
    cdef str type_str = type(dt64)
    cdef int dt_reso = np.get_datetime64_unit(dt64)
    cdef str dt_reso_str
    try:
        dt_reso_str = nptime_unit_int2str(dt_reso)
    except Exception as err:
        raise ValueError(
            "Cannot convert %s from '%d' unit to int64 '%s' ticks.\n"
            "Supported units: %d→'Y', %d→'M', %d→'W', %d→'D', %d→'h', %d→'m', "
            "%d→'s', %d→'ms', %d→'us', %d→'ns'" % (
                type_str,
                dt_reso,
                to_unit,
                DT_NPY_UNIT_YY,
                DT_NPY_UNIT_MM,
                DT_NPY_UNIT_WW,
                DT_NPY_UNIT_DD,
                DT_NPY_UNIT_HH,
                DT_NPY_UNIT_MI,
                DT_NPY_UNIT_SS,
                DT_NPY_UNIT_MS,
                DT_NPY_UNIT_US,
                DT_NPY_UNIT_NS,
            )
        ) from err
    else:
        raise ValueError(
            "Cannot convert %s from '%s' to int64 '%s' ticks.\n"
            "Supported units: 'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'" 
            % (type_str, dt_reso_str, to_unit)
        )

# NumPy: timedelta64 -----------------------------------------------------------------------------------
# . type check
cdef inline bint is_td64(object obj) except -1:
    """Check if an object is an instance or subclass of `np.timedelta64` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> isinstance(obj, np.timedelta64)
    """
    return np.is_timedelta64_object(obj)

cdef inline bint assure_td64(object obj) except -1:
    """Assure an object is an instance of np.timedelta64."""
    if not np.is_timedelta64_object(obj):
        raise TypeError(
            "Expects an instance of 'np.timedelta64', "
            "instead got %s." % type(obj)
        )
    return True

# . conversion
cdef inline np.npy_int64 td64_as_int64_us(object td64, np.npy_int64 offset=0):
    """Convert a np.timedelta64 to int64 microsecond ('us') ticks `<'int'>`.

    :param dt64 `<'np.timedelta64'>`: The timedelta64 to convert.
    :param offset `<'int'>`: An optional offset added after conversion. Defaults to `0`.
    :returns `<'int'>`: The int64 value representing the timedelta64 in microseconds.
    
    ## Equivalent
    >>> td64.astype("timedelta64[us]").astype("int64") + offset
    """
    # Get unit & value
    assure_td64(td64)
    cdef:
        np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(td64)
        np.npy_int64 value = np.get_timedelta64_value(td64)
        np.npy_int64 q, r

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:   # nanosecond
        with cython.cdivision(True):
            q = value / 1_000; r = value % 1_000
            if r < 0:
                q -= 1
        return q + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:   # microsecond
        return value + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:   # millisecond
        return value * US_MILLISECOND + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:    # second
        return value * US_SECOND + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:    # minute
        return value * US_MINUTE + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:    # hour
        return value * US_HOUR + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:    # day
        return value * US_DAY + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:    # week
        return value * US_DAY * 7 + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:    # month
        return _td64_M_as_int64_D(value, np.NPY_DATETIMEUNIT.NPY_FR_us, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:    # year
        return _td64_Y_as_int64_D(value, np.NPY_DATETIMEUNIT.NPY_FR_us, offset)

    # Unsupported unit
    _raise_dt64_to_reso_error(td64, "us")

cdef inline np.npy_int64 _td64_Y_as_int64_D(np.npy_int64 value, int to_reso, np.npy_int64 offset=0):
    """(internal) Convert np.timedelta[Y] to int64 day ticks using an average 
    year (365.2425 D), then further adjust the resolution by `to_reso` and
    add `offset` `<'int'>`.
    
    :param value `<'int'>`: The year value of the np.timedelta64[Y].
    :param to_reso `<'int'>`: Post-conversion `NPY_DATETIMEUNIT` unit 
        adjustment applied to the day counts.
    :param offset `<'int'>`: Optional value to add at the end. Defaults to `0`.
    :returns `<'int'>`: The int64 value after conversion.
    """
    # Exact rational forms to avoid FP
    #   365.2425 d                      = 146,097 / 400
    #   8,765.82 h  = 24 * 365.2425     = 438,291 / 50
    #   525,949.2 m = 1440 * 365.2425   = 2,629,746 / 5
    cdef np.npy_int64 q, r
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_D:   # day
        value *= 146_097
        with cython.cdivision(True):
            q = value / 400; r = value % 400
            if r < 0:
                q -= 1
        return q + offset
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_h:   # hour
        value *= 438_291
        with cython.cdivision(True):
            q = value / 50; r = value % 50
            if r < 0:
                q -= 1
        return q + offset
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:   # minute
        value *= 2_629_746
        with cython.cdivision(True):
            q = value / 5; r = value % 5
            if r < 0:
                q -= 1
        return q + offset
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:   # second
        return value * TD64_YY_SECOND + offset
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return value * TD64_YY_MILLISECOND + offset
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return value * TD64_YY_MICROSECOND + offset
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return value * TD64_YY_NANOSECOND + offset

    # Unsupported conversion resolution
    raise AssertionError("Unsupported conversion unit '%d' for timedelta64." % to_reso)

cdef inline np.npy_int64 _td64_M_as_int64_D(np.npy_int64 value, int to_reso, np.npy_int64 offset=0):
    """(internal) Convert np.timedelta[M] to int64 day ticks using an 
    average month (365.2425 / 12 = 30.436875 D), then further adjust 
    the resolution by `to_reso` and add `offset` `<'int'>`.
    
    :param value `<'int'>`: The month value of the np.timedelta64[M].
    :param to_reso `<'int'>`: Post-conversion `NPY_DATETIMEUNIT` unit 
        adjustment applied to the day counts.
    :param offset `<'int'>`: Optional value to add at the end. Defaults to `0`.
    :returns `<'int'>`: The int64 value after conversion.
    """
    # Exact rational forms (avoid FP):
    #   30.436875 d = 365.2425 / 12 d   = 48699 / 1600 
    #   730.485 h   = 24 * 30.436875    = 146097 / 200
    #   43,829.1 m  = 1440 * 30.436875  = 438291 / 10
    cdef np.npy_int64 q, r
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_D:   # day
        value *= 48_699
        with cython.cdivision(True):
            q = value / 1_600; r = value % 1_600
            if r < 0:
                q -= 1
        return q + offset
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_h:   # hour
        value *= 146_097
        with cython.cdivision(True):
            q = value / 200; r = value % 200
            if r < 0:
                q -= 1
        return q + offset
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:   # minute
        value *= 438_291
        with cython.cdivision(True):
            q = value / 10; r = value % 10
            if r < 0:
                q -= 1
        return q + offset
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:   # second
        return value * TD64_MM_SECOND + offset
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return value * TD64_MM_MILLISECOND + offset
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return value * TD64_MM_MICROSECOND + offset
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return value * TD64_MM_NANOSECOND + offset

    # Unsupported conversion resolution
    raise AssertionError("Unsupported conversion unit '%d' for timedelta64." % to_reso)

cdef inline datetime.timedelta td64_to_td(object td64, object tdclass=None):
    """Convert np.timedelta64 to timedelta `<'datetime.timedelta'>`.

    :param td64 `<'np.timedelta64'>`: The timedelta64 to convert.
    :param tdclass `<'type[datetime.timedelta]/None'>`: Optional custom timedelta class. Defaults to `None`.
        if `None` uses python's built-in `datetime.timedelta` as the constructor.
    :returns `<'datetime.timedelta'>`: The resulting timedelta (or subclass if `tdclass` is specified).
    """
    return td_fr_us(td64_as_int64_us(td64, 0), tdclass)

# NumPy: ndarray ---------------------------------------------------------------------------------------
# . type check
cdef inline bint is_arr(object obj) except -1:
    """Check if an object is an instance or subclass of `np.ndarray` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> isinstance(obj, np.ndarray)
    """
    return np.PyArray_Check(obj)

cdef inline bint assure_1dim_arr(np.ndarray arr) except -1:
    """Assure the array is 1-dimensional."""
    if arr.ndim != 1:
        raise ValueError("Expects 1-D ndarray, instead got %d-dimensional array." % arr.ndim)
    return True

cdef inline np.ndarray assure_arr_contiguous(np.ndarray arr):
    """Ensure that an ndarray is C-contiguous in memory `<'np.ndarray'>`.

    :returns `<'np.ndarray'>`: The original array if already contiguous; 
        otherwise, returns a contiguous copy.
    """
    if np.PyArray_IS_C_CONTIGUOUS(arr):
        return arr
    return np.PyArray_GETCONTIGUOUS(arr)

# . dtype
cdef inline np.ndarray arr_assure_int64(np.ndarray arr, bint copy=True):
    """Ensure that a 1-D array is contiguous and dtype int64 `<'ndarray[int64]'>`.

    :param arr `<'ndarray'>`: The 1-D array to ensure.
    :param copy `<'bool'>`: Whether to always create a copy even if the array is
        already dtype int64. Defaults to `True`.
    :returns `<'ndarray[int64]'>`:
        If the array is already the right dtype, return it directly or a copy
        depending on the `copy` flag. Otherwise, cast the array to int64 
        (new buffer) and return the result.
    """
    # Assure 1-D
    assure_1dim_arr(arr)

    # Assure contiguous
    if not np.PyArray_IS_C_CONTIGUOUS(arr):
        arr = np.PyArray_GETCONTIGUOUS(arr)
        copy = False
    
    # Assure dtype
    if np.PyArray_TYPE(arr) == np.NPY_TYPES.NPY_INT64:
        return np.PyArray_Copy(arr) if copy else arr
    else:
        return np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)

cdef inline np.ndarray arr_assure_int64_like(np.ndarray arr, bint copy=True):
    """Ensure that a 1-D array is contiguous and dtype 
    int64 / datetime64[*] / timedelta64[*] `<'ndarray[int64*]'>`.

    :param arr `<'ndarray'>`: The 1-D array to ensure.
    :param copy `<'bool'>`: Whether to always create a copy even if the array is
        already dtype int64 / datetime64[*] / timedelta64[*]. Defaults to `True`.
    :returns `<'ndarray[int64]'>`:
        If the array is already the right dtype, return it directly or a copy
        depending on the `copy` flag. Otherwise, cast the array to int64 
        (new buffer) and return the result.
    """
    # Assure 1-D
    assure_1dim_arr(arr)

    # Assure contiguous
    if not np.PyArray_IS_C_CONTIGUOUS(arr):
        arr = np.PyArray_GETCONTIGUOUS(arr)
        copy = False

    # Assure dtype
    if np.PyArray_TYPE(arr) in (
        np.NPY_TYPES.NPY_INT64,
        np.NPY_TYPES.NPY_DATETIME,
        np.NPY_TYPES.NPY_TIMEDELTA,
    ):
        return np.PyArray_Copy(arr) if copy else arr
    else:
        return np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)

cdef inline np.ndarray arr_assure_float64(np.ndarray arr, bint copy=True):
    """Ensure that a 1-D array is contiguous and dtype float64 `<'ndarray[float64]'>`.

    :param arr `<'ndarray'>`: The 1-D array to ensure.
    :param copy `<'bool'>`: Whether to always create a copy even if the array is
        already dtype float64. Defaults to `True`.
    :returns `<'ndarray[float64]'>`:
        If the array is already the right dtype, return it directly or a copy
        depending on the `copy` flag. Otherwise, cast the array to float64 
        (new buffer) and return the result.
    """
    # Assure 1-D
    assure_1dim_arr(arr)

    # Assure contiguous
    if not np.PyArray_IS_C_CONTIGUOUS(arr):
        arr = np.PyArray_GETCONTIGUOUS(arr)
        copy = False

    # Assure dtype
    if np.PyArray_TYPE(arr) == np.NPY_TYPES.NPY_FLOAT64:
        return np.PyArray_Copy(arr) if copy else arr
    else:
        return np.PyArray_Cast(arr, np.NPY_TYPES.NPY_FLOAT64)

# . create
cdef inline np.ndarray arr_zero_int64(np.npy_intp size):
    """Create a 1-D ndarray[int64] filled with zero `<'ndarray[int64]'>`.

    :param size `<'int'>`: The size of the new array.
    :returns `<'ndarray[int64]'>`: A new 1-D ndarray of the specified size, filled with zeros.

    ## Equivalent
    >>> np.zeros(size, dtype="int64")
    """
    if size < 0:
        raise ValueError("The size of a new array must be a positive integer.")
    else:
        return np.PyArray_ZEROS(1, [size], np.NPY_TYPES.NPY_INT64, 0)

cdef inline np.ndarray arr_fill_int64(np.npy_int64 value, np.npy_intp size):
    """Create a 1-D ndarray[int64] filled with a specific integer `<'ndarray[int64]'>`.

    :param value `<'int'>`: The integer value to fill the array with.
    :param size `<'int'>`: The size of the new array.
    :returns `<'ndarray[int64]'>`: A new 1-D ndarray of the specified size, filled with the given integer.

    ## Equivalent
    >>> np.full(size, value, dtype="int64")
    """
    # Fast-path
    if value == 0:
        return arr_zero_int64(size)

    # New array
    if size < 1:
        raise ValueError("The size of a new array must be a positive integer.")
    cdef:
        np.ndarray arr = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_intp i
    for i in range(size):
        arr_ptr[i] = value
    return arr

# . range
cdef inline np.ndarray arr_clamp(np.ndarray arr, np.npy_int64 minimum, np.npy_int64 maximum, np.npy_int64 offset=0, bint copy=True):
    """Clamp the values of a 1-D ndarray between 'minimum' and 'maximum' value `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param minimum `<'int'>`: The minimum value to clamp to.
    :param maximum `<'int'>`: The maximum value to clamp to.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice:
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.clip(arr, minimum, maximum) + offset
    """
    # Validate
    if minimum > maximum:
        raise ValueError("minimum value cannot be greater than maximum value.")

    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v
    
    # No offset
    if offset == 0:
        for i in range(size):
            v = out_ptr[i]
            # Preserve NaT
            if v == LLONG_MIN:
                continue
            # Clamp value
            if v > maximum:
                out_ptr[i] = maximum
            elif v < minimum:
                out_ptr[i] = minimum

    # With offset
    else:
        for i in range(size):
            v = out_ptr[i]
            # Preserve NaT
            if v == LLONG_MIN:
                continue
            # Clamp value
            if v > maximum:
                v = maximum
            elif v < minimum:
                v = minimum
            out_ptr[i] = v + offset

    return out

cdef inline np.ndarray arr_min(np.ndarray arr, np.npy_int64 value, np.npy_int64 offset=0, bint copy=True):
    """Take the elementwise minimum of a 1-D array with the scalar 'value' `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param value <'int'>: The scalar to compare against.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.minimum(arr, value) + offset
    """
    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v

    # No offset
    if offset == 0:
        for i in range(size):
            v = out_ptr[i]
            # Preserve NaT
            if v == LLONG_MIN:
                continue
            # Min value
            if v > value:
                out_ptr[i] = value

    # With offset
    else:
        for i in range(size):
            v = out_ptr[i]
            # Preserve NaT
            if v == LLONG_MIN:
                continue
            # Min value
            if v > value:
                v = value
            out_ptr[i] = v + offset

    return out

cdef inline np.ndarray arr_max(np.ndarray arr, np.npy_int64 value, np.npy_int64 offset=0, bint copy=True):
    """Take the elementwise maximum of a 1-D array with the scalar 'value' `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param value <'int'>: The scalar to compare against.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.maximum(arr, value) + offset
    """
    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v

    # No offset
    if offset == 0:
        for i in range(size):
            v = out_ptr[i]
            # Preserve NaT
            if v == LLONG_MIN:
                continue
            # Max value
            if v < value:
                out_ptr[i] = value

    # With offset
    else:
        for i in range(size):
            v = out_ptr[i]
            # Preserve NaT
            if v == LLONG_MIN:
                continue
            # Max value
            if v < value:
                v = value
            out_ptr[i] = v + offset

    return out

cdef inline np.ndarray arr_min_arr(np.ndarray arr1, np.ndarray arr2, np.npy_int64 offset=0, bint copy=True):
    """Take the elementwise minimum of two 1-D arrays <'ndarray[int64]'>.

    :param arr1 `<'np.ndarray'>`: The first 1-D array. If dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param arr2 `<'np.ndarray'>`: The second 1-D array. Same casting rules as 'arr1'.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify `arr1` in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) propagate: if either operand is NaT, the result is NaT.
      LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.minimum(arr1, arr2) + offset
    """
    # Validate
    cdef np.npy_intp size = arr1.shape[0]
    if size != arr2.shape[0]:
        raise ValueError(
            "Cannot compare ndarrays with different lengths.\n"
            "  - arr1 length: %d\n"
            "  - arr2 length: %d" % (size, arr2.shape[0])
        )

    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr1, copy)
        np.ndarray arr = arr_assure_int64_like(arr2, False)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_intp i
        np.npy_int64 v1, v2

    # No offset
    if offset == 0:
        for i in range(size):
            # Propagate NaT
            v1 = out_ptr[i]
            if v1 == LLONG_MIN:
                continue
            v2 = arr_ptr[i]
            if v2 == LLONG_MIN:
                out_ptr[i] = v2
                continue
            # Min value
            if v1 > v2:
                out_ptr[i] = v2

    # With offset
    else:
        for i in range(size):
            # Propagate NaT
            v1 = out_ptr[i]
            if v1 == LLONG_MIN:
                continue
            v2 = arr_ptr[i]
            if v2 == LLONG_MIN:
                out_ptr[i] = v2
                continue
            # Min value
            if v1 > v2:
                out_ptr[i] = v2 + offset
            else:
                out_ptr[i] = v1 + offset
        
    return out

cdef inline np.ndarray arr_max_arr(np.ndarray arr1, np.ndarray arr2, np.npy_int64 offset=0, bint copy=True):
    """Take the elementwise maximum of two 1-D arrays <'ndarray[int64]'>.

    :param arr1 `<'np.ndarray'>`: The first 1-D array.
    :param arr2 `<'np.ndarray'>`: The second 1-D array.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify `arr1` in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) propagate: if either operand is NaT, the result is NaT.
      LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.maximum(arr1, arr2) + offset
    """
    # Validate
    cdef np.npy_intp size = arr1.shape[0]
    if size != arr2.shape[0]:
        raise ValueError(
            "Cannot compare ndarrays with different lengths.\n"
            "  - arr1 length: %d\n"
            "  - arr2 length: %d" % (size, arr2.shape[0])
        )

    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr1, copy)
        np.ndarray arr = arr_assure_int64_like(arr2, False)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_intp i
        np.npy_int64 v1, v2

    # No offset
    if offset == 0:
        for i in range(size):
            # Propagate NaT
            v1 = out_ptr[i]
            if v1 == LLONG_MIN:
                continue
            v2 = arr_ptr[i]
            if v2 == LLONG_MIN:
                out_ptr[i] = v2
                continue
            # Max value
            if v1 < v2:
                out_ptr[i] = v2

    # With offset
    else:
        for i in range(size):
            # Propagate NaT
            v1 = out_ptr[i]
            if v1 == LLONG_MIN:
                continue
            v2 = arr_ptr[i]
            if v2 == LLONG_MIN:
                out_ptr[i] = v2
                continue
            # Max value
            if v1 < v2:
                out_ptr[i] = v2 + offset
            else:
                out_ptr[i] = v1 + offset

    return out

# . arithmetic
cdef inline np.ndarray arr_abs(np.ndarray arr, np.npy_int64 offset=0, bint copy=True):
    """Take the elementwise absolute of a 1-D array `<'ndarray[int64]'`>

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.abs(arr) + offset
    """
    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v

    # No offset
    if offset == 0:
        for i in range(size):
            v = out_ptr[i]
            # Preserve NaT
            if v == LLONG_MIN:
                continue
            # Abs value
            if v < 0:
                out_ptr[i] = -v
    
    # With offset
    else:
        for i in range(size):
            v = out_ptr[i]
            # Preserve NaT
            if v == LLONG_MIN:
                continue
            # Abs value
            if v < 0:
                v = -v
            out_ptr[i] = v + offset

    return out

cdef inline np.ndarray arr_neg(np.ndarray arr, np.npy_int64 offset=0, bint copy=True):
    """Negate the values of a 1-D array `<'ndarray[int64]'>`

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> -arr + offset
    """
    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v

    # Compute
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Negate value
        out_ptr[i] = -v + offset

    return out

cdef inline np.ndarray arr_add(np.ndarray arr, np.npy_int64 value, bint copy=True):
    """Add a 'value' to the 1-D array `<'np.ndarray'>`

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param value `<'int'>`: The value to add to each element.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr + value
    """
    # Fast-path
    cdef np.ndarray out = arr_assure_int64_like(arr, copy)
    if value == 0:
        return out

    # Setup
    cdef:
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v

    # Compute
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Add value
        out_ptr[i] = v + value

    return out

cdef inline np.ndarray arr_mul(np.ndarray arr, np.npy_int64 factor, np.npy_int64 offset=0, bint copy=True):
    """Multiply the values of a 1-D array by the 'factor' `<'np.ndarray'>`

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: The scalar multiplier.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    
    ## Equivalent
    >>> arr * factor + offset
    """
    # Fast-path
    if factor == 1:
        return arr_add(arr, offset, copy)

    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v

    # Zero factor
    if factor == 0:
        for i in range(size):
            v = out_ptr[i]
            # Preserve NaT
            if v == LLONG_MIN:
                continue
            # Set to offset
            out_ptr[i] = offset

    # Compute
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Multiply value
        out_ptr[i] = v * factor + offset

    return out

cdef inline np.ndarray arr_mod(np.ndarray arr, np.npy_int64 factor, np.npy_int64 offset=0, bint copy=True):
    """Elementwise modulo of a 1-D array by `factor` with Python semantics 
    (remainder has the same sign as the divisor) `<'np.ndarray'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr % factor + offset
    """
    # Validate
    if factor == 0:
        raise ZeroDivisionError("arr_mod: division by zero.")

    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v
        bint neg_f = factor < 0

    # Fast path: factor == ±1
    if factor == 1 or factor == -1:
        for i in range(size):
            v = out_ptr[i]
            # Preserve NaT
            if v == LLONG_MIN:
                continue
            # Set to offset
            out_ptr[i] = offset
        return out

    # Compute
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Modulo value
        with cython.cdivision(True):
            v = v % factor
        if v != 0 and ((v < 0) != neg_f):
            v += factor
        out_ptr[i] = v + offset

    return out

cdef inline np.ndarray arr_div_even(np.ndarray arr, np.npy_int64 factor, np.npy_int64 offset=0, bint copy=True):
    """Divide a 1-D array by `factor` and rounds to the nearest integers (half-to-even) <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.round(arr / factor, 0) + offset
    """
    # Validate / Fast-path
    if factor == 0:
        raise ZeroDivisionError("arr_div_even: division by zero.")
    if factor == 1:  # factor == 1 → q = v (no rounding) + offset
        return arr_add(arr, offset, copy)
    if factor == -1:  # factor == -1 → q = -v (exact) + offset
        return arr_neg(arr, offset, copy)

    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, q, r, abs_f, abs_r, half

    # Compute
    abs_f = -factor if factor < 0 else factor
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Divide value
        with cython.cdivision(True):
            q = v / factor; r = v % factor
        if r != 0:
            abs_r = -r if r < 0 else r
            half  = abs_f - abs_r
            if abs_r > half or (abs_r == half and (q & 1) != 0):
                q += 1 if ((v ^ factor) >= 0) else -1
        out_ptr[i] = q + offset

    return out

cdef inline np.ndarray arr_div_even_mul(np.ndarray arr, np.npy_int64 factor, np.npy_int64 multiple, np.npy_int64 offset=0, bint copy=True):
    """Divide a 1-D array by `factor` and rounds to the nearest integers (half-to-even),
    then scale by `multiple` <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param multiple `<'int'>`: The scalar multiplier.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.round(arr / factor, 0) * multiple + offset
    """
    # Validate / Fast-path
    if multiple == 0:
        return arr_mul(arr, 0, offset, copy)
    if multiple == 1:
        return arr_div_even(arr, factor, offset, copy)
    if factor == 0:
        raise ZeroDivisionError("arr_div_even_mul: division by zero.")
    if factor == 1:  # factor == 1 → q = v (no rounding) * multiple + offset
        return arr_mul(arr, multiple, offset, copy)
    if factor == -1:  # factor == -1 → q = v (exact) * -multiple + offset
        return arr_mul(arr, -multiple, offset, copy)

    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, q, r, abs_f, abs_r, half

    # Compute
    abs_f = -factor if factor < 0 else factor
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Divide & Multiply
        with cython.cdivision(True):
            q = v / factor; r = v % factor
        if r != 0:
            abs_r = -r if r < 0 else r
            half  = abs_f - abs_r
            if abs_r > half or (abs_r == half and (q & 1) != 0):
                q += 1 if ((v ^ factor) >= 0) else -1
        out_ptr[i] = q * multiple + offset

    return out

cdef inline np.ndarray arr_div_up(np.ndarray arr, np.npy_int64 factor, np.npy_int64 offset=0, bint copy=True):
    """Divide a 1-D array by `factor` and rounds to the nearest integers (half-up / away-from-zero) <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    """
    # Validate / Fast-path
    if factor == 0:
        raise ZeroDivisionError("arr_div_up: division by zero.")
    if factor == 1:  # factor == 1 → q = v (no rounding) + offset
        return arr_add(arr, offset, copy)
    if factor == -1:  # factor == -1 → q = -v (exact) + offset
        return arr_neg(arr, offset, copy)

    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, q, r, abs_f, abs_r, half

    # Compute
    abs_f = -factor if factor < 0 else factor
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Divide value
        with cython.cdivision(True):
            q = v / factor; r = v % factor
        if r != 0:
            abs_r = -r if r < 0 else r
            half  = abs_f - abs_r
            if abs_r >= half:
                q += 1 if ((v ^ factor) >= 0) else -1
        out_ptr[i] = q + offset

    return out

cdef inline np.ndarray arr_div_up_mul(np.ndarray arr, np.npy_int64 factor, np.npy_int64 multiple, np.npy_int64 offset=0, bint copy=True):
    """Divide a 1-D array by `factor` and rounds to nearest away from 
    zero (half-up), then scale by 'multiple' <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param multiple `<'int'>`: The scalar multiplier.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    """
    # Validate / Fast-path
    if multiple == 0:
        return arr_mul(arr, 0, offset, copy)
    if multiple == 1:
        return arr_div_up(arr, factor, offset, copy)
    if factor == 0:
        raise ZeroDivisionError("arr_div_up_mul: division by zero.")
    if factor == 1:  # factor == 1 → q = v (no rounding) * multiple + offset
        return arr_mul(arr, multiple, offset, copy)
    if factor == -1:  # factor == -1 → q = v (exact) * -multiple + offset
        return arr_mul(arr, -multiple, offset, copy)

    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, q, r, abs_f, abs_r, half

    # Compute
    abs_f = -factor if factor < 0 else factor
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Divide & Multiply
        with cython.cdivision(True):
            q = v / factor; r = v % factor
        if r != 0:
            abs_r = -r if r < 0 else r
            half  = abs_f - abs_r
            if abs_r >= half:
                q += 1 if ((v ^ factor) >= 0) else -1
        out_ptr[i] = q * multiple + offset

    return out

cdef inline np.ndarray arr_div_down(np.ndarray arr, np.npy_int64 factor, np.npy_int64 offset=0, bint copy=True):
    """Divide a 1-D array by `factor` and rounds to the nearest 
    integers (half-down / toward-zero) <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    """
    # Validate / Fast-path
    if factor == 0:
        raise ZeroDivisionError("arr_div_down: division by zero.")
    if factor == 1:  # factor == 1 → q = v (no rounding) + offset
        return arr_add(arr, offset, copy)
    if factor == -1:  # factor == -1 → q = -v (exact) + offset
        return arr_neg(arr, offset, copy)

    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, q, r, abs_f, abs_r, half

    # Compute
    abs_f = -factor if factor < 0 else factor
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Divide value
        with cython.cdivision(True):
            q = v / factor; r = v % factor
        if r != 0:
            abs_r = -r if r < 0 else r
            half  = abs_f - abs_r
            if abs_r > half:
                q += 1 if ((v ^ factor) >= 0) else -1
        out_ptr[i] = q + offset

    return out

cdef inline np.ndarray arr_div_down_mul(np.ndarray arr, np.npy_int64 factor, np.npy_int64 multiple, np.npy_int64 offset=0, bint copy=True):
    """Divide a 1-D array by `factor` and rounds to the nearest integers (half-down),
    then scale by `multiple` <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param multiple `<'int'>`: The scalar multiplier.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    """
    # Validate / Fast-path
    if multiple == 0:
        return arr_mul(arr, 0, offset, copy)
    if multiple == 1:
        return arr_div_down(arr, factor, offset, copy)
    if factor == 0:
        raise ZeroDivisionError("arr_div_down_mul: division by zero.")
    if factor == 1:  # factor == 1 → q = v (no rounding) * multiple + offset
        return arr_mul(arr, multiple, offset, copy)
    if factor == -1:  # factor == -1 → q = v (exact) * -multiple + offset
        return arr_mul(arr, -multiple, offset, copy)

    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, q, r, abs_f, abs_r, half

    # Compute
    abs_f = -factor if factor < 0 else factor
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Divide & Multiply
        with cython.cdivision(True):
            q = v / factor; r = v % factor
        if r != 0:
            abs_r = -r if r < 0 else r
            half  = abs_f - abs_r
            if abs_r > half:
                q += 1 if ((v ^ factor) >= 0) else -1
        out_ptr[i] = q * multiple + offset

    return out

cdef inline np.ndarray arr_div_ceil(np.ndarray arr, np.npy_int64 factor, np.npy_int64 offset=0, bint copy=True):
    """Divide a 1-D array by `factor` and ceils up to the nearest integers `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.ceil(arr / factor) + offset
    """
    # Validate / Fast-path
    if factor == 0:
        raise ZeroDivisionError("arr_div_ceil: division by zero.")
    if factor == 1:  # factor == 1 → q = v (no rounding) + offset
        return arr_add(arr, offset, copy)
    if factor == -1:  # factor == -1 → q = -v (exact) + offset
        return arr_neg(arr, offset, copy)

    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, q, r

    # Compute
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Divide value
        with cython.cdivision(True):
            q = v / factor; r = v % factor
        if r != 0 and ((v ^ factor) >= 0):
            q += 1
        out_ptr[i] = q + offset

    return out

cdef inline np.ndarray arr_div_ceil_mul(np.ndarray arr, np.npy_int64 factor, np.npy_int64 multiple, np.npy_int64 offset=0, bint copy=True):
    """Divide a 1-D array by `factor` and ceils up to the nearest integers,
    then scale by `multiple` <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param multiple `<'int'>`: The scalar multiplier.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.ceil(arr / factor) * multiple + offset
    """
    # Validate / Fast-path
    if multiple == 0:
        return arr_mul(arr, 0, offset, copy)
    if multiple == 1:
        return arr_div_ceil(arr, factor, offset, copy)
    if factor == 0:
        raise ZeroDivisionError("arr_div_ceil_mul: division by zero.")
    if factor == 1:  # factor == 1 → q = v (no rounding) * multiple + offset
        return arr_mul(arr, multiple, offset, copy)
    if factor == -1:  # factor == -1 → q = v (exact) * -multiple + offset
        return arr_mul(arr, -multiple, offset, copy)

    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, q, r

    # Compute
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Divide value
        with cython.cdivision(True):
            q = v / factor; r = v % factor
        if r != 0 and ((v ^ factor) >= 0):
            q += 1
        out_ptr[i] = q * multiple + offset

    return out

cdef inline np.ndarray arr_div_floor(np.ndarray arr, np.npy_int64 factor, np.npy_int64 offset=0, bint copy=True):
    """Divide a 1-D array by `factor` and floors down to the nearest integers `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.floor(arr / factor) + offset
    """
    # Validate / Fast-path
    if factor == 0:
        raise ZeroDivisionError("arr_div_floor: division by zero.")
    if factor == 1:  # factor == 1 → q = v (no rounding) + offset
        return arr_add(arr, offset, copy)
    if factor == -1:  # factor == -1 → q = -v (exact) + offset
        return arr_neg(arr, offset, copy)

    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, q, r

    # Compute
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Divide value
        with cython.cdivision(True):
            q = v / factor; r = v % factor
        if r != 0 and ((v ^ factor) < 0):
            q -= 1
        out_ptr[i] = q + offset

    return out

cdef inline np.ndarray arr_div_floor_mul(np.ndarray arr, np.npy_int64 factor, np.npy_int64 multiple, np.npy_int64 offset=0, bint copy=True):
    """Divide a 1-D array by `factor` and floors down to the nearest integers,
    then scale by `multiple` <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param multiple `<'int'>`: The scalar multiplier.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.floor(arr / factor) * multiple + offset
    """
    # Validate / Fast-path
    if multiple == 0:
        return arr_mul(arr, 0, offset, copy)
    if multiple == 1:
        return arr_div_floor(arr, factor, offset, copy)
    if factor == 0:
        raise ZeroDivisionError("arr_div_floor_mul: division by zero.")
    if factor == 1:  # factor == 1 → q = v (no rounding) * multiple + offset
        return arr_mul(arr, multiple, offset, copy)
    if factor == -1:  # factor == -1 → q = v (exact) * -multiple + offset
        return arr_mul(arr, -multiple, offset, copy)

    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, q, r

    # Compute
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Divide value
        with cython.cdivision(True):
            q = v / factor; r = v % factor
        if r != 0 and ((v ^ factor) < 0):
            q -= 1
        out_ptr[i] = q * multiple + offset

    return out

cdef inline np.ndarray arr_div_trunc(np.ndarray arr, np.npy_int64 factor, np.npy_int64 offset=0, bint copy=True):
    """Divide a 1-D array by `factor` and and truncate toward zero `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    """
    # Validate / Fast-path
    if factor == 0:
        raise ZeroDivisionError("arr_div_trunc: division by zero.")
    if factor == 1:  # factor == 1 → q = v (no rounding) + offset
        return arr_add(arr, offset, copy)
    if factor == -1:  # factor == -1 → q = -v (exact) + offset
        return arr_neg(arr, offset, copy)

    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, q

    # Compute
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Divide value
        with cython.cdivision(True):
            q = v / factor
        out_ptr[i] = q + offset

    return out

cdef inline np.ndarray arr_div_trunc_mul(np.ndarray arr, np.npy_int64 factor, np.npy_int64 multiple, np.npy_int64 offset=0, bint copy=True):
    """Divide a 1-D array by `factor` and truncate toward zero,
    then scale by `multiple` <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param multiple `<'int'>`: The scalar multiplier.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    """
    # Validate / Fast-path
    if multiple == 0:
        return arr_mul(arr, 0, offset, copy)
    if multiple == 1:
        return arr_div_trunc(arr, factor, offset, copy)
    if factor == 0:
        raise ZeroDivisionError("arr_div_trunc_mul: division by zero.")
    if factor == 1:  # factor == 1 → q = v (no rounding) * multiple + offset
        return arr_mul(arr, multiple, offset, copy)
    if factor == -1:  # factor == -1 → q = v (exact) * -multiple + offset
        return arr_mul(arr, -multiple, offset, copy)
        
    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, q

    # Compute
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Divide value
        with cython.cdivision(True):
            q = v / factor
        out_ptr[i] = q * multiple + offset

    return out

cdef inline np.ndarray arr_add_arr(np.ndarray arr1, np.ndarray arr2, np.npy_int64 offset=0, bint copy=True):
    """Elementwise addition of two 1-D arrays <'ndarray[int64]'>.

    :param arr1 `<'np.ndarray'>`: The first 1-D array. If dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param arr2 `<'np.ndarray'>`: The second 1-D array. Same casting rules as 'arr1'.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify `arr1` in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) propagate: if either operand is NaT, the result is NaT.
      LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr1 + arr2 + offset
    """
    # Validate
    cdef np.npy_intp size = arr1.shape[0]
    if size != arr2.shape[0]:
        raise ValueError(
            "Cannot perform addition on ndarrays with different lengths.\n"
            "  - arr1 length: %d\n"
            "  - arr2 length: %d" % (size, arr2.shape[0])
        )
        
    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr1, copy)
        np.ndarray arr = arr_assure_int64_like(arr2, False)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_intp i
        np.npy_int64 v1, v2

    # Compute
    for i in range(size):
        # Propagate NaT
        v1 = out_ptr[i]
        if v1 == LLONG_MIN:
            continue
        v2 = arr_ptr[i]
        if v2 == LLONG_MIN:
            out_ptr[i] = v2
            continue
        # Add values
        out_ptr[i] = v1 + v2 + offset

    return out

cdef inline np.ndarray arr_sub_arr(np.ndarray arr1, np.ndarray arr2, np.npy_int64 offset=0, bint copy=True):
    """Elementwise subtraction of two 1-D arrays <'ndarray[int64]'>.

    :param arr1 `<'np.ndarray'>`: The first 1-D array. If dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param arr2 `<'np.ndarray'>`: The second 1-D array. Same casting rules as 'arr1'.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify `arr1` in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) propagate: if either operand is NaT, the result is NaT.
      LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr1 - arr2 + offset
    """
    # Validate
    cdef np.npy_intp size = arr1.shape[0]
    if size != arr2.shape[0]:
        raise ValueError(
            "Cannot perform subtraction on ndarrays with different lengths.\n"
            "  - arr1 length: %d\n"
            "  - arr2 length: %d" % (size, arr2.shape[0])
        )

    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr1, copy)
        np.ndarray arr = arr_assure_int64_like(arr2, False)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_intp i
        np.npy_int64 v1, v2

    # Compute
    for i in range(size):
        # Propagate NaT
        v1 = out_ptr[i]
        if v1 == LLONG_MIN:
            continue
        v2 = arr_ptr[i]
        if v2 == LLONG_MIN:
            out_ptr[i] = v2
            continue
        # Subtract values
        out_ptr[i] = v1 - v2 + offset

    return out

cdef inline np.ndarray arr_mul_arr(np.ndarray arr1, np.ndarray arr2, np.npy_int64 offset=0, bint copy=True):
    """Elementwise multiplication of two 1-D arrays <'ndarray[int64]'>.

    :param arr1 `<'np.ndarray'>`: The first 1-D array. If dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param arr2 `<'np.ndarray'>`: The second 1-D array. Same casting rules as 'arr1'.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify `arr1` in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) propagate: if either operand is NaT, the result is NaT.
      LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr1 * arr2 + offset
    """
    # Validate
    cdef np.npy_intp size = arr1.shape[0]
    if size != arr2.shape[0]:
        raise ValueError(
            "Cannot perform multiplication on ndarrays with different lengths.\n"
            "  - arr1 length: %d\n"
            "  - arr2 length: %d" % (size, arr2.shape[0])
        )

    # Setup
    cdef:
        np.ndarray out = arr_assure_int64_like(arr1, copy)
        np.ndarray arr = arr_assure_int64_like(arr2, False)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_intp i
        np.npy_int64 v1, v2

    # Compute
    for i in range(size):
        # Propagate NaT
        v1 = out_ptr[i]
        if v1 == LLONG_MIN:
            continue
        v2 = arr_ptr[i]
        if v2 == LLONG_MIN:
            out_ptr[i] = v2
            continue
        # Multiply values
        out_ptr[i] = v1 * v2 + offset

    return out

# . comparison
cdef inline np.ndarray arr_eq(np.ndarray arr, np.npy_int64 value):
    """Elementwise equal comparison of a 1-D array to a scalar 'value' `<'ndarray[bool]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param value `<'int'>`: The scalar value to compare against.
    :returns `<'ndarray[bool]'>`: The boolean result array.
    
    ## Notice
    - For datetime64/timedelta64 inputs, `value` is interpreted in the 
      array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT is represented as LLONG_MIN and is compared as a normal integer
      (unlike NumPy's datetime behavior where NaT comparisons always yield False).

    ## Equivalent
    >>> arr == value
    """
    # Setup
    arr = arr_assure_int64_like(arr, False)
    cdef:
        # Target array
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_intp size = arr.shape[0]
        np.npy_intp i
        # Output array
        np.ndarray out = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_BOOL, 0)
        np.npy_bool* out_ptr = <np.npy_bool*> np.PyArray_DATA(out)

    # Compare
    for i in range(size):
        out_ptr[i] = arr_ptr[i] == value
    return out

cdef inline np.ndarray arr_gt(np.ndarray arr, np.npy_int64 value):
    """Elementwise greater-than comparison of a 1-D array to a scalar 'value' `<'ndarray[bool]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param value `<'int'>`: The scalar value to compare against.
    :returns `<'ndarray[bool]'>`: The boolean result array.
    
    ## Notice
    - For datetime64/timedelta64 inputs, `value` is interpreted in the 
      array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT is represented as LLONG_MIN and is compared as a normal integer
      (unlike NumPy's datetime behavior where NaT comparisons always yield False).

    ## Equivalent
    >>> arr > value
    """
    # Setup
    arr = arr_assure_int64_like(arr, False)
    cdef:
        # Target array
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_intp size = arr.shape[0]
        np.npy_intp i
        # Output array
        np.ndarray out = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_BOOL, 0)
        np.npy_bool* out_ptr = <np.npy_bool*> np.PyArray_DATA(out)

    # Compare
    for i in range(size):
        out_ptr[i] = arr_ptr[i] > value
    return out

cdef inline np.ndarray arr_ge(np.ndarray arr, np.npy_int64 value):
    """Elementwise greater-than-or-equal comparison of a 1-D array to a scalar 'value' `<'ndarray[bool]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param value `<'int'>`: The scalar value to compare against.
    :returns `<'ndarray[bool]'>`: The boolean result array.
    
    ## Notice
    - For datetime64/timedelta64 inputs, `value` is interpreted in the 
      array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT is represented as LLONG_MIN and is compared as a normal integer
      (unlike NumPy's datetime behavior where NaT comparisons always yield False).

    ## Equivalent
    >>> arr >= value
    """
    # Setup
    arr = arr_assure_int64_like(arr, False)
    cdef:
        # Target array
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_intp size = arr.shape[0]
        np.npy_intp i
        # Output array
        np.ndarray out = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_BOOL, 0)
        np.npy_bool* out_ptr = <np.npy_bool*> np.PyArray_DATA(out)

    # Compare
    for i in range(size):
        out_ptr[i] = arr_ptr[i] >= value
    return out

cdef inline np.ndarray arr_lt(np.ndarray arr, np.npy_int64 value):
    """Elementwise less-than comparison of a 1-D array to a scalar 'value' `<'ndarray[bool]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param value `<'int'>`: The scalar value to compare against.
    :returns `<'ndarray[bool]'>`: The boolean result array.
    
    ## Notice
    - For datetime64/timedelta64 inputs, `value` is interpreted in the 
      array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT is represented as LLONG_MIN and is compared as a normal integer
      (unlike NumPy's datetime behavior where NaT comparisons always yield False).

    ## Equivalent
    >>> arr < value
    """
    # Setup
    arr = arr_assure_int64_like(arr, False)
    cdef:
        # Target array
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_intp size = arr.shape[0]
        np.npy_intp i
        # Output array
        np.ndarray out = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_BOOL, 0)
        np.npy_bool* out_ptr = <np.npy_bool*> np.PyArray_DATA(out)

    # Compare
    for i in range(size):
        out_ptr[i] = arr_ptr[i] < value
    return out

cdef inline np.ndarray arr_le(np.ndarray arr, np.npy_int64 value):
    """Elementwise less-than-or-equal comparison of a 1-D array to a scalar 'value' `<'ndarray[bool]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param value `<'int'>`: The scalar value to compare against.
    :returns `<'ndarray[bool]'>`: The boolean result array.
    
    ## Notice
    - For datetime64/timedelta64 inputs, `value` is interpreted in the 
      array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT is represented as LLONG_MIN and is compared as a normal integer
      (unlike NumPy's datetime behavior where NaT comparisons always yield False).

    ## Equivalent
    >>> arr <= value
    """
    # Setup
    arr = arr_assure_int64_like(arr, False)
    cdef:
        # Target array
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_intp size = arr.shape[0]
        np.npy_intp i
        # Output array
        np.ndarray out = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_BOOL, 0)
        np.npy_bool* out_ptr = <np.npy_bool*> np.PyArray_DATA(out)

    # Compare
    for i in range(size):
        out_ptr[i] = arr_ptr[i] <= value
    return out

cdef inline np.ndarray arr_eq_arr(np.ndarray arr1, np.ndarray arr2):
    """Elementwise equal comparison of two 1-D arrays `<'ndarray[bool]'>`.

    :param arr1 `<'np.ndarray'>`: The first 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param arr2 `<'np.ndarray'>`: The second 1-D array. Same casting rules as 'arr1'.
    :returns `<'ndarray[bool]'>`: The boolean result array.
    
    ## Notice
    - For datetime64/timedelta64 inputs, comparison is performed on underlying int64 ticks.
    - NaT is represented as LLONG_MIN and is compared as a normal integer
      (unlike NumPy's datetime behavior where NaT comparisons always yield False).

    ## Equivalent
    >>> arr1 == arr2
    """
    # Validate
    cdef np.npy_intp size = arr1.shape[0]
    if size != arr2.shape[0]:
        raise ValueError(
            "Cannot compare ndarrays with different lengths.\n"
            "  - arr1 length: %d\n"
            "  - arr2 length: %d" % (size, arr2.shape[0])
        )

    # Setup
    arr1 = arr_assure_int64_like(arr1, False)
    arr2 = arr_assure_int64_like(arr2, False)
    cdef:
        # Target arrays
        np.npy_int64* arr1_ptr = <np.npy_int64*> np.PyArray_DATA(arr1)
        np.npy_int64* arr2_ptr = <np.npy_int64*> np.PyArray_DATA(arr2)
        np.npy_intp i
        # Output array
        np.ndarray out = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_BOOL, 0)
        np.npy_bool* out_ptr = <np.npy_bool*> np.PyArray_DATA(out)
    
    # Compare
    for i in range(size):
        out_ptr[i] = arr1_ptr[i] == arr2_ptr[i]
    return out

cdef inline np.ndarray arr_gt_arr(np.ndarray arr1, np.ndarray arr2):
    """Elementwise greater-than comparison of two 1-D arrays `<'ndarray[bool]'>`.

    :param arr1 `<'np.ndarray'>`: The first 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param arr2 `<'np.ndarray'>`: The second 1-D array. Same casting rules as 'arr1'.
    :returns `<'ndarray[bool]'>`: The boolean result array.
    
    ## Notice
    - For datetime64/timedelta64 inputs, comparison is performed on underlying int64 ticks.
    - NaT is represented as LLONG_MIN and is compared as a normal integer
      (unlike NumPy's datetime behavior where NaT comparisons always yield False).

    ## Equivalent
    >>> arr1 > arr2
    """
    # Validate
    cdef np.npy_intp size = arr1.shape[0]
    if size != arr2.shape[0]:
        raise ValueError(
            "Cannot compare ndarrays with different lengths.\n"
            "  - arr1 length: %d\n"
            "  - arr2 length: %d" % (size, arr2.shape[0])
        )

    # Setup
    arr1 = arr_assure_int64_like(arr1, False)
    arr2 = arr_assure_int64_like(arr2, False)
    cdef:
        # Target arrays
        np.npy_int64* arr1_ptr = <np.npy_int64*> np.PyArray_DATA(arr1)
        np.npy_int64* arr2_ptr = <np.npy_int64*> np.PyArray_DATA(arr2)
        np.npy_intp i
        # Output array
        np.ndarray out = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_BOOL, 0)
        np.npy_bool* out_ptr = <np.npy_bool*> np.PyArray_DATA(out)

    # Compare
    for i in range(size):
        out_ptr[i] = arr1_ptr[i] > arr2_ptr[i]
    return out

cdef inline np.ndarray arr_ge_arr(np.ndarray arr1, np.ndarray arr2):
    """Elementwise greater-than-or-equal comparison of two 1-D arrays `<'ndarray[bool]'>`.

    :param arr1 `<'np.ndarray'>`: The first 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param arr2 `<'np.ndarray'>`: The second 1-D array. Same casting rules as 'arr1'.
    :returns `<'ndarray[bool]'>`: The boolean result array.
    
    ## Notice
    - For datetime64/timedelta64 inputs, comparison is performed on underlying int64 ticks.
    - NaT is represented as LLONG_MIN and is compared as a normal integer
      (unlike NumPy's datetime behavior where NaT comparisons always yield False).

    ## Equivalent
    >>> arr1 >= arr2
    """
    # Validate
    cdef np.npy_intp size = arr1.shape[0]
    if size != arr2.shape[0]:
        raise ValueError(
            "Cannot compare ndarrays with different lengths.\n"
            "  - arr1 length: %d\n"
            "  - arr2 length: %d" % (size, arr2.shape[0])
        )

    # Setup
    arr1 = arr_assure_int64_like(arr1, False)
    arr2 = arr_assure_int64_like(arr2, False)
    cdef:
        # Target arrays
        np.npy_int64* arr1_ptr = <np.npy_int64*> np.PyArray_DATA(arr1)
        np.npy_int64* arr2_ptr = <np.npy_int64*> np.PyArray_DATA(arr2)
        np.npy_intp i
        # Output array
        np.ndarray out = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_BOOL, 0)
        np.npy_bool* out_ptr = <np.npy_bool*> np.PyArray_DATA(out)

    # Compare
    for i in range(size):
        out_ptr[i] = arr1_ptr[i] >= arr2_ptr[i]
    return out

# NumPy: ndarray[datetime64] ---------------------------------------------------------------------------
# . type check
cdef inline bint is_dt64arr(np.ndarray arr) except -1:
    """Check if the array is dtype of 'datetime64[*]' `<'bool'>`.
    
    ## Equivalent
    >>> isinstance(arr.dtype, np.dtypes.DateTime64DType)
    """
    return np.PyArray_TYPE(arr) == np.NPY_TYPES.NPY_DATETIME

cdef inline bint assure_dt64arr(np.ndarray arr) except -1:
    """Assure the array is dtype of 'datetime64[*]'"""
    if not is_dt64arr(arr):
        raise TypeError(
            "Expects an instance of 'np.ndarray[datetime64[*]]', "
            "instead got 'np.ndarray[%s]'." % arr.dtype
        )
    return True

# . range check
cdef inline bint is_dt64arr_ns_safe(np.ndarray arr, int arr_reso=-1) except -1:
    """Check if a 1-D array can be safely represented as datetime64[ns] `<'bool'>`.

    A value is **ns-safe** if, when interpreted using `arr_reso`, 
    it lies strictly inside the datetime64[ns] valid range
    (1677-09-22 .. 2262-04-10, endpoints excluded). 
    NaT (LLONG_MIN) is ignored.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :returns `<'bool'>`: True if all non-NaT values are ns-safe; otherwise False.
    """
    # Get array resolution
    if arr_reso < 0:
        if np.PyArray_TYPE(arr) != np.NPY_TYPES.NPY_DATETIME:
            _raise_missing_arr_reso_error(arr)
        arr_reso = get_arr_nptime_unit(arr)

    # Get min & max range
    cdef np.npy_int64 minimum, maximum
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        minimum, maximum = DT64_NS_NS_MIN, DT64_NS_NS_MAX
    elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        minimum, maximum = DT64_NS_US_MIN, DT64_NS_US_MAX
    elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        minimum, maximum = DT64_NS_MS_MIN, DT64_NS_MS_MAX
    elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        minimum, maximum = DT64_NS_SS_MIN, DT64_NS_SS_MAX
    elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        minimum, maximum = DT64_NS_MI_MIN, DT64_NS_MI_MAX
    elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        minimum, maximum = DT64_NS_HH_MIN, DT64_NS_HH_MAX
    elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        minimum, maximum = DT64_NS_DD_MIN, DT64_NS_DD_MAX
    elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        minimum, maximum = DT64_NS_WW_MIN, DT64_NS_WW_MAX
    elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        minimum, maximum = DT64_NS_MM_MIN, DT64_NS_MM_MAX
    elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        minimum, maximum = DT64_NS_YY_MIN, DT64_NS_YY_MAX
    else:
        raise ValueError(
            "Unsupported datetime resolution '%s'. "
            "Accepts: Y, M, W, D, h, m, s, ms, us, ns."
            % nptime_unit_int2str(arr_reso)
        )

    # Setup
    arr = arr_assure_int64_like(arr, False)
    cdef: 
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_intp size = arr.shape[0]
        np.npy_intp i
        np.npy_int64 v

    # Check
    for i in range(size):
        v = arr_ptr[i]
        if not minimum < v < maximum and v != LLONG_MIN:
            return False
    return True

# . access
cdef inline np.ndarray dt64arr_year(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Extract civil year numbers from a 1-D ndarray[datetime64[*]] `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: The civil year for each element.
        NaT values (LLONG_MIN) are preserved as the year values.

    ## Equivalent
    >>> arr.astype('datetime64[Y]').astype('int64') + 1970 + offset
    """
    cdef np.ndarray out = dt64arr_as_int64_Y(arr, arr_reso, 0, copy)
    return arr_add(out, EPOCH_YEAR + offset, False)

cdef inline np.ndarray dt64arr_quarter(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Extract quarter numbers (1..4) from a 1-D ndarray[datetime64[*]] `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: The quarter number for each element.
        NaT values (LLONG_MIN) are preserved as the quarter values.

    ## Equivalent
    >>> arr = arr.astype('datetime64[M]').astype('int64')
        (arr % 12) // 3 + 1 + offset
    """
    cdef:
        np.ndarray out = dt64arr_as_int64_M(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, r
        
    for i in range(size):
        v = out_ptr[i]
        # Propagate NaT
        if v == LLONG_MIN:
            continue
        # Convert to quarter
        with cython.cdivision(True):
            r = v % 12
            if r < 0:
                r += 12
            out_ptr[i] = (r / 3) + 1 + offset

    return out

cdef inline np.ndarray dt64arr_month(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Extract month numbers (1..12) from a 1-D ndarray[datetime64[*]] <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: The month number for each element.
        NaT values (LLONG_MIN) are preserved as the month values.

    ## Equivalent
    >>> arr.astype('datetime64[M]').astype('int64') % 12 + 1 + offset
    """
    cdef np.ndarray out = dt64arr_as_int64_M(arr, arr_reso, 0, copy)
    return arr_mod(out, 12, offset + 1, False)

cdef inline np.ndarray dt64arr_weekday(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Extract weekday numbers (0=Mon .. 6=Sun) from a 1-D ndarray[datetime64[*]] <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: The weekday number for each element.
        NaT values (LLONG_MIN) are preserved as the weekday values.

    ## Equivalent
    >>> v = arr.astype('datetime64[D]').astype('int64')
        ((v % 7 + 7) % 7 + 3) % 7 + offset
    """
    cdef:
        np.ndarray out = dt64arr_as_int64_D(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, r
    
    for i in range(size):
        v = out_ptr[i]
        # Propagate NaT
        if v == LLONG_MIN:
            continue
        # Convert to weekday (0=Monday, 6=Sunday)
        with cython.cdivision(True):
            r = v % 7
        if r < 0:
            r += 7
        # Anchor to Thursday (3)
        r += 3  
        if r >= 7:
            r -= 7
        out_ptr[i] = r + offset

    return out

cdef inline np.ndarray dt64arr_day(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Extract day-of-month (1..31) from a 1-D ndarray[datetime64[*]] <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: The day-of-month for each element.
        NaT values (LLONG_MIN) are preserved as the day values.
    """
    cdef:
        np.ndarray out = dt64arr_as_int64_D(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, n, r, n400, n100, n4, n1, yy, mm, days_bf

    for i in range(size):
        v = out_ptr[i]
        # Propagate NaT
        if v == LLONG_MIN:
            continue

        # Convert to 0-based offset from 0001-01-01
        n = v + EPOCH_DAY - 1

        with cython.cdivision(True):
            # Number of complete 400-year cycles
            n400 = n / 146_097; r = n % 146_097
            if r < 0:
                n400 -= 1; r += 146_097
            n = r
            # Number of complete 100-year cycles within the 400-year cycle
            n100 = n / 36_524;  n %= 36_524
            # Number of complete 4-year cycles within the 100-year cycle
            n4   = n / 1_461;   n %= 1_461
            # Number of complete years within the 4-year cycle
            n1   = n / 365;     n %= 365

        # End-of-cycle dates map directly to December 31 (day=31)
        if n100 == 4 or n1 == 4:  # end-of-cycle dates
            out_ptr[i] = 31 + offset
            continue

        # Compute day of month (1..31)
        yy = n400 * 400 + n100 * 100 + n4 * 4 + n1 + 1
        mm = (n + 50) >> 5  # initial 1..12 estimate
        days_bf = days_bf_month(yy, mm)
        if days_bf > n:
            days_bf = days_bf_month(yy, mm - 1)
        out_ptr[i] = (n - days_bf + 1) + offset

    return out

cdef inline np.ndarray dt64arr_hour(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Extract hour-of-day (0..23) from a 1-D ndarray[datetime64[*]] <'ndarray[int64]'>.
    
    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: hour-of-day for each element.
        NaT values (LLONG_MIN) are preserved as the hour values.

    ## Equivalent
    >>> arr.astype('datetime64[h]').astype('int64') % 24 + offset
    """
    cdef np.ndarray out = dt64arr_as_int64_h(arr, arr_reso, 0, copy)
    return arr_mod(out, 24, offset, False)

cdef inline np.ndarray dt64arr_minute(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Extract minute-of-hour (0..59) from a 1-D ndarray[datetime64[*]] <'ndarray[int64]'>.
    
    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: minute-of-hour for each element.
        NaT values (LLONG_MIN) are preserved as the minute values.

    ## Equivalent
    >>> arr.astype('datetime64[m]').astype('int64') % 60 + offset
    """
    cdef np.ndarray out = dt64arr_as_int64_m(arr, arr_reso, 0, copy)
    return arr_mod(out, 60, offset, False)

cdef inline np.ndarray dt64arr_second(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Extract second-of-minute (0..59) from a 1-D ndarray[datetime64[*]] <'ndarray[int64]'>.
    
    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: second-of-minute for each element.
        NaT values (LLONG_MIN) are preserved as the second values.

    ## Equivalent
    >>> arr.astype('datetime64[s]').astype('int64') % 60 + offset
    """
    cdef np.ndarray out = dt64arr_as_int64_s(arr, arr_reso, 0, copy)
    return arr_mod(out, 60, offset, False)

cdef inline np.ndarray dt64arr_millisecond(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Extract millisecond-of-second (0..999) from a 1-D ndarray[datetime64[*]] <'ndarray[int64]'>.
    
    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: millisecond-of-second for each element.
        NaT values (LLONG_MIN) are preserved as the millisecond values.

    ## Equivalent
    >>> arr.astype('datetime64[ms]').astype('int64') % 1000 + offset
    """
    cdef np.ndarray out = dt64arr_as_int64_ms(arr, arr_reso, 0, copy)
    return arr_mod(out, 1_000, offset, False)

cdef inline np.ndarray dt64arr_microsecond(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Extract microsecond-of-second (0..999,999) from a 1-D ndarray[datetime64[*]] <'ndarray[int64]'>.
    
    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: microsecond-of-second for each element.
        NaT values (LLONG_MIN) are preserved as the microsecond values.

    ## Equivalent
    >>> arr.astype('datetime64[us]').astype('int64') % 1_000_000 + offset
    """
    cdef np.ndarray out = dt64arr_as_int64_us(arr, arr_reso, 0, copy)
    return arr_mod(out, 1_000_000, offset, False)

cdef inline np.ndarray dt64arr_nanosecond(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Extract nanosecond-of-microsecond (0..999) from a 1-D ndarray[datetime64[*]] <'ndarray[int64]'>.
    
    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: nanosecond-of-microsecond for each element.
        NaT values (LLONG_MIN) are preserved as the nanosecond values.

    ## Equivalent
    >>> arr.astype('datetime64[ns]').astype('int64') % 1000 + offset
    """
    cdef np.ndarray out = dt64arr_as_int64_ns(arr, arr_reso, 0, copy)
    return arr_mod(out, 1_000, offset, False)

cdef inline np.ndarray dt64arr_times(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Extract time-of-day (elapsed since midnight) in the same unit as `arr` <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: Time-of-day ticks for each element in the same unit.
        NaT values (LLONG_MIN) are preserved as the time-of-day values.
    """
    # Get array resolution
    if arr_reso < 0:
        if np.PyArray_TYPE(arr) != np.NPY_TYPES.NPY_DATETIME:
            _raise_missing_arr_reso_error(arr)
        arr_reso = get_arr_nptime_unit(arr)

    # Compute times
    arr = arr_assure_int64(arr, copy)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_mod(arr, NS_DAY, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_mod(arr, US_DAY, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_mod(arr, MS_DAY, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:   # second
        return arr_mod(arr, SS_DAY, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:   # minute
        return arr_mod(arr, 1440, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_h:   # hour
        return arr_mod(arr, 24, offset, False)
    if arr_reso in (
        np.NPY_DATETIMEUNIT.NPY_FR_D,  # day
        np.NPY_DATETIMEUNIT.NPY_FR_W,  # week
        np.NPY_DATETIMEUNIT.NPY_FR_M,  # month
        np.NPY_DATETIMEUNIT.NPY_FR_Y,  # year
    ):
        return arr_mul(arr, 0, offset, False)

    # Unsupported resolution
    _raise_missing_arr_reso_error(arr)

# . calendar
cdef inline np.ndarray dt64arr_isocalendar(np.ndarray arr, int arr_reso=-1, bint copy=True):
    """Return ISO-8601 calendar components for each element of a 1-D datetime64 array `<'ndarray[int64]'>`.

    Output is a (N, 3) int64 array with columns: [iso_year, iso_week, iso_weekday].
    ISO weekday is 1..7 for Mon..Sun. Week numbering follows ISO-8601:
    week 1 is the week containing January 4 (i.e., the first Thursday).
    The earilies date is clamp to '0001-01-01'

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: Shape (N, 3) array: [iso_year, iso_week, iso_weekday] per element.
        NaT values (LLONG_MIN) are propragated as the ISO calendar values.

    Output Example:
    >>> [[1936   11    7]
         [1936   12    1]
         [1936   12    2]
         ...
         [2003   42    6]
         [2003   42    7]
         [2003   43    1]]
    """
    arr = dt64arr_as_int64_D(arr, arr_reso, 0, copy)
    cdef:
        # Target array
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_intp size = arr.shape[0]
        np.npy_intp i
        # Output array
        np.ndarray out = np.PyArray_EMPTY(2, [size, 3], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_int64 v
        ymd _ymd
        iso _iso

    for i in range(size):
        v = arr_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            out_ptr[0] = out_ptr[1] = out_ptr[2] = v
        # Calculate ISO calendar (year, week, weekday)
        else:
            _ymd = ymd_fr_ord(v + EPOCH_DAY)
            _iso = ymd_isocalendar(_ymd.year, _ymd.month, _ymd.day)
            out_ptr[0] = _iso.year
            out_ptr[1] = _iso.week
            out_ptr[2] = _iso.weekday
        # Increment +3
        out_ptr += 3
        
    return out

cdef inline np.ndarray dt64arr_isoyear(np.ndarray arr, int arr_reso=-1, bint copy=True):
    """Extract ISO-8601 year numbers from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The ISO year for each element.
        NaT values (LLONG_MIN) are preserved as the year values.
    """
    cdef:
        np.ndarray out = dt64arr_as_int64_D(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v
        ymd _ymd

    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Calculate ISO year
        _ymd = ymd_fr_ord(v + EPOCH_DAY)
        out_ptr[i] = ymd_isoyear(_ymd.year, _ymd.month, _ymd.day)

    return out

cdef inline np.ndarray dt64arr_isoweek(np.ndarray arr, int arr_reso=-1, bint copy=True):
    """Extract ISO-8601 week numbers from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The ISO week for each element.
        NaT values (LLONG_MIN) are preserved as the week values.
    """
    cdef:
        np.ndarray out = dt64arr_as_int64_D(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v
        ymd _ymd

    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Calculate ISO week
        _ymd = ymd_fr_ord(v + EPOCH_DAY)
        out_ptr[i] = ymd_isoweek(_ymd.year, _ymd.month, _ymd.day)

    return out

cdef inline np.ndarray dt64arr_isoweekday(np.ndarray arr, int arr_reso=-1, bint copy=True):
    """Extract ISO-8601 weekday numbers (1=Mon .. 7=Sun) from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The ISO weekday for each element.
        NaT values (LLONG_MIN) are preserved as the weekday values.
    """
    cdef:
        np.ndarray out = dt64arr_as_int64_D(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v
        ymd _ymd

    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Calculate ISO weekday
        _ymd = ymd_fr_ord(v + EPOCH_DAY)
        out_ptr[i] = ymd_isoweekday(_ymd.year, _ymd.month, _ymd.day)

    return out

cdef inline np.ndarray dt64arr_is_leap_year(np.ndarray arr, int arr_reso=-1, bint copy=True):
    """Elementwise check for leap years of a 1-D datetime64 array `<'ndarray[bool]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[bool]'>`: Boolean indicator of leap year for each element.
        NaT values (LLONG_MIN) are emitted as `False`.
    """
    arr = dt64arr_as_int64_Y(arr, arr_reso, 0, copy)
    cdef:
        # Target array
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_intp size = arr.shape[0]
        np.npy_intp i
        # Output array
        np.ndarray out = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_BOOL, 0)
        np.npy_bool* out_ptr = <np.npy_bool*> np.PyArray_DATA(out)
        np.npy_int64 v

    for i in range(size):
        v = arr_ptr[i]
        # Handle NaT
        if v == LLONG_MIN:
            out_ptr[i] = False
            continue
        # Check leap year
        out_ptr[i] = is_leap_year(v + EPOCH_YEAR)

    return out

cdef inline np.ndarray dt64arr_is_long_year(np.ndarray arr, int arr_reso=-1, bint copy=True):
    """Elementwise check for long years of a 1-D datetime64 array `<'ndarray[bool]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[bool]'>`: Boolean indicator of long year for each element.
        NaT values (LLONG_MIN) are emitted as `False`.
    """
    arr = dt64arr_as_int64_Y(arr, arr_reso, 0, copy)
    cdef:
        # Target array
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_intp size = arr.shape[0]
        np.npy_intp i
        # Output array
        np.ndarray out = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_BOOL, 0)
        np.npy_bool* out_ptr = <np.npy_bool*> np.PyArray_DATA(out)
        np.npy_int64 v

    for i in range(size):
        v = arr_ptr[i]
        # Handle NaT
        if v == LLONG_MIN:
            out_ptr[i] = False
            continue
        # Check long year
        out_ptr[i] = is_long_year(v + EPOCH_YEAR)

    return out

cdef inline np.ndarray dt64arr_leap_bt_years(np.ndarray arr, np.npy_int64 year, int arr_reso=-1, bint copy=True):
    """Compute the number of leap years between the target `year` and elements
    of a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param year `<'int'>`: The target year to compute against.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    
    :returns `<'ndarray[int64]'>`: The total number of leap years for each element.
        NaT values (LLONG_MIN) are preserved as the leap years value.
    """
    cdef:
        np.ndarray out = dt64arr_as_int64_Y(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v

    for i in range(size):
        v = out_ptr[i]
        # Propagate NaT
        if v == LLONG_MIN:
            continue
        # Calculate leap years between
        out_ptr[i] = leaps_bt_years(v + EPOCH_YEAR, year)

    return out

cdef inline np.ndarray dt64arr_days_in_year(np.ndarray arr, int arr_reso=-1, bint copy=True):
    """Extract the number of days in the year (365/366) from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The number of days in the year for each element.
        NaT values (LLONG_MIN) are preserved as the days-in-year value.
    """
    cdef:
        np.ndarray out = dt64arr_as_int64_Y(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v

    for i in range(size):
        v = out_ptr[i]
        # Propagate NaT
        if v == LLONG_MIN:
            continue
        # Get days in year
        out_ptr[i] = days_in_year(v + EPOCH_YEAR)

    return out

cdef inline np.ndarray dt64arr_days_bf_year(np.ndarray arr, int arr_reso=-1, bint copy=True):
    """Extract the number of days strictly before January 1 of year under the 
    proleptic Gregorian rules from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The number of days before year for each element.
        NaT values (LLONG_MIN) are preserved as the days-before-year value.
    """
    cdef:
        np.ndarray out = dt64arr_as_int64_Y(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v

    for i in range(size):
        v = out_ptr[i]
        # Propagate NaT
        if v == LLONG_MIN:
            continue
        # Get days before year
        out_ptr[i] = days_bf_year(v + EPOCH_YEAR)

    return out

cdef inline np.ndarray dt64arr_day_of_year(np.ndarray arr, int arr_reso=-1, bint copy=True):
    """Extract the 1-based ordinal day-of-year from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The day-of-year for each element.
        NaT values (LLONG_MIN) are preserved as the year values.
    """
    cdef:
        np.ndarray out = dt64arr_as_int64_D(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v
        ymd _ymd

    for i in range(size):
        v = out_ptr[i]
        # Propagate NaT
        if v == LLONG_MIN:
            continue
        # Get days of year
        _ymd = ymd_fr_ord(v + EPOCH_DAY)
        out_ptr[i] = day_of_year(_ymd.year, _ymd.month, _ymd.day)

    return out

cdef inline np.ndarray dt64arr_days_in_quarter(np.ndarray arr, int arr_reso=-1, bint copy=True):
    """Extract number of days in calendar quarter under the proleptic Gregorian rules
    from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The number of days in calendar quarter for each element.
        NaT values (LLONG_MIN) are preserved as the days values.
    """
    cdef:
        np.ndarray out = dt64arr_as_int64_D(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v
        ymd _ymd

    for i in range(size):
        v = out_ptr[i]
        # Propagate NaT
        if v == LLONG_MIN:
            continue
        # Compute days in quarter
        _ymd = ymd_fr_ord(v + EPOCH_DAY)
        out_ptr[i] = days_in_quarter(_ymd.year, _ymd.month)

    return out

cdef inline np.ndarray dt64arr_days_bf_quarter(np.ndarray arr, int arr_reso=-1, bint copy=True):
    """Extract number of days strictly before the first day of the 
    calendar quarter under the proleptic Gregorian rules from a 1-D 
    datetime64 array `<'ndarray[int64]'>`
    
    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The number of days before calendar quarter for each element.
        NaT values (LLONG_MIN) are preserved as the days values.
    """
    cdef:
        np.ndarray out = dt64arr_as_int64_D(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v
        ymd _ymd

    for i in range(size):
        v = out_ptr[i]
        # Propagate NaT
        if v == LLONG_MIN:
            continue
        # Compute days before quarter
        _ymd = ymd_fr_ord(v + EPOCH_DAY)
        out_ptr[i] = days_bf_quarter(_ymd.year, _ymd.month)

    return out

cdef inline np.ndarray dt64arr_day_of_quarter(np.ndarray arr, int arr_reso=-1, bint copy=True):
    """Extract the 1-based ordinal day-of-quarter from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The the number of days between the 
        1st day of the quarter and the date for each element. 
        NaT values (LLONG_MIN) are preserved as the day values.
    """
    cdef:
        np.ndarray out = dt64arr_as_int64_D(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v
        ymd _ymd

    for i in range(size):
        v = out_ptr[i]
        # Propagate NaT
        if v == LLONG_MIN:
            continue
        # Compute days of quarter
        _ymd = ymd_fr_ord(v + EPOCH_DAY)
        out_ptr[i] = day_of_quarter(_ymd.year, _ymd.month, _ymd.day)

    return out

cdef inline np.ndarray dt64arr_days_in_month(np.ndarray arr, int arr_reso=-1, bint copy=True):
    """Extract number of days in calendar month under the proleptic Gregorian rules
    from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The number of days in calendar month for each element.
        NaT values (LLONG_MIN) are preserved as the days values.
    """
    cdef:
        np.ndarray out = dt64arr_as_int64_D(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v
        ymd _ymd

    for i in range(size):
        v = out_ptr[i]
        # Propagate NaT
        if v == LLONG_MIN:
            continue
        # Compute days in month
        _ymd = ymd_fr_ord(v + EPOCH_DAY)
        out_ptr[i] = days_in_month(_ymd.year, _ymd.month)

    return out

cdef inline np.ndarray dt64arr_days_bf_month(np.ndarray arr, int arr_reso=-1, bint copy=True):
    """Extract number of days strictly before the first day of the 
    calendar month under the proleptic Gregorian rules from a 1-D 
    datetime64 array `<'ndarray[int64]'>`
    
    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The number of days before calendar month for each element.
        NaT values (LLONG_MIN) are preserved as the days values.
    """
    cdef:
        np.ndarray out = dt64arr_as_int64_D(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v
        ymd _ymd

    for i in range(size):
        v = out_ptr[i]
        # Propagate NaT
        if v == LLONG_MIN:
            continue
        # Compute days before month
        _ymd = ymd_fr_ord(v + EPOCH_DAY)
        out_ptr[i] = days_bf_month(_ymd.year, _ymd.month)

    return out

# . conversion: int64
cdef inline np.ndarray dt64arr_fr_int64(np.npy_int64 value, np.npy_intp size, str unit):
    """Create a 1-D datetime64 array filled with the specified integer `value` `<'ndarray[datetime64]'>`.

    :param value `<'int'>`: The integer value of the datetime64 array.
    :param size `<'int'>`: The length of the datetime64 array.
    :param unit <'str'>: Time unit in its string form:
        'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'Y', etc.

    ## Equivalent
    >>> np.full(size, value, dtype=f"datetime64[{unit}]")
    """
    return arr_fill_int64(value, size).astype(nptime_unit_str2dt64(unit))

cdef inline np.ndarray dt64arr_as_int64(np.ndarray arr, str as_unit, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Convert a 1-D ndarray[datetime64[*]] to int64 ticks in the requested unit `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param as_unit <'str'>: Target unit, supports:
        'ns','us','ms','s','m','h','D', 'W', 'M', 'Q', 'Y' and 'min' (alias for minutes).

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    - For 'W' (weeks) uses Thursday anchoring (same as Numpy).

    ## Equivalent
    >>> arr.astype(f"datetime64[{as_unit}]").astype("int64") + offset
    """
    if as_unit is None:
        _raise_invalid_nptime_str_unit_error(as_unit)
    cdef:
        Py_ssize_t size = str_len(as_unit)
        Py_UCS4 ch

    # Unit: 's', 'm', 'h', 'D', 'M', 'Y'
    if size == 1:
        ch = str_read(as_unit, 0)
        if ch == "s":
            return dt64arr_as_int64_s(arr, arr_reso, offset, copy)
        if ch == "m":
            return dt64arr_as_int64_m(arr, arr_reso, offset, copy)
        if ch == "h":
            return dt64arr_as_int64_h(arr, arr_reso, offset, copy)
        if ch == "D":
            return dt64arr_as_int64_D(arr, arr_reso, offset, copy)
        if ch == "W":
            return dt64arr_as_int64_W(arr, arr_reso, offset, copy)
        if ch == "M":
            return dt64arr_as_int64_M(arr, arr_reso, offset, copy)
        if ch == "Q":
            return dt64arr_as_int64_Q(arr, arr_reso, offset, copy)
        if ch == "Y":
            return dt64arr_as_int64_Y(arr, arr_reso, offset, copy)

    # Unit: 'ns', 'us', 'ms'
    elif size == 2 and str_read(as_unit, 1) == "s":
        ch = str_read(as_unit, 0)
        if ch == "n":
            return dt64arr_as_int64_ns(arr, arr_reso, offset, copy)
        if ch == "u":
            return dt64arr_as_int64_us(arr, arr_reso, offset, copy)
        if ch == "m":
            return dt64arr_as_int64_ms(arr, arr_reso, offset, copy)

    # Unit: 'min' for pandas compatibility
    elif size == 3 and as_unit == "min":
        return dt64arr_as_int64_m(arr, arr_reso, offset, copy)

    # Unsupported unit
    _raise_dt64arr_to_reso_error(arr, arr_reso, as_unit)

cdef inline np.ndarray dt64arr_as_int64_Y(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Convert a 1-D ndarray[datetime64[*]] to int64 year ticks (Y) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[Y]").astype("int64") + offset
    """
    # Get array resolution
    if arr_reso < 0:
        if np.PyArray_TYPE(arr) != np.NPY_TYPES.NPY_DATETIME:
            _raise_missing_arr_reso_error(arr)
        arr_reso = get_arr_nptime_unit(arr)

    # Fast-path: datetime64[Y]
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_Y:
        arr = arr_assure_int64(arr, copy)
        return arr_add(arr, offset, False)

    # Fast-path: datetime64[M]
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_M:
        arr = arr_assure_int64(arr, copy)
        return arr_div_floor(arr, 12, offset, False)

    # Setup
    cdef:
        np.ndarray out = dt64arr_as_int64_D(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, n, r, n400, n100, n4, n1, yy
    
    # Compute
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue

        # Convert to 0-based offset from 0001-01-01
        n = v + EPOCH_DAY - 1

        with cython.cdivision(True):
            # Number of complete 400-year cycles
            n400 = n / 146_097; r = n % 146_097
            if r < 0:
                n400 -= 1; r += 146_097
            n = r
            # Number of complete 100-year cycles within the 400-year cycle
            n100 = n / 36_524;  n %= 36_524
            # Number of complete 4-year cycles within the 100-year cycle
            n4   = n / 1_461;   n %= 1_461
            # Number of complete years within the 4-year cycle
            n1   = n / 365;     n %= 365

        # Compute the year
        yy = n400 * 400 + n100 * 100 + n4 * 4 + n1 + 1
        if n100 == 4 or n1 == 4:  # end-of-cycle dates
            yy -= 1
        out_ptr[i] = yy - EPOCH_YEAR + offset
            
    return out

cdef inline np.ndarray dt64arr_as_int64_Q(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Convert a 1-D ndarray[datetime64[*]] to int64 quarter ticks (Q) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> (arr.astype("datetime64[M]").astype("int64") // 3) + offset
    """
    cdef:
        np.ndarray out = dt64arr_as_int64_M(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, q, r
    
    # Compute
    for i in range(size):
        v = out_ptr[i]
        # preserve NaT value
        if v == LLONG_MIN:
            continue
        # convert months to quarters
        with cython.cdivision(True):
            q = v / 3; r = v % 3
        if r < 0:
            q -= 1
        out_ptr[i] = q + offset

    return out

cdef inline np.ndarray dt64arr_as_int64_M(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Convert a 1-D ndarray[datetime64[*]] to int64 month ticks (M) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[M]").astype("int64") + offset
    """
    # Get array resolution
    if arr_reso < 0:
        if np.PyArray_TYPE(arr) != np.NPY_TYPES.NPY_DATETIME:
            _raise_missing_arr_reso_error(arr)
        arr_reso = get_arr_nptime_unit(arr)

    # Fast-path: datetime64[Y]
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_Y:
        arr = arr_assure_int64(arr, copy)
        return arr_mul(arr, 12, offset, False)

    # Fast-path: datetime64[M]
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_M:
        arr = arr_assure_int64(arr, copy)
        return arr_add(arr, offset, False)

    # Setup
    cdef:
        np.ndarray out = dt64arr_as_int64_D(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, n, r, n400, n100, n4, n1, yy, mm

    # Compute
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue

        # Convert to 0-based offset from 0001-01-01
        n = v + EPOCH_DAY - 1

        with cython.cdivision(True):
            # Number of complete 400-year cycles
            n400 = n / 146_097; r = n % 146_097
            if r < 0:
                n400 -= 1; r += 146_097
            n = r
            # Number of complete 100-year cycles within the 400-year cycle
            n100 = n / 36_524;  n %= 36_524
            # Number of complete 4-year cycles within the 100-year cycle
            n4   = n / 1_461;   n %= 1_461
            # Number of complete years within the 4-year cycle
            n1   = n / 365;     n %= 365

        # Compute year & month
        yy = n400 * 400 + n100 * 100 + n4 * 4 + n1 + 1
        if n100 == 4 or n1 == 4:  # end-of-cycle dates
            yy -= 1; mm = 12
        else:
            mm = (n + 50) >> 5  # initial 1..12 estimate
            if days_bf_month(yy, mm) > n:
                mm -= 1

        # Total months since epoch
        out_ptr[i] = (yy - EPOCH_YEAR) * 12 + (mm - 1) + offset

    return out

cdef inline np.ndarray dt64arr_as_int64_W(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Convert a 1-D ndarray[datetime64[*]] to int64 week ticks (W) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[W]").astype("int64") + offset
    """
    cdef:
        np.ndarray out = dt64arr_as_int64_D(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, q, r

    # Compute
    for i in range(size):
        v = out_ptr[i]
        # preserve NaT value
        if v == LLONG_MIN:
            continue
        # convert days to weeks
        with cython.cdivision(True):
            q = v / 7; r = v % 7
        if r < 0:
            q -= 1
        out_ptr[i] = q + offset

    return out

cdef inline np.ndarray dt64arr_as_int64_D(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Convert a 1-D ndarray[datetime64[*]] to int64 day ticks (D) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[D]").astype("int64") + offset
    """
    # Get array resolution
    if arr_reso < 0:
        if np.PyArray_TYPE(arr) != np.NPY_TYPES.NPY_DATETIME:
            _raise_missing_arr_reso_error(arr)
        arr_reso = get_arr_nptime_unit(arr)

    # Conversion
    arr = arr_assure_int64(arr, copy)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_div_floor(arr, NS_DAY, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_div_floor(arr, US_DAY, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_div_floor(arr, MS_DAY, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr_div_floor(arr, SS_DAY, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr_div_floor(arr, 1440, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr_div_floor(arr, 24, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr_add(arr, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return arr_mul(arr, 7, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _dt64arr_M_as_int64_D(arr, 1, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _dt64arr_Y_as_int64_D(arr, 1, offset, False)

    # Unsupported array resolution
    _raise_dt64arr_to_reso_error(arr, arr_reso, "D")

cdef inline np.ndarray dt64arr_as_int64_h(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Convert a 1-D ndarray[datetime64[*]] to int64 hour ticks (h) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[h]").astype("int64") + offset
    """
    # Get array resolution
    if arr_reso < 0:
        if np.PyArray_TYPE(arr) != np.NPY_TYPES.NPY_DATETIME:
            _raise_missing_arr_reso_error(arr)
        arr_reso = get_arr_nptime_unit(arr)

    # Conversion
    arr = arr_assure_int64(arr, copy)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_div_floor(arr, NS_HOUR, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_div_floor(arr, US_HOUR, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_div_floor(arr, MS_HOUR, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr_div_floor(arr, SS_HOUR, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr_div_floor(arr, 60, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr_add(arr, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr_mul(arr, 24, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return arr_mul(arr, 24 * 7, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _dt64arr_M_as_int64_D(arr, 24, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _dt64arr_Y_as_int64_D(arr, 24, offset, False)
    
    # Unsupported array resolution
    _raise_dt64arr_to_reso_error(arr, arr_reso, "h")

cdef inline np.ndarray dt64arr_as_int64_m(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Convert a 1-D ndarray[datetime64[*]] to int64 minute ticks (m) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[m]").astype("int64") + offset
    """
    # Get array resolution
    if arr_reso < 0:
        if np.PyArray_TYPE(arr) != np.NPY_TYPES.NPY_DATETIME:
            _raise_missing_arr_reso_error(arr)
        arr_reso = get_arr_nptime_unit(arr)

    # Conversion
    arr = arr_assure_int64(arr, copy)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_div_floor(arr, NS_MINUTE, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_div_floor(arr, US_MINUTE, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_div_floor(arr, MS_MINUTE, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr_div_floor(arr, SS_MINUTE, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr_add(arr, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr_mul(arr, 60, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr_mul(arr, 1_440, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return arr_mul(arr, 1_440 * 7, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _dt64arr_M_as_int64_D(arr, 1_440, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _dt64arr_Y_as_int64_D(arr, 1_440, offset, False)
    
    # Unsupported array resolution
    _raise_dt64arr_to_reso_error(arr, arr_reso, "m")

cdef inline np.ndarray dt64arr_as_int64_s(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Convert a 1-D ndarray[datetime64[*]] to int64 second ticks (s) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[s]").astype("int64") + offset
    """
    # Get array resolution
    if arr_reso < 0:
        if np.PyArray_TYPE(arr) != np.NPY_TYPES.NPY_DATETIME:
            _raise_missing_arr_reso_error(arr)
        arr_reso = get_arr_nptime_unit(arr)

    # Conversion
    arr = arr_assure_int64(arr, copy)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_div_floor(arr, NS_SECOND, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_div_floor(arr, US_SECOND, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_div_floor(arr, MS_SECOND, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr_add(arr, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr_mul(arr, SS_MINUTE, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr_mul(arr, SS_HOUR, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr_mul(arr, SS_DAY, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return arr_mul(arr, SS_DAY * 7, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _dt64arr_M_as_int64_D(arr, SS_DAY, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _dt64arr_Y_as_int64_D(arr, SS_DAY, offset, False)
    
    # Unsupported array resolution
    _raise_dt64arr_to_reso_error(arr, arr_reso, "s")

cdef inline np.ndarray dt64arr_as_int64_ms(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Convert a 1-D ndarray[datetime64[*]] to int64 millisecond ticks (ms) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[ms]").astype("int64") + offset
    """
    # Get array resolution
    if arr_reso < 0:
        if np.PyArray_TYPE(arr) != np.NPY_TYPES.NPY_DATETIME:
            _raise_missing_arr_reso_error(arr)
        arr_reso = get_arr_nptime_unit(arr)

    # Conversion
    arr = arr_assure_int64(arr, copy)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_div_floor(arr, NS_MILLISECOND, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_div_floor(arr, US_MILLISECOND, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_add(arr, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr_mul(arr, MS_SECOND, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr_mul(arr, MS_MINUTE, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr_mul(arr, MS_HOUR, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr_mul(arr, MS_DAY, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return arr_mul(arr, MS_DAY * 7, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _dt64arr_M_as_int64_D(arr, MS_DAY, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _dt64arr_Y_as_int64_D(arr, MS_DAY, offset, False)
    
    # Unsupported array resolution
    _raise_dt64arr_to_reso_error(arr, arr_reso, "ms")

cdef inline np.ndarray dt64arr_as_int64_us(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Convert a 1-D ndarray[datetime64[*]] to int64 microsecond ticks (us) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[us]").astype("int64") + offset
    """
    # Get array resolution
    if arr_reso < 0:
        if np.PyArray_TYPE(arr) != np.NPY_TYPES.NPY_DATETIME:
            _raise_missing_arr_reso_error(arr)
        arr_reso = get_arr_nptime_unit(arr)

    # Conversion
    arr = arr_assure_int64(arr, copy)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_div_floor(arr, NS_MICROSECOND, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_add(arr, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_mul(arr, US_MILLISECOND, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr_mul(arr, US_SECOND, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr_mul(arr, US_MINUTE, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr_mul(arr, US_HOUR, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr_mul(arr, US_DAY, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return arr_mul(arr, US_DAY * 7, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _dt64arr_M_as_int64_D(arr, US_DAY, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _dt64arr_Y_as_int64_D(arr, US_DAY, offset, False)
    
    # Unsupported array resolution
    _raise_dt64arr_to_reso_error(arr, arr_reso, "us")

cdef inline np.ndarray dt64arr_as_int64_ns(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Convert a 1-D ndarray[datetime64[*]] to int64 nanosecond ticks (ns) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[ns]").astype("int64") + offset
    """
    # Get array resolution
    if arr_reso < 0:
        if np.PyArray_TYPE(arr) != np.NPY_TYPES.NPY_DATETIME:
            _raise_missing_arr_reso_error(arr)
        arr_reso = get_arr_nptime_unit(arr)

    # Conversion
    arr = arr_assure_int64(arr, copy)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_add(arr, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_mul(arr, NS_MICROSECOND, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_mul(arr, NS_MILLISECOND, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr_mul(arr, NS_SECOND, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr_mul(arr, NS_MINUTE, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr_mul(arr, NS_HOUR, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr_mul(arr, NS_DAY, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return arr_mul(arr, NS_DAY * 7, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _dt64arr_M_as_int64_D(arr, NS_DAY, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _dt64arr_Y_as_int64_D(arr, NS_DAY, offset, False)
        
    # Unsupported array resolution
    _raise_dt64arr_to_reso_error(arr, arr_reso, "ns")

cdef inline np.ndarray _dt64arr_Y_as_int64_D(np.ndarray arr, np.npy_int64 factor=1, np.npy_int64 offset=0, bint copy=True):
    """(internal) Convert an ndarray[datetime64[Y]] to int64 day ticks (D), then
    scale by `factor` and add `offset` `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: 1-D array with dtype datetime64[Y].
    :param factor `<'int'>`: Post-conversion scale applied to day counts. Defaults to `1`.
    :param offset `<'int'>`: Optional value added after scaling. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    """
    # Setup
    cdef:
        np.ndarray out = arr_assure_int64(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v

    # Compute
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:            
            continue
        # Convert to day resolution
        out_ptr[i] = _dt64_Y_as_int64_D(v, factor, offset)

    return out

cdef inline np.ndarray _dt64arr_M_as_int64_D(np.ndarray arr, np.npy_int64 factor=1, np.npy_int64 offset=0, bint copy=True):
    """(internal) Convert an ndarray[datetime64[M]] to int64 day ticks (D), then
    scale by `factor` and add `offset` `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: 1-D array with dtype datetime64[M].
    :param factor `<'int'>`: Post-conversion scale applied to day counts. Defaults to `1`.
    :param offset `<'int'>`: Value added after scaling. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    """
    # Setup
    cdef:
        np.ndarray out = arr_assure_int64(arr, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, yy_ep, yy, mm, leaps

    # Compute
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Convert to day resolution
        out_ptr[i] = _dt64_M_as_int64_D(v, factor, offset)

    return out

cdef inline np.ndarray dt64arr_as_W_iso(np.ndarray arr, int weekday, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Convert a 1-D ndarray[datetime64[*]] to int64 week ticks (W) aligned to an ISO weekday `<'ndarray[int64]'>`.

    NumPy's datetime64[W] is **Thursday-anchored** (1970-01-01 is a Thursday).
    This function aligns weeks to any ISO weekday: 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat, 7=Sun.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param weekday <'int'>: ISO weekday to align to, value is clamp to [1, 7].
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    Equivalent to (conceptually):
    >>> arr = arr.astype("datetime64[D]").astype("int64")
        (arr + (4 - weekday)) // 7 + offset
    """

    # Fast-path
    cdef np.npy_int64 adj = 4 - min(max(weekday, 1), 7)
    if adj == 0:
        return dt64arr_as_int64_W(arr, arr_reso, offset, copy)

    # Setup
    cdef:
        np.ndarray out = dt64arr_as_int64_D(arr, arr_reso, 0, copy)
        np.npy_int64* out_ptr = <np.npy_int64*> np.PyArray_DATA(out)
        np.npy_intp size = out.shape[0]
        np.npy_intp i
        np.npy_int64 v, q, r

    # Compute
    for i in range(size):
        v = out_ptr[i]
        # Preserve NaT
        if v == LLONG_MIN:
            continue
        # Aligned to the specified weekday
        v += adj
        # Convert to week resolution
        with cython.cdivision(True):
            q = v / 7; r = v % 7
        if r < 0:
            q -= 1
        out_ptr[i] = q + offset

    return out

cdef inline np.ndarray dt64arr_to_ord(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Convert a 1-D ndarray[datetime64[*]] to int64 proleptic 
    Gregorian ordinals `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    - Day 1 is '0001-01-01' (ordinal=1).

    ## Equivalent
    >>> arr.astype("datetime64[D]").astype("int64") + 719163 + offset
    """
    return dt64arr_as_int64_D(arr, arr_reso, EPOCH_DAY + offset, copy)

# . conversion: float64
cdef inline np.ndarray dt64arr_to_ts(np.ndarray arr, int arr_reso=-1, bint copy=True):
    """Convert a 1-D ndarray[datetime64[*]] to float64 Unix timestamps `<'ndarray[float64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[float64]'>`: The result timestamps array.

    ## Notice
    - NaT values (LLONG_MIN) are emitted as NaN.
    - When down-converting from finer units (e.g., ns → us), sub-microsecond 
      parts are **truncated** toward floor.

    ## Equivalent
    >>> arr.astype("datetime64[us]").astype("int64") / 1_000_000
    """
    arr = dt64arr_as_int64_us(arr, arr_reso, 0, copy)  # int64[us]
    cdef:
        # Target array
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_intp size = arr.shape[0]
        np.npy_intp i
        np.npy_int64 v, q, r
        # Output array
        np.ndarray out = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_FLOAT64, 0)
        np.npy_float64* out_ptr = <np.npy_float64*> np.PyArray_DATA(out)
        # Pre-computed
        np.npy_float64 inv_us = 1.0 / <np.npy_float64> US_SECOND

    for i in range(size):
        v = arr_ptr[i]
        # Preserve NaT 
        if v == LLONG_MIN:
            out_ptr[i] = math.NAN
            continue
        # Convert to seconds
        #: r in C semantics can be negative; normalize to 0 <= r < US_SECOND
        with cython.cdivision(True):
            r = v % US_SECOND
            if r < 0:
                r += US_SECOND
            #: (v - r) is an exact multiple of US_SECOND
            q = (v - r) / US_SECOND
        #: Combine whole seconds and sub-second fraction
        out_ptr[i] = (<np.npy_float64> r * inv_us) + <np.npy_float64> q

    return out

# . conversion: unit
cdef inline np.ndarray dt64arr_as_unit(np.ndarray arr, str as_unit, int arr_reso=-1, bint limit=False, bint copy=True):
    """Convert a 1-D datetime64 array to a specifc NumPy datetime64 unit `<'ndarray[datetime64]'>`

    :param arr `<'np.ndarray'>`: The datetime64 array to convert. 
        Also support int64 array ticks along with `arr_reso` argument (see below).

    :param as_unit `<'str'>`: Traget datetime64 unit (e.g. `'ns'`, `'us'`, `'ms'`, `'s'`).

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param limit `<'bool'>`: Whether to limit conversions to sub-second units. Defaults to `False`.

        - If `False` (default), all native datetime64 conversions are supported.
        - If `True`, conversions are **restricted** to sub-second units only:
          `{"s", "ms", "us", "ns"}` for both source and target. A `ValueError`
          is raised if either side is outside that set.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'np.ndarray[datetime64]'>`: An array with dtype `datetime64[as_unit]` 
        `NaT` values are preserved.

    ## Equivalent
    >>> arr.astype(f"datetime64[{as_unit}]")
    """
    if as_unit is None:
        _raise_invalid_nptime_str_unit_error(as_unit)
    # Get array resolution
    cdef bint is_dt64 = np.PyArray_TYPE(arr) == np.NPY_TYPES.NPY_DATETIME
    if arr_reso < 0:
        if not is_dt64:
            _raise_missing_arr_reso_error(arr)
        arr_reso = get_arr_nptime_unit(arr)
    elif not is_dt64:
        arr = arr.astype(nptime_unit_int2dt64(arr_reso))
        copy = False
    #: from now on, 'arr' can only be datetime64 
    #: array under the 'arr_reso' resolution.
    cdef int tg_unit = nptime_unit_str2int(as_unit)

    # Check limit
    if limit:
        if tg_unit not in (
            np.NPY_DATETIMEUNIT.NPY_FR_ns,
            np.NPY_DATETIMEUNIT.NPY_FR_us,
            np.NPY_DATETIMEUNIT.NPY_FR_ms,
            np.NPY_DATETIMEUNIT.NPY_FR_s,
        ) or arr_reso not in (
            np.NPY_DATETIMEUNIT.NPY_FR_ns,
            np.NPY_DATETIMEUNIT.NPY_FR_us,
            np.NPY_DATETIMEUNIT.NPY_FR_ms,
            np.NPY_DATETIMEUNIT.NPY_FR_s,
        ):
            raise ValueError(
                "Cannot convert ndarray from datetime64[%s] to datetime64[%s].\n"
                "Conversion limits to datetime units between: 's', 'ms', 'us', 'ns'." 
                % (nptime_unit_int2str(arr_reso), nptime_unit_int2str(tg_unit))
            )

    # Fast-path: same unit
    if arr_reso == tg_unit:
        return arr_add(arr, 0, copy)  # honor 'copy' flag

    # To nanosecond [ns]
    cdef bint is_ns_safe
    if tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        # check 'ns' overflow
        is_ns_safe = is_dt64arr_ns_safe(arr, arr_reso)
        # my_unit [us] -> tg_unit [ns]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:
            if not is_ns_safe:
                return arr_add(arr, 0, copy)  # honor 'copy' flag
            return arr.astype(DT64_DTYPE_NS)
        # my_unit [M, Y] -> tg_unit [ns]
        elif arr_reso in (
            np.NPY_DATETIMEUNIT.NPY_FR_M,
            np.NPY_DATETIMEUNIT.NPY_FR_Y,
        ):
            if not is_ns_safe:
                arr = dt64arr_as_int64_us(arr, arr_reso, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            arr = dt64arr_as_int64_ns(arr, arr_reso, 0, copy)
            return arr.astype(DT64_DTYPE_NS)
        # my_unit [rest] -> tg_unit [ns]
        else:
            if not is_ns_safe:
                return arr.astype(DT64_DTYPE_US)
            return arr.astype(DT64_DTYPE_NS)

    # To microsecond [us]
    if tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        # my_unit [ns] -> tg_unit [us]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_floor(arr, NS_MICROSECOND, 0, copy)
        # my_unit [M, Y] -> tg_unit [us]
        elif arr_reso in (
            np.NPY_DATETIMEUNIT.NPY_FR_M,
            np.NPY_DATETIMEUNIT.NPY_FR_Y,
        ):
            arr = dt64arr_as_int64_us(arr, arr_reso, 0, copy)
        # my_unit [rest] -> tg_unit [us]
        return arr.astype(DT64_DTYPE_US)
    
    # To millisecond [ms]
    if tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        # my_unit [ns] -> tg_unit [ms]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_floor(arr, NS_MILLISECOND, 0, copy)
        # my_unit [M, Y] -> tg_unit [ms]
        elif arr_reso in (
            np.NPY_DATETIMEUNIT.NPY_FR_M,
            np.NPY_DATETIMEUNIT.NPY_FR_Y,
        ):
            arr = dt64arr_as_int64_ms(arr, arr_reso, 0, copy)
        # my_unit [rest] -> tg_unit [ms]
        return arr.astype(DT64_DTYPE_MS)

    # To second [s]
    if tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        # my_unit [ns] -> tg_unit [s]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_floor(arr, NS_SECOND, 0, copy)
        # my_unit [M, Y] -> tg_unit [s]
        elif arr_reso in (
            np.NPY_DATETIMEUNIT.NPY_FR_M,
            np.NPY_DATETIMEUNIT.NPY_FR_Y,
        ):
            arr = dt64arr_as_int64_s(arr, arr_reso, 0, copy)
        # my_unit [rest] -> tg_unit [s]
        return arr.astype(DT64_DTYPE_SS)

    # To minute [m]
    if tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        # my_unit [ns] -> tg_unit [m]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_floor(arr, NS_MINUTE, 0, copy)
        # my_unit [M, Y] -> tg_unit [m]
        elif arr_reso in (
            np.NPY_DATETIMEUNIT.NPY_FR_M,
            np.NPY_DATETIMEUNIT.NPY_FR_Y,
        ):
            arr = dt64arr_as_int64_m(arr, arr_reso, 0, copy)
        # my_unit [rest] -> tg_unit [m]
        return arr.astype(DT64_DTYPE_MI)

    # To hour [h]
    if tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        # my_unit [ns] -> tg_unit [h]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_floor(arr, NS_HOUR, 0, copy)
        # my_unit [M, Y] -> tg_unit [h]
        elif arr_reso in (
            np.NPY_DATETIMEUNIT.NPY_FR_M,
            np.NPY_DATETIMEUNIT.NPY_FR_Y,
        ):
            arr = dt64arr_as_int64_h(arr, arr_reso, 0, copy)
        # my_unit [rest] -> tg_unit [h]
        return arr.astype(DT64_DTYPE_HH)

    # To day [D]
    if tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        # my_unit [ns] -> tg_unit [D]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_floor(arr, NS_DAY, 0, copy)
        # my_unit [M, Y] -> tg_unit [D]
        elif arr_reso in (
            np.NPY_DATETIMEUNIT.NPY_FR_M,
            np.NPY_DATETIMEUNIT.NPY_FR_Y,
        ):
            arr = dt64arr_as_int64_D(arr, arr_reso, 0, copy)
        # my_unit [rest] -> tg_unit [D]
        return arr.astype(DT64_DTYPE_DD)

    # To week [W]
    if tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_W:
        # my_unit [ns] -> tg_unit [W]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_floor(arr, NS_DAY * 7, 0, copy)
        # my_unit [M, Y] -> tg_unit [W]
        elif arr_reso in (
            np.NPY_DATETIMEUNIT.NPY_FR_M,
            np.NPY_DATETIMEUNIT.NPY_FR_Y,
        ):
            arr = dt64arr_as_int64_W(arr, arr_reso, 0, copy)
        # my_unit [rest] -> tg_unit [W]
        return arr.astype(DT64_DTYPE_WW)

    # To month [M]
    if tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_M:
        # my_unit [ns, Y] -> tg_unit [M]
        if arr_reso in (
            np.NPY_DATETIMEUNIT.NPY_FR_ns,
            np.NPY_DATETIMEUNIT.NPY_FR_Y,
        ):
            arr = dt64arr_as_int64_M(arr, arr_reso, 0, copy)
        # my_unit [rest] -> tg_unit [M]
        return arr.astype(DT64_DTYPE_MM)

    # To year [Y]
    if tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:
        arr = dt64arr_as_int64_Y(arr, arr_reso, 0, copy)
        return arr.astype(DT64_DTYPE_YY)

    # Fallback
    return arr.astype(nptime_unit_int2dt64(tg_unit))

# . arithmetic
cdef inline np.ndarray dt64arr_round(np.ndarray arr, str to_unit, int arr_reso=-1, bint copy=True):
    """Round a 1-D datetime64 array to the nearest multiple of `to_unit` (ties-to-even) `<'ndarray[datetime64]'>`

    This function will try to preserve the nanosecond datetime64 resolution 
    when `safe`, and `downgrade` to microsecond unit only to avoid overflow.

    :param arr `<'np.ndarray'>`: The datetime64 array to round. 
        Also support int64 array ticks along with `arr_reso` argument (see below).

    :param to_unit `<'str'>`: Target rounding granularity. 
        Supports: `'ns', 'us', 'ms', 's', 'm', 'h', 'D'`.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'np.ndarray[datetime64]'>`: The rounded array.
        `NaT` values are preserved.
    """
    if to_unit is None:
        _raise_invalid_nptime_str_unit_error(to_unit)
    # Get array resolution
    cdef bint is_dt64 = np.PyArray_TYPE(arr) == np.NPY_TYPES.NPY_DATETIME
    if arr_reso < 0:
        if not is_dt64:
            _raise_missing_arr_reso_error(arr)
        arr_reso = get_arr_nptime_unit(arr)
    elif not is_dt64:
        arr = arr.astype(nptime_unit_int2dt64(arr_reso))
        copy = False
    #: from now on, 'arr' can only be datetime64 
    #: array under the 'arr_reso' resolution.
    cdef int tg_unit = nptime_unit_str2int(to_unit)

    # Fast-path
    #: source is coarser or equal to target (no rounding needed)
    if arr_reso <= tg_unit:
        return arr_add(arr, 0, copy)  # honor `copy` flag

    # To microsecond [us]
    if tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        # my_unit [ns] -> tg_unit [us]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_even(arr, NS_MICROSECOND, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            else:
                arr = arr_div_even_mul(arr, NS_MICROSECOND, NS_MICROSECOND, 0, copy)
                return arr.astype(DT64_DTYPE_NS)
        # Coarser/equal units are handled by fast-path.

    # To millisecond [ms]
    elif tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        # my_unit [ns] -> tg_unit [ms]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_even_mul(arr, NS_MILLISECOND, US_MILLISECOND, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            else:
                arr = arr_div_even_mul(arr, NS_MILLISECOND, NS_MILLISECOND, 0, copy)
                return arr.astype(DT64_DTYPE_NS)
        # my_unit [us] -> tg_unit [ms]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_div_even_mul(arr, US_MILLISECOND, US_MILLISECOND, 0, copy)
            return arr.astype(DT64_DTYPE_US)
        # Coarser/equal units are handled by fast-path.

    # To second [s]
    elif tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        # my_unit [ns] -> tg_unit [s]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_even_mul(arr, NS_SECOND, US_SECOND, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            else:
                arr = arr_div_even_mul(arr, NS_SECOND, NS_SECOND, 0, copy)
                return arr.astype(DT64_DTYPE_NS)
        # my_unit [us] -> tg_unit [s]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_div_even_mul(arr, US_SECOND, US_SECOND, 0, copy)
            return arr.astype(DT64_DTYPE_US)
        # my_unit [ms] -> tg_unit [s]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_div_even_mul(arr, MS_SECOND, MS_SECOND, 0, copy)
            return arr.astype(DT64_DTYPE_MS)
        # Coarser/equal units are handled by fast-path.

    # To minute [m]
    elif tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        # my_unit [ns] -> tg_unit [m]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_even_mul(arr, NS_MINUTE, US_MINUTE, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            else:
                arr = arr_div_even_mul(arr, NS_MINUTE, NS_MINUTE, 0, copy)
                return arr.astype(DT64_DTYPE_NS)
        # my_unit [us] -> tg_unit [m]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_div_even_mul(arr, US_MINUTE, US_MINUTE, 0, copy)
            return arr.astype(DT64_DTYPE_US)
        # my_unit [ms] -> tg_unit [m]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_div_even_mul(arr, MS_MINUTE, MS_MINUTE, 0, copy)
            return arr.astype(DT64_DTYPE_MS)
        # my_unit [s] -> tg_unit [m]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:
            arr = arr_div_even_mul(arr, SS_MINUTE, SS_MINUTE, 0, copy)
            return arr.astype(DT64_DTYPE_SS)
        # Coarser/equal units are handled by fast-path.
    
    # To hour [h]
    elif tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        # my_unit [ns] -> tg_unit [h]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_even_mul(arr, NS_HOUR, US_HOUR, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            else:
                arr = arr_div_even_mul(arr, NS_HOUR, NS_HOUR, 0, copy)
                return arr.astype(DT64_DTYPE_NS)
        # my_unit [us] -> tg_unit [h]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_div_even_mul(arr, US_HOUR, US_HOUR, 0, copy)
            return arr.astype(DT64_DTYPE_US)
        # my_unit [ms] -> tg_unit [h]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_div_even_mul(arr, MS_HOUR, MS_HOUR, 0, copy)
            return arr.astype(DT64_DTYPE_MS)
        # my_unit [s] -> tg_unit [h]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:
            arr = arr_div_even_mul(arr, SS_HOUR, SS_HOUR, 0, copy)
            return arr.astype(DT64_DTYPE_SS)
        # my_unit [m] -> tg_unit [h]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:
            arr = arr_div_even_mul(arr, 60, 60, 0, copy)
            return arr.astype(DT64_DTYPE_MI)
        # Coarser/equal units are handled by fast-path.

    # To day [D]
    elif tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        # my_unit [ns] -> tg_unit [D]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_even_mul(arr, NS_DAY, US_DAY, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            else:
                arr = arr_div_even_mul(arr, NS_DAY, NS_DAY, 0, copy)
                return arr.astype(DT64_DTYPE_NS)
        # my_unit [us] -> tg_unit [D]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_div_even_mul(arr, US_DAY, US_DAY, 0, copy)
            return arr.astype(DT64_DTYPE_US)
        # my_unit [ms] -> tg_unit [D]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_div_even_mul(arr, MS_DAY, MS_DAY, 0, copy)
            return arr.astype(DT64_DTYPE_MS)
        # my_unit [s] -> tg_unit [D]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:
            arr = arr_div_even_mul(arr, SS_DAY, SS_DAY, 0, copy)
            return arr.astype(DT64_DTYPE_SS)
        # my_unit [m] -> tg_unit [D]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:
            arr = arr_div_even_mul(arr, 1440, 1440, 0, copy)
            return arr.astype(DT64_DTYPE_MI)
        # my_unit [h] -> tg_unit [D]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_h:
            arr = arr_div_even_mul(arr, 24, 24, 0, copy)
            return arr.astype(DT64_DTYPE_HH)
        # Coarser/equal units are handled by fast-path.

    # Invalid unit
    _raise_arr_dt64_rcl_error("round", arr_reso, to_unit)

cdef inline np.ndarray dt64arr_ceil(np.ndarray arr, str to_unit, int arr_reso=-1, bint copy=True):
    """Ceil a 1-D datetime64 array to the nearest multiple of `to_unit` `<'ndarray[datetime64]'>`.

    This function will try to preserve the nanosecond datetime64 resolution 
    when `safe`, and `downgrade` to microsecond unit only to avoid overflow.

    :param arr `<'np.ndarray'>`: The datetime64 array to ceil. 
        Also support int64 array ticks along with `arr_reso` argument (see below).

    :param to_unit `<'str'>`: Target ceiling granularity. 
        Supports: `'ns', 'us', 'ms', 's', 'm', 'h', 'D'`.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'np.ndarray[datetime64]'>`: The ceiled array.
        `NaT` values are preserved.
    """
    if to_unit is None:
        _raise_invalid_nptime_str_unit_error(to_unit)
    # Get array resolution
    cdef bint is_dt64 = np.PyArray_TYPE(arr) == np.NPY_TYPES.NPY_DATETIME
    if arr_reso < 0:
        if not is_dt64:
            _raise_missing_arr_reso_error(arr)
        arr_reso = get_arr_nptime_unit(arr)
    elif not is_dt64:
        arr = arr.astype(nptime_unit_int2dt64(arr_reso))
        copy = False
    #: from now on, 'arr' can only be datetime64 
    #: array under the 'arr_reso' resolution.
    cdef int tg_unit = nptime_unit_str2int(to_unit)

    # Fast-path
    #: source is coarser or equal to target (no rounding needed)
    if arr_reso <= tg_unit:
        return arr_add(arr, 0, copy)  # honor `copy` flag

    # To microsecond [us]
    if tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        # my_unit [ns] -> tg_unit [us]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_ceil(arr, NS_MICROSECOND, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            else:
                arr = arr_div_ceil_mul(arr, NS_MICROSECOND, NS_MICROSECOND, 0, copy)
                return arr.astype(DT64_DTYPE_NS)
        # Coarser/equal units are handled by fast-path.

    # To millisecond [ms]
    elif tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        # my_unit [ns] -> tg_unit [ms]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_ceil_mul(arr, NS_MILLISECOND, US_MILLISECOND, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            else:
                arr = arr_div_ceil_mul(arr, NS_MILLISECOND, NS_MILLISECOND, 0, copy)
                return arr.astype(DT64_DTYPE_NS)
        # my_unit [us] -> tg_unit [ms]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_div_ceil_mul(arr, US_MILLISECOND, US_MILLISECOND, 0, copy)
            return arr.astype(DT64_DTYPE_US)
        # Coarser/equal units are handled by fast-path.

    # To second [s]
    elif tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        # my_unit [ns] -> tg_unit [s]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_ceil_mul(arr, NS_SECOND, US_SECOND, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            else:
                arr = arr_div_ceil_mul(arr, NS_SECOND, NS_SECOND, 0, copy)
                return arr.astype(DT64_DTYPE_NS)
        # my_unit [us] -> tg_unit [s]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_div_ceil_mul(arr, US_SECOND, US_SECOND, 0, copy)
            return arr.astype(DT64_DTYPE_US)
        # my_unit [ms] -> tg_unit [s]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_div_ceil_mul(arr, MS_SECOND, MS_SECOND, 0, copy)
            return arr.astype(DT64_DTYPE_MS)
        # Coarser/equal units are handled by fast-path.

    # To minute [m]
    elif tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        # my_unit [ns] -> tg_unit [m]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_ceil_mul(arr, NS_MINUTE, US_MINUTE, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            else:
                arr = arr_div_ceil_mul(arr, NS_MINUTE, NS_MINUTE, 0, copy)
                return arr.astype(DT64_DTYPE_NS)
        # my_unit [us] -> tg_unit [m]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_div_ceil_mul(arr, US_MINUTE, US_MINUTE, 0, copy)
            return arr.astype(DT64_DTYPE_US)
        # my_unit [ms] -> tg_unit [m]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_div_ceil_mul(arr, MS_MINUTE, MS_MINUTE, 0, copy)
            return arr.astype(DT64_DTYPE_MS)
        # my_unit [s] -> tg_unit [m]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:
            arr = arr_div_ceil_mul(arr, SS_MINUTE, SS_MINUTE, 0, copy)
            return arr.astype(DT64_DTYPE_SS)
        # Coarser/equal units are handled by fast-path.
    
    # To hour [h]
    elif tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        # my_unit [ns] -> tg_unit [h]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_ceil_mul(arr, NS_HOUR, US_HOUR, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            else:
                arr = arr_div_ceil_mul(arr, NS_HOUR, NS_HOUR, 0, copy)
                return arr.astype(DT64_DTYPE_NS)
        # my_unit [us] -> tg_unit [h]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_div_ceil_mul(arr, US_HOUR, US_HOUR, 0, copy)
            return arr.astype(DT64_DTYPE_US)
        # my_unit [ms] -> tg_unit [h]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_div_ceil_mul(arr, MS_HOUR, MS_HOUR, 0, copy)
            return arr.astype(DT64_DTYPE_MS)
        # my_unit [s] -> tg_unit [h]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:
            arr = arr_div_ceil_mul(arr, SS_HOUR, SS_HOUR, 0, copy)
            return arr.astype(DT64_DTYPE_SS)
        # my_unit [m] -> tg_unit [h]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:
            arr = arr_div_ceil_mul(arr, 60, 60, 0, copy)
            return arr.astype(DT64_DTYPE_MI)
        # Coarser/equal units are handled by fast-path.

    # To day [D]
    elif tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        # my_unit [ns] -> tg_unit [D]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_ceil_mul(arr, NS_DAY, US_DAY, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            else:
                arr = arr_div_ceil_mul(arr, NS_DAY, NS_DAY, 0, copy)
                return arr.astype(DT64_DTYPE_NS)
        # my_unit [us] -> tg_unit [D]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_div_ceil_mul(arr, US_DAY, US_DAY, 0, copy)
            return arr.astype(DT64_DTYPE_US)
        # my_unit [ms] -> tg_unit [D]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_div_ceil_mul(arr, MS_DAY, MS_DAY, 0, copy)
            return arr.astype(DT64_DTYPE_MS)
        # my_unit [s] -> tg_unit [D]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:
            arr = arr_div_ceil_mul(arr, SS_DAY, SS_DAY, 0, copy)
            return arr.astype(DT64_DTYPE_SS)
        # my_unit [m] -> tg_unit [D]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:
            arr = arr_div_ceil_mul(arr, 1440, 1440, 0, copy)
            return arr.astype(DT64_DTYPE_MI)
        # my_unit [h] -> tg_unit [D]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_h:
            arr = arr_div_ceil_mul(arr, 24, 24, 0, copy)
            return arr.astype(DT64_DTYPE_HH)
        # Coarser/equal units are handled by fast-path.

    # Invalid unit
    _raise_arr_dt64_rcl_error("ceil", arr_reso, to_unit)

cdef inline np.ndarray dt64arr_floor(np.ndarray arr, str to_unit, int arr_reso=-1, bint copy=True):
    """Floor a 1-D datetime64 array to the nearest multiple of `to_unit` `<'ndarray[datetime64]'>`.
    
    This function will try to preserve the nanosecond datetime64 resolution 
    when `safe`, and `downgrade` to microsecond unit only to avoid overflow.

    :param arr `<'np.ndarray'>`: The datetime64 array to floor. 
        Also support int64 array ticks along with `arr_reso` argument (see below).

    :param to_unit `<'str'>`: Target flooring granularity.
        Supports: `'ns', 'us', 'ms', 's', 'm', 'h', 'D'`.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `datetime64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'np.ndarray[datetime64]'>`: The floored array.
        `NaT` values are preserved.
    """
    if to_unit is None:
        _raise_invalid_nptime_str_unit_error(to_unit)
    # Get array resolution
    cdef bint is_dt64 = np.PyArray_TYPE(arr) == np.NPY_TYPES.NPY_DATETIME
    if arr_reso < 0:
        if not is_dt64:
            _raise_missing_arr_reso_error(arr)
        arr_reso = get_arr_nptime_unit(arr)
    elif not is_dt64:
        arr = arr.astype(nptime_unit_int2dt64(arr_reso))
        copy = False
    #: from now on, 'arr' can only be datetime64 
    #: array under the 'arr_reso' resolution.
    cdef int tg_unit = nptime_unit_str2int(to_unit)

    # Fast-path
    #: source is coarser or equal to target (no rounding needed)
    if arr_reso <= tg_unit:
        return arr_add(arr, 0, copy)  # honor `copy` flag

    # To microsecond [us]
    if tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        # my_unit [ns] -> tg_unit [us]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_floor(arr, NS_MICROSECOND, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            else:
                arr = arr_div_floor_mul(arr, NS_MICROSECOND, NS_MICROSECOND, 0, copy)
                return arr.astype(DT64_DTYPE_NS)
        # Coarser/equal units are handled by fast-path.

    # To millisecond [ms]
    elif tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        # my_unit [ns] -> tg_unit [ms]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_floor_mul(arr, NS_MILLISECOND, US_MILLISECOND, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            else:
                arr = arr_div_floor_mul(arr, NS_MILLISECOND, NS_MILLISECOND, 0, copy)
                return arr.astype(DT64_DTYPE_NS)
        # my_unit [us] -> tg_unit [ms]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_div_floor_mul(arr, US_MILLISECOND, US_MILLISECOND, 0, copy)
            return arr.astype(DT64_DTYPE_US)
        # Coarser/equal units are handled by fast-path.

    # To second [s]
    elif tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        # my_unit [ns] -> tg_unit [s]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_floor_mul(arr, NS_SECOND, US_SECOND, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            else:
                arr = arr_div_floor_mul(arr, NS_SECOND, NS_SECOND, 0, copy)
                return arr.astype(DT64_DTYPE_NS)
        # my_unit [us] -> tg_unit [s]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_div_floor_mul(arr, US_SECOND, US_SECOND, 0, copy)
            return arr.astype(DT64_DTYPE_US)
        # my_unit [ms] -> tg_unit [s]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_div_floor_mul(arr, MS_SECOND, MS_SECOND, 0, copy)
            return arr.astype(DT64_DTYPE_MS)
        # Coarser/equal units are handled by fast-path.

    # To minute [m]
    elif tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        # my_unit [ns] -> tg_unit [m]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_floor_mul(arr, NS_MINUTE, US_MINUTE, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            else:
                arr = arr_div_floor_mul(arr, NS_MINUTE, NS_MINUTE, 0, copy)
                return arr.astype(DT64_DTYPE_NS)
        # my_unit [us] -> tg_unit [m]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_div_floor_mul(arr, US_MINUTE, US_MINUTE, 0, copy)
            return arr.astype(DT64_DTYPE_US)
        # my_unit [ms] -> tg_unit [m]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_div_floor_mul(arr, MS_MINUTE, MS_MINUTE, 0, copy)
            return arr.astype(DT64_DTYPE_MS)
        # my_unit [s] -> tg_unit [m]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:
            arr = arr_div_floor_mul(arr, SS_MINUTE, SS_MINUTE, 0, copy)
            return arr.astype(DT64_DTYPE_SS)
        # Coarser/equal units are handled by fast-path.
    
    # To hour [h]
    elif tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        # my_unit [ns] -> tg_unit [h]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_floor_mul(arr, NS_HOUR, US_HOUR, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            else:
                arr = arr_div_floor_mul(arr, NS_HOUR, NS_HOUR, 0, copy)
                return arr.astype(DT64_DTYPE_NS)
        # my_unit [us] -> tg_unit [h]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_div_floor_mul(arr, US_HOUR, US_HOUR, 0, copy)
            return arr.astype(DT64_DTYPE_US)
        # my_unit [ms] -> tg_unit [h]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_div_floor_mul(arr, MS_HOUR, MS_HOUR, 0, copy)
            return arr.astype(DT64_DTYPE_MS)
        # my_unit [s] -> tg_unit [h]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:
            arr = arr_div_floor_mul(arr, SS_HOUR, SS_HOUR, 0, copy)
            return arr.astype(DT64_DTYPE_SS)
        # my_unit [m] -> tg_unit [h]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:
            arr = arr_div_floor_mul(arr, 60, 60, 0, copy)
            return arr.astype(DT64_DTYPE_MI)
        # Coarser/equal units are handled by fast-path.

    # To day [D]
    elif tg_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        # my_unit [ns] -> tg_unit [D]
        if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, DT_NPY_UNIT_NS):
                arr = arr_div_floor_mul(arr, NS_DAY, US_DAY, 0, copy)
                return arr.astype(DT64_DTYPE_US)
            else:
                arr = arr_div_floor_mul(arr, NS_DAY, NS_DAY, 0, copy)
                return arr.astype(DT64_DTYPE_NS)
        # my_unit [us] -> tg_unit [D]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_div_floor_mul(arr, US_DAY, US_DAY, 0, copy)
            return arr.astype(DT64_DTYPE_US)
        # my_unit [ms] -> tg_unit [D]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_div_floor_mul(arr, MS_DAY, MS_DAY, 0, copy)
            return arr.astype(DT64_DTYPE_MS)
        # my_unit [s] -> tg_unit [D]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:
            arr = arr_div_floor_mul(arr, SS_DAY, SS_DAY, 0, copy)
            return arr.astype(DT64_DTYPE_SS)
        # my_unit [m] -> tg_unit [D]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:
            arr = arr_div_floor_mul(arr, 1440, 1440, 0, copy)
            return arr.astype(DT64_DTYPE_MI)
        # my_unit [h] -> tg_unit [D]
        elif arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_h:
            arr = arr_div_floor_mul(arr, 24, 24, 0, copy)
            return arr.astype(DT64_DTYPE_HH)
        # Coarser/equal units are handled by fast-path.

    # Invalid unit
    _raise_arr_dt64_rcl_error("floor", arr_reso, to_unit)

# . errors
cdef inline bint _raise_missing_arr_reso_error(np.ndarray arr) except -1:
    """(internal) Raise error for array missing resolution information.
    
    :param arr `<'np.ndarray'>`: The array missing resolution information.
    """
    raise ValueError(
        "missing time resolution for <ndarray[%s]>. "
        "Pleae provide the `arr_reso` (NPY_DATETIMEUNIT enum) of the array." % arr.dtype
    )

cdef inline bint _raise_dt64arr_to_reso_error(np.ndarray arr, int arr_reso, str to_unit) except -1:
    """(internal) Raise error for unsupported conversion unit for datetime-like array.
    
    :param arr `<'np.ndarray'>`: The datetime-like array.
    :param arr_reso `<'int'>`: The resolusion of the datetime-like array.
    :param to_unit `<'str'>`: The target conversion numpy unit.
    """
    cdef str dtype_str = str(arr.dtype)
    cdef str arr_reso_str
    try:
        arr_reso_str = nptime_unit_int2str(arr_reso)
    except Exception as err:
        raise ValueError(
            "Cannot convert ndarray[%s] from '%d' unit to int64 '%s' ticks.\n"
            "Supported units: %d→'Y', %d→'M', %d→'W', %d→'D', %d→'h', %d→'m', "
            "%d→'s', %d→'ms', %d→'us', %d→'ns'" % (
                dtype_str, 
                arr_reso,
                to_unit,
                DT_NPY_UNIT_YY,
                DT_NPY_UNIT_MM,
                DT_NPY_UNIT_WW,
                DT_NPY_UNIT_DD,
                DT_NPY_UNIT_HH,
                DT_NPY_UNIT_MI,
                DT_NPY_UNIT_SS,
                DT_NPY_UNIT_MS,
                DT_NPY_UNIT_US,
                DT_NPY_UNIT_NS,
            )
        ) from err
    else:
        raise ValueError(
            "Cannot convert ndarray[%s[%s]] to int64 '%s' ticks.\n"
            "Supported units: 'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'" 
            % (dtype_str, arr_reso_str, to_unit)
        )

cdef inline bint _raise_arr_dt64_rcl_error(str ops, int arr_reso, str to_unit) except -1:
    """(internal) Raise error for unsupported unit for 'dt64arr_round/ceil/floor()' functions.
    
    :param arr `<'np.ndarray'>`: The datetime64 array.
    :param arr_reso `<'int'>`: The resolusion of the datetime array.
    :param to_unit `<'str'>`: The target round/ceil/floor numpy unit.
    """
    raise ValueError(
        "Failed: <'ndarray[datetime64[%s]]'> %s to '%s' resolution is not supported." 
        % (nptime_unit_int2str(arr_reso), ops, to_unit)
    )

# NumPy: ndarray[timedelta64] --------------------------------------------------------------------------
# . type check
cdef inline bint is_td64arr(np.ndarray arr) except -1:
    """Check if the array is dtype of 'timedelta64[*]' `<'bool'>`.
    
    ## Equivalent
    >>> isinstance(arr.dtype, np.dtypes.TimeDelta64DType)
    """
    return np.PyArray_TYPE(arr) == np.NPY_TYPES.NPY_TIMEDELTA

cdef inline bint assure_td64arr(np.ndarray arr) except -1:
    """Assure the array is dtype of 'timedelta64[*]'."""
    if not is_td64arr(arr):
        raise TypeError(
            "Expects instance of 'np.ndarray[timedelta64[*]]', "
            "instead got 'np.ndarray[%s]'." % arr.dtype
        )
    return True

# . conversion
cdef inline np.ndarray td64arr_as_int64_us(np.ndarray arr, int arr_reso=-1, np.npy_int64 offset=0, bint copy=True):
    """Convert a 1-D ndarray[timedelta64[*]] to int64 microsecond ticks (us) `<'ndarray[int64]'>`.
    
    :param arr `<'np.ndarray'>`: The 1-D array dtype of timedelta64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.
    
        - If not specified and `arr` is `timedelta64[*]`, the array's 
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("timedelta64[us]").astype("int64") + offset
    """
    # Get array resolution
    if arr_reso < 0:
        if np.PyArray_TYPE(arr) != np.NPY_TYPES.NPY_TIMEDELTA:
            _raise_missing_arr_reso_error(arr)
        arr_reso = get_arr_nptime_unit(arr)

    # Conversion
    arr = arr_assure_int64(arr, copy)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:   # nanosecond
        return arr_div_floor(arr, NS_MICROSECOND, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:   # microsecond
        return arr_add(arr, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:   # millisecond
        return arr_mul(arr, US_MILLISECOND, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:    # second
        return arr_mul(arr, US_SECOND, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:    # minute
        return arr_mul(arr, US_MINUTE, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_h:    # hour
        return arr_mul(arr, US_HOUR, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_D:    # day
        return arr_mul(arr, US_DAY, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_W:    # week
        return arr_mul(arr, US_DAY * 7, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_M:    # month
        return _td64arr_M_as_int64_D(arr, np.NPY_DATETIMEUNIT.NPY_FR_us, offset, False)
    if arr_reso == np.NPY_DATETIMEUNIT.NPY_FR_Y:    # year
        return _td64arr_Y_as_int64_D(arr, np.NPY_DATETIMEUNIT.NPY_FR_us, offset, False)

    # Unsupported array resolution
    _raise_dt64arr_to_reso_error(arr, arr_reso, "us")

cdef inline np.ndarray _td64arr_Y_as_int64_D(np.ndarray arr, int to_reso, np.npy_int64 offset=0, bint copy=True):
    """(internal) Convert a 1-D ndarray[timedelta64[Y]] to int64 day ticks using an 
    average year (365.2425 D), then further adjust the resolution by `to_reso` and 
    add `offset` `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: 1-D array with dtype timedelta64[Y].
    :param to_reso `<'int'>`: Post-conversion `NPY_DATETIMEUNIT` unit 
        adjustment applied to the day counts.
    :param offset `<'int'>`: Optional value to add at the end. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.
    """
    # Exact rational forms to avoid FP
    #   365.2425 d                      = 146,097 / 400
    #   8,765.82 h  = 24 * 365.2425     = 438,291 / 50
    #   525,949.2 m = 1440 * 365.2425   = 2,629,746 / 5
    arr = arr_assure_int64(arr, copy)
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_D:   # day
        return arr_div_floor(arr_mul(arr, 146_097, 0, False), 400, offset, False)
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_h:   # hour
        return arr_div_floor(arr_mul(arr, 438_291, 0, False), 50, offset, False)
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:   # minute
        return arr_div_floor(arr_mul(arr, 2_629_746, 0, False), 5, offset, False)
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:   # second
        return arr_mul(arr, TD64_YY_SECOND, offset, False)
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_mul(arr, TD64_YY_MILLISECOND, offset, False)
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_mul(arr, TD64_YY_MICROSECOND, offset, False)
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_mul(arr, TD64_YY_NANOSECOND, offset, False)

    # Unsupported conversion resolution
    raise AssertionError("Unsupported conversion unit '%d' for timedelta64 array." % to_reso)

cdef inline np.ndarray _td64arr_M_as_int64_D(np.ndarray arr, int to_reso, np.npy_int64 offset=0, bint copy=True):
    """(internal) Convert a 1-D ndarray[timedelta64[M]] to int64 day ticks using an 
    average month (365.2425 / 12 = 30.436875 D), then further adjust the resolution 
    by `to_reso` and add `offset` `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: 1-D array with dtype timedelta64[M].
    :param to_reso `<'int'>`: Post-conversion `NPY_DATETIMEUNIT` unit 
        adjustment applied to the day counts.
    :param offset `<'int'>`: Optional value to add at the end. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.
    """
    # Exact rational forms (avoid FP):
    #   30.436875 d = 365.2425 / 12 d   = 48699 / 1600 
    #   730.485 h   = 24 * 30.436875    = 146097 / 200
    #   43,829.1 m  = 1440 * 30.436875  = 438291 / 10
    arr = arr_assure_int64(arr, copy)
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_D:   # day
        return arr_div_floor(arr_mul(arr, 48_699, 0, False), 1_600, offset, False)
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_h:   # hour
        return arr_div_floor(arr_mul(arr, 146_097, 0, False), 200, offset, False)
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_m:   # minute
        return arr_div_floor(arr_mul(arr, 438_291, 0, False), 10, offset, False)
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_s:   # second
        return arr_mul(arr, TD64_MM_SECOND, offset, False)
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_mul(arr, TD64_MM_MILLISECOND, offset, False)
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_mul(arr, TD64_MM_MICROSECOND, offset, False)
    if to_reso == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_mul(arr, TD64_MM_NANOSECOND, offset, False)

    # Unsupported conversion resolution
    raise AssertionError("Unsupported conversion unit '%d' for timedelta64 array." % to_reso)
