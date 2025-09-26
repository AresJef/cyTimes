# cython: language_level=3

# Cython imports
cimport cython
cimport numpy as np
from libc.limits cimport LLONG_MIN
from libc.time cimport (
    strftime, time_t,
    gmtime as libc_gmtime,
    localtime as libc_localtime
)
from cpython cimport datetime
from cpython.time cimport time as unix_time
from cpython.exc cimport PyErr_SetFromErrno
from cpython.pyport cimport PY_SSIZE_T_MAX
from cpython.unicode cimport (
    PyUnicode_Count,
    PyUnicode_AsUTF8,
    PyUnicode_DecodeUTF8,
    PyUnicode_ReadChar as str_read,
    PyUnicode_GET_LENGTH as str_len,
    PyUnicode_FromOrdinal as str_chr,
    PyUnicode_Substring as str_substr,
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
    # . timezone
    datetime.tzinfo UTC
    object _LOCAL_TZ
    dict _TIMEZONE_MAP
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
    #: numpy 
    np.npy_int64 NP_NAT = LLONG_MIN

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
cdef inline long long math_mod(long long num, long long factor, long long offset=0):
    """Computes the modulo of a number by the factor, handling
    negative numbers according to Python's modulo semantics `<'int'>`.

    Equivalent to:
    >>> (num % factor) + offset
    """
    if factor == 0:
        raise ZeroDivisionError("division by zero for 'utils.math_mod()'.")
    if factor == -1 and num == LLONG_MIN:
        # (LLONG_MIN % -1) == 0 in Python semantics; avoid UB by skipping '%'
        return offset

    cdef:
        bint neg_f = factor < 0
        long long r
    
    with cython.cdivision(True):
        r = num % factor
        if r != 0:
            if not neg_f:
                if r < 0:
                    r += factor
            else:
                if r > 0:
                    r += factor
    return r + offset

cdef inline long long math_round_div(long long num, long long factor, long long offset=0):
    """Divides a number by the factor and rounds the result to
    the nearest integer (half away from zero), handling negative
    numbers according to Python's division semantics `<'int'>`.

    Equivalent to:
    >>> round(num / factor, 0) + offset
    """
    if factor == 0:
        raise ZeroDivisionError("division by zero for 'utils.math_round_div()'.")
    if factor == -1 and num == LLONG_MIN:
        raise OverflowError("math_round_div: result does not fit in signed 64-bit")

    cdef:
        bint neg_f = factor < 0
        long long abs_f = -factor if neg_f else factor
        long long q, r, abs_r
    
    with cython.cdivision(True):
        q = num // factor
        r = num % factor
        abs_r = -r if r < 0 else r
        if abs_r * 2 >= abs_f:
            if (not neg_f and num >= 0) or (neg_f and num < 0):
                q += 1
            else:
                q -= 1
    return q + offset

cdef inline long long math_ceil_div(long long num, long long factor, long long offset=0):
    """Divides a number by the factor and rounds the result up
    to the nearest integer, handling negative numbers according
    to Python's division semantics `<'int'>`.

    Equivalent to:
    >>> math.ceil(num / factor) + offset
    """
    if factor == 0:
        raise ZeroDivisionError("division by zero for 'utils.math_ceil_div()'.")
    if factor == -1 and num == LLONG_MIN:
        raise OverflowError("math_ceil_div: result does not fit in signed 64-bit")

    cdef long long q, r
    with cython.cdivision(True):
        q = num // factor
        r = num % factor
        if r != 0:
            if factor > 0:
                if num > 0:
                    q += 1
            else:
                if num < 0:
                    q += 1
    return q + offset

cdef inline long long math_floor_div(long long num, long long factor, long long offset=0):
    """Divides a number by the factor and rounds the result
    down to the nearest integer, handling negative numbers
    according to Python's division semantics `<'int'>`.

    Equivalent to:
    >>> math.floor(num / factor) + offset
    """
    if factor == 0:
        raise ZeroDivisionError("division by zero for 'utils.math_floor_div()'.")
    if factor == -1 and num == LLONG_MIN:
        raise OverflowError("math_floor_div: result does not fit in signed 64-bit")

    cdef long long q, r
    with cython.cdivision(True):
        q = num // factor
        r = num % factor
        if r != 0:
            if factor > 0:
                if num < 0:
                    q -= 1
            else:
                if num > 0:
                    q -= 1
    return q + offset

# Parser -----------------------------------------------------------------------------------------------
# . check
cdef inline Py_ssize_t str_count(str s, str substr) except -1:
    """Get the number of occurrences of a 'substr' in an unicode `<'int'>`.

    Equivalent to:
    >>> s.count(substr)
    """
    return PyUnicode_Count(s, substr, 0, PY_SSIZE_T_MAX)

cdef inline bint is_iso_sep(Py_UCS4 ch) except -1:
    """Check if the passed in 'ch' is an ISO format date/time seperator (" " or "T") `<'bool'>`"""
    return ch in (" ", "t", "T")

cdef inline bint is_isodate_sep(Py_UCS4 ch) except -1:
    """Check if the passed in 'ch' is an ISO format date fields separator ("-" or "/") `<'bool'>`"""
    return ch in ("-", "/")

cdef inline bint is_isoweek_sep(Py_UCS4 ch) except -1:
    """Check if the passed in 'ch' is an ISO format week number identifier ("W") `<'bool'>`"""
    return ch in ("W", "w")

cdef inline bint is_isotime_sep(Py_UCS4 ch) except -1:
    """Check if the passed in 'ch' is an ISO format time fields separator (":") `<'bool'>`"""
    return ch == ":"

cdef inline bint is_ascii_digit(Py_UCS4 ch) except -1:
    """Check if the passed in 'ch' is an ASCII digit [0-9] `<'bool'>`"""
    return "0" <= ch <= "9"
    
cdef inline bint is_ascii_alpha_upper(Py_UCS4 ch) except -1:
    """Check if the passed in 'ch' is an ASCII alpha in uppercase [A-Z] `<'bool'>`."""
    return "A" <= ch <= "Z"

cdef inline bint is_ascii_alpha_lower(Py_UCS4 ch) except -1:
    """Check if the passed in 'ch' is an ASCII alpha in lowercase [a-z] `<'bool'>`."""
    return "a" <= ch <= "z"

cdef inline bint is_ascii_alpha(Py_UCS4 ch) except -1:
    """Check if the passed in 'ch' is an ASCII alpha [a-zA-Z] `<'bool'>`."""
    return is_ascii_alpha_lower(ch) or is_ascii_alpha_upper(ch)

# . parse
cdef inline int parse_isoyear(str data, Py_ssize_t pos, Py_ssize_t size) except -2:
    """Parse ISO format year component (YYYY) from a string,
    returns `-1` for invalid ISO years `<'int'>`.

    This function extracts and parses the year component from an ISO date string.
    It reads four characters starting at the specified position and converts them
    into an integer representing the year. The function ensures that the parsed
    year is valid (i.e., between '0001' and '9999').

    :param data `<'str'>`: The input string containing the ISO year to parse.
    :param pos `<'int'>`: The starting position in the string of the ISO year.
    :param size `<'int'>`: The length of the input 'data' string.
        If 'size <= 0', the function computes the size of the 'data' string internally.
    """
    # Validate
    if size <= 0:
        size = str_len(data)
    if size - pos < 4:
        return -1  # exit: invalid

    # Parse value
    cdef Py_UCS4 c0 = str_read(data, pos)
    if not is_ascii_digit(c0):
        return -1  # exit: invalid
    cdef Py_UCS4 c1 = str_read(data, pos + 1)
    if not is_ascii_digit(c1):
        return -1  # exit: invalid
    cdef Py_UCS4 c2 = str_read(data, pos + 2)
    if not is_ascii_digit(c2):
        return -1  # exit: invalid
    cdef Py_UCS4 c3 = str_read(data, pos + 3)
    if not is_ascii_digit(c3):
        return -1  # exit: invalid

    # Convert to integer
    cdef int year = (ord(c0) - 48) * 1000 \
                  + (ord(c1) - 48) *  100 \
                  + (ord(c2) - 48) *   10 \
                  + (ord(c3) - 48)
    return year if year > 0 else -1

cdef inline int parse_isomonth(str data, Py_ssize_t pos, Py_ssize_t size)  except -2:
    """Parse ISO format month component (MM) from a string,
    returns `-1` for invalid ISO months `<'int'>`.

    This function extracts and parses the month component from an ISO date string.
    It reads two characters starting at the specified position and converts them
    into an integer representing the month. The function ensures that the parsed
    month is valid (i.e., between '01' and '12').

    :param data `<'str'>`: The input string containing the ISO month to parse.
    :param pos `<'int'>`: The starting position in the string of the ISO month.
    :param size `<'int'>`: The length of the input 'data' string.
        If 'size <= 0', the function computes the size of the 'data' string internally.
    """
    # Validate
    if size <= 0:
        size = str_len(data)
    if size - pos < 2:
        return -1  # exit: invalid

    # Parse value
    cdef Py_UCS4 c0 = str_read(data, pos)
    if not is_ascii_digit(c0):
        return -1  # exit: invalid
    cdef Py_UCS4 c1 = str_read(data, pos + 1)
    if not is_ascii_digit(c1):
        return -1  # exit: invalid

    # Convert to integer
    cdef int month = (ord(c0) - 48) * 10 + (ord(c1) - 48)
    return month if 1 <= month <= 12 else -1
    
cdef inline int parse_isoday(str data, Py_ssize_t pos, Py_ssize_t size) except -2:
    """Parse ISO format day component (DD) from a string,
    returns `-1` for invalid ISO days `<'int'>`.

    This function extracts and parses the day component from an ISO date string.
    It reads two characters starting at the specified position and converts them
    into an integer representing the day. The function ensures that the parsed day
    is valid (i.e., between '01' and '31').

    :param data `<'str'>`: The input string containing the ISO day to parse.
    :param pos `<'int'>`: The starting position in the string of the ISO day.
    :param size `<'int'>`: The length of the input 'data' string.
        If 'size <= 0', the function computes the size of the 'data' string internally.
    """
    # Validate
    if size <= 0:
        size = str_len(data)
    if size - pos < 2:
        return -1  # exit: invalid

    # Parse value
    cdef Py_UCS4 c0 = str_read(data, pos)
    if not is_ascii_digit(c0):
        return -1  # exit: invalid
    cdef Py_UCS4 c1 = str_read(data, pos + 1)
    if not is_ascii_digit(c1):
        return -1  # exit: invalid

    # Convert to integer
    cdef int day = (ord(c0) - 48) * 10 + (ord(c1) - 48)
    return day if 1 <= day <= 31 else -1

cdef inline int parse_isoweek(str data, Py_ssize_t pos, Py_ssize_t size) except -2:
    """Parse an ISO format week number component (WW) from a string,
    returns `-1` for invalid ISO week number `<'int'>`.

    This function extracts and parses the week number from an ISO date string.
    It reads two characters starting at the specified position and converts them
    into an integer representing the week number. The function ensures that the
    parsed week number is valid (i.e., between '01' and '53').

    :param data `<'str'>`: The input string containing the ISO week number to parse.
    :param pos `<'int'>`: The starting position in the string of the ISO week number.
    :param size `<'int'>`: The length of the input 'data' string.
        If 'size <= 0', the function computes the size of the 'data' string internally.
    """
    # Validate
    if size <= 0:
        size = str_len(data)
    if size - pos < 2:
        return -1  # exit: invalid

    # Parse value
    cdef Py_UCS4 c0 = str_read(data, pos)
    if not is_ascii_digit(c0):
        return -1  # exit: invalid
    cdef Py_UCS4 c1 = str_read(data, pos + 1)
    if not is_ascii_digit(c1):
        return -1  # exit: invalid

    # Convert to integer
    cdef int week = (ord(c0) - 48) * 10 + (ord(c1) - 48)
    return week if 1 <= week <= 53 else -1

cdef inline int parse_isoweekday(str data, Py_ssize_t pos, Py_ssize_t size) except -2:
    """Parse an ISO format weekday component (D) from a string,
    returns `-1` for invalid ISO weekdays `<'int'>`.

    This function extracts and parses the weekday component from an ISO date string.
    It reads a single character at the specified position and converts it into an
    integer representing the ISO weekday, where Monday is 1 and Sunday is 7.

    :param data `<'str'>`: The input string containing the ISO weekday to parse.
    :param pos `<'int'>`: The starting position in the string of the ISO weekday.
    :param size `<'int'>`: The length of the input 'data' string.
        If 'size <= 0', the function computes the size of the 'data' string internally.
    """
    # Validate
    if size <= 0:
        size = str_len(data)
    if size - pos < 1:
        return -1  # exit: invalid

    # Parse value
    cdef Py_UCS4 ch = str_read(data, pos)
    if not "1" <= ch <= "7":
        return -1  # exit: invalid

    # Convert to integer
    return ord(ch) - 48

cdef inline int parse_isoyearday(str data, Py_ssize_t pos, Py_ssize_t size) except -2:
    """Parse an ISO format day of the year component (DDD) from a string,
    returns `-1` for invalid ISO day of the year `<'int'>`.

    This function extracts and parses the day of the year from an ISO date string.
    It reads three characters starting at the specified position and converts them
    into an integer representing the day of the year. The function ensures that the
    parsed days are valid (i.e., between '001' and '366').

    :param data `<'str'>`: The input string containing the ISO day of the year to parse.
    :param pos `<'int'>`: The starting position in the string of the ISO day of the year.
    :param size `<'int'>`: The length of the input 'data' string.
        If 'size <= 0', the function computes the size of the 'data' string internally.
    """
    # Validate
    if size <= 0:
        size = str_len(data)
    if size - pos < 3:
        return -1  # exit: invalid

    # Parse value
    cdef Py_UCS4 c0 = str_read(data, pos)
    if not is_ascii_digit(c0):
        return -1  # exit: invalid
    cdef Py_UCS4 c1 = str_read(data, pos + 1)
    if not is_ascii_digit(c1):
        return -1  # exit: invalid
    cdef Py_UCS4 c2 = str_read(data, pos + 2)
    if not is_ascii_digit(c2):
        return -1  # exit: invalid

    # Convert to integer
    cdef int days = (ord(c0) - 48) * 100 \
                    + (ord(c1) - 48) * 10 \
                    + (ord(c2) - 48)
    return days if 1 <= days <= 366 else -1    

cdef inline int parse_isohour(str data, Py_ssize_t pos, Py_ssize_t size) except -2:
    """Parse an ISO format hour (HH) component from a string,
    returns `-1` for invalid ISO hours `<'int'>`.

    This function extracts and parses the hour component from a time string in ISO format.
    It reads two characters starting at the specified position and converts them into an
    integer representing the hours. The function ensures that the parsed hours are valid
    (i.e., between '00' and '23').

    :param data `<'str'>`: The input string containing the ISO hour to parse.
    :param pos `<'int'>`: The starting position in the string of the ISO hour.
    :param size `<'int'>`: The length of the input 'data' string.
        If 'size <= 0', the function computes the size of the 'data' string internally.
    """
    # Validate
    if size <= 0:
        size = str_len(data)
    if size - pos < 2:
        return -1  # exit: invalid

    # Parse value
    cdef Py_UCS4 c0 = str_read(data, pos)
    if not is_ascii_digit(c0):
        return -1  # exit: invalid
    cdef Py_UCS4 c1 = str_read(data, pos + 1)
    if not is_ascii_digit(c1):
        return -1  # exit: invalid

    # Convert to integer
    cdef int hour = (ord(c0) - 48) * 10 + (ord(c1) - 48)
    return hour if 0 <= hour <= 23 else -1

cdef inline int parse_isominute(str data, Py_ssize_t pos, Py_ssize_t size) except -2:
    """Parse an ISO format minute (MM) component from a string,
    returns `-1` for invalid ISO minutes `<'int'>`.

    This function extracts and parses the minute component from a time string in ISO format.
    It reads two characters starting at the specified position and converts them into an
    integer representing the minutes. The function ensures that the parsed minutes are valid
    (i.e., between '00' and '59').

    :param data `<'str'>`: The input string containing the ISO minute to parse.
    :param pos `<'int'>`: The starting position in the string of the ISO minute.
    :param size `<'int'>`: The length of the input 'data' string.
        If 'size <= 0', the function computes the size of the 'data' string internally.
    """
    # Validate
    if size <= 0:
        size = str_len(data)
    if size - pos < 2:
        return -1  # exit: invalid

    # Parse value
    cdef Py_UCS4 c0 = str_read(data, pos)
    if not is_ascii_digit(c0):
        return -1  # exit: invalid
    cdef Py_UCS4 c1 = str_read(data, pos + 1)
    if not is_ascii_digit(c1):
        return -1  # exit: invalid

    # Convert to integer
    cdef int minute = (ord(c0) - 48) * 10 + (ord(c1) - 48)
    return minute if 0 <= minute <= 59 else -1

cdef inline int parse_isosecond(str data, Py_ssize_t pos, Py_ssize_t size) except -2:
    """Parse an ISO format second (SS) component from a string,
    returns `-1` for invalid ISO seconds `<'int'>`.

    This function extracts and parses the second component from a time string in ISO format.
    It reads two characters starting at the specified position and converts them into an
    integer representing the seconds. The function ensures that the parsed seconds are valid
    (i.e., between '00' and '59').

    :param data `<'str'>`: The input string containing the ISO second to parse.
    :param pos `<'int'>`: The starting position in the string of the ISO second.
    :param size `<'int'>`: The length of the input 'data' string.
        If 'size <= 0', the function computes the size of the 'data' string internally.
    """
    return parse_isominute(data, pos, size)

cdef inline int parse_isofraction(str data, Py_ssize_t pos, Py_ssize_t size) except -2:
    """Parse an ISO fractional time component (fractions of a second) from a string,
    returns `-1` for invalid ISO fraction `<'int'>`.

    This function extracts and parses a fractional time component in ISO format (e.g.,
    the fractional seconds in "2023-11-25T14:30:15.123456Z"). It reads up to six digits
    after the starting position, padding with zeros if necessary to ensure a six-digit
    integer representation.

    :param data `<'str'>`: The input string containing the ISO fraction to parse.
    :param pos `<'int'>`: The starting position in the string of the ISO fraction.
    :param size `<'int'>`: The length of the input 'data' string.
        If 'size <= 0', the function computes the size of the 'data' string internally.
    """
    # Validate
    if size <= 0:
        size = str_len(data)

    # Parse value
    cdef Py_ssize_t idx = 0
    cdef int val = 0
    cdef Py_UCS4 ch
    while pos < size and idx < 6:
        ch = str_read(data, pos)
        if not is_ascii_digit(ch):
            break
        val = val * 10 + (ord(ch) - 48)
        pos += 1
        idx += 1

    # Pad with trailing zeros
    if idx == 0:
        return -1  # exit: invalid
    if idx == 6:
        return val
    if idx == 5:
        return val * 10
    if idx == 4:
        return val * 100
    if idx == 3:
        return val * 1000
    if idx == 2:
        return val * 10000
    return val * 100000
    
cdef inline unsigned long long slice_to_uint(str data, Py_ssize_t start, Py_ssize_t size) except -2:
    """Slice a substring from a string and convert to an unsigned integer `<'int'>`.

    This function slices a portion of the input string 'data' starting
    at 'start' and spanning 'size' characters. The sliced substring is
    validated to ensure it contains only ASCII digits, before converting
    to unsigned integer.

    :param data `<'str'>`: The input string to slice and convert.
    :param start `<'int'>`: The starting index for slicing the string.
    :param size `<'int'>`: The number of characters to slice from 'start'.
    """
    # Validate
    if size <= 0:
        raise ValueError("size must be >= 1, got %d" % size)
    cdef Py_ssize_t length = str_len(data)
    if start < 0 or start + size > length:
        raise ValueError("slice_to_uint out of range (start=%d size=%d len=%d)" % (start, size, length))

    # Parse value
    cdef unsigned long long val = 0
    cdef Py_ssize_t i
    cdef Py_UCS4 ch
    cdef unsigned long long digit

    for i in range(size):
        ch = str_read(data, start + i)
        if not is_ascii_digit(ch):
            raise ValueError("invalid character '%s' as an integer." % str_chr(ch))
        digit = ord(ch) - 48
        val = val * 10 + digit
    return val

# Time -------------------------------------------------------------------------------------------------
cdef inline tm tm_gmtime(double ts) except * nogil:
    """Convert a timestamp to 'struct:tm' expressing UTC time `<'struct:tm'>`.

    This function takes a Unix timestamp 'ts' and converts it into 'struct:tm'
    representing the UTC time. It is equivalent to 'time.gmtime(ts)'' in Python
    but implemented in Cython for efficiency.
    """
    cdef time_t tic = <time_t> ts
    cdef tm* t = libc_gmtime(&tic)
    if t is NULL:
        _raise_from_errno()
    # Fix 0-based date values (and the 1900-based year).
    # See tmtotuple() in https://github.com/python/cpython/blob/master/Modules/timemodule.c
    t.tm_year += 1900
    t.tm_mon += 1
    if t.tm_sec > 59:  # clamp leap seconds
        t.tm_sec = 59
    t.tm_wday = (t.tm_wday + 6) % 7
    t.tm_yday += 1
    return t[0]

cdef inline tm tm_localtime(double ts) except * nogil:
    """Convert a timestamp to 'struct:tm' expressing local time `<'struct:tm'>`.

    This function takes a Unix timestamp 'ts' and converts it into 'struct:tm' 
    representing the local time. It is equivalent to 'time.localtime(ts)' in 
    Python but implemented in Cython for efficiency.
    """
    cdef time_t tic = <time_t> ts
    cdef tm* t = libc_localtime(&tic)
    if t is NULL:
        _raise_from_errno()
    # Fix 0-based date values (and the 1900-based year).
    # See tmtotuple() in https://github.com/python/cpython/blob/master/Modules/timemodule.c
    t.tm_year += 1900
    t.tm_mon += 1
    if t.tm_sec > 59:  # clamp leap seconds
        t.tm_sec = 59
    t.tm_wday = (t.tm_wday + 6) % 7
    t.tm_yday += 1
    return t[0]

cdef inline long long ts_gmtime(double ts) except *:
    """Convert a timestamp to UTC seconds since the Unix Epoch `<'int'>`.

    This function converts a Unix timestamp 'ts' to integer in
    seconds since the Unix Epoch, representing the UTC time.
    """
    cdef tm _tm = tm_gmtime(ts)
    cdef long long ordinal = ymd_to_ordinal(_tm.tm_year, _tm.tm_mon, _tm.tm_mday)
    return (
        (ordinal - EPOCH_DAY) * SS_DAY
        + _tm.tm_hour * SS_HOUR
        + _tm.tm_min * SS_MINUTE
        + _tm.tm_sec
    )

cdef inline long long ts_localtime(double ts) except *:
    """Convert a timestamp to local seconds since the Unix Epoch `<'int'>`.

    This function converts a Unix timestamp 'ts' to integer in
    seconds since the Unix Epoch, representing the local time.
    """
    cdef tm _tm = tm_localtime(ts)
    cdef long long ordinal = ymd_to_ordinal(_tm.tm_year, _tm.tm_mon, _tm.tm_mday)
    return (
        (ordinal - EPOCH_DAY) * SS_DAY
        + _tm.tm_hour * SS_HOUR
        + _tm.tm_min * SS_MINUTE
        + _tm.tm_sec
    )

cdef inline int _raise_from_errno() except -1 with gil:
    """(internal) Error handling for 'ts_gmtime/ts_localtime' functions."""
    PyErr_SetFromErrno(RuntimeError)
    return <int> -1  # type: ignore

# . conversion
cdef inline str tm_strftime(tm t, str fmt):
    """Convert 'struct:tm' to string according to the given format `<'str'>`."""
    # Revert fields to 0-based
    t.tm_year -= 1900
    t.tm_mon -= 1
    t.tm_wday = (t.tm_wday + 1) % 7
    t.tm_yday -= 1

    # Perform strftime
    cdef char buffer[256]
    cdef Py_ssize_t size = strftime(buffer, sizeof(buffer), PyUnicode_AsUTF8(fmt), &t)
    if size == 0:
        raise OverflowError("strftime format is too long:\n'%s'" % fmt)
    return PyUnicode_DecodeUTF8(buffer, size, NULL)

cdef inline tm tm_fr_us(long long val) except *:
    """Create 'struct:tm' from `EPOCH` microseconds (int) `<'struct:tm'>`."""
    # Compute ymd & hmsf
    val += EPOCH_MICROSECOND
    cdef:
        ymd _ymd = ymd_fr_ordinal(math_floor_div(val, US_DAY))
        hmsf _hmsf = hmsf_fr_us(val)

    # New 'struct:tm'
    cdef int yy = _ymd.year
    cdef int mm = _ymd.month
    cdef int dd = _ymd.day
    cdef int wday = ymd_weekday(yy, mm, dd)
    cdef int yday = days_bf_month(yy, mm) + dd
    return tm(
        _hmsf.second, _hmsf.minute, _hmsf.hour,  # sec, min, hour
        dd, mm, yy,  # day, mon, year
        wday, yday, -1,# wday, yday, isdst
    )

cdef inline tm tm_fr_seconds(double val) except *:
    """Create 'struct:tm' from `EPOCH` seconds (float) `<'struct:tm'>`."""
    return tm_fr_us(<long long> int(val * US_SECOND))

cdef inline hmsf hmsf_fr_us(long long val) except *:
    """Create 'struct:hmsf' from microseconds (int) `<'struct:hmsf'>`.
    
    Notice that the orgin of the microseconds must be 0,
    and `NOT` the Unix Epoch (1970-01-01 00:00:00).
    """
    if val <= 0:
        return hmsf(0, 0, 0, 0)

    val = math_mod(val, US_DAY)
    cdef int hh = math_floor_div(val, US_HOUR)
    val -= hh * US_HOUR
    cdef int mi = math_floor_div(val, US_MINUTE)
    val -= mi * US_MINUTE
    cdef long long ss = math_floor_div(val, US_SECOND)
    return hmsf(hh, mi, ss, val - ss * US_SECOND)

cdef inline hmsf hmsf_fr_seconds(double val) except *:
    """Create 'struct:hmsf' from seconds (float) `<'struct:hmsf'>`.
    
    Notice that the orgin of the seconds must be 0,
    and `NOT` the Unix Epoch (1970-01-01 00:00:00).
    """
    return hmsf_fr_us(<long long> int(val * US_SECOND))

# Calendar ---------------------------------------------------------------------------------------------
# . year
cdef inline bint is_leap_year(int year) except -1 nogil: 
    """Check if the passed in 'year' is a leap year `<'bool'>`."""
    if year < 1:
        return False
    # Fast path: if not divisible by 4, it's not a leap year
    if year & 3:       # bitwise AND, much cheaper than division
        return False
    # Divisible by 4; centuries are not leap years unless divisible by 400
    if year % 100:
        return True
    return year % 400 == 0

cdef inline bint is_long_year(int year) except -1 nogil:
    """Check if the passed in 'year' is a long year `<'bool'>`.

    #### Long year: maximum ISO week number equal 53.
    """
    if year < 1 or year > 9999:
        return False

    # 0=Mon..6=Sun
    cdef int jan1 = ymd_weekday(year, 1, 1)
    if jan1 == 3:  # Thursday
        return 1

    cdef int dec31 = ymd_weekday(year, 12, 31)
    return dec31 == 3  # Thursday

cdef inline int leap_bt_year(int year1, int year2) except -1 nogil:
    """Compute the number of leap years between 'year1' & 'year2' `<'int'>`."""
    # Normalize order
    if year1 > year2:
        year1, year2 = year2, year1

    # Fast exits
    if year2 <= 0 or year1 == year2:
        return 0

    # Count difference; negative/zero years are treated as 0 by _leaps_upto
    return leap_years(year2, False) - leap_years(year1, False)

cdef inline int leap_years(int y, bint inclusive) except -1 nogil:
    """Count leap years in the range [1 .. y] `<'int'>`.
    
    If inclusive == False, exclusive of y itself.
    I.e., _leaps_upto(1) == 0; _leaps_upto(5) counts only year 4.
    """
    if y <= 1:
        return 0
    if not inclusive:
        y -= 1
    return y // 4 - y // 100 + y // 400

cdef inline int days_in_year(int year) except -1 nogil:
    """Compute the maximum days (365, 366) in the 'year' `<'int'>`."""
    return 366 if is_leap_year(year) else 365

cdef inline int days_bf_year(int year) except -1 nogil:
    """Compute days from 0001-01-01 up to the start of 'year' `<'int'>`."""
    if year <= 1:
        return 0
    cdef int y = year - 1
    # Every year contributes 365 days, plus 1 day per leap year
    return y * 365 + (y // 4) - (y // 100) + (y // 400)

cdef inline int days_of_year(int year, int month, int day) except -1 nogil:
    """Compute the number of days between the 1st day of
    the 'year' and the current Y/M/D `<'int'>`.
    """
    if day > 28:
        day = min(day, days_in_month(year, month))
    elif day < 1:
        day = 1
    return days_bf_month(year, month) + day

# . quarter
cdef inline int quarter_of_month(int month) except -1 nogil:
    """Compute the quarter (1-4) of the passed in 'month' `<'int'>`."""
    # Jan - Mar
    if month <= 3:
        return 1
    # Oct - Dec
    if month >= 10:
        return 4
    # Apr - Sep
    return (month + 2) // 3

cdef inline int days_in_quarter(int year, int month) except -1 nogil:
    """Compute the maximum days in the quarter `<'int'>`."""
    cdef:
        int quarter = quarter_of_month(month)
        int days = DAYS_IN_QUARTER[quarter]
    if quarter == 1 and is_leap_year(year):
        days += 1
    return days

cdef inline int days_bf_quarter(int year, int month) except -1 nogil:
    """Compute the number of days between the 1st day of the
    'year' and the 1st day of the quarter `<'int'>`.
    """
    cdef:
        int quarter = quarter_of_month(month)
        int days = DAYS_BR_QUARTER[quarter - 1]
    if quarter > 1 and is_leap_year(year):
        days += 1
    return days

cdef inline int days_of_quarter(int year, int month, int day) except -1 nogil:
    """Compute the number of days between the 1st day 
    of the quarter and the current Y/M/D `<'int'>`.
    """
    return days_of_year(year, month, day) - days_bf_quarter(year, month)

# . month
cdef inline int days_in_month(int year, int month) except -1 nogil:
    """Compute the maximum days in the month `<'int'>`."""
    if not 1 < month < 12:
        return 31
    cdef int days = DAYS_IN_MONTH[month]
    return days + 1 if month == 2 and is_leap_year(year) else days

cdef inline int days_bf_month(int year, int month) except -1 nogil:
    """Compute the number of days between the 1st day of the
    'year' and the 1st day of the 'month' `<'int'>`.
    """
    # January
    if month <= 1:
        return 0
    # February
    if month == 2:
        return 31
    # Rest
    cdef int days = DAYS_BR_MONTH[month - 1] if month < 12 else 334
    if is_leap_year(year):
        days += 1
    return days

# . week
cdef inline int ymd_weekday(int year, int month, int day) except -1 nogil:
    """Compute the weekday (0=Mon...6=Sun) `<'int'>`."""
    return (ymd_to_ordinal(year, month, day) + 6) % 7

# . iso
cdef inline iso ymd_isocalendar(int year, int month, int day) noexcept nogil:
    """Compute the ISO calendar `<'struct:iso'>`."""
    cdef iso out
    # Fast saturation for invalid lower bound (avoid probing year-1)
    if year < 1:
        out.year, out.week, out.weekday = 1, 1, 1
        return out

    # Core values (helpers are inline/nogil)
    cdef:
        int ordinal = ymd_to_ordinal(year, month, day)
        int ord_1st = iso_week1_monday_ordinal(year)
        int days = ordinal - ord_1st
        int weeks, weekday, next_1st

    # Floor division for potential negative 'days' (divisor > 0)
    if days >= 0:
        weeks = days // 7
    else:
        weeks = - ((-days + 6) // 7)

    # Resolve ISO year/week without ever probing out-of-range years
    if weeks < 0:
        if year > 1:
            year -= 1
            weeks = (ordinal - iso_week1_monday_ordinal(year)) // 7 + 1
        else:
            # Can't go before year 1; pin to ISO week 1
            weeks = 1
    elif weeks >= 52:
        if year < 9999:
            next_1st = iso_week1_monday_ordinal(year + 1)
            if ordinal >= next_1st:
                year += 1
                weeks = 1
            else:
                weeks += 1
        else:
            # No look-ahead beyond 9999
            weeks += 1
    else:
        weeks += 1

    # ISO weekday: 1..7 (Mon..Sun) — compute from 'ordinal' to avoid negative modulo
    weekday = ((ordinal + 6) % 7) + 1
    out.year, out.week, out.weekday = year, weeks, weekday
    return out

cdef inline int ymd_isoyear(int year, int month, int day) except -1 nogil:
    """Compute the ISO calendar year (0-10000) `<'int'>`."""
    cdef iso _iso = ymd_isocalendar(year, month, day)
    return _iso.year

cdef inline int ymd_isoweek(int year, int month, int day) except -1 nogil:
    """Compute the ISO calendar week number (1-53) `<'int'>`."""
    cdef iso _iso = ymd_isocalendar(year, month, day)
    return _iso.week

cdef inline int ymd_isoweekday(int year, int month, int day) except -1 nogil:
    """Compute the ISO weekday (1=Mon...7=Sun) `<'int'>`."""
    return ymd_weekday(year, month, day) + 1

# . Y/M/D
cdef inline int ymd_to_ordinal(int year, int month, int day) except -1 nogil:
    """Convert 'Y/M/D' to Gregorian ordinal days `<'int'>`."""
    if day > 28:
        day = min(day, days_in_month(year, month))
    elif day < 1:
        day = 1
    return days_bf_year(year) + days_bf_month(year, month) + day

cdef inline ymd ymd_fr_ordinal(int val) noexcept nogil:
    """Create 'struct:ymd' from Gregorian ordinal days `<'struct:ymd'>`.

    Faster, branch-light version using 400y cycles and 153-day month blocks.
    Ordinal 'val' is clamped to [1..ORDINAL_MAX]; 1 -> 0001-01-01.
    """
    cdef ymd out
    # Clamp year
    if val <= 1:
        out.year, out.month, out.day = 1, 1, 1
        return out
    if val >= ORDINAL_MAX:
        out.year, out.month, out.day = 9999, 12, 31
        return out

    # Hinnant-style decomposition -------------------------------------------------
    cdef int n = val - 1

    # Add 306 so that March is month 1 internally (makes month calc trivial)
    cdef int z = n + 306

    # 400-year era and day-of-era
    cdef int era = z // 146097                    # [0, ..]
    cdef int doe = z - era * 146097               # [0, 146096]

    # years-of-era (0..399)
    cdef int yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    cdef int y    = yoe + era * 400               # provisional year base

    # day-of-year (0..365)
    cdef int doy = doe - (365 * yoe + yoe // 4 - yoe // 100 + yoe // 400)

    # month partition in 153-day buckets (0..11), then convert to 1..12
    cdef int mp = (5 * doy + 2) // 153
    cdef int dd = doy - (153 * mp + 2) // 5 + 1
    cdef int mm = mp + 3 if mp < 10 else mp - 9

    # final year adjustment for Jan/Feb
    cdef int yy = y + (mm <= 2)

    # Output
    out.year, out.month, out.day = yy, mm, dd
    return out

cdef inline ymd ymd_fr_isocalendar(int year, int week, int weekday) noexcept nogil:
    """Create 'struct:ymd' from ISO calendar values (ISO year, ISO week, ISO weekday) `<'struct:ymd'>`."""
    # Saturate ISO year
    if year < 1:
        return ymd_fr_ordinal(1)
    elif year > 9999:
        return ymd_fr_ordinal(ORDINAL_MAX)

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

    # Compute Jan 1 ordinal and long-year flag
    cdef int day1 = ymd_to_ordinal(year, 1, 1)   # Gregorian ordinal of Jan 1
    cdef int wkd_mon0 = (day1 + 6) % 7           # 0=Mon..6=Sun
    cdef int leap = 0
    if year % 4 == 0:
        if year % 400 == 0:
            leap = 1
        elif year % 100 != 0:
            leap = 1
    cdef bint long_year = (wkd_mon0 == 3) or (leap and wkd_mon0 == 2)

    # Normalize week, safely handling week 53
    if week == 53 and not long_year:
        # Rolls into next ISO year; if out of range, saturate to max date.
        if year == 9999:
            return ymd_fr_ordinal(ORDINAL_MAX)
        year += 1
        day1 = ymd_to_ordinal(year, 1, 1)
        week = 1
        # recompute wkd_mon0/leap/long_year not needed further
    elif not long_year and week > 52:
        week = 52

    # ISO week 1 Monday: Monday on/before Jan 4
    cdef int d = day1 + 3               # ordinal(Jan 4)
    cdef int mon1 = d - ((d + 6) % 7)   # Monday on/before d

    # Target ordinal
    cdef int ordv = mon1 + (week - 1) * 7 + (weekday - 1)
    return ymd_fr_ordinal(ordv)

cdef inline ymd ymd_fr_days_of_year(int year, int days) noexcept nogil:
    """Create 'struct:ymd' from the year and days of the year `<'struct:ymd'>`."""
    cdef ymd out
    # Clamp year
    if year < 1:
        out.year, out.month, out.day = 1, 1, 1
        return out
    elif year > 9999:
        out.year, out.month, out.day = 9999, 12, 31
        return out

    # Clamp days into valid range
    cdef int leap = is_leap_year(year)
    cdef int maxd = 365 + leap
    if days < 1:
        days = 1
    elif days > maxd:
        days = maxd

    # Jan / Feb fast paths (no branches beyond these)
    if days <= 31:
        out.year, out.month, out.day = year, 1, days
        return out
    if days <= 59 + leap:
        out.year, out.month, out.day = year, 2, days - 31
        return out

    # March..December: compare against non-leap cumulative table
    # Binary search on DAYS_BR_MONTH[3..12]
    cdef int target = days - leap
    cdef int lo = 3
    cdef int hi = 12
    cdef int mid
    while lo < hi:
        mid = (lo + hi) >> 1
        if target <= DAYS_BR_MONTH[mid]:
            hi = mid
        else:
            lo = mid + 1
    out.year, out.month = year, lo
    out.day = target - DAYS_BR_MONTH[lo - 1]
    return out

cdef inline int iso_week1_monday_ordinal(int year) except -1 nogil:
    """Gregorian ordinal for the Monday starting ISO week 1 of year `<'int'>`."""
    cdef int d = ymd_to_ordinal(year, 1, 1) + 3  # == ordinal of Jan 4
    # Monday on or before d:  d - ((d + 6) % 7)   (Mon=0 with ordinal 1=Mon)
    return d - ((d + 6) % 7)

# datetime.date ----------------------------------------------------------------------------------------
# . generate
cdef inline datetime.date date_new(int year=1, int month=1, int day=1):
    """Create a new `<'datetime.date'>`.
    
    Equivalent to:
    >>> datetime.date(year, month, day)
    """
    # Clamp year
    if year < 1:
        return datetime.date_new(1, 1, 1)
    if year > 9999:
        return datetime.date_new(9999, 12, 31)

    # Clamp month & day
    month = min(max(month, 1), 12)
    if day > 28:
        day = min(day, days_in_month(year, month))
    elif day < 1:
        day = 1

    # New date        
    return datetime.date_new(year, month, day)

cdef inline datetime.date date_now(object tz=None):
    """Get the current date `<'datetime.date'>`.
    
    Equivalent to:
    >>> datetime.date.today()
    """
    # With timezone
    if tz is not None:
        return date_fr_dt(datetime.datetime_from_timestamp(unix_time(), tz))

    # Without timezone
    cdef tm _tm = tm_localtime(unix_time())
    return datetime.date_new(_tm.tm_year, _tm.tm_mon, _tm.tm_mday)

# . type check
cdef inline bint is_date(object obj) except -1:
    """Check if an object is an instance of datetime.date `<'bool'>`.
    
    Equivalent to:
    >>> isinstance(obj, datetime.date)
    """
    return datetime.PyDate_Check(obj)

cdef inline bint is_date_exact(object obj) except -1:
    """Check if an object is the exact datetime.date type `<'bool'>`.
    
    Equivalent to:
    >>> type(obj) is datetime.date
    """
    return datetime.PyDate_CheckExact(obj)

# . conversion: to
cdef inline tm date_to_tm(datetime.date date) except *:
    """Convert date to `<'struct:tm'>`.
    
    #### All time fields are set to 0.
    """
    cdef:
        int yy = date.year
        int mm = date.month
        int dd = date.day
    return tm(
        0, 0, 0, # sec, min, hour
        dd, mm, yy,  # day, mon, year
        ymd_weekday(yy, mm, dd),  # wday
        days_bf_month(yy, mm) + dd,  # yday
        -1  # isdst
    )

cdef inline str date_to_strformat(datetime.date date, str fmt):
    """Convert date to string according to the given format `<'str'>`.

    Equivalent to:
    >>> date.strftime(fmt)
    """
    return tm_strftime(date_to_tm(date), fmt)

cdef inline str date_to_isoformat(datetime.date date):
    """Convert date to string in ISO format ('%Y-%m-%d') `<'str'>`."""
    return "%04d-%02d-%02d" % (date.year, date.month, date.day)

cdef inline long long date_to_us(datetime.date date):
    """Convert date to `EPOCH` microseconds `<'int'>`."""
    cdef long long ordinal = date_to_ordinal(date)
    return (ordinal - EPOCH_DAY) * US_DAY

cdef inline double date_to_seconds(datetime.date date):
    """Convert date to `EPOCH` seconds `<'float'>`."""
    cdef long long ordinal = date_to_ordinal(date)
    return (ordinal - EPOCH_DAY) * SS_DAY

cdef inline int date_to_ordinal(datetime.date date) except -1:
    """Convert date to Gregorian ordinal days `<'int'>`."""
    return ymd_to_ordinal(date.year, date.month, date.day)

cdef inline double date_to_ts(datetime.date date):
    """Convert date to `EPOCH` timestamp `<'float'>`."""
    cdef:
        long long ordinal = date_to_ordinal(date)
        long long ss = (ordinal - EPOCH_DAY) * SS_DAY
    return <double> ss * 2 - ts_localtime(ss)

# . conversion: from
cdef inline datetime.date date_fr_us(long long val):
    """Create date from `EPOCH` microseconds (int) `<'datetime.date'>`."""
    return date_fr_ordinal(math_floor_div(val + EPOCH_MICROSECOND, US_DAY))

cdef inline datetime.date date_fr_seconds(double val):
    """Create date from `EPOCH` seconds (float) `<'datetime.date'>`."""
    cdef long long sec = int(val)
    return date_fr_ordinal(sec // 86_400 + EPOCH_DAY)

cdef inline datetime.date date_fr_ordinal(int val):
    """Create date from Gregorian ordinal days `<'datetime.date'>`."""
    cdef ymd _ymd = ymd_fr_ordinal(val)
    return datetime.date_new(_ymd.year, _ymd.month, _ymd.day)

cdef inline datetime.date date_fr_ts(double val):
    """Create date from `EPOCH` timestamp (float) `<'datetime.date'>`."""
    return datetime.date_from_timestamp(val)

cdef inline datetime.date date_fr_date(datetime.date date):
    """Create date from another date (include subclass) `<'datetime.date'>`."""
    return datetime.date_new(date.year, date.month, date.day)

cdef inline datetime.date date_fr_dt(datetime.datetime dt):
    """Create date from datetime (include subclass) `<'datetime.date'>`."""
    return datetime.date_new(dt.year, dt.month, dt.day)

# datetime.datetime ------------------------------------------------------------------------------------
# . generate
cdef inline datetime.datetime dt_new(
    int year=1, int month=1, int day=1,
    int hour=0, int minute=0, int second=0,
    int microsecond=0, object tz=None, int fold=0,
):
    """Create a new datetime `<'datetime.datetime'>`.
    
    Equivalent to:
    >>> datetime.datetime(year, month, day, hour, minute, second, microsecond, tz, fold)
    """
    fold = 1 if fold == 1 else 0
    # Clamp year
    if year < 1:
        return datetime.datetime_new(1, 1, 1, 0, 0, 0, 0, tz, fold)
    if year > 9_999:
        return datetime.datetime_new(9_999, 12, 31, 23, 59, 59, 999999, tz, fold)
    
    # Clamp month & day
    month = min(max(month, 1), 12)
    if day > 28:
        day = min(day, days_in_month(year, month))
    elif day < 1:
        day = 1

    # Clamp h/m/s/f
    hour = min(max(hour, 0), 23)
    minute = min(max(minute, 0), 59)
    second = min(max(second, 0), 59)
    microsecond = min(max(microsecond, 0), 999_999)

    # New datetime
    return datetime.datetime_new(year, month, day, hour, minute, second, microsecond, tz, fold)

cdef inline datetime.datetime dt_now(object tz=None):
    """Get the current datetime `<'datetime.datetime'>`.
    
    Equivalent to:
    >>> datetime.datetime.now(tz)
    """
    # With timezone
    if tz is not None:
        return datetime.datetime_from_timestamp(unix_time(), tz)

    # Without timezone
    cdef: 
        double ts = unix_time()
        long long us = int(ts * 1_000_000)
        tm _tm = tm_localtime(ts)
    return datetime.datetime_new(
        _tm.tm_year, _tm.tm_mon, _tm.tm_mday, 
        _tm.tm_hour, _tm.tm_min, _tm.tm_sec, 
        us % 1_000_000, None, 0,
    )

# . type check
cdef inline bint is_dt(object obj) except -1:
    """Check if an object is an instance of datetime.datetime `<'bool'>`.

    Equivalent to:
    >>> isinstance(obj, datetime.datetime)
    """
    return datetime.PyDateTime_Check(obj)

cdef inline bint is_dt_exact(object obj) except -1:
    """Check if an object is the exact datetime.datetime type `<'bool'>`.

    Equivalent to:
    >>> type(obj) is datetime.datetime
    """
    return datetime.PyDateTime_CheckExact(obj)

# . tzinfo
cdef inline str dt_tzname(datetime.datetime dt):
    """Get the tzinfo 'tzname' of the datetime `<'str/None'>`.
    
    Equivalent to:
    >>> dt.tzname()
    """
    return tz_name(dt.tzinfo, dt)

cdef inline datetime.timedelta dt_dst(datetime.datetime dt):
    """Get the tzinfo 'dst' of the datetime `<'datetime.timedelta/None'>`.
    
    Equivalent to:
    >>> dt.dst()
    """
    return tz_dst(dt.tzinfo, dt)

cdef inline datetime.timedelta dt_utcoffset(datetime.datetime dt):
    """Get the tzinfo 'utcoffset' of the datetime `<'datetime.timedelta/None'>`.
    
    Equivalent to:
    >>> dt.utcoffset()
    """
    return tz_utcoffset(dt.tzinfo, dt)

cdef inline datetime.datetime dt_normalize_tz(datetime.datetime dt):
    """Normalize the datetime to its tzinfo `<'datetime.datetime'>`.
    
    This function is designed to handle ambiguous 
    datetime by normalizing it to its timezone.
    """
    tz = dt.tzinfo
    if tz is None:
        return dt

    cdef int off0, off1
    if dt.fold == 1:
        off0 = tz_utcoffset_seconds(tz, dt_replace_fold(dt, 0))
        off1 = tz_utcoffset_seconds(tz, dt)
    else:
        off0 = tz_utcoffset_seconds(tz, dt)
        off1 = tz_utcoffset_seconds(tz, dt_replace_fold(dt, 1))

    if off0 == off1:
        return dt

    if off1 > off0:
        return dt_add(dt, 0, off1 - off0, 0)

    raise ValueError("datetime '%s' is ambiguous." % dt)

# . conversion: to
cdef inline tm dt_to_tm(datetime.datetime dt, bint utc=False) except *:
    """Convert datetime to `<'struct:tm'>`.
    
    If 'dt' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from the result.
    """
    cdef:
        object tz = dt.tzinfo
        int yy, mm, dd, isdst
    
    # No timezone
    if tz is None:
        isdst = 0 if utc else -1
        yy, mm, dd = dt.year, dt.month, dt.day

    # Has timezone
    else:
        if utc:
            utc_off = tz_utcoffset(tz, dt)
            if utc_off is not None:
                dt = dt_add(
                    dt,
                    -datetime.timedelta_days(utc_off),
                    -datetime.timedelta_seconds(utc_off),
                    -datetime.timedelta_microseconds(utc_off),
                )
            isdst = 0
        else:
            dst_off = tz_dst(tz, dt)
            if dst_off is not None:
                isdst = 1 if dst_off else 0
            else:
                isdst = -1
        yy, mm, dd = dt.year, dt.month, dt.day

    # New 'struct:tm'
    return tm(
        dt.second, dt.minute, dt.hour,  # sec, min, hour
        dd, mm, yy,  # day, mon, year
        ymd_weekday(yy, mm, dd),  # wday
        days_bf_month(yy, mm) + dd,  # yday
        isdst,  # isdst
    )

cdef inline str dt_to_ctime(datetime.datetime dt):
    """Convert datetime to string in C time format `<'str'>`.
    
    - ctime format: 'Tue Oct  1 08:19:05 2024'
    """
    cdef int yy, mm, dd, wkd
    cdef str weekday, month
    
    # Weekday
    yy, mm, dd = dt.year, dt.month, dt.day
    wkd = ymd_weekday(yy, mm, dd)
    if wkd == 0:
        weekday = "Mon"
    elif wkd == 1:
        weekday = "Tue"
    elif wkd == 2:
        weekday = "Wed"
    elif wkd == 3:
        weekday = "Thu"
    elif wkd == 4:
        weekday = "Fri"
    elif wkd == 5:
        weekday = "Sat"
    else:
        weekday = "Sun"

    # Month
    if mm == 1:
        month = "Jan"
    elif mm == 2:
        month = "Feb"
    elif mm == 3:
        month = "Mar"
    elif mm == 4:
        month = "Apr"
    elif mm == 5:
        month = "May"
    elif mm == 6:
        month = "Jun"
    elif mm == 7:
        month = "Jul"
    elif mm == 8:
        month = "Aug"
    elif mm == 9:
        month = "Sep"
    elif mm == 10:
        month = "Oct"
    elif mm == 11:
        month = "Nov"
    else:
        month = "Dec"

    # Fromat
    return "%s %s %2d %02d:%02d:%02d %04d" % (
        weekday, month, dd, dt.hour, dt.minute, dt.second, yy
    )

cdef inline str dt_to_strformat(datetime.datetime dt, str fmt):
    """Convert datetime to string according to the given format `<'str'>`.
    
    Equivalent to:
    >>> dt.strftime(fmt)
    """
    cdef:
        list fmt_l = []
        Py_ssize_t size, idx
        str frepl, zrepl, Zrepl
        Py_UCS4 ch

    # Escape format
    size, idx = str_len(fmt), 0
    frepl, zrepl, Zrepl = None, None, None
    while idx < size:
        ch = str_read(fmt, idx)
        idx += 1
        # Format identifier: '%'
        if ch == "%":
            if idx < size:
                ch = str_read(fmt, idx)
                idx += 1
                # . fraction replacement
                if ch == "f":
                    if frepl is None:
                        frepl = "%06d" % dt.microsecond
                    fmt_l.append(frepl)
                # . utc offset replacement
                elif ch == "z":
                    if zrepl is None:
                        utcfmt = tz_utcformat(dt.tzinfo, dt)
                        zrepl = "" if utcfmt is None else utcfmt
                    fmt_l.append(zrepl)
                # . timezone name replacement
                elif ch == "Z":
                    if Zrepl is None:
                        tzname = dt_tzname(dt)
                        Zrepl = "" if tzname is None else tzname
                    fmt_l.append(Zrepl)
                # . not applicable
                else:
                    fmt_l.append("%")
                    fmt_l.append(str_chr(ch))
            # eof
            else:
                fmt_l.append("%")
        # Normal character
        else:
            fmt_l.append(str_chr(ch))

    # Format to string
    return tm_strftime(dt_to_tm(dt, False), "".join(fmt_l))

cdef inline str dt_to_isoformat(datetime.datetime dt, str sep="T", bint utc=False):
    """Convert datetime to string in ISO format `<'str'>`.

    If 'dt' is timezone-aware, setting 'utc=True'
    adds the UTC at the end of the ISO format.
    """
    cdef:
        int us = dt.microsecond
        str utc_fmt = tz_utcformat(dt.tzinfo, dt) if utc else None

    if us == 0:
        if utc_fmt is None:
            return "%04d-%02d-%02d%s%02d:%02d:%02d" % (
                dt.year, dt.month, dt.day, sep,
                dt.hour, dt.minute, dt.second,
            )
        return "%04d-%02d-%02d%s%02d:%02d:%02d%s" % (
            dt.year, dt.month, dt.day, sep,
            dt.hour, dt.minute, dt.second, utc_fmt,
        )
    else:
        if utc_fmt is None:
            return "%04d-%02d-%02d%s%02d:%02d:%02d.%06d" % (
                dt.year, dt.month, dt.day, sep,
                dt.hour, dt.minute, dt.second, us,
            )
        return "%04d-%02d-%02d%s%02d:%02d:%02d.%06d%s" % (
            dt.year, dt.month, dt.day, sep,
            dt.hour, dt.minute, dt.second, us, utc_fmt,
        )    

cdef inline long long dt_to_us(datetime.datetime dt, bint utc=False):
    """Convert datetime to `EPOCH` microseconds `<'int'>`.

    If 'dt' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from total mircroseconds.
    """
    cdef:
        long long ordinal = dt_to_ordinal(dt, False)
        long long hh = dt.hour
        long long mi = dt.minute
        long long ss = dt.second
        long long us = dt.microsecond
        
    us += (
        (ordinal - EPOCH_DAY) * US_DAY
        + hh * US_HOUR
        + mi * US_MINUTE
        + ss * US_SECOND
    )
    if utc:
        utc_off = dt_utcoffset(dt)
        if utc_off is not None:
            us -= td_to_us(utc_off)
    return us

cdef inline double dt_to_seconds(datetime.datetime dt, bint utc=False):
    """Convert datetime to `EPOCH` seconds `<'float'>`.
    
    If 'dt' is timezone-aware, setting 'utc=True' 
    substracts 'utcoffset' from total seconds.
    """
    cdef:
        double ordinal = dt_to_ordinal(dt, False)
        double hh = dt.hour
        double mi = dt.minute
        double ss = dt.second
        double us = dt.microsecond

    ss += (
        (ordinal - EPOCH_DAY) * SS_DAY
        + hh * SS_HOUR
        + mi * SS_MINUTE
        + us / 1_000_000
    )
    if utc:
        utc_off = dt_utcoffset(dt)
        if utc_off is not None:
            ss -= td_to_seconds(utc_off)
    return ss

cdef inline int dt_to_ordinal(datetime.datetime dt, bint utc=False) except -1:
    """Convert datetime to Gregorian ordinal days `<'int'>`.
    
    If 'dt' is timezone-aware, setting 'utc=True' 
    substracts 'utcoffset' from total days.
    """
    cdef int ordinal, seconds
    ordinal = ymd_to_ordinal(dt.year, dt.month, dt.day)

    if utc:
        utc_off = dt_utcoffset(dt)
        if utc_off is not None:
            seconds = dt.hour * SS_HOUR + dt.minute * SS_MINUTE + dt.second
            seconds -= (
                datetime.timedelta_days(utc_off) * SS_DAY
                + datetime.timedelta_seconds(utc_off)
            )
            # UTC offset move 1 day backward
            if seconds < 0:
                ordinal -= 1
            # UTC offset move 1 day forward
            elif seconds >= SS_DAY:
                ordinal += 1
    return ordinal

cdef inline long long dt_to_posix(datetime.datetime dt):
    """Convert datetime to POSIX Time (seconds) `<'int'>`."""
    # Compute 'EPOCH' seconds
    cdef long long ordinal, hh, mi, ss, t
    ordinal = dt_to_ordinal(dt, False)
    hh, mi, ss = dt.hour, dt.minute, dt.second
    t = (ordinal - EPOCH_DAY) * SS_DAY + hh * SS_HOUR + mi * SS_MINUTE + ss
    
    # Adjustment for Daylight Saving
    cdef long long adj1, adj2, u1, u2, t1, t2
    adj1 = ts_localtime(t) - t
    u1 = t - adj1
    t1 = ts_localtime(u1)
    if t == t1:
        # We found one solution, but it may not be the one we need.
        # Look for an earlier solution (if `fold` is 0), or a later
        # one (if `fold` is 1).
        u2 = u1 - SS_DAY if dt.fold == 0 else u1 + SS_DAY
        adj2 = ts_localtime(u2) - u2
        if adj1 == adj2:
            return u1
    else:
        adj2 = t1 - u1
    
    # Final adjustment
    u2 = t - adj2
    t2 = ts_localtime(u2)
    if t == t2:
        return u2
    if t == t1:
        return u1
    # We have found both offsets adj1 and adj2,
    # but neither t - adj1 nor t - adj2 is the
    # solution. This means t is in the gap.
    return max(u1, u2) if dt.fold == 0 else min(u1, u2)

cdef inline double dt_to_ts(datetime.datetime dt):
    """Convert datetime to `EPOCH` timestamp `<'float'>`.
    
    Equivalent to:
    >>> dt.timestamp()
    """
    # With timezone
    cdef double ts
    utc_off = dt_utcoffset(dt)
    if utc_off is not None:
        ts = dt_to_seconds(dt, False)
        ts -= td_to_seconds(utc_off)
        return ts

    # Without timezone
    ts = <double> dt_to_posix(dt)
    return ts + (<double> dt.microsecond) / 1_000_000

cdef inline long long dt_as_epoch(datetime.datetime dt, str unit, bint utc=False):
    """Convert datetime to `EPOCH` integer according to the given unit resolution `<'int'>`.

    Supported units: 'Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us'.

    If 'dt' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from the result.
    """
    cdef:
        Py_ssize_t size = str_len(unit)
        long long ordinal, hh, mi, ss, us
        Py_UCS4 ch

    # Adjustment for UTC
    if utc:
        utc_off = tz_utcoffset(dt.tzinfo, dt)
        if utc_off is not None:
            dt = dt_add(
                dt,
                -datetime.timedelta_days(utc_off),
                -datetime.timedelta_seconds(utc_off),
                -datetime.timedelta_microseconds(utc_off),
            )

    # Unit: 's', 'm', 'h', 'D', 'W', 'M', 'Q', 'Y'
    if size == 1:
        ch = str_read(unit, 0)
        # . second
        if ch == "s":
            ordinal = dt_to_ordinal(dt, False)
            hh, mi, ss = dt.hour, dt.minute, dt.second
            return (ordinal - EPOCH_DAY) * SS_DAY + hh * SS_HOUR + mi * SS_MINUTE + ss
        # . minute
        if ch == "m":
            ordinal = dt_to_ordinal(dt, False)
            hh, mi = dt.hour, dt.minute
            return (ordinal - EPOCH_DAY) * 1_440 + hh * 60 + mi
        # . hour
        if ch == "h":
            ordinal = dt_to_ordinal(dt, False)
            return (ordinal - EPOCH_DAY) * 24 + dt.hour
        # . day
        if ch == "D":
            return dt_to_ordinal(dt, False) - EPOCH_DAY
        # . week
        if ch == "W":
            ordinal = dt_to_ordinal(dt, False)
            return (ordinal - EPOCH_DAY) // 7
        # . month
        if ch == "M":
            return (dt.year - EPOCH_YEAR) * 12 + dt.month - 1
        # . quarter
        if ch == "Q":
            return (dt.year - EPOCH_YEAR) * 4 + quarter_of_month(dt.month) - 1
        # . year
        if ch == "Y":
            return dt.year - EPOCH_YEAR

    # Unit: 'ns', 'us', 'ms'
    elif size == 2 and str_read(unit, 1) == "s":
        ch = str_read(unit, 0)
        # . nanosecond
        if ch == "n":
            ordinal = dt_to_ordinal(dt, False)
            hh, mi, ss, us = dt.hour, dt.minute, dt.second, dt.microsecond
            return (
                (ordinal - EPOCH_DAY) * NS_DAY
                + hh * NS_HOUR
                + mi * NS_MINUTE
                + ss * NS_SECOND
                + us * NS_MICROSECOND
            )
        # . microsecond
        if ch == "u":
            ordinal = dt_to_ordinal(dt, False)
            hh, mi, ss, us = dt.hour, dt.minute, dt.second, dt.microsecond
            return (
                (ordinal - EPOCH_DAY) * US_DAY
                + hh * US_HOUR
                + mi * US_MINUTE
                + ss * US_SECOND
                + us
            )
        # . millisecond
        if ch == "m":
            ordinal = dt_to_ordinal(dt, False)
            hh, mi, ss, us = dt.hour, dt.minute, dt.second, dt.microsecond
            return (
                (ordinal - EPOCH_DAY) * MS_DAY
                + hh * MS_HOUR
                + mi * MS_MINUTE
                + ss * MS_SECOND
                + us // 1_000
            )

    # Unit: 'min' for pandas compatibility
    elif size == 3 and unit == "min":
        ordinal = dt_to_ordinal(dt, False)
        return (ordinal - EPOCH_DAY) * 1_440 + dt.hour * 60 + dt.minute

    # Unsupported unit
    raise ValueError("unsupported datetime unit '%s'." % unit)

cdef inline long long dt_as_epoch_iso_W(datetime.datetime dt, int weekday, bint utc=False):
    """Convert datetime to `EPOCH` integer under 'W' (weeks) resolution `<'int'>`.

    Different from 'dt_as_epoch(dt, "W")', which aligns the weekday
    to Thursday (the weekday of 1970-01-01). This function allows
    specifying the ISO 'weekday' (1=Monday, 7=Sunday) for alignment.

    For example: if 'weekday=1', the result represents the Monday-aligned
    weeks since EPOCH (1970-01-01).

    If 'dt' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from the result.
    """
    cdef int ordinal = dt_to_ordinal(dt, utc)
    cdef int wkd_off = 4 - min(max(weekday, 1), 7)
    return (ordinal - EPOCH_DAY + wkd_off) // 7

# . conversion: from
cdef inline datetime.datetime dt_fr_us(long long val, object tz=None):
    """Create datetime from `EPOCH` microseconds (int) `<'datetime.datetime'>`."""
    # Compute ymd & hmsf
    val += EPOCH_MICROSECOND
    cdef: 
        ymd _ymd = ymd_fr_ordinal(math_floor_div(val, US_DAY))
        hmsf _hmsf = hmsf_fr_us(val)

    # New datetime
    return datetime.datetime_new(
        _ymd.year, _ymd.month, _ymd.day, 
        _hmsf.hour, _hmsf.minute, _hmsf.second, 
        _hmsf.microsecond, tz, 0
    )

cdef inline datetime.datetime dt_fr_seconds(double val, object tz=None):
    """Create datetime from `EPOCH` seconds (float) `<'datetime.datetime'>`."""
    return dt_fr_us(<long long> int(val * US_SECOND), tz)

cdef inline datetime.datetime dt_fr_ordinal(int val, object tz=None):
    """Create datetime from Gregorian ordinal days (int) `<'datetime.datetime'>`.
    
    Equivalent to:
    >>> datetime.datetime.fromordinal(val).replace(tzinfo=tz)
    """
    cdef ymd _ymd = ymd_fr_ordinal(val)
    return datetime.datetime_new(_ymd.year, _ymd.month, _ymd.day, 0, 0, 0, 0, tz, 0)

cdef inline datetime.datetime dt_fr_ts(double val, object tz=None):
    """Create datetime from `EPOCH` timestamp (float) `<'datetime.datetime'>`.
    
    Equivalent to:
    >>> datetime.datetime.fromtimestamp(val, tz)
    """
    return datetime.datetime_from_timestamp(val, tz)

cdef inline datetime.datetime dt_combine(datetime.date date=None, datetime.time time=None, tz: object = None):
    """Create datetime by combining date & time `<'datetime.datetime'>`.

    - If 'date' is None, use current local date.
    - If 'time' is None, all time fields are set to 0.
    """
    # Date
    cdef tm _tm
    cdef int yy, mm, dd
    if date is None:
        _tm = tm_localtime(unix_time())
        yy, mm, dd = _tm.tm_year, _tm.tm_mon, _tm.tm_mday
    else:
        yy, mm, dd = date.year, date.month, date.day

    # Time
    cdef int hh, mi, ss, us, fold
    if time is None:
        hh, mi, ss, us, fold = 0, 0, 0, 0, 0
    else:
        hh, mi, ss, us, fold = time.hour, time.minute, time.second, time.microsecond, time.fold
        if tz is None:
            tz = time.tzinfo

    # Combine
    return datetime.datetime_new(yy, mm, dd, hh, mi, ss, us, tz, fold)

cdef inline datetime.datetime dt_fr_date(datetime.date date, object tz=None):
    """Create datetime from date (include subclass) `<'datetime.datetime'>`.
    
    #### All time fields are set to 0.
    """
    return datetime.datetime_new(date.year, date.month, date.day, 0, 0, 0, 0, tz, 0)

cdef inline datetime.datetime dt_fr_time(datetime.time time):
    """Create datetime from time (include subclass) `<'datetime.datetime'>`.
    
    #### Date fields are set to 1970-01-01.
    """
    return datetime.datetime_new(
        1970, 1, 1, 
        time.hour, time.minute, time.second, 
        time.microsecond, time.tzinfo, time.fold
    )

cdef inline datetime.datetime dt_fr_dt(datetime.datetime dt):
    """Create datetime from another datetime (include subclass) `<'datetime.datetime'>`."""
    return datetime.datetime_new(
        dt.year, dt.month, dt.day, 
        dt.hour, dt.minute, dt.second, 
        dt.microsecond, dt.tzinfo, dt.fold
    )

# . manipulation
cdef inline datetime.datetime dt_add(datetime.datetime dt, int days=0, int seconds=0, int microseconds=0):
    """Add delta to datetime `<'datetime.datetime'>`.
    
    Equivalent to:
    >>> dt + datetime.timedelta(days, seconds, microseconds)
    """
    cdef: 
        long long dd, hh, mi, ss, us
        ymd _ymd
        hmsf _hmsf
    # Fast-path
    if seconds == 0 and microseconds == 0:
        # No adjustment
        if days == 0:
            return dt  # exit

        # Add 'day' delta
        _ymd = ymd_fr_ordinal(dt_to_ordinal(dt, False) + days)
        hh, mi, ss, us = dt.hour, dt.minute, dt.second, dt.microsecond

    # Compute full delta
    else:
        # Add 'us' delta
        dd, ss, us = days, seconds, microseconds
        us += dd * US_DAY + ss * US_SECOND
        us += dt_to_us(dt, False) + EPOCH_MICROSECOND
        _ymd = ymd_fr_ordinal(math_floor_div(us, US_DAY))
        _hmsf = hmsf_fr_us(us)
        hh, mi, ss, us = _hmsf.hour, _hmsf.minute, _hmsf.second, _hmsf.microsecond

    # New datetime
    return datetime.datetime_new(
        _ymd.year, _ymd.month, _ymd.day,
        hh, mi, ss, us, dt.tzinfo, dt.fold,
    )

cdef inline datetime.datetime dt_replace_tz(datetime.datetime dt, object tz):
    """Replace the datetime timezone `<'datetime.datetime'>`.

    Equivalent to:
    >>> dt.replace(tzinfo=tz)
    """
    # Same tzinfo
    if tz is datetime.datetime_tzinfo(dt):
        return dt

    # New datetime
    return datetime.datetime_new(
        dt.year, dt.month, dt.day, 
        dt.hour, dt.minute, dt.second, 
        dt.microsecond, tz, dt.fold,
    )

cdef inline datetime.datetime dt_replace_fold(datetime.datetime dt, int fold):
    """Replace the datetime fold `<'datetime.datetime'>`.

    Equivalent to:
    >>> dt.replace(fold=fold)
    """
    # Same fold
    if fold not in (0, 1) or fold == dt.fold:
        return dt

    # New datetime
    return datetime.datetime_new(
        dt.year, dt.month, dt.day, 
        dt.hour, dt.minute, dt.second, 
        dt.microsecond, dt.tzinfo, fold,
    )

cdef inline datetime.datetime dt_astimezone(datetime.datetime dt, object tz=None):
    """Convert the datetime timezone `<'datetime.datetime'>`.
    
    Equivalent to:
    >>> dt.astimezone(tz)
    """
    mytz = datetime.datetime_tzinfo(dt)
    if tz is None:
        tz = tz_local(dt)
        if mytz is None:
            # since 'dt' is timezone-naive, we 
            # simply localize to the local timezone.
            return dt_replace_tz(dt, tz)  # exit

    if mytz is None:
        mytz = tz_local(dt)
        if mytz is tz:
            # Since 'dt' is timezone-naive, we
            # simply localize to the target timezone.
            return dt_replace_tz(dt, tz)  # exit
        myoffset = tz_utcoffset(mytz, dt)
    elif mytz is tz:
        return dt  # exit
    else:
        myoffset = tz_utcoffset(mytz, dt)
        if myoffset is None:
            mytz = tz_local(dt_replace_tz(dt, None))
            if mytz is tz:
                return dt  # exit
            myoffset = tz_utcoffset(mytz, dt)

    # Convert to UTC time
    cdef long long us = dt_to_us(dt, False)
    us -= td_to_us(myoffset)
    dt = dt_fr_us(us, tz)

    # Convert from UTC to tz's local time
    return tz.fromutc(dt)

# datetime.time ----------------------------------------------------------------------------------------
# . generate
cdef inline datetime.time time_new(
    int hour=0, int minute=0, int second=0,
    int microsecond=0, object tz=None, int fold=0,
):
    """Create a new time `<'datetime.time'>`.
    
    Equivalent to:
    >>> datetime.time(hour, minute, second, microsecond, tz, fold)
    """
    # Clamp h/m/s/f
    hour = min(max(hour, 0), 23)
    minute = min(max(minute, 0), 59)
    second = min(max(second, 0), 59)
    microsecond = min(max(microsecond, 0), 999_999)

    # New time
    return datetime.time_new(
        hour, minute, second, microsecond,
        tz, 1 if fold == 1 else 0,
    )

cdef inline datetime.time time_now(object tz=None):
    """Get the current time `<'datetime.time'>`.
    
    Equivalent to:
    >>> datetime.datetime.now(tz).time()
    """
    # With timezone
    if tz is not None:
        return time_fr_dt(datetime.datetime_from_timestamp(unix_time(), tz))

    # Without timezone
    cdef: 
        double ts = unix_time()
        long long us = int(ts * 1_000_000)
        tm _tm = tm_localtime(ts)
    return datetime.time_new(
        _tm.tm_hour, _tm.tm_min, _tm.tm_sec, us % 1_000_000, None, 0
    )

# . type check
cdef inline bint is_time(object obj) except -1:
    """Check if an object is an instance of datetime.time `<'bool'>`.
    
    Equivalent to:
    >>> isinstance(obj, datetime.time)
    """
    return datetime.PyTime_Check(obj)

cdef inline bint is_time_exact(object obj) except -1:
    """Check if an object is the exact datetime.time type `<'bool'>`.

    Equivalent to:
    >>> type(obj) is datetime.time
    """
    return datetime.PyTime_CheckExact(obj)

# . tzinfo
cdef inline str time_tzname(datetime.time time):
    """Get the tzinfo 'tzname' of the time `<'str/None'>`.
    
    Equivalent to:
    >>> time.tzname()
    """
    return tz_name(time.tzinfo, None)

cdef inline datetime.timedelta time_dst(datetime.time time):
    """Get the tzinfo 'dst' of the time `<'datetime.timedelta/None'>`.
    
    Equivalent to:
    >>> time.dst()
    """
    return tz_dst(time.tzinfo, None)

cdef inline datetime.timedelta time_utcoffset(datetime.time time):
    """Get the tzinfo 'utcoffset' of the time `<'datetime.timedelta/None'>`.
    
    Equivalent to:
    >>> time.utcoffset()
    """
    return tz_utcoffset(time.tzinfo, None)

# . conversion
cdef inline str time_to_isoformat(datetime.time time, bint utc=False):
    """Convert time to string in ISO format `<'str'>`.

    If 'time' is timezone-aware, setting 'utc=True' 
    adds the UTC at the end of the ISO format.
    """
    cdef:
        int us = time.microsecond
        str utc_fmt = tz_utcformat(time.tzinfo, None) if utc else None

    if us == 0:
        if utc_fmt is None:
            return "%02d:%02d:%02d" % (time.hour, time.minute, time.second)
        return "%02d:%02d:%02d%s" % (time.hour, time.minute, time.second, utc_fmt)
    else:
        if utc_fmt is None:
            return "%02d:%02d:%02d.%06d" % (time.hour, time.minute, time.second, us)
        return "%02d:%02d:%02d.%06d%s" % (time.hour, time.minute, time.second, us, utc_fmt)

cdef inline long long time_to_us(datetime.time time, bint utc=False):
    """Convert time to microseconds `<'int'>`.
    
    If 'time' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from total mircroseconds.
    """
    cdef:
        long long hh = time.hour
        long long mi = time.minute
        long long ss = time.second
        long long us = time.microsecond

    us += hh * US_HOUR + mi * US_MINUTE + ss * US_SECOND
    if utc:
        utc_off = time_utcoffset(time)
        if utc_off is not None:
            us -= td_to_us(utc_off)
    return us

cdef inline double time_to_seconds(datetime.time time, bint utc=False):
    """Convert time to seconds `<'float'>`.
    
    If 'time' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from total seconds.
    """
    cdef:
        double hh = time.hour
        double mi = time.minute
        double ss = time.second
        double us = time.microsecond
    
    ss += hh * SS_HOUR + mi * SS_MINUTE + us / 1_000_000
    if utc:
        utc_off = time_utcoffset(time)
        if utc_off is not None:
            ss -= td_to_seconds(utc_off)
    return ss

cdef inline datetime.time time_fr_us(long long val, object tz=None):
    """Create time from microseconds (int) `<'datetime.time'>`."""
    cdef hmsf _hmsf = hmsf_fr_us(val)
    return datetime.time_new(
        _hmsf.hour, _hmsf.minute, _hmsf.second, _hmsf.microsecond, tz, 0
    )

cdef inline datetime.time time_fr_seconds(double val, object tz=None):
    """Create time from seconds (float) `<'datetime.time'>`."""
    return time_fr_us(<long long> int(val * US_SECOND), tz)

cdef inline datetime.time time_fr_time(datetime.time time):
    """Create time from another time (include subclass) `<'datetime.time'>`."""
    return datetime.time_new(
        time.hour, time.minute, time.second, 
        time.microsecond, time.tzinfo, time.fold
    )

cdef inline datetime.time time_fr_dt(datetime.datetime dt):
    """Create time from datetime (include subclass) `<'datetime.time'>`."""
    return datetime.time_new(
        dt.hour, dt.minute, dt.second, 
        dt.microsecond, dt.tzinfo, dt.fold
    )

# datetime.timedelta -----------------------------------------------------------------------------------
# . generate
cdef inline datetime.timedelta td_new(int days=0, int seconds=0, int microseconds=0):
    """Create a new timedelta `<'datetime.timedelta'>`.
    
    Equivalent to:
    >>> datetime.timedelta(days, seconds, microseconds)
    """
    return datetime.timedelta_new(days, seconds, microseconds)

# . type check
cdef inline bint is_td(object obj) except -1:
    """Check if an object is an instance of datetime.timedelta `<'bool'>`.

    Equivalent to:
    >>> isinstance(obj, datetime.timedelta)
    """
    return datetime.PyDelta_Check(obj)

cdef inline bint is_td_exact(object obj) except -1:
    """Check if an object is the exact datetime.timedelta type `<'bool'>`.

    Equivalent to:
    >>> type(obj) is datetime.timedelta    
    """
    return datetime.PyDelta_CheckExact(obj)

# . conversion
cdef inline str td_to_isoformat(datetime.timedelta td):
    """Convert timedelta to string in ISO format `<'str'>`."""
    cdef:
        long long days = td.day
        long long hours, minutes
        long long seconds = days * SS_DAY + td.second  # total seconds (may be negative)
        long long us = td.microsecond             # microseconds (0..999999)
        bint neg

    # Determine sign and normalize to absolute value without multiplying by 1e6.
    if seconds < 0:
        neg = True
        if us == 0:
            # Pure second-resolution negative
            seconds = -seconds
        else:
            # Borrow 1 second: |-X sec, +us| =>
            # (|X|-1) seconds and (1_000_000 - us) microseconds
            seconds = -seconds - 1
            us = 1_000_000 - us
    else:
        neg = False

    # Split absolute seconds into H:M:S
    hours = math_floor_div(seconds, SS_HOUR)
    seconds -= hours * SS_HOUR
    minutes = math_floor_div(seconds, SS_MINUTE)
    seconds -= minutes * SS_MINUTE

    # Emit
    if us == 0:
        if not neg:
            return "%02d:%02d:%02d" % (hours, minutes, seconds)
        return "-%02d:%02d:%02d" % (hours, minutes, seconds)
    elif not neg:
        return "%02d:%02d:%02d.%06d" % (hours, minutes, seconds, us)
    else:
        return "-%02d:%02d:%02d.%06d" % (hours, minutes, seconds, us)

cdef inline str td_to_utcformat(datetime.timedelta td):
    """Convert timedelta to string in UTC format ('+/-HHMM') `<'str'>`."""
    cdef:
        long long days = td.day
        long long hours, minutes
        long long seconds = days * SS_DAY + td.second
        bint neg

    # Determine sign
    if seconds < 0:
        neg = True
        seconds = -seconds
    else:
        neg = False

    # Split absolute seconds into H:M
    hours = math_floor_div(seconds, SS_HOUR)
    seconds -= hours * SS_HOUR
    minutes = math_floor_div(seconds, SS_MINUTE)

    # Emit
    if neg:
        return "-%02d%02d" % (hours, minutes)
    else:
        return "+%02d%02d" % (hours, minutes)

cdef inline long long td_to_us(datetime.timedelta td):
    """Convert timedelta to microseconds `<'int'>`."""
    cdef:
        long long days = td.day
        long long seconds = td.second
        long long us = td.microsecond
    return days * US_DAY + seconds * US_SECOND + us

cdef inline double td_to_seconds(datetime.timedelta td):
    """Convert timedelta to seconds `<'float'>`."""
    cdef:
        double days = td.day
        double seconds = td.second
        double us = td.microsecond
    return days * SS_DAY + seconds + us / 1_000_000

cdef inline datetime.timedelta td_fr_us(long long val):
    """Create timedelta from microseconds (int) `<'datetime.timedelta'>`."""
    cdef long long days, seconds
    days = math_floor_div(val, US_DAY)
    val -= days * US_DAY
    seconds = math_floor_div(val, US_SECOND)
    val -= seconds * US_SECOND
    return datetime.timedelta_new(days, seconds, val)

cdef inline datetime.timedelta td_fr_seconds(double val):
    """Create timedelta from seconds (float) `<'datetime.timedelta'>`."""
    return td_fr_us(<long long> int(val * US_SECOND))

cdef inline datetime.timedelta td_fr_td(datetime.timedelta td):
    """Create timedelta from another timedelta (include subclass) `<'datetime.timedelta'>`."""
    return datetime.timedelta_new(td.day, td.second, td.microsecond)

# datetime.tzinfo --------------------------------------------------------------------------------------
# . generate
cdef inline object tz_new(int hours=0, int minites=0, int seconds=0):
    """Create a new timezone `<'datetime.timezone'>`.
    
    Equivalent to:
    >>> datetime.timezone(datetime.timedelta(hours=hours, minutes=minites))
    """
    cdef long long offset = hours * 3_600 + minites * 60 + seconds
    if not -86_340 <= offset <= 86_340:
        raise ValueError(
            "timezone offset '%s' (seconds) out of range, "
            "must between -86340 and 86340." % offset
        )
    return datetime.timezone_new(datetime.timedelta_new(0, offset, 0), None)

cdef inline object tz_local(datetime.datetime dt=None):
    """Get the local timezone `<'datetime.timezone'>`."""
    return _LOCAL_TZ

cdef inline int tz_local_seconds(datetime.datetime dt=None) except -200_000:
    """Get the local timezone offset in total seconds `<'int'>`."""
    # Convert to timestamp
    cdef:
        double ts
        long long ts1, ts2
        tm loc, gmt

    if dt is not None:
        if dt.tzinfo is None:
            ts1 = dt_to_posix(dt)
            # . detect gap
            ts2 = dt_to_posix(dt_replace_fold(dt, 1-dt.fold))
            if ts2 != ts1 and (ts2 > ts1) == dt.fold:
                ts = <double> ts2
            else:
                ts = <double> ts1
        else:
            ts = dt_to_seconds(dt, True)
    else:
        ts = unix_time()

    # Compute offset
    loc, gmt = tm_localtime(ts), tm_gmtime(ts)
    return (
        (loc.tm_mday - gmt.tm_mday) * SS_DAY
        + (loc.tm_hour - gmt.tm_hour) * SS_HOUR
        + (loc.tm_min - gmt.tm_min) * SS_MINUTE
        + (loc.tm_sec - gmt.tm_sec)
    )

cdef object tz_parse(object tz)

# . type check
cdef inline bint is_tz(object obj) except -1:
    """Check if an object is an instance of datetime.tzinfo `<'bool'>`.
    
    Equivalent to:
    >>> isinstance(obj, datetime.tzinfo)
    """
    return datetime.PyTZInfo_Check(obj)

cdef inline bint is_tz_exact(object obj) except -1:
    """Check if an object is the exact datetime.tzinfo type `<'bool'>`.

    Equivalent to:
    >>> type(obj) is datetime.date
    """
    return datetime.PyTZInfo_CheckExact(obj)

# . access
cdef inline str tz_name(object tz, datetime.datetime dt=None):
    """Access the name of the tzinfo `<'str/None'>`.
    
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
                "expects instance of 'datetime.tzinfo', "
                "instead got %s." % type(tz)
            ) from err
        raise err

cdef inline datetime.timedelta tz_dst(object tz, datetime.datetime dt=None):
    """Access the 'dst' of the tzinfo `<'datetime.timedelta'>`.

    Equivalent to:
    >>> tz.dst(dt)
    """
    if tz is None:
        return None
    try:
        return tz.dst(dt)
    except Exception as err:
        if not is_tz(tz):
            raise TypeError(
                "expects instance of 'datetime.tzinfo', "
                "instead got %s." % type(tz)
            ) from err
        raise err

cdef inline datetime.timedelta tz_utcoffset(object tz, datetime.datetime dt=None):
    """Access the 'utcoffset' of the tzinfo `<'datetime.timedelta'>`.

    Equivalent to:
    >>> tz.utcoffset(dt)
    """
    if tz is None:
        return None
    try:
        return tz.utcoffset(dt)
    except Exception as err:
        if not is_tz(tz):
            raise TypeError(
                "expects instance of 'datetime.tzinfo', "
                "instead got %s." % type(tz)
            ) from err
        raise err

cdef inline int tz_utcoffset_seconds(object tz, datetime.datetime dt=None) except -200_000:
    """Access the 'utcoffset' of the tzinfo in total seconds `<'int'>`.

    #### Returns `-100_000` if utcoffset is None.

    Equivalent to:
    >>> tz.utcoffset(dt).total_seconds()
    """
    offset = tz_utcoffset(tz, dt)
    if offset is None:
        return -100_000
    return datetime.timedelta_days(offset) * SS_DAY + datetime.timedelta_seconds(offset)

cdef inline str tz_utcformat(object tz, datetime.datetime dt=None):
    """Access tzinfo as string in UTC format ('+/-HHMM') `<'str/None'>`."""
    offset = tz_utcoffset(tz, dt)
    if offset is None:
        return None
    return td_to_utcformat(offset)

# NumPy: share -----------------------------------------------------------------------------------------
# . time unit
cdef inline str map_nptime_unit_int2str(int unit):
    """Map numpy datetime64/timedelta64 unit from integer
    to the corresponding string representation `<'str'>`."""
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
    # if unit == np.NPY_DATETIMEUNIT.NPY_FR_B:
    #     return "B"

    # Unsupported unit
    raise ValueError("unknown datetime unit '%d'." % unit)

cdef inline int map_nptime_unit_str2int(str unit) except -1:
    """Map numpy datetime64/timedelta64 unit from string
    representation to the corresponding integer `<'int'>`."""
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
    raise ValueError("unknown datetime unit '%s'." % unit)

cdef inline int get_arr_nptime_unit(np.ndarray arr) except -1:
    """Get ndarray[datetime64/timedelta64] unit from the,
    returns the unit in `<'int'>`."""
    cdef int dtype = np.PyArray_TYPE(arr)
    if dtype not in (np.NPY_TYPES.NPY_DATETIME, np.NPY_TYPES.NPY_TIMEDELTA):
        raise TypeError(
            "expects instance of 'np.ndarray[datetime64/timedelta64]', "
            "instead got '%s'." % arr.dtype
        )

    cdef int ndim = arr.ndim
    cdef np.NPY_DATETIMEUNIT unit
    if ndim == 1:
        if arr.shape[0] == 0:
            return parse_arr_nptime_unit(arr)
        return np.get_datetime64_unit(arr[0])
    if ndim == 2:
        if arr.shape[1] == 0:
            return parse_arr_nptime_unit(arr)
        return np.get_datetime64_unit(arr[0, 0])
    if ndim == 3:
        if arr.shape[2] == 0:
            return parse_arr_nptime_unit(arr)
        return np.get_datetime64_unit(arr[0, 0, 0])
    if ndim == 4:
        if arr.shape[3] == 0:
            return parse_arr_nptime_unit(arr)
        return np.get_datetime64_unit(arr[0, 0, 0, 0])
    # Invalid
    raise ValueError("support <'ndarray'> with 1-4 dimensions, got ndim '%d'." % ndim)

cdef inline int parse_arr_nptime_unit(np.ndarray arr) except -1:
    """Parse ndarray[datetime64/timedelta64] unit from the,
    returns the unit in `<'int'>`."""
    # Validate
    if np.PyArray_TYPE(arr) not in (
        np.NPY_TYPES.NPY_DATETIME,
        np.NPY_TYPES.NPY_TIMEDELTA,
    ):
        raise TypeError(
            "expects instance of 'np.ndarray[datetime64/timedelta64]', "
            "instead got '%s'." % arr.dtype
        )

    # Parse
    cdef str dtype_str = arr.dtype.str
    cdef Py_ssize_t size = str_len(dtype_str)
    if size < 6:
        raise ValueError("unable to parse datetime unit from '%s'." % dtype_str)
    try:
        return map_nptime_unit_str2int(str_substr(dtype_str, 4, size - 1))
    except ValueError as err:
        raise ValueError("unable to parse datetime unit from '%s'." % dtype_str) from err

# NumPy: datetime64 ------------------------------------------------------------------------------------
# . type check
cdef inline bint is_dt64(object obj) except -1:
    """Check if an object is an instance of np.datetime64 `<'bool'>`.

    Equivalent to:
    >>> isinstance(obj, np.datetime64)
    """
    return np.is_datetime64_object(obj)

cdef inline bint validate_dt64(object obj) except -1:
    """Validate if an object is an instance of np.datetime64,
    and raises `TypeError` if not."""
    if not np.is_datetime64_object(obj):
        raise TypeError(
            "expects instance of 'np.datetime64', "
            "instead got %s." % type(obj)
        )
    return True

# . conversion
cdef inline np.npy_int64 dt64_as_int64_us(object dt64, np.npy_int64 offset=0):
    """Cast np.datetime64 to int64 under 'us' (microsecond) resolution `<'int'>`.
    
    Equivalent to:
    >>> dt64.astype("datetime64[us]").astype("int64") + offset
    """
    # Get unit & value
    validate_dt64(dt64)
    cdef np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(dt64)
    cdef np.npy_int64 val = np.get_datetime64_value(dt64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // 1_000 + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val * US_MILLISECOND + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val * US_SECOND + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * US_MINUTE + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * US_HOUR + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * US_DAY + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return _dt64_W_as_int64_D(val, US_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _dt64_M_as_int64_D(val, US_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _dt64_Y_as_int64_D(val, US_DAY, offset)
    # . unsupported
    _raise_dt64_as_int64_unit_error("us", unit)

cdef inline np.npy_int64 _dt64_Y_as_int64_D(np.npy_int64 val, np.npy_int64 factor=1, np.npy_int64 offset=0):
    """(internal) Convert the value of np.datetime64[Y] to int64 under 'D' resolution `<'int'>`."""
    cdef:
        np.npy_int64 year  = val + EPOCH_YEAR
        np.npy_int64 y_1   = year - 1
        np.npy_int64 leaps = (
            (y_1 // 4 - 1970 // 4)
            - (y_1 // 100 - 1970 // 100)
            + (y_1 // 400 - 1970 // 400)
        )
    return (val * 365 + leaps) * factor + offset

cdef inline np.npy_int64 _dt64_M_as_int64_D(np.npy_int64 val, np.npy_int64 factor=1, np.npy_int64 offset=0):
    """(internal) Convert the value of np.datetime64[M] to int64 under 'D' resolution `<'int'>`."""
    cdef:
        np.npy_int64 year_ep = val // 12
        np.npy_int64 year    = year_ep + EPOCH_YEAR
        np.npy_int64 month   = val % 12 + 1
        np.npy_int64 y_1     = year - 1
        np.npy_int64 leaps   = (
            (y_1 // 4 - 1970 // 4)
            - (y_1 // 100 - 1970 // 100)
            + (y_1 // 400 - 1970 // 400)
        )
    return (year_ep * 365 + leaps + days_bf_month(year, month)) * factor + offset

cdef inline np.npy_int64 _dt64_W_as_int64_D(np.npy_int64 val, np.npy_int64 factor=1, np.npy_int64 offset=0):
    """(internal) Convert the value of np.datetime64[W] to int64 under 'D' resolution `<'int'>`."""
    return val * (7 * factor) + offset

cdef inline datetime.datetime dt64_to_dt(object dt64, tz: object=None):
    """Convert np.datetime64 to datetime `<'datetime.datetime'>`."""
    return dt_fr_us(dt64_as_int64_us(dt64, 0), tz)

# . errors: internal
cdef inline bint _raise_dt64_as_int64_unit_error(str reso, int unit, bint is_dt64=True) except -1:
    """(internal) Raise unsupported unit for dt/td_as_int*() function."""
    obj_type = "datetime64" if is_dt64 else "timedelta64"
    try:
        unit_str = map_nptime_unit_int2str(unit)
    except Exception as err:
        raise ValueError(
            "cannot cast %s to an integer under '%s' resolution.\n"
            "%s with datetime unit '%d' is not supported."
            % (obj_type, reso, obj_type, unit)
        ) from err
    else:
        raise ValueError(
            "cannot cast %s[%s] to an integer under '%s' resolution.\n"
            "%s with datetime unit '%s' is not supported."
            % (obj_type, unit_str, reso, obj_type, unit_str)
        )

# NumPy: timedelta64 -----------------------------------------------------------------------------------
# . type check
cdef inline bint is_td64(object obj) except -1:
    """Check if an object is an instance of np.timedelta64 `<'bool'>`.

    Equivalent to:
    >>> isinstance(obj, np.timedelta64)
    """
    return np.is_timedelta64_object(obj)

cdef inline bint validate_td64(object obj) except -1:
    """Validate if an object is an instance of np.timedelta64,
    and raises `TypeError` if not."""
    if not np.is_timedelta64_object(obj):
        raise TypeError(
            "expects instance of 'np.timedelta64', "
            "instead got %s." % type(obj)
        )
    return True

# . conversion
cdef inline np.npy_int64 td64_as_int64_us(object td64, np.npy_int64 offset=0):
    """Cast np.timedelta64 to int64 under 'us' (microsecond) resolution `<'int'>`.
    
    Equivalent to:
    >>> td64.astype("timedelta64[us]").astype("int64") + offset
    """
    # Get unit & value
    validate_td64(td64)
    cdef np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(td64)
    cdef np.npy_int64 val = np.get_timedelta64_value(td64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // 1_000 + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val * US_MILLISECOND + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val * US_SECOND + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * US_MINUTE + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * US_HOUR + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * US_DAY + offset
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return _td64_W_as_int64_D(val, US_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _td64_M_as_int64_D(val, US_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _td64_Y_as_int64_D(val, US_DAY, offset)
    # . unsupported unit
    _raise_dt64_as_int64_unit_error("us", unit, False)

cdef inline np.npy_int64 _td64_Y_as_int64_D(np.npy_int64 val, np.npy_int64 factor=1, np.npy_int64 offset=0):
    """(internal) Convert the value of np.timedelta[Y] to int64 under 'D' resolution `<'int'>`."""
    # Average number of days in a year: 365.2425
    # We use integer arithmetic by scaling to avoid floating-point inaccuracies.
    # Multiply by 3652425 and divide by 10000 to represent 365.2425 days/year.
    if factor == 1:  # day
        return val * 3_652_425 // 10_000 + offset  # val * 365.2425
    if factor == 24:  # hour
        return val * 876_582 // 100 + offset  # val * 8765.82 (365.2425 * 24)
    if factor == 1_440:  # minute
        return val * 5_259_492 // 10 + offset  # val * 525949.2 (365.2425 * 1440)
    if factor == SS_DAY:  # second
        return val * TD64_YY_SECOND + offset
    if factor == MS_DAY:  # millisecond
        return val * TD64_YY_MILLISECOND + offset
    if factor == US_DAY:  # microsecond
        return val * TD64_YY_MICROSECOND + offset
    if factor == NS_DAY:  # nanosecond
        return val * TD64_YY_NANOSECOND + offset
    raise AssertionError("unsupported factor '%d' for timedelta unit 'Y' conversion." % factor)

cdef inline np.npy_int64 _td64_M_as_int64_D(np.npy_int64 val, np.npy_int64 factor=1, np.npy_int64 offset=0):
    """(internal) Convert the value of np.timedelta[M] to int64 under 'D' resolution `<'int'>`."""
    # Average number of days in a month: 30.436875 (365.2425 / 12)
    # We use integer arithmetic by scaling to avoid floating-point inaccuracies.
    # Multiply by 30436875 and divide by 1000000 to represent 30.436875 days/month.
    if factor == 1: #  day
        return val * 30_436_875 // 1_000_000 + offset  # val * 30.436875
    if factor == 24: #  hour
        return val * 730_485 // 1_000 + offset  # val * 730.485 (30.436875 * 24)
    if factor == 1_440:  # minute
        return val * 438_291 // 10 + offset  # val * 43829.1 (30.436875 * 1440)
    if factor == SS_DAY:  # second
        return val * TD64_MM_SECOND + offset
    if factor == MS_DAY:  # millisecond
        return val * TD64_MM_MILLISECOND + offset
    if factor == US_DAY:  # microsecond
        return val * TD64_MM_MICROSECOND + offset
    if factor == NS_DAY:  # nanosecond
        return val * TD64_MM_NANOSECOND + offset
    raise AssertionError("unsupported factor '%d' for timedelta unit 'M' conversion." % factor)

cdef inline np.npy_int64 _td64_W_as_int64_D(np.npy_int64 val, np.npy_int64 factor=1, np.npy_int64 offset=0):
    """(internal) Convert the value of np.timedelta[W] to int64 under 'D' resolution `<'int'>`."""
    return val * (7 * factor) + offset

cdef inline datetime.timedelta td64_to_td(object td64):
    """Convert np.timedelta64 to timedelta `<'datetime.timedelta'>`.

    Equivalent to:
    >>> us = td64.astype("timedelta64[us]").astype("int64")
    >>> datetime.timedelta(microseconds=int(us))
    """
    return td_fr_us(td64_as_int64_us(td64, 0))

# NumPy: ndarray ---------------------------------------------------------------------------------------
# . type check
cdef inline bint is_arr(object obj) except -1:
    """Check if an object is an instance of np.ndarray `<'bool'>`.

    Equivalent to:
    >>> isinstance(obj, np.ndarray)
    """
    return np.PyArray_Check(obj)

# . dtype
cdef inline np.ndarray arr_assure_int64(np.ndarray arr):
    """Assure the given ndarray is dtype of 'int64' `<'ndarray[int64]'>`.

    Automatically cast the 'arr' to 'int64' if not the correct dtype.
    """
    if np.PyArray_TYPE(arr) == np.NPY_TYPES.NPY_INT64:
        return arr
    return np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)

cdef inline np.ndarray arr_assure_int64_like(np.ndarray arr):
    """Assure the given ndarray is dtype of [int64/datetime64/timedelta64] `<'ndarray'>`.

    The data of an 'int64-like' array can be directly accessed as 'np.npy_int64*'
    
    Automatically cast the 'arr' to 'int64' if not the correct dtype.
    """
    if np.PyArray_TYPE(arr) in (
        np.NPY_TYPES.NPY_INT64,
        np.NPY_TYPES.NPY_DATETIME,
        np.NPY_TYPES.NPY_TIMEDELTA,
    ):
        return arr
    return np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)

cdef inline np.ndarray arr_assure_float64(np.ndarray arr):
    """Assure the given ndarray is dtype of 'float64' `<'ndarray[int64]'>`.

    Automatically cast the 'arr' to 'flaot64' if not the correct dtype.
    """
    if np.PyArray_TYPE(arr) == np.NPY_TYPES.NPY_FLOAT64:
        return arr
    return np.PyArray_Cast(arr, np.NPY_TYPES.NPY_FLOAT64)

# . create
cdef inline np.ndarray arr_zero_int64(np.npy_int64 size):
    """Create an 1-dimensional ndarray[int64] filled with zero `<'ndarray[int64]'>`.

    Equivalent to:
    >>> np.zeros(size, dtype="int64")
    """
    # Validate
    if size < 1:
        raise ValueError("ndarray size must be an integer greater than 0.")

    # New array
    return np.PyArray_ZEROS(1, [size], np.NPY_TYPES.NPY_INT64, 0)

cdef inline np.ndarray arr_fill_int64(np.npy_int64 value, np.npy_int64 size):
    """Create a new 1-dimensional ndarray[int64] filled with 'value' `<'ndarray[int64]'>`.

    Equivalent to:
    >>> np.full(size, value, dtype="int64")
    """
    # Fast-path
    if value == 0:
        return arr_zero_int64(size)

    # Validate
    if size < 1:
        raise ValueError("ndarray size must be an integer greater than 0.")

    # New array
    cdef:
        np.ndarray arr = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        arr_ptr[i] = value
    return arr

# . range
cdef inline np.ndarray arr_clip(np.ndarray arr, np.npy_int64 minimum, np.npy_int64 maximum, np.npy_int64 offset=0):
    """Clip the values of an ndarray between 'minimum' and 'maximum' value `<'ndarray[int64]'>`.

    Before compute, this function will try to cast the array to 'int64',
    if it is not dtype of [int64/datetime64/timedelta64].

    Equivalent to:
    >>> np.clip(arr, minimum, maximum) + offset
    """
    # Validate
    arr = arr_assure_int64_like(arr)

    # Compute
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i, v

    for i in range(size):
        v = arr_ptr[i]
        if v > maximum:
            v = maximum
        elif v < minimum:
            v = minimum
        res_ptr[i] = v + offset
    return res

cdef inline np.ndarray arr_min(np.ndarray arr, np.npy_int64 value, np.npy_int64 offset=0):
    """Get the minimum values between the ndarray and the 'value `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64' 
    if it is not dtype of [int64/datetime64/timedelta64].

    Equivalent to:
    >>> np.minimum(arr, value) + offset
    """
    # Validate
    arr = arr_assure_int64_like(arr)

    # Compute
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = min(arr_ptr[i], value) + offset
    return res

cdef inline np.ndarray arr_max(np.ndarray arr, np.npy_int64 value, np.npy_int64 offset=0):
    """Get the maximum values between the ndarray and the 'value' `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64' 
    if it is not dtype of [int64/datetime64/timedelta64].

    Equivalent to:
    >>> np.maximum(arr, value) + offset
    """
    # Validate
    arr = arr_assure_int64_like(arr)

    # Compute
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = max(arr_ptr[i], value) + offset
    return res

cdef inline np.ndarray arr_min_arr(np.ndarray arr1, np.ndarray arr2, np.npy_int64 offset=0):
    """Get the minimum values between two ndarrays `<'ndarray[int64]'>`.

    Before compute, this function will cast the arrays to 'int64'
    if they are not dtype of [int64/datetime64/timedelta64].

    Equivalent to:
    >>> np.minimum(arr1, arr2) + offset
    """
    # Validate
    cdef np.npy_int64 size = arr1.shape[0]
    if size != arr2.shape[0]:
        raise ValueError("cannot compare ndarrays with different shapes.")
    arr1 = arr_assure_int64_like(arr1)
    arr2 = arr_assure_int64_like(arr2)

    # Compute
    cdef:
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr1_ptr = <np.npy_int64*> np.PyArray_DATA(arr1)
        np.npy_int64* arr2_ptr = <np.npy_int64*> np.PyArray_DATA(arr2)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = min(arr1_ptr[i], arr2_ptr[i]) + offset
    return res

cdef inline np.ndarray arr_max_arr(np.ndarray arr1, np.ndarray arr2, np.npy_int64 offset=0):
    """Get the maxmimum values between two ndarrays `<'ndarray[int64]'>`.

    Before compute, this function will cast the arrays to 'int64'
    if they are not dtype of [int64/datetime64/timedelta64].

    Equivalent to:
    >>> np.maximum(arr1, arr2) + offset
    """
    # Validate
    cdef np.npy_int64 size = arr1.shape[0]
    if size != arr2.shape[0]:
        raise ValueError("cannot compare ndarrays with different shapes.")
    arr1 = arr_assure_int64_like(arr1)
    arr2 = arr_assure_int64_like(arr2)

    # Compute
    cdef:
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr1_ptr = <np.npy_int64*> np.PyArray_DATA(arr1)
        np.npy_int64* arr2_ptr = <np.npy_int64*> np.PyArray_DATA(arr2)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = max(arr1_ptr[i], arr2_ptr[i]) + offset
    return res

# . arithmetic
cdef inline np.ndarray arr_abs(np.ndarray arr, np.npy_int64 offset=0):
    """Compute the absolute values of the ndarray `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not dtype of [int64/datetime64/timedelta64].

    Equivalent to:
    >>> np.abs(arr) + offset
    """
    # Validate
    arr = arr_assure_int64_like(arr)

    # Compute
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i, v

    for i in range(size):
        v = arr_ptr[i]
        if v < 0:
            res_ptr[i] = -v + offset
        else:
            res_ptr[i] = v + offset
    return res

cdef inline np.ndarray arr_add(np.ndarray arr, np.npy_int64 value):
    """Add the value to the ndarray `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64' 
    if it is not dtype of [int64/datetime64/timedelta64].

    Equivalent to:
    >>> arr + value
    """
    # Fast-path
    if value == 0:
        return arr_assure_int64(arr)

    # Validate
    arr = arr_assure_int64_like(arr)

    # Compute
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = arr_ptr[i] + value
    return res

cdef inline np.ndarray arr_mul(np.ndarray arr, np.npy_int64 factor, np.npy_int64 offset=0):
    """Multiply the values of the ndarray by the factor `<'ndarray[int64]'>`.
    
    Before compute, this function will cast the array to 'int64' 
    if it is not dtype of [int64/datetime64/timedelta64].
    
    Equivalent to:
    >>> arr * factor + offset
    """
    # Fast-path
    if factor == 1 and offset == 0:
        return arr_assure_int64(arr)

    # Validate
    arr = arr_assure_int64_like(arr)

    # Compute
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = arr_ptr[i] * factor + offset
    return res

cdef inline np.ndarray arr_div(np.ndarray arr, np.npy_float64 factor, np.npy_float64 offset=0):
    """Divides the values of the ndarray by the factor, handling negative
    numbers according to Python's division semantics `<'ndarray[float64]'>`.

    Before compute, this function will cast the array to 'float64'
    if it is not in 'float64' dtype.

    Equivalent to:
    >>> arr / factor + offset
    """
    # Fast-path
    if factor == 1 and offset == 0:
        return arr_assure_float64(arr)

    # Validate
    arr = arr_assure_float64(arr)

    # Compute
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_FLOAT64, 0)
        np.npy_float64* res_ptr = <np.npy_float64*> np.PyArray_DATA(res)
        np.npy_float64* arr_ptr = <np.npy_float64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    with cython.cdivision(True):
        for i in range(size):
            res_ptr[i] = arr_ptr[i] / factor + offset
    return res

cdef inline np.ndarray arr_mod(np.ndarray arr, np.npy_int64 factor, np.npy_int64 offset=0):
    """Computes the modulo of the values of the ndarray by the
    factor, handling negative numbers according to Python's modulo
    semantics `<'ndarray[int64]'>`.

    Before computation, this function will cast the array to 'int64'
    if it is not already in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> arr % factor + offset
    """
    # Validate
    arr = arr_assure_int64_like(arr)
    if factor == 0:
        raise ZeroDivisionError("division by zero in 'arr_mod()'.")

    # Prepare for computation
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        bint neg_f = factor < 0
        np.npy_int64 i, r

    with cython.cdivision(True):
        for i in range(size):
            r = arr_ptr[i] % factor 
            if r != 0:
                if not neg_f:
                    if r < 0:
                        r += factor
                else:
                    if r > 0:
                        r += factor
            res_ptr[i] = r
    return res

cdef inline np.ndarray arr_round_div(np.ndarray arr, np.npy_int64 factor, np.npy_int64 offset=0):
    """Divides the values of the ndarray by the factor and rounds to the
    nearest integers (half away from zero), handling negative numbers
    according to Python's division semantics `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> np.round(arr / factor, 0) + offset
    """
    # Fast-path
    if factor == 1 and offset == 0:
        return arr_assure_int64(arr)

    # Validate
    arr = arr_assure_int64_like(arr)
    if factor == 0:
        raise ZeroDivisionError("division by zero for 'utils.arr_round_div()'.")

    # Compute
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        bint neg_f = factor < 0
        np.npy_int64 abs_f = -factor if neg_f else factor
        np.npy_int64 i, n, q, r, abs_r

    with cython.cdivision(True):
        for i in range(size):
            n = arr_ptr[i]
            q = n // factor
            r = n % factor
            abs_r = -r if r < 0 else r
            if abs_r * 2 >= abs_f:
                if (not neg_f and n >= 0) or (neg_f and n < 0):
                    q += 1
                else:
                    q -= 1
            res_ptr[i] = q + offset
    return res

cdef inline np.ndarray arr_ceil_div(np.ndarray arr, np.npy_int64 factor, np.npy_int64 offset=0):
    """Divides the values of the ndarray by the factor and rounds
    up to the nearest integers, handling negative numbers according
    to Python's division semantics `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> np.ceil(arr / factor) + offset
    """
    # Fast-path
    if factor == 1 and offset == 0:
        return arr_assure_int64(arr)

    # Validate
    arr = arr_assure_int64_like(arr)
    if factor == 0:
        raise ZeroDivisionError("division by zero for 'utils.arr_ceil_div()'.")

    # Compute
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        bint neg_f = factor < 0
        np.npy_int64 i, n, q, r

    with cython.cdivision(True):
        for i in range(size):
            n = arr_ptr[i]
            q = n // factor
            r = n % factor
            if r != 0:
                if not neg_f:
                    if n > 0:
                        q += 1
                else:
                    if n < 0:
                        q += 1
            res_ptr[i] = q + offset
    return res

cdef inline np.ndarray arr_floor_div(np.ndarray arr, np.npy_int64 factor, np.npy_int64 offset=0):
    """Divides the values of the ndarray by the factor and rounds down
    to the nearest integers, handling negative numbers according to
    Python's division semantics `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> np.floor(arr / factor) + offset
    """
    # Fast-path
    if factor == 1 and offset == 0:
        return arr_assure_int64(arr)

    # Validate
    arr = arr_assure_int64_like(arr)
    if factor == 0:
        raise ZeroDivisionError("division by zero for 'utils.arr_floor_div()'.")

    # Compute
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        bint neg_f = factor < 0
        np.npy_int64 i, n, q, r
    
    with cython.cdivision(True):
        for i in range(size):
            n = arr_ptr[i]
            q = n // factor
            r = n % factor
            if r != 0:
                if not neg_f:
                    if n < 0:
                        q -= 1
                else:
                    if n > 0:
                        q -= 1
            res_ptr[i] = q + offset
    return res

cdef inline np.ndarray arr_round_to_mul(np.ndarray arr, np.npy_int64 factor, np.npy_int64 multiple = 0, np.npy_int64 offset=0):
    """Round to multiple. Divides the values of the ndarray by the factor and rounds
    to the nearest integers (half away from zero), handling negative numbers according
    to Python's division semantics. Finally multiply the the multiple. Argument
    'multiple' defaults to `0`, which means if not specified, it uses factor as
    the multiple `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> np.round(arr / factor, 0) * multiple + offset
    """
    # Fast-path
    if factor == 1 and multiple == 1 and offset == 0:
        return arr_assure_int64(arr)

    # Validate
    arr = arr_assure_int64_like(arr)
    if factor == 0:
        raise ZeroDivisionError("cannot round to multiple 0.")
    if multiple == 0:
        multiple = factor

    # Compute
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        bint neg_f = factor < 0
        np.npy_int64 abs_f = -factor if neg_f else factor
        np.npy_int64 i, n, q, r, abs_r

    with cython.cdivision(True):
        for i in range(size):
            n = arr_ptr[i]
            q = n // factor
            r = n % factor
            abs_r = -r if r < 0 else r
            if abs_r * 2 >= abs_f:
                if (not neg_f and n >= 0) or (neg_f and n < 0):
                    q += 1
                else:
                    q -= 1
            res_ptr[i] = q * multiple + offset
    return res

cdef inline np.ndarray arr_ceil_to_mul(np.ndarray arr, np.npy_int64 factor, np.npy_int64 multiple = 0, np.npy_int64 offset=0):
    """Ceil to multiple. Divides the values of the ndarray by the factor and
    rounds up to the nearest integers, handling negative numbers according to
    Python's division semantics. Finally multiply the the multiple. Argument
    'multiple' defaults to `0`, which means if not specified, it uses factor
    as the multiple `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> np.ceil(arr / factor) * multiple + offset
    """
    # Fast-path
    if factor == 1 and multiple == 1 and offset == 0:
        return arr_assure_int64(arr)

    # Validate
    arr = arr_assure_int64_like(arr)
    if factor == 0:
        raise ZeroDivisionError("cannot ceil to multiple 0.")
    if multiple == 0:
        multiple = factor

    # Compute
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        bint neg_f = factor < 0
        np.npy_int64 i, n, q, r

    with cython.cdivision(True):
        for i in range(size):
            n = arr_ptr[i]
            q = n // factor
            r = n % factor
            if r != 0:
                if not neg_f:
                    if n > 0:
                        q += 1
                else:
                    if n < 0:
                        q += 1
            res_ptr[i] = q * multiple + offset
    return res

cdef inline np.ndarray arr_floor_to_mul(np.ndarray arr, np.npy_int64 factor, np.npy_int64 multiple = 0, np.npy_int64 offset=0):
    """Floor to multiple. Divides the values of the ndarray by the factor and
    rounds down to the nearest integers, handling negative numbers according to
    Python's division semantics. Finally multiply the the multiple. Argument
    multiple defaults to '0', which means if not specified, it uses factor as
    the multiple `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> np.floor(arr / factor) * multiple + offset
    """
    # Fast-path
    if factor == 1 and multiple == 1 and offset == 0:
        return arr_assure_int64(arr)
    
    # Validate
    arr = arr_assure_int64_like(arr)
    if factor == 0:
        raise ZeroDivisionError("cannot floor to multiple 0.")
    if multiple == 0:
        multiple = factor

    # Compute
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        bint neg_f = factor < 0
        np.npy_int64 i, n, q, r

    with cython.cdivision(True):
        for i in range(size):
            n = arr_ptr[i]
            q = n // factor
            r = n % factor
            if r != 0:
                if not neg_f:
                    if n < 0:
                        q -= 1
                else:
                    if n > 0:
                        q -= 1
            res_ptr[i] = q * multiple + offset
    return res

cdef inline np.ndarray arr_add_arr(np.ndarray arr1, np.ndarray arr2, np.npy_int64 offset=0):
    """Addition between two ndarrays `<'ndarray[int64]'>`.

    Before compute, this function will cast the arrays to 'int64' 
    if they are not in [int64/datetime64/timedelta64] dtype.

    Equivalent to:
    >>> arr1 + arr2 + offset
    """
    # Validate
    cdef np.npy_int64 size = arr1.shape[0]
    if size != arr2.shape[0]:
        raise ValueError("cannot perform addition on ndarrays with different shapes.")
    arr1 = arr_assure_int64_like(arr1)
    arr2 = arr_assure_int64_like(arr2)

    # Compute
    cdef:
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr1_ptr = <np.npy_int64*> np.PyArray_DATA(arr1)
        np.npy_int64* arr2_ptr = <np.npy_int64*> np.PyArray_DATA(arr2)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = arr1_ptr[i] + arr2_ptr[i] + offset
    return res

cdef inline np.ndarray arr_sub_arr(np.ndarray arr1, np.ndarray arr2, np.npy_int64 offset=0):
    """Subtraction between two ndarrays `<'ndarray[int64]'>`.

    Before compute, this function will cast the arrays to 'int64' 
    if they are not in [int64/datetime64/timedelta64] dtype.

    Equivalent to:
    >>> arr1 - arr2 + offset
    """
    # Validate
    cdef np.npy_int64 size = arr1.shape[0]
    if size != arr2.shape[0]:
        raise ValueError("cannot perform subtraction on ndarrays with different shapes.")
    arr1 = arr_assure_int64_like(arr1)
    arr2 = arr_assure_int64_like(arr2)

    # Compute
    cdef:
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr1_ptr = <np.npy_int64*> np.PyArray_DATA(arr1)
        np.npy_int64* arr2_ptr = <np.npy_int64*> np.PyArray_DATA(arr2)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = arr1_ptr[i] - arr2_ptr[i] + offset
    return res

# . comparison
cdef inline np.ndarray arr_equal_to(np.ndarray arr, np.npy_int64 value):
    """Check if the values of the ndarray are equal to the 'value' `<'ndarray[bool]'>`.
    
    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> arr == value
    """
    # Validate
    arr = arr_assure_int64_like(arr)

    # Compute
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_BOOL, 0)
        np.npy_bool* res_ptr = <np.npy_bool*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = arr_ptr[i] == value
    return res

cdef inline np.ndarray arr_greater_than(np.ndarray arr, np.npy_int64 value):
    """Check if the values of the ndarray are greater than the 'value' `<'ndarray[bool]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> arr > value
    """
    # Validate
    arr = arr_assure_int64_like(arr)

    # Compute
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_BOOL, 0)
        np.npy_bool* res_ptr = <np.npy_bool*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = arr_ptr[i] > value
    return res

cdef inline np.ndarray arr_less_than(np.ndarray arr, np.npy_int64 value):
    """Check if the values of the ndarray are less than the 'value' `<'ndarray[bool]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> arr < value
    """
    # Validate
    arr = arr_assure_int64_like(arr)

    # Compute
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_BOOL, 0)
        np.npy_bool* res_ptr = <np.npy_bool*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = arr_ptr[i] < value
    return res

cdef inline np.ndarray arr_equal_to_arr(np.ndarray arr1, np.ndarray arr2):
    """Check if the values of two ndarrays are equal `<'ndarray[bool]'>`

    Equivalent to:
    >>> arr1 == arr2
    """
    # Validate
    cdef np.npy_int64 size = arr1.shape[0]
    if size != arr2.shape[0]:
        raise ValueError("cannot perform comparison on ndarrays with different shapes.")
    arr1 = arr_assure_int64_like(arr1)
    arr2 = arr_assure_int64_like(arr2)

    # Compute
    cdef:
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_BOOL, 0)
        np.npy_bool* res_ptr = <np.npy_bool*> np.PyArray_DATA(res)
        np.npy_int64* arr1_ptr = <np.npy_int64*> np.PyArray_DATA(arr1)
        np.npy_int64* arr2_ptr = <np.npy_int64*> np.PyArray_DATA(arr2)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = arr1_ptr[i] == arr2_ptr[i]
    return res

# NumPy: ndarray[datetime64] ---------------------------------------------------------------------------
# . type check
cdef inline bint is_dt64arr(np.ndarray arr) except -1:
    """Check if the given array is dtype of 'datetime64' `<'bool'>`.
    
    Equivalent to:
    >>> isinstance(arr.dtype, np.dtypes.DateTime64DType)
    """
    return np.PyArray_TYPE(arr) == np.NPY_TYPES.NPY_DATETIME

cdef inline bint validate_dt64arr(np.ndarray arr) except -1:
    """Validate if the given array is dtype of 'datetime64',
    raises `TypeError` if dtype is incorrect.
    """
    if not is_dt64arr(arr):
        raise TypeError(
            "expects instance of 'np.ndarray[datetime64]', "
            "instead got '%s'." % arr.dtype
        )
    return True

# . range check
cdef inline bint is_dt64arr_ns_safe(np.ndarray arr, str arr_unit=None) except -1:
    """Check if the ndarray[datetime64] is within 
    nanoseconds conversion range `<'bool'>`.

    Safe range: between '1677-09-22' and '2262-04-10' (+/- 1 day from limits)
    """
    # Validate array
    if np.PyArray_SIZE(arr) == 0:
        return arr  # exit: empty array

    # Get time unit
    cdef int unit
    cdef int dtype = np.PyArray_TYPE(arr)
    if dtype != np.NPY_TYPES.NPY_DATETIME:
        if arr_unit is None:
            raise ValueError(
                "cannot check if <'ndarray[%s]'> is within nanosecond conversion "
                "range without specifying the array datetime unit." % arr.dtype
            )
        unit = map_nptime_unit_str2int(arr_unit)
        arr = arr_assure_int64(arr)
    elif arr_unit is not None:
        unit = map_nptime_unit_str2int(arr_unit)
    else:
        unit = get_arr_nptime_unit(arr)
    #: arr can only be only datetime64 or int64 afterwords

    # Get min & max range
    cdef np.npy_int64 minimum, maximum
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        minimum, maximum = DT64_NS_NS_MIN, DT64_NS_NS_MAX
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        minimum, maximum = DT64_NS_US_MIN, DT64_NS_US_MAX
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        minimum, maximum = DT64_NS_MS_MIN, DT64_NS_MS_MAX
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        minimum, maximum = DT64_NS_SS_MIN, DT64_NS_SS_MAX
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        minimum, maximum = DT64_NS_MI_MIN, DT64_NS_MI_MAX
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        minimum, maximum = DT64_NS_HH_MIN, DT64_NS_HH_MAX
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        minimum, maximum = DT64_NS_DD_MIN, DT64_NS_DD_MAX
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        minimum, maximum = DT64_NS_WW_MIN, DT64_NS_WW_MAX
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        minimum, maximum = DT64_NS_MM_MIN, DT64_NS_MM_MAX
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        minimum, maximum = DT64_NS_YY_MIN, DT64_NS_YY_MAX
    else:
        raise ValueError(
            "cannot check <'ndarray[%s]'> nanosecond conversion "
            "range, datetime unit '%s' is not supported." 
            % (arr.dtype, map_nptime_unit_int2str(unit))
        )

    # Check range
    cdef: 
        np.npy_int64 size = arr.shape[0]
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        if not minimum < arr_ptr[i] < maximum:
            return False
    return True

# . access
cdef inline np.ndarray dt64arr_year(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Get the year values of the ndarray[datetime64] `<'ndarray[int64]'>`."""
    # Cast to int64[Y]
    arr = dt64arr_as_int64_Y(arr, arr_unit, 0)

    # Add back epoch
    cdef: 
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = arr_ptr[i] + EPOCH_YEAR + offset
    return res

cdef inline np.ndarray dt64arr_quarter(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Get the quarter values of the ndarray[datetime64] `<'ndarray[int64]'>`."""
    # Cast to int64[M]
    arr = dt64arr_as_int64_M(arr, arr_unit, 0)

    # Compute quarter
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i
        
    for i in range(size):
        res_ptr[i] = quarter_of_month(arr_ptr[i] % 12 + 1) + offset
    return res

cdef inline np.ndarray dt64arr_month(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Get the month values of the ndarray[datetime64] `<'ndarray[int64]'>`."""
    # Cast to int64[M]
    arr = dt64arr_as_int64_M(arr, arr_unit, 0)

    # Compute quarter
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i
    
    for i in range(size):
        res_ptr[i] = arr_ptr[i] % 12 + 1 + offset
    return res

cdef inline np.ndarray dt64arr_weekday(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Get the weekday values of the ndarray[datetime64] `<'ndarray[int64]'>`."""
    # Cast to int64[M]
    arr = dt64arr_as_int64_D(arr, arr_unit, 0)

    # Compute quarter
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i
    
    for i in range(size):
        res_ptr[i] = (arr_ptr[i] + EPOCH_DAY + 6) % 7 + offset
    return res

cdef inline np.ndarray dt64arr_day(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Get the day values of the ndarray[datetime64] `<'ndarray[int64]'>`."""
    # Cast to int64[M]
    arr = dt64arr_as_int64_D(arr, arr_unit, 0)

    # Compute quarter
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i, n, n400, n100, n4, n1, year, month, day, days_bf

    for i in range(size):
        # Convert to ordinal
        n = arr_ptr[i] + EPOCH_DAY - 1
        # Number of complete 400-year cycles
        n400 = n // 146_097
        n -= n400 * 146_097
        # Number of complete 100-year cycles within the 400-year cycle
        n100 = n // 36_524
        n -= n100 * 36_524
        # Number of complete 4-year cycles within the 100-year cycle
        n4 = n // 1_461
        n -= n4 * 1_461
        # Number of complete years within the 4-year cycle
        n1 = n // 365
        n -= n1 * 365
        # Compute the year
        year = n400 * 400 + n100 * 100 + n4 * 4 + n1 + 1
        # Adjust for end-of-cycle dates
        if n100 == 4 or n1 == 4:
            day = 31
        else:
            month = (n + 50) >> 5
            days_bf = days_bf_month(year, month)
            if days_bf > n:
                days_bf = days_bf_month(year, month - 1)
            day = n - days_bf + 1
        # Compute total months
        res_ptr[i] = day + offset
    return res

cdef inline np.ndarray dt64arr_hour(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Get the hour values of the ndarray[datetime64] `<'ndarray[int64]'>`."""
    # Cast to int64[M]
    arr = dt64arr_as_int64_h(arr, arr_unit, 0)

    # Compute quarter
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i
    
    for i in range(size):
        res_ptr[i] = (arr_ptr[i] + EPOCH_HOUR) % 24 + offset
    return res

cdef inline np.ndarray dt64arr_minute(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Get the minute values of the ndarray[datetime64] `<'ndarray[int64]'>`."""
    # Cast to int64[M]
    arr = dt64arr_as_int64_m(arr, arr_unit, 0)

    # Compute quarter
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i
    
    for i in range(size):
        res_ptr[i] = (arr_ptr[i] + EPOCH_MINUTE) % 60 + offset
    return res

cdef inline np.ndarray dt64arr_second(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Get the second values of the ndarray[datetime64] `<'ndarray[int64]'>`."""
    # Cast to int64[M]
    arr = dt64arr_as_int64_s(arr, arr_unit, 0)

    # Compute quarter
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i
    
    for i in range(size):
        res_ptr[i] = arr_ptr[i] % 60 + offset
    return res

cdef inline np.ndarray dt64arr_millisecond(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Get the millisecond values of the ndarray[datetime64] `<'ndarray[int64]'>`."""
    # Cast to int64[M]
    arr = dt64arr_as_int64_ms(arr, arr_unit, 0)

    # Compute quarter
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i
    
    for i in range(size):
        res_ptr[i] = arr_ptr[i] % 1_000 + offset
    return res

cdef inline np.ndarray dt64arr_microsecond(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Get the microsecond values of the ndarray[datetime64] `<'ndarray[int64]'>`."""
    # Cast to int64[M]
    arr = dt64arr_as_int64_us(arr, arr_unit, 0)

    # Compute quarter
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i
    
    for i in range(size):
        res_ptr[i] = arr_ptr[i] % 1_000_000 + offset
    return res

cdef inline np.ndarray dt64arr_nanosecond(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Get the nanosecond values of the ndarray[datetime64] `<'ndarray[int64]'>`."""
    # Cast to int64[M]
    arr = dt64arr_as_int64_ns(arr, arr_unit, 0)

    # Compute quarter
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i
    
    for i in range(size):
        res_ptr[i] = arr_ptr[i] % 1_000 + offset
    return res

cdef inline np.ndarray dt64arr_times(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Get the times values of the ndarray[datetime64] `<'ndarray[int64]'>`."""
    # Empty array
    if np.PyArray_SIZE(arr) == 0:
        return arr  # exit: empty array

    # Get time unit
    cdef int unit
    cdef int dtype = np.PyArray_TYPE(arr)
    if dtype != np.NPY_TYPES.NPY_DATETIME:
        if arr_unit is None:
            _raise_dt64arr_convert_arr_unit_error(arr, "int64 under 'Y' resolution")
        unit = map_nptime_unit_str2int(arr_unit)
        arr = arr_assure_int64(arr)
    elif arr_unit is not None:
        unit = map_nptime_unit_str2int(arr_unit)
    else:
        unit = get_arr_nptime_unit(arr)
    #: arr can only be only datetime64 or int64 afterwords

    # Get factor
    cdef np.npy_int64 factor
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_mod(arr, NS_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_mod(arr, US_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_mod(arr, MS_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr_mod(arr, SS_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr_mod(arr, 1440, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr_mod(arr, 24, offset)
    if unit in (
        np.NPY_DATETIMEUNIT.NPY_FR_D,  # day
        np.NPY_DATETIMEUNIT.NPY_FR_W,  # week
        np.NPY_DATETIMEUNIT.NPY_FR_M,  # month
        np.NPY_DATETIMEUNIT.NPY_FR_Y,  # year
    ):
        if offset == 0:
            return arr_zero_int64(arr.shape[0])
        else:
            return arr_fill_int64(offset, arr.shape[0])
    # . unsupported
    unit_str = map_nptime_unit_int2str(unit)
    raise ValueError(
        "cannot access the time value from ndarray[datetime64[%s]].\n"
        "Array with dtype of 'datetime64[%s]' is not supported" % (unit_str, unit_str)
    )

# . calendar
cdef inline np.ndarray dt64arr_isocalendar(np.ndarray arr, str arr_unit=None):
    """Get the ISO calendar values of the ndarray[datetime64] `<'ndarray[int64]'>`.
    
    Returns a 2-dimensional array where each row contains
    the ISO year, week number, and weekday values.

    Example:
    >>> [[1936   11    7]
        [1936   12    1]
        [1936   12    2]
        ...
        [2003   42    6]
        [2003   42    7]
        [2003   43    1]]
    """
    # Cast to int64[D]
    arr = dt64arr_as_int64_D(arr, arr_unit, 0)

    # Compute days in month
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(2, [size, 3], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        _ymd = ymd_fr_ordinal(arr_ptr[i] + EPOCH_DAY)
        _iso = ymd_isocalendar(_ymd.year, _ymd.month, _ymd.day)
        res_ptr[i * 3] = _iso.year
        res_ptr[i * 3 + 1] = _iso.week
        res_ptr[i * 3 + 2] = _iso.weekday
    return res

cdef inline np.ndarray dt64arr_is_leap_year(np.ndarray arr, str arr_unit=None):
    """Check if the ndarray[datetime64] are leap years `<'ndarray[bool]'>`."""
    # Cast to int64[Y]
    arr = dt64arr_as_int64_Y(arr, arr_unit, 0)

    # Check leap year
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_BOOL, 0)
        np.npy_bool* res_ptr = <np.npy_bool*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = is_leap_year(arr_ptr[i] + EPOCH_YEAR)
    return res

cdef inline np.ndarray dt64arr_is_long_year(np.ndarray arr, str arr_unit=None):
    """Check if the ndarray[datetime64] are long years 
    (maximum ISO week number equal 53) `<'ndarray[bool]'>`.
    """
    # Cast to int64[Y]
    arr = dt64arr_as_int64_Y(arr, arr_unit, 0)

    # Check long year
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_BOOL, 0)
        np.npy_bool* res_ptr = <np.npy_bool*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = is_long_year(arr_ptr[i] + EPOCH_YEAR)
    return res

cdef inline np.ndarray dt64arr_leap_bt_year(np.ndarray arr, np.npy_int64 year, str arr_unit=None):
    """Calcuate the number of leap years between the ndarray[datetime64]
    and the passed in 'year' value `<'ndarray[int64]'>`."""
    # Cast to int64[Y]
    arr = dt64arr_as_int64_Y(arr, arr_unit, 0)

    # Compute leap years
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = leap_bt_year(arr_ptr[i] + EPOCH_YEAR, year)
    return res

cdef inline np.ndarray dt64arr_days_in_year(np.ndarray arr, str arr_unit=None):
    """Get the maximum days (365, 366) in the year
    of the ndarray[datetime64] `<'ndarray[int64]'>`."""
    # Cast to int64[Y]
    arr = dt64arr_as_int64_Y(arr, arr_unit, 0)

    # Compute days in year
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = days_in_year(arr_ptr[i] + EPOCH_YEAR)
    return res

cdef inline np.ndarray dt64arr_days_bf_year(np.ndarray arr, str arr_unit=None):
    """Get the number of days between the np.ndarray[datetime64]
    and the 1st day of the 1AD `<'ndarray[int64]'>`.
    """
    # Cast to int64[Y]
    arr = dt64arr_as_int64_Y(arr, arr_unit, 0)

    # Compute days before year
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = days_bf_year(arr_ptr[i] + EPOCH_YEAR)
    return res

cdef inline np.ndarray dt64arr_days_of_year(np.ndarray arr, str arr_unit=None):
    """Get the number of days between the np.ndarray[datetime64]
    and the 1st day of the array years `<'ndarray[int64]'>`.
    """
    # Cast to int64[D]
    arr = dt64arr_as_int64_D(arr, arr_unit, 0)

    # Compute days of year
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        _ymd = ymd_fr_ordinal(arr_ptr[i] + EPOCH_DAY)
        res_ptr[i] = days_of_year(_ymd.year, _ymd.month, _ymd.day)
    return res

cdef inline np.ndarray dt64arr_days_in_quarter(np.ndarray arr, str arr_unit=None):
    """Get the maximum days in the quarter of the np.npdarray[datetime64] `<'int'>`."""
    # Cast to int64[D]
    arr = dt64arr_as_int64_D(arr, arr_unit, 0)

    # Compute days in quarter
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        _ymd = ymd_fr_ordinal(arr_ptr[i] + EPOCH_DAY)
        res_ptr[i] = days_in_quarter(_ymd.year, _ymd.month)
    return res

cdef inline np.ndarray dt64arr_days_bf_quarter(np.ndarray arr, str arr_unit=None):
    """Get the number of days between the 1st day of the year 
    of the np.ndarray[datetime64] and the 1st day of its quarter `<'int'>`.
    """
    # Cast to int64[D]
    arr = dt64arr_as_int64_D(arr, arr_unit, 0)

    # Compute days in quarter
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        _ymd = ymd_fr_ordinal(arr_ptr[i] + EPOCH_DAY)
        res_ptr[i] = days_bf_quarter(_ymd.year, _ymd.month)
    return res

cdef inline np.ndarray dt64arr_days_of_quarter(np.ndarray arr, str arr_unit=None):
    """Get the number of days between the 1st day of the quarter
    of the np.ndarray[datetime64] and the its date `<'int'>`."""
    # Cast to int64[D]
    arr = dt64arr_as_int64_D(arr, arr_unit, 0)

    # Compute days in quarter
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i

    for i in range(size):
        _ymd = ymd_fr_ordinal(arr_ptr[i] + EPOCH_DAY)
        res_ptr[i] = days_of_quarter(_ymd.year, _ymd.month, _ymd.day)
    return res

cdef inline np.ndarray dt64arr_days_in_month(np.ndarray arr, str arr_unit=None):
    """Get the maximum days in the month of the ndarray[datetime64] <'ndarray[int64]'>."""
    # Cast to int64[M]
    arr = dt64arr_as_int64_M(arr, arr_unit, 0)

    # Compute days in month
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i, v, yy, mm, dd

    for i in range(size):
        v = arr_ptr[i]
        mm = v % 12 + 1  # 1-based index
        if not 1 < mm < 12:
            dd = 31
        else:
            dd = DAYS_IN_MONTH[mm]
            if mm == 2 and is_leap_year(v // 12 + EPOCH_YEAR):  # add back epoch
                dd += 1
        res_ptr[i] = dd
    return res

cdef inline np.ndarray dt64arr_days_bf_month(np.ndarray arr, str arr_unit=None):
    """Get the number of days between the 1st day of the
    np.ndarray[datetime64] and the 1st day of its month `<'int'>`.
    """
    # Cast to int64[M]
    arr = dt64arr_as_int64_M(arr, arr_unit, 0)

    # Compute days before month
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i, v, yy, mm, dd

    for i in range(size):
        v = arr_ptr[i]
        mm = v % 12 + 1  # 1-based index
        if mm <= 0:
            dd = 0
        elif mm == 2:
            dd = 31
        else:
            dd = DAYS_BR_MONTH[mm - 1]
            if is_leap_year(v // 12 + EPOCH_YEAR):  # add back epoch
                dd += 1
        res_ptr[i] = dd
    return res

# . conversion: int64
cdef inline np.ndarray dt64arr_fr_int64(np.npy_int64 val, str unit, np.npy_int64 size):
    """Create an ndarray[datetime64] from the passed in 
    integer and array size `<'ndarray[datetime64]'>`.

    Equivalent to:
    >>> np.array([val for _ in range(size)], dtype="datetime64[%s]" % unit)
    """
    # Validate
    if size < 1:
        raise ValueError("array size must be greater than 0.")
    unit = map_nptime_unit_int2str(map_nptime_unit_str2int(unit))

    # New array
    cdef:
        np.ndarray arr = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i
    
    for i in range(size):
        arr_ptr[i] = val
    return arr.astype("datetime64[%s]" % unit)

cdef inline np.ndarray dt64arr_as_int64(np.ndarray arr, str unit, str arr_unit=None, np.npy_int64 offset=0):
    """Cast np.ndarray[datetime64] to int64 according to the given
    'unit' resolution `<'ndarray[int64]'>`.

    Supported units: 'Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'.

    Equivalent to:
    >>> arr.astype(f"datetime64[unit]").astype("int64") + offset
    """
    cdef:
        Py_ssize_t size = str_len(unit)
        Py_UCS4 ch

    # Unit: 's', 'm', 'h', 'D'/'d', 'M', 'Y'
    if size == 1:
        ch = str_read(unit, 0)
        if ch == "s":
            return dt64arr_as_int64_s(arr, arr_unit, offset)
        if ch == "m":
            return dt64arr_as_int64_m(arr, arr_unit, offset)
        if ch == "h":
            return dt64arr_as_int64_h(arr, arr_unit, offset)
        if ch in ("D", "d"):
            return dt64arr_as_int64_D(arr, arr_unit, offset)
        if ch == "W":
            return dt64arr_as_int64_W(arr, arr_unit, offset)
        if ch == "M":
            return dt64arr_as_int64_M(arr, arr_unit, offset)
        if ch == "Q":
            return dt64arr_as_int64_Q(arr, arr_unit, offset)
        if ch == "Y":
            return dt64arr_as_int64_Y(arr, arr_unit, offset)

    # Unit: 'ns', 'us', 'ms'
    elif size == 2 and str_read(unit, 1) == "s":
        ch = str_read(unit, 0)
        if ch == "n":
            return dt64arr_as_int64_ns(arr, arr_unit, offset)
        if ch == "u":
            return dt64arr_as_int64_us(arr, arr_unit, offset)
        if ch == "m":
            return dt64arr_as_int64_ms(arr, arr_unit, offset)

    # Unit: 'min' for pandas compatibility
    elif size == 3 and unit == "min":
        return dt64arr_as_int64_m(arr, arr_unit, offset)

    # Unsupported unit
    raise ValueError(
        "cannot cast <'ndarray[%s]'> to int64 under '%s' resolution.\n"
        "Supported resolutions: ['Y', 'M', 'D', 'h', 'm', 's', 'ms', 'us', 'ns']." % (arr.dtype, unit)
    )

cdef inline np.ndarray dt64arr_as_int64_Y(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Cast np.ndarray[datetime64] to int64 under 'Y' (year)
    resolution `<'ndarray[int64]>`.
    
    Equivalent to:
    >>> arr.astype("datetime64[Y]").astype("int64") + offset
    """
    # Empty array
    if np.PyArray_SIZE(arr) == 0:
        return arr  # exit: empty array

    # Get time unit
    cdef int unit
    cdef int dtype = np.PyArray_TYPE(arr)
    if dtype != np.NPY_TYPES.NPY_DATETIME:
        if arr_unit is None:
            _raise_dt64arr_convert_arr_unit_error(arr, "int64 under 'Y' resolution")
        unit = map_nptime_unit_str2int(arr_unit)
        arr = arr_assure_int64(arr)
    elif arr_unit is not None:
        unit = map_nptime_unit_str2int(arr_unit)
    else:
        unit = get_arr_nptime_unit(arr)
    #: arr can only be only datetime64 or int64 afterwords

    # Fast-path: datetime64[Y]
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:
        if dtype == np.NPY_TYPES.NPY_DATETIME:
            arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
        return arr if offset == 0 else arr_add(arr, offset)
    
    # Fast-path: datetime64[M]
    cdef np.npy_int64 size = arr.shape[0]
    cdef np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
    cdef np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
    cdef np.npy_int64* arr_ptr
    cdef np.npy_int64 i
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:
        arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        for i in range(size):
            res_ptr[i] = arr_ptr[i] // 12 + offset
        return res
    
    # All the rest resolutions
    cdef np.npy_int64 n, n400, n100, n4, n1, year
    if unit not in (
        np.NPY_DATETIMEUNIT.NPY_FR_ns,
        np.NPY_DATETIMEUNIT.NPY_FR_us,
        np.NPY_DATETIMEUNIT.NPY_FR_ms,
        np.NPY_DATETIMEUNIT.NPY_FR_s,
        np.NPY_DATETIMEUNIT.NPY_FR_m,
        np.NPY_DATETIMEUNIT.NPY_FR_h,
        np.NPY_DATETIMEUNIT.NPY_FR_D,
        np.NPY_DATETIMEUNIT.NPY_FR_W,
    ):
        _raise_dt64arr_as_int64_unit_error("Y", unit)
    arr = dt64arr_as_int64_D(arr, arr_unit, 0)
    arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
    for i in range(size):
        # Convert to ordinal
        n = arr_ptr[i] + EPOCH_DAY - 1
        # Number of complete 400-year cycles
        n400 = n // 146_097
        n -= n400 * 146_097
        # Number of complete 100-year cycles within the 400-year cycle
        n100 = n // 36_524
        n -= n100 * 36_524
        # Number of complete 4-year cycles within the 100-year cycle
        n4 = n // 1_461
        n -= n4 * 1_461
        # Number of complete years within the 4-year cycle
        n1 = n // 365
        n -= n1 * 365
        # Compute the year
        year = n400 * 400 + n100 * 100 + n4 * 4 + n1 + 1
        # Adjust for end-of-cycle dates
        if n100 == 4 or n1 == 4:
            year -= 1
        res_ptr[i] = year - EPOCH_YEAR + offset
    return res

cdef inline np.ndarray dt64arr_as_int64_Q(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Cast np.ndarray[datetime64] to int64 under 'Q' (quarter)
    resolution `<'ndarray[int64]'>`.
    """
    # Cast to int64[M]
    arr = dt64arr_as_int64_M(arr, arr_unit, 0)

    # Compute quarter
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i, v
        
    for i in range(size):
        v = arr_ptr[i]
        res_ptr[i] = (v // 12 * 4) + (v % 12) // 3 + offset
    return res

cdef inline np.ndarray dt64arr_as_int64_M(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Cast np.ndarray[datetime64] to int64 under 'M' (month)
    resolution `<'ndarray[int64]'>`.
    
    Equivalent to:
    >>> arr.astype("datetime64[M]").astype("int64") + offset
    """
    # Empty array
    if np.PyArray_SIZE(arr) == 0:
        return arr  # exit: empty array

    # Get time unit
    cdef int unit
    cdef int dtype = np.PyArray_TYPE(arr)
    if dtype != np.NPY_TYPES.NPY_DATETIME:
        if arr_unit is None:
            _raise_dt64arr_convert_arr_unit_error(arr, "int64 under 'M' resolution")
        unit = map_nptime_unit_str2int(arr_unit)
        arr = arr_assure_int64(arr)
    elif arr_unit is not None:
        unit = map_nptime_unit_str2int(arr_unit)
    else:
        unit = get_arr_nptime_unit(arr)
    #: arr can only be only datetime64 or int64 afterwords

    # Fast-path: datetime64[M]
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:
        if dtype == np.NPY_TYPES.NPY_DATETIME:
            arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
        return arr if offset == 0 else arr_add(arr, offset)

    # Fast-path: datetime64[Y]
    cdef np.npy_int64 size = arr.shape[0]
    cdef np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
    cdef np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
    cdef np.npy_int64* arr_ptr
    cdef np.npy_int64 i
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:
        arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        for i in range(size):
            res_ptr[i] = arr_ptr[i] * 12 + offset
        return res

    # All the rest resolutions
    cdef np.npy_int64 n, n400, n100, n4, n1, year, month
    if unit not in (
        np.NPY_DATETIMEUNIT.NPY_FR_ns,
        np.NPY_DATETIMEUNIT.NPY_FR_us,
        np.NPY_DATETIMEUNIT.NPY_FR_ms,
        np.NPY_DATETIMEUNIT.NPY_FR_s,
        np.NPY_DATETIMEUNIT.NPY_FR_m,
        np.NPY_DATETIMEUNIT.NPY_FR_h,
        np.NPY_DATETIMEUNIT.NPY_FR_D,
        np.NPY_DATETIMEUNIT.NPY_FR_W,
    ):
        _raise_dt64arr_as_int64_unit_error("M", unit)
    arr = dt64arr_as_int64_D(arr, arr_unit, 0)
    arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
    for i in range(size):
        # Convert to ordinal
        n = arr_ptr[i] + EPOCH_DAY - 1
        # Number of complete 400-year cycles
        n400 = n // 146_097
        n -= n400 * 146_097
        # Number of complete 100-year cycles within the 400-year cycle
        n100 = n // 36_524
        n -= n100 * 36_524
        # Number of complete 4-year cycles within the 100-year cycle
        n4 = n // 1_461
        n -= n4 * 1_461
        # Number of complete years within the 4-year cycle
        n1 = n // 365
        n -= n1 * 365
        # Compute the year
        year = n400 * 400 + n100 * 100 + n4 * 4 + n1
        # Adjust for end-of-cycle dates
        if n100 == 4 or n1 == 4:
            month = 12
        else:
            year, month = year + 1, (n + 50) >> 5
            if days_bf_month(year, month) > n:
                month -= 1
        # Compute total months
        res_ptr[i] = (year - EPOCH_YEAR) * 12 + (month - 1) + offset
    return res

cdef inline np.ndarray dt64arr_as_int64_W(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Cast np.ndarray[datetime64] to int64 under 'W' (week)
    resolution `<'ndarray[int64]'>`.

    Equivalent to:
    >>> arr.astype("datetime64[W]").astype("int64") + offset
    """
    # Cast to int64[D]
    arr = dt64arr_as_int64_D(arr, arr_unit, 0)

    # Compute week
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i, v

    for i in range(size):
        v = arr_ptr[i]
        res_ptr[i] = v // 7 + offset
    return res

cdef inline np.ndarray dt64arr_as_int64_D(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Cast np.ndarray[datetime64] to int64 under 'D' (day)
    resolution `<'ndarray[int64]'>`.
    
    Equivalent to:
    >>> arr.astype("datetime64[D]").astype("int64") + offset
    """
    # Empty array
    if np.PyArray_SIZE(arr) == 0:
        return arr  # exit: empty array

    # Get time unit
    cdef int unit
    cdef int dtype = np.PyArray_TYPE(arr)
    if dtype != np.NPY_TYPES.NPY_DATETIME:
        if arr_unit is None:
            _raise_dt64arr_convert_arr_unit_error(arr, "int64 under 'D' resolution")
        unit = map_nptime_unit_str2int(arr_unit)
        arr = arr_assure_int64(arr)
    elif arr_unit is not None:
        unit = map_nptime_unit_str2int(arr_unit)
    else:
        unit = get_arr_nptime_unit(arr)
    #: arr can only be only datetime64 or int64 afterwords

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_floor_div(arr, NS_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_floor_div(arr, US_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_floor_div(arr, MS_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr_floor_div(arr, SS_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr_floor_div(arr, 1440, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr_floor_div(arr, 24, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr_assure_int64(arr) if offset == 0 else arr_add(arr, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return _dt64arr_W_as_int64_D(arr, 1, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _dt64arr_M_as_int64_D(arr, 1, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _dt64arr_Y_as_int64_D(arr, 1, offset)
    # . unsupported
    _raise_dt64arr_as_int64_unit_error("D", unit)

cdef inline np.ndarray dt64arr_as_int64_h(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Cast np.ndarray[datetime64] to int64 under 'h' (hour)
    resolution `<'ndarray[int64]'>`.
    
    Equivalent to:
    >>> arr.astype("datetime64[h]").astype("int64") + offset
    """
    # Empty array
    if np.PyArray_SIZE(arr) == 0:
        return arr  # exit: empty array

    # Get time unit
    cdef int unit
    cdef int dtype = np.PyArray_TYPE(arr)
    if dtype != np.NPY_TYPES.NPY_DATETIME:
        if arr_unit is None:
            _raise_dt64arr_convert_arr_unit_error(arr, "int64 under 'h' resolution")
        unit = map_nptime_unit_str2int(arr_unit)
        arr = arr_assure_int64(arr)
    elif arr_unit is not None:
        unit = map_nptime_unit_str2int(arr_unit)
    else:
        unit = get_arr_nptime_unit(arr)
    #: arr can only be only datetime64 or int64 afterwords

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_floor_div(arr, NS_HOUR, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_floor_div(arr, US_HOUR, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_floor_div(arr, MS_HOUR, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr_floor_div(arr, SS_HOUR, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr_floor_div(arr, 60, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr_assure_int64(arr) if offset == 0 else arr_add(arr, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr_mul(arr, 24, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return _dt64arr_W_as_int64_D(arr, 24, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _dt64arr_M_as_int64_D(arr, 24, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _dt64arr_Y_as_int64_D(arr, 24, offset)
    # . unsupported
    _raise_dt64arr_as_int64_unit_error("h", unit)

cdef inline np.ndarray dt64arr_as_int64_m(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Cast np.ndarray[datetime64] to int64 under 'm' (minute)
    resolution `<'ndarray[int64]'>`.
    
    Equivalent to:
    >>> arr.astype("datetime64[m]").astype("int64") + offset
    """
    # Empty array
    if np.PyArray_SIZE(arr) == 0:
        return arr  # exit: empty array

    # Get time unit
    cdef int unit
    cdef int dtype = np.PyArray_TYPE(arr)
    if dtype != np.NPY_TYPES.NPY_DATETIME:
        if arr_unit is None:
            _raise_dt64arr_convert_arr_unit_error(arr, "int64 under 'm' resolution")
        unit = map_nptime_unit_str2int(arr_unit)
        arr = arr_assure_int64(arr)
    elif arr_unit is not None:
        unit = map_nptime_unit_str2int(arr_unit)
    else:
        unit = get_arr_nptime_unit(arr)
    #: arr can only be only datetime64 or int64 afterwords

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_floor_div(arr, NS_MINUTE, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_floor_div(arr, US_MINUTE, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_floor_div(arr, MS_MINUTE, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr_floor_div(arr, SS_MINUTE, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr_assure_int64(arr) if offset == 0 else arr_add(arr, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr_mul(arr, 60, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr_mul(arr, 1_440, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return _dt64arr_W_as_int64_D(arr, 1_440, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _dt64arr_M_as_int64_D(arr, 1_440, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _dt64arr_Y_as_int64_D(arr, 1_440, offset)
    # . unsupported
    _raise_dt64arr_as_int64_unit_error("m", unit)

cdef inline np.ndarray dt64arr_as_int64_s(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Cast np.ndarray[datetime64] to int64 under 's' (second)
    resolution `<'ndarray[int64]'>`.
    
    Equivalent to:
    >>> arr.astype("datetime64[s]").astype("int64") + offset
    """
    # Empty array
    if np.PyArray_SIZE(arr) == 0:
        return arr  # exit: empty array

    # Get time unit
    cdef int unit
    cdef int dtype = np.PyArray_TYPE(arr)
    if dtype != np.NPY_TYPES.NPY_DATETIME:
        if arr_unit is None:
            _raise_dt64arr_convert_arr_unit_error(arr, "int64 under 's' resolution")
        unit = map_nptime_unit_str2int(arr_unit)
        arr = arr_assure_int64(arr)
    elif arr_unit is not None:
        unit = map_nptime_unit_str2int(arr_unit)
    else:
        unit = get_arr_nptime_unit(arr)
    #: arr can only be only datetime64 or int64 afterwords

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_floor_div(arr, NS_SECOND, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_floor_div(arr, US_SECOND, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_floor_div(arr, MS_SECOND, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr_assure_int64(arr) if offset == 0 else arr_add(arr, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr_mul(arr, SS_MINUTE, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr_mul(arr, SS_HOUR, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr_mul(arr, SS_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return _dt64arr_W_as_int64_D(arr, SS_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _dt64arr_M_as_int64_D(arr, SS_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _dt64arr_Y_as_int64_D(arr, SS_DAY, offset)
    # . unsupported
    _raise_dt64arr_as_int64_unit_error("s", unit)

cdef inline np.ndarray dt64arr_as_int64_ms(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Cast np.ndarray[datetime64] to int64 under 'ms' (millisecond)
    resolution `<'ndarray[int64]'>`.
    
    Equivalent to:
    >>> arr.astype("datetime64[ms]").astype("int64") + offset
    """
    # Empty array
    if np.PyArray_SIZE(arr) == 0:
        return arr  # exit: empty array

    # Get time unit
    cdef int unit
    cdef int dtype = np.PyArray_TYPE(arr)
    if dtype != np.NPY_TYPES.NPY_DATETIME:
        if arr_unit is None:
            _raise_dt64arr_convert_arr_unit_error(arr, "int64 under 'ms' resolution")
        unit = map_nptime_unit_str2int(arr_unit)
        arr = arr_assure_int64(arr)
    elif arr_unit is not None:
        unit = map_nptime_unit_str2int(arr_unit)
    else:
        unit = get_arr_nptime_unit(arr)
    #: arr can only be only datetime64 or int64 afterwords

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_floor_div(arr, NS_MILLISECOND, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_floor_div(arr, US_MILLISECOND, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_assure_int64(arr) if offset == 0 else arr_add(arr, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr_mul(arr, MS_SECOND, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr_mul(arr, MS_MINUTE, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr_mul(arr, MS_HOUR, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr_mul(arr, MS_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return _dt64arr_W_as_int64_D(arr, MS_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _dt64arr_M_as_int64_D(arr, MS_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _dt64arr_Y_as_int64_D(arr, MS_DAY, offset)
    # . unsupported
    _raise_dt64arr_as_int64_unit_error("ms", unit)

cdef inline np.ndarray dt64arr_as_int64_us(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Cast np.ndarray[datetime64] to int64 under 'us' (microsecond)
    resolution `<'ndarray[int64]'>`.
    
    Equivalent to:
    >>> arr.astype("datetime64[us]").astype("int64") + offset
    """
    # Empty array
    if np.PyArray_SIZE(arr) == 0:
        return arr  # exit: empty array

    # Get time unit
    cdef int unit
    cdef int dtype = np.PyArray_TYPE(arr)
    if dtype != np.NPY_TYPES.NPY_DATETIME:
        if arr_unit is None:
            _raise_dt64arr_convert_arr_unit_error(arr, "int64 under 'us' resolution")
        unit = map_nptime_unit_str2int(arr_unit)
        arr = arr_assure_int64(arr)
    elif arr_unit is not None:
        unit = map_nptime_unit_str2int(arr_unit)
    else:
        unit = get_arr_nptime_unit(arr)
    #: arr can only be only datetime64 or int64 afterwords

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_floor_div(arr, NS_MICROSECOND, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_assure_int64(arr) if offset == 0 else arr_add(arr, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_mul(arr, US_MILLISECOND, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr_mul(arr, US_SECOND, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr_mul(arr, US_MINUTE, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr_mul(arr, US_HOUR, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr_mul(arr, US_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return _dt64arr_W_as_int64_D(arr, US_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _dt64arr_M_as_int64_D(arr, US_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _dt64arr_Y_as_int64_D(arr, US_DAY, offset)
    # . unsupported
    _raise_dt64arr_as_int64_unit_error("us", unit)

cdef inline np.ndarray dt64arr_as_int64_ns(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Cast np.ndarray[datetime64] to int64 under 'ns' (nanosecond)
    resolution `<'ndarray[int64]'>`.
    
    Equivalent to:
    >>> arr.astype("datetime64[ns]").astype("int64") + offset
    """
    # Empty array
    if np.PyArray_SIZE(arr) == 0:
        return arr  # exit: empty array

    # Get time unit
    cdef int unit
    cdef int dtype = np.PyArray_TYPE(arr)
    if dtype != np.NPY_TYPES.NPY_DATETIME:
        if arr_unit is None:
            _raise_dt64arr_convert_arr_unit_error(arr, "int64 under 'ns' resolution")
        unit = map_nptime_unit_str2int(arr_unit)
        arr = arr_assure_int64(arr)
    elif arr_unit is not None:
        unit = map_nptime_unit_str2int(arr_unit)
    else:
        unit = get_arr_nptime_unit(arr)
    #: arr can only be only datetime64 or int64 afterwords

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_assure_int64(arr) if offset == 0 else arr_add(arr, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_mul(arr, NS_MICROSECOND, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_mul(arr, NS_MILLISECOND, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr_mul(arr, NS_SECOND, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr_mul(arr, NS_MINUTE, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr_mul(arr, NS_HOUR, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr_mul(arr, NS_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return _dt64arr_W_as_int64_D(arr, NS_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _dt64arr_M_as_int64_D(arr, NS_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _dt64arr_Y_as_int64_D(arr, NS_DAY, offset)
    # . unsupported
    _raise_dt64arr_as_int64_unit_error("ns", unit)

cdef inline np.ndarray _dt64arr_Y_as_int64_D(np.ndarray arr, np.npy_int64 factor=1, np.npy_int64 offset=0):
    """(internal) Cast np.ndarray[datetime64[Y]] to int64
    under 'D' (day) resolution `<'ndarray[int64]'>`.
    """
    arr = arr_assure_int64_like(arr)
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i, year, y_1, year_since_epoch, leaps

    for i in range(size):
        year_since_epoch = arr_ptr[i]
        year = year_since_epoch + EPOCH_YEAR
        # Compute leap years
        y_1 = year - 1
        leaps = (
            (y_1 // 4 - 1970 // 4)
            - (y_1 // 100 - 1970 // 100)
            + (y_1 // 400 - 1970 // 400)
        )
        # Compute total days
        res_ptr[i] = (year_since_epoch * 365 + leaps) * factor + offset
    return res

cdef inline np.ndarray _dt64arr_M_as_int64_D(np.ndarray arr, np.npy_int64 factor=1, np.npy_int64 offset=0):
    """(internal) Cast np.ndarray[datetime64[M]] to int64
    under 'D' (day) resolution `<'ndarray[int64]'>`.
    """
    arr = arr_assure_int64_like(arr)
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 i, v
        np.npy_int64 year, y_1, year_ep, leaps
        np.npy_int64 month, days_bf_mm

    for i in range(size):
        v = arr_ptr[i]
        year_ep = v // 12
        year = year_ep + EPOCH_YEAR
        month = v % 12 + 1
        # Compute leap years
        y_1 = year - 1
        leaps = (
            (y_1 // 4 - 1970 // 4)
            - (y_1 // 100 - 1970 // 100)
            + (y_1 // 400 - 1970 // 400)
        )
        # Compute total days
        res_ptr[i] = (year_ep * 365 + leaps + days_bf_month(year, month)) * factor + offset
    return res

cdef inline np.ndarray _dt64arr_W_as_int64_D(np.ndarray arr, np.npy_int64 factor=1, np.npy_int64 offset=0):
    """(internal) Cast np.ndarray[datetime64[W]] to int64
    under 'D' (day) resolution `<'ndarray[int64]'>`.
    """
    return arr_mul(arr, 7 * factor, offset)

cdef inline np.ndarray dt64arr_as_iso_W(np.ndarray arr, int weekday, str arr_unit=None, np.npy_int64 offset=0):
    """Cast np.ndarray[datetime64] to int64 with 'W' (week)
    resolution, aligned to the specified ISO 'weekday' `<'ndarray[int64]'>`.

    NumPy aligns datetime64[W] to Thursday (the weekday of 1970-01-01).
    This function allows specifying the ISO 'weekday' (1=Monday, 7=Sunday)
    for alignment.

    For example: if 'weekday=1', the result represents the Monday-aligned
    weeks since EPOCH (1970-01-01).
    """
    # Cast to int64[D]
    arr = dt64arr_as_int64_D(arr, arr_unit, 0)

    # Compute week
    cdef:
        np.npy_int64 size = arr.shape[0]
        np.ndarray res = np.PyArray_EMPTY(1, [size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr_ptr = <np.npy_int64*> np.PyArray_DATA(arr)
        np.npy_int64 wkd_off = 4 - min(max(weekday, 1), 7)
        np.npy_int64 i

    for i in range(size):
        res_ptr[i] = (arr_ptr[i] + wkd_off) // 7 + offset
    return res

cdef inline np.ndarray dt64arr_to_ordinal(np.ndarray arr, str arr_unit=None):
    """Convert np.ndarray[datetime64] to proleptic Gregorian 
    ordinals `<'ndarray[int64]'>`.

    '0001-01-01' is day 1 (ordinal=1).
    """
    return dt64arr_as_int64_D(arr, arr_unit, EPOCH_DAY)

# . conversion: float64
cdef inline np.ndarray dt64arr_to_ts(np.ndarray arr, str arr_unit=None):
    """Convert np.ndarray[datetime64] to timestamps `<'ndarray[float64]'>`.

    Fractional seconds are rounded to the nearest microsecond.
    """
    # Empty array
    if np.PyArray_SIZE(arr) == 0:
        return arr  # exit: empty array

    # Convert to seconds[float64]
    arr = dt64arr_as_int64_us(arr, arr_unit, 0)  # int64[us]
    return arr_div(arr, US_SECOND)  # float64[s]

# . conversion: unit
cdef inline np.ndarray dt64arr_as_unit(np.ndarray arr, str unit, str arr_unit=None, bint limit=False):
    """Convert np.ndarray[datetime64] to the specified unit `<'ndarray[datetime64]'>`.

    - 'limit=False': supports all datetime64 native conversions.
    - 'limit=True': only supports conversion between ['s', 'ms', 'us', 'ns'].

    Equivalent to:
    >>> arr.astype(f"datetime64[unit]")
    """
    # Validate array
    if np.PyArray_SIZE(arr) == 0:
        return arr  # exit: empty array

    # Get time unit
    cdef int my_unit, to_unit
    cdef int dtype = np.PyArray_TYPE(arr)
    if dtype != np.NPY_TYPES.NPY_DATETIME:
        if arr_unit is None:
            _raise_dt64arr_convert_arr_unit_error(arr, "int64")
        my_unit = map_nptime_unit_str2int(arr_unit)
        arr_unit = map_nptime_unit_int2str(my_unit)
        arr = arr.astype("datetime64[%s]" % arr_unit)
    elif arr_unit is not None:
        my_unit = map_nptime_unit_str2int(arr_unit)
    else:
        my_unit = get_arr_nptime_unit(arr)
    to_unit = map_nptime_unit_str2int(unit)

    # Check limit
    if limit:
        if to_unit not in (
            np.NPY_DATETIMEUNIT.NPY_FR_ns,
            np.NPY_DATETIMEUNIT.NPY_FR_us,
            np.NPY_DATETIMEUNIT.NPY_FR_ms,
            np.NPY_DATETIMEUNIT.NPY_FR_s,
        ) or my_unit not in (
            np.NPY_DATETIMEUNIT.NPY_FR_ns,
            np.NPY_DATETIMEUNIT.NPY_FR_us,
            np.NPY_DATETIMEUNIT.NPY_FR_ms,
            np.NPY_DATETIMEUNIT.NPY_FR_s,
        ):
            raise ValueError(
                "cannot convert ndarray from datetime64[%s] to datetime64[%s].\n"
                "Conversion limits to datetime units between ['s', 'ms', 'us', 'ns']." 
                % (map_nptime_unit_int2str(my_unit), map_nptime_unit_int2str(to_unit))
            )

    # Fast-path: same unit
    if my_unit == to_unit:
        return arr

    # To nanosecond
    cdef bint ns_safe
    if to_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        # check 'ns' overflow
        ns_safe = is_dt64arr_ns_safe(arr, arr_unit)
        # my_unit [us] -> to_unit [ns]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
            if not ns_safe:
                return arr
            return arr.astype("datetime64[ns]")
        # my_unit [M, Y] -> to_unit [ns]
        elif my_unit in (
            np.NPY_DATETIMEUNIT.NPY_FR_M,
            np.NPY_DATETIMEUNIT.NPY_FR_Y,
        ):
            if not ns_safe:
                arr = dt64arr_as_int64_us(arr, arr_unit, 0)
                return arr.astype("datetime64[us]")
            arr = dt64arr_as_int64_ns(arr, arr_unit, 0)
            return arr.astype("datetime64[ns]")
        # my_unit [rest] -> to_unit [ns]
        if not ns_safe:
            return arr.astype("datetime64[us]")
        return arr.astype("datetime64[ns]")

    # To microsecond
    if to_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        # my_unit [ns] -> to_unit [us]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_floor_div(arr, NS_MICROSECOND)
        # my_unit [M, Y] -> to_unit [us]
        elif my_unit in (
            np.NPY_DATETIMEUNIT.NPY_FR_M,
            np.NPY_DATETIMEUNIT.NPY_FR_Y,
        ):
            arr = dt64arr_as_int64_us(arr, arr_unit, 0)
        # my_unit [rest] -> to_unit [us]
        return arr.astype("datetime64[us]")

    # To millisecond
    if to_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        # my_unit [ns] -> to_unit [ms]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_floor_div(arr, NS_MILLISECOND)
        # my_unit [M, Y] -> to_unit [ms]
        elif my_unit in (
            np.NPY_DATETIMEUNIT.NPY_FR_M,
            np.NPY_DATETIMEUNIT.NPY_FR_Y,
        ):
            arr = dt64arr_as_int64_ms(arr, arr_unit, 0)
        # my_unit [rest] -> to_unit [ms]
        return arr.astype("datetime64[ms]")

    # To second
    if to_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        # my_unit [ns] -> to_unit [s]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_floor_div(arr, NS_SECOND)
        # my_unit [M, Y] -> to_unit [s]
        elif my_unit in (
            np.NPY_DATETIMEUNIT.NPY_FR_M,
            np.NPY_DATETIMEUNIT.NPY_FR_Y,
        ):
            arr = dt64arr_as_int64_s(arr, arr_unit, 0)
        # my_unit [rest] -> to_unit [s]
        return arr.astype("datetime64[s]")

    # To minute
    if to_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        # my_unit [ns] -> to_unit [s]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_floor_div(arr, NS_MINUTE)
        # my_unit [M, Y] -> to_unit [m]
        elif my_unit in (
            np.NPY_DATETIMEUNIT.NPY_FR_M,
            np.NPY_DATETIMEUNIT.NPY_FR_Y,
        ):
            arr = dt64arr_as_int64_m(arr, arr_unit, 0)
        # my_unit [rest] -> to_unit [m]
        return arr.astype("datetime64[m]")

    # To hour
    if to_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        # my_unit [ns] -> to_unit [h]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_floor_div(arr, NS_HOUR)
        # my_unit [M, Y] -> to_unit [h]
        elif my_unit in (
            np.NPY_DATETIMEUNIT.NPY_FR_M,
            np.NPY_DATETIMEUNIT.NPY_FR_Y,
        ):
            arr = dt64arr_as_int64_h(arr, arr_unit, 0)
        # my_unit [rest] -> to_unit [h]
        return arr.astype("datetime64[h]")

    # To day
    if to_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        # my_unit [ns] -> to_unit [D]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_floor_div(arr, NS_DAY)
        # my_unit [M, Y] -> to_unit [D]
        elif my_unit in (
            np.NPY_DATETIMEUNIT.NPY_FR_M,
            np.NPY_DATETIMEUNIT.NPY_FR_Y,
        ):
            arr = dt64arr_as_int64_D(arr, arr_unit, 0)
        # my_unit [rest] -> to_unit [D]
        return arr.astype("datetime64[D]")

    # To week
    if to_unit == np.NPY_DATETIMEUNIT.NPY_FR_W:
        # my_unit [ns] -> to_unit [W]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if not is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_floor_div(arr, NS_DAY * 7)
        # my_unit [M, Y] -> to_unit [W]
        elif my_unit in (
            np.NPY_DATETIMEUNIT.NPY_FR_M,
            np.NPY_DATETIMEUNIT.NPY_FR_Y,
        ):
            arr = dt64arr_as_int64_W(arr, arr_unit, 0)
        # my_unit [rest] -> to_unit [D]
        return arr.astype("datetime64[W]")

    # To month
    if to_unit == np.NPY_DATETIMEUNIT.NPY_FR_M:
        # my_unit [ns, Y] -> to_unit [M]
        if my_unit in (
            np.NPY_DATETIMEUNIT.NPY_FR_ns,
            np.NPY_DATETIMEUNIT.NPY_FR_Y,
        ):
            arr = dt64arr_as_int64_M(arr, arr_unit, 0)
        # my_unit [rest] -> to_unit [M]
        return arr.astype("datetime64[M]")

    # To year
    if to_unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:
        arr = dt64arr_as_int64_Y(arr, arr_unit, 0)
        return arr.astype("datetime64[Y]")

    # Rest
    return arr.astype("datetime64[%s]" % map_nptime_unit_int2str(to_unit))

# . arithmetic
cdef inline np.ndarray dt64arr_round(np.ndarray arr, str unit, str arr_unit=None):
    """Perform round operation on the np.ndarray[datetime64] to the 
    specified unit `<'ndarray[datetime64]'>`.

    - Supported array resolution: 'ns', 'us', 'ms', 's', 'm', 'h', 'D'
    - Supported units: 'ns', 'us', 'ms', 's', 'm', 'h', 'D'.
    """
    # Validate array
    if np.PyArray_SIZE(arr) == 0:
        return arr  # exit: empty array

    # Get time unit
    cdef int my_unit, to_unit
    cdef int dtype = np.PyArray_TYPE(arr)
    if dtype != np.NPY_TYPES.NPY_DATETIME:
        if arr_unit is None:
            _raise_dt64arr_convert_arr_unit_error(arr, "int64")
        my_unit = map_nptime_unit_str2int(arr_unit)
    elif arr_unit is not None:
        my_unit = map_nptime_unit_str2int(arr_unit)
    else:
        my_unit = get_arr_nptime_unit(arr)
    to_unit = map_nptime_unit_str2int(unit)

    # Fast-path
    #: 1. same or my_unit is lower than to_unit (ms is lower than us)
    #: 2. to nanosecond (ns is the highest supported unit)
    if my_unit <= to_unit or to_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        if dtype == np.NPY_TYPES.NPY_DATETIME:
            return arr
        return arr.astype("datetime64[%s]" % map_nptime_unit_int2str(my_unit))

    # To microsecond
    if to_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        # my_unit [ns] -> to_unit [us]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_round_to_mul(arr, NS_MICROSECOND)
                to_dtype = "datetime64[ns]"
            else:
                arr = arr_round_div(arr, NS_MICROSECOND)
                to_dtype = "datetime64[us]"
        # my_unit [D...us] are lower units, which is covered 
        # by fast-path. Other units are not supported.
        else:
            _raise_dt64arr_rcl_unsupport_unit_error("round", to_unit, my_unit)

    # To millisecond
    elif to_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        # my_unit [ns] -> to_unit [ms]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_round_to_mul(arr, NS_MILLISECOND)
                to_dtype = "datetime64[ns]"
            else:
                arr = arr_round_to_mul(arr, NS_MILLISECOND, US_MILLISECOND)
                to_dtype = "datetime64[us]"
        # my_unit [us] -> to_unit [ms]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_round_to_mul(arr, US_MILLISECOND)
            to_dtype = "datetime64[us]"
        # my_unit [D...ms] are lower units, which is covered
        # by fast-path. Other units are not supported.
        else:
            _raise_dt64arr_rcl_unsupport_unit_error("round", to_unit, my_unit)

    # To second
    elif to_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        # my_unit [ns] -> to_unit [s]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_round_to_mul(arr, NS_SECOND)
                to_dtype = "datetime64[ns]"
            else:
                arr = arr_round_to_mul(arr, NS_SECOND, US_SECOND)
                to_dtype = "datetime64[us]"
        # my_unit [us] -> to_unit [s]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_round_to_mul(arr, US_SECOND)
            to_dtype = "datetime64[us]"
        # my_unit [ms] -> to_unit [s]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_round_to_mul(arr, MS_SECOND)
            to_dtype = "datetime64[ms]"
        # my_unit [D...s] are lower units, which is covered
        # by fast-path. Other units are not supported.
        else:
            _raise_dt64arr_rcl_unsupport_unit_error("round", to_unit, my_unit)

    # To minute
    elif to_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        # my_unit [ns] -> to_unit [m]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_round_to_mul(arr, NS_MINUTE)
                to_dtype = "datetime64[ns]"
            else:
                arr = arr_round_to_mul(arr, NS_MINUTE, US_MINUTE)
                to_dtype = "datetime64[us]"
        # my_unit [us] -> to_unit [m]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_round_to_mul(arr, US_MINUTE)
            to_dtype = "datetime64[us]"
        # my_unit [ms] -> to_unit [m]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_round_to_mul(arr, MS_MINUTE)
            to_dtype = "datetime64[ms]"
        # my_unit [s] -> to_unit [m]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
            arr = arr_round_to_mul(arr, SS_MINUTE)
            to_dtype = "datetime64[s]"
        # my_unit [D...m] are lower units, which is covered
        # by fast-path. Other units are not supported.
        else:
            _raise_dt64arr_rcl_unsupport_unit_error("round", to_unit, my_unit)
    
    # To hour
    elif to_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        # my_unit [ns] -> to_unit [h]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_round_to_mul(arr, NS_HOUR)
                to_dtype = "datetime64[ns]"
            else:
                arr = arr_round_to_mul(arr, NS_HOUR, US_HOUR)
                to_dtype = "datetime64[us]"
        # my_unit [us] -> to_unit [h]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_round_to_mul(arr, US_HOUR)
            to_dtype = "datetime64[us]"
        # my_unit [ms] -> to_unit [h]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_round_to_mul(arr, MS_HOUR)
            to_dtype = "datetime64[ms]"
        # my_unit [s] -> to_unit [h]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
            arr = arr_round_to_mul(arr, SS_HOUR)
            to_dtype = "datetime64[s]"
        # my_unit [m] -> to_unit [h]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
            arr = arr_round_to_mul(arr, 60)
            to_dtype = "datetime64[m]"
        # my_unit [D...h] are lower units, which is covered
        # by fast-path. Other units are not supported.
        else:
            _raise_dt64arr_rcl_unsupport_unit_error("round", to_unit, my_unit)

    # To day
    elif to_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        # my_unit [ns] -> to_unit [D]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_round_to_mul(arr, NS_DAY)
                to_dtype = "datetime64[ns]"
            else:
                arr = arr_round_to_mul(arr, NS_DAY, US_DAY)
                to_dtype = "datetime64[us]"
        # my_unit [us] -> to_unit [D]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_round_to_mul(arr, US_DAY)
            to_dtype = "datetime64[us]"
        # my_unit [ms] -> to_unit [D]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_round_to_mul(arr, MS_DAY)
            to_dtype = "datetime64[ms]"
        # my_unit [s] -> to_unit [D]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
            arr = arr_round_to_mul(arr, SS_DAY)
            to_dtype = "datetime64[s]"
        # my_unit [m] -> to_unit [D]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
            arr = arr_round_to_mul(arr, 1440)
            to_dtype = "datetime64[m]"
        # my_unit [h] -> to_unit [D]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
            arr = arr_round_to_mul(arr, 24)
            to_dtype = "datetime64[h]"
        # my_unit [D] are lower units, which is covered
        # by fast-path. Other units are not supported.
        else:
            _raise_dt64arr_rcl_unsupport_unit_error("round", to_unit, my_unit)

    # Invalid unit
    else:
        _raise_dt64arr_rcl_unsupport_unit_error("round", to_unit, my_unit)

    # Convert to datetime64
    return arr.astype(to_dtype)

cdef inline np.ndarray dt64arr_ceil(np.ndarray arr, str unit, str arr_unit=None):
    """Perform ceil operation on the np.ndarray[datetime64] to the 
    specified unit `<'ndarray[datetime64]'>`.

    - Supported units: 'ns', 'us', 'ms', 's', 'm', 'h', 'D'.
    - Supported array resolution: 'ns', 'us', 'ms', 's', 'm', 'h', 'D'
    """
    # Validate array
    if np.PyArray_SIZE(arr) == 0:
        return arr  # exit: empty array

    # Get time unit
    cdef int my_unit, to_unit
    cdef int dtype = np.PyArray_TYPE(arr)
    if dtype != np.NPY_TYPES.NPY_DATETIME:
        if arr_unit is None:
            _raise_dt64arr_convert_arr_unit_error(arr, "int64")
        my_unit = map_nptime_unit_str2int(arr_unit)
    elif arr_unit is not None:
        my_unit = map_nptime_unit_str2int(arr_unit)
    else:
        my_unit = get_arr_nptime_unit(arr)
    to_unit = map_nptime_unit_str2int(unit)

    # Fast-path
    #: 1. same or my_unit is lower than to_unit (ms is lower than us)
    #: 2. to nanosecond (ns is the highest supported unit)
    if my_unit <= to_unit or to_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        if dtype == np.NPY_TYPES.NPY_DATETIME:
            return arr
        return arr.astype("datetime64[%s]" % map_nptime_unit_int2str(my_unit))

    # To microsecond
    if to_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        # my_unit [ns] -> to_unit [us]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_ceil_to_mul(arr, NS_MICROSECOND)
                to_dtype = "datetime64[ns]"
            else:
                arr = arr_ceil_div(arr, NS_MICROSECOND)
                to_dtype = "datetime64[us]"
        # my_unit [D...us] are lower units, which is covered 
        # by fast-path. Other units are not supported.
        else:
            _raise_dt64arr_rcl_unsupport_unit_error("ceil", to_unit, my_unit)

    # To millisecond
    elif to_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        # my_unit [ns] -> to_unit [ms]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_ceil_to_mul(arr, NS_MILLISECOND)
                to_dtype = "datetime64[ns]"
            else:
                arr = arr_ceil_to_mul(arr, NS_MILLISECOND, US_MILLISECOND)
                to_dtype = "datetime64[us]"
        # my_unit [us] -> to_unit [ms]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_ceil_to_mul(arr, US_MILLISECOND)
            to_dtype = "datetime64[us]"
        # my_unit [D...ms] are lower units, which is covered
        # by fast-path. Other units are not supported.
        else:
            _raise_dt64arr_rcl_unsupport_unit_error("ceil", to_unit, my_unit)

    # To second
    elif to_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        # my_unit [ns] -> to_unit [s]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_ceil_to_mul(arr, NS_SECOND)
                to_dtype = "datetime64[ns]"
            else:
                arr = arr_ceil_to_mul(arr, NS_SECOND, US_SECOND)
                to_dtype = "datetime64[us]"
        # my_unit [us] -> to_unit [s]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_ceil_to_mul(arr, US_SECOND)
            to_dtype = "datetime64[us]"
        # my_unit [ms] -> to_unit [s]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_ceil_to_mul(arr, MS_SECOND)
            to_dtype = "datetime64[ms]"
        # my_unit [D...s] are lower units, which is covered
        # by fast-path. Other units are not supported.
        else:
            _raise_dt64arr_rcl_unsupport_unit_error("ceil", to_unit, my_unit)

    # To minute
    elif to_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        # my_unit [ns] -> to_unit [m]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_ceil_to_mul(arr, NS_MINUTE)
                to_dtype = "datetime64[ns]"
            else:
                arr = arr_ceil_to_mul(arr, NS_MINUTE, US_MINUTE)
                to_dtype = "datetime64[us]"
        # my_unit [us] -> to_unit [m]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_ceil_to_mul(arr, US_MINUTE)
            to_dtype = "datetime64[us]"
        # my_unit [ms] -> to_unit [m]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_ceil_to_mul(arr, MS_MINUTE)
            to_dtype = "datetime64[ms]"
        # my_unit [s] -> to_unit [m]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
            arr = arr_ceil_to_mul(arr, SS_MINUTE)
            to_dtype = "datetime64[s]"
        # my_unit [D...m] are lower units, which is covered
        # by fast-path. Other units are not supported.
        else:
            _raise_dt64arr_rcl_unsupport_unit_error("ceil", to_unit, my_unit)
    
    # To hour
    elif to_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        # my_unit [ns] -> to_unit [h]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_ceil_to_mul(arr, NS_HOUR)
                to_dtype = "datetime64[ns]"
            else:
                arr = arr_ceil_to_mul(arr, NS_HOUR, US_HOUR)
                to_dtype = "datetime64[us]"
        # my_unit [us] -> to_unit [h]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_ceil_to_mul(arr, US_HOUR)
            to_dtype = "datetime64[us]"
        # my_unit [ms] -> to_unit [h]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_ceil_to_mul(arr, MS_HOUR)
            to_dtype = "datetime64[ms]"
        # my_unit [s] -> to_unit [h]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
            arr = arr_ceil_to_mul(arr, SS_HOUR)
            to_dtype = "datetime64[s]"
        # my_unit [m] -> to_unit [h]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
            arr = arr_ceil_to_mul(arr, 60)
            to_dtype = "datetime64[m]"
        # my_unit [D...h] are lower units, which is covered
        # by fast-path. Other units are not supported.
        else:
            _raise_dt64arr_rcl_unsupport_unit_error("ceil", to_unit, my_unit)

    # To day
    elif to_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        # my_unit [ns] -> to_unit [D]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_ceil_to_mul(arr, NS_DAY)
                to_dtype = "datetime64[ns]"
            else:
                arr = arr_ceil_to_mul(arr, NS_DAY, US_DAY)
                to_dtype = "datetime64[us]"
        # my_unit [us] -> to_unit [D]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_ceil_to_mul(arr, US_DAY)
            to_dtype = "datetime64[us]"
        # my_unit [ms] -> to_unit [D]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_ceil_to_mul(arr, MS_DAY)
            to_dtype = "datetime64[ms]"
        # my_unit [s] -> to_unit [D]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
            arr = arr_ceil_to_mul(arr, SS_DAY)
            to_dtype = "datetime64[s]"
        # my_unit [m] -> to_unit [D]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
            arr = arr_ceil_to_mul(arr, 1440)
            to_dtype = "datetime64[m]"
        # my_unit [h] -> to_unit [D]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
            arr = arr_ceil_to_mul(arr, 24)
            to_dtype = "datetime64[h]"
        # my_unit [D] are lower units, which is covered
        # by fast-path. Other units are not supported.
        else:
            _raise_dt64arr_rcl_unsupport_unit_error("ceil", to_unit, my_unit)

    # Invalid unit
    else:
        _raise_dt64arr_rcl_unsupport_unit_error("ceil", to_unit, my_unit)

    # Convert to datetime64
    return arr.astype(to_dtype)

cdef inline np.ndarray dt64arr_floor(np.ndarray arr, str unit, str arr_unit=None):
    """Perform floor operation on the np.ndarray[datetime64] to the 
    specified unit `<'ndarray[datetime64]'>`.

    - Supported units: 'ns', 'us', 'ms', 's', 'm', 'h', 'D'.
    - Supported array resolution: 'ns', 'us', 'ms', 's', 'm', 'h', 'D'
    """
    # Validate array
    if np.PyArray_SIZE(arr) == 0:
        return arr  # exit: empty array

    # Get time unit
    cdef int my_unit, to_unit
    cdef int dtype = np.PyArray_TYPE(arr)
    if dtype != np.NPY_TYPES.NPY_DATETIME:
        if arr_unit is None:
            _raise_dt64arr_convert_arr_unit_error(arr, "int64")
        my_unit = map_nptime_unit_str2int(arr_unit)
    elif arr_unit is not None:
        my_unit = map_nptime_unit_str2int(arr_unit)
    else:
        my_unit = get_arr_nptime_unit(arr)
    to_unit = map_nptime_unit_str2int(unit)

    # Fast-path
    #: 1. same or my_unit is lower than to_unit (ms is lower than us)
    #: 2. to nanosecond (ns is the highest supported unit)
    if my_unit <= to_unit or to_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        if dtype == np.NPY_TYPES.NPY_DATETIME:
            return arr
        return arr.astype("datetime64[%s]" % map_nptime_unit_int2str(my_unit))

    # To microsecond
    if to_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        # my_unit [ns] -> to_unit [us]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_floor_to_mul(arr, NS_MICROSECOND)
                to_dtype = "datetime64[ns]"
            else:
                arr = arr_floor_div(arr, NS_MICROSECOND)
                to_dtype = "datetime64[us]"
        # my_unit [D...us] are lower units, which is covered 
        # by fast-path. Other units are not supported.
        else:
            _raise_dt64arr_rcl_unsupport_unit_error("floor", to_unit, my_unit)

    # To millisecond
    elif to_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        # my_unit [ns] -> to_unit [ms]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_floor_to_mul(arr, NS_MILLISECOND)
                to_dtype = "datetime64[ns]"
            else:
                arr = arr_floor_to_mul(arr, NS_MILLISECOND, US_MILLISECOND)
                to_dtype = "datetime64[us]"
        # my_unit [us] -> to_unit [ms]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_floor_to_mul(arr, US_MILLISECOND)
            to_dtype = "datetime64[us]"
        # my_unit [D...ms] are lower units, which is covered
        # by fast-path. Other units are not supported.
        else:
            _raise_dt64arr_rcl_unsupport_unit_error("floor", to_unit, my_unit)

    # To second
    elif to_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        # my_unit [ns] -> to_unit [s]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_floor_to_mul(arr, NS_SECOND)
                to_dtype = "datetime64[ns]"
            else:
                arr = arr_floor_to_mul(arr, NS_SECOND, US_SECOND)
                to_dtype = "datetime64[us]"
        # my_unit [us] -> to_unit [s]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_floor_to_mul(arr, US_SECOND)
            to_dtype = "datetime64[us]"
        # my_unit [ms] -> to_unit [s]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_floor_to_mul(arr, MS_SECOND)
            to_dtype = "datetime64[ms]"
        # my_unit [D...s] are lower units, which is covered
        # by fast-path. Other units are not supported.
        else:
            _raise_dt64arr_rcl_unsupport_unit_error("floor", to_unit, my_unit)

    # To minute
    elif to_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        # my_unit [ns] -> to_unit [m]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_floor_to_mul(arr, NS_MINUTE)
                to_dtype = "datetime64[ns]"
            else:
                arr = arr_floor_to_mul(arr, NS_MINUTE, US_MINUTE)
                to_dtype = "datetime64[us]"
        # my_unit [us] -> to_unit [m]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_floor_to_mul(arr, US_MINUTE)
            to_dtype = "datetime64[us]"
        # my_unit [ms] -> to_unit [m]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_floor_to_mul(arr, MS_MINUTE)
            to_dtype = "datetime64[ms]"
        # my_unit [s] -> to_unit [m]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
            arr = arr_floor_to_mul(arr, SS_MINUTE)
            to_dtype = "datetime64[s]"
        # my_unit [D...m] are lower units, which is covered
        # by fast-path. Other units are not supported.
        else:
            _raise_dt64arr_rcl_unsupport_unit_error("floor", to_unit, my_unit)
    
    # To hour
    elif to_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        # my_unit [ns] -> to_unit [h]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_floor_to_mul(arr, NS_HOUR)
                to_dtype = "datetime64[ns]"
            else:
                arr = arr_floor_to_mul(arr, NS_HOUR, US_HOUR)
                to_dtype = "datetime64[us]"
        # my_unit [us] -> to_unit [h]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_floor_to_mul(arr, US_HOUR)
            to_dtype = "datetime64[us]"
        # my_unit [ms] -> to_unit [h]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_floor_to_mul(arr, MS_HOUR)
            to_dtype = "datetime64[ms]"
        # my_unit [s] -> to_unit [h]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
            arr = arr_floor_to_mul(arr, SS_HOUR)
            to_dtype = "datetime64[s]"
        # my_unit [m] -> to_unit [h]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
            arr = arr_floor_to_mul(arr, 60)
            to_dtype = "datetime64[m]"
        # my_unit [D...h] are lower units, which is covered
        # by fast-path. Other units are not supported.
        else:
            _raise_dt64arr_rcl_unsupport_unit_error("floor", to_unit, my_unit)

    # To day
    elif to_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        # my_unit [ns] -> to_unit [D]
        if my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
            if is_dt64arr_ns_safe(arr, "ns"):
                arr = arr_floor_to_mul(arr, NS_DAY)
                to_dtype = "datetime64[ns]"
            else:
                arr = arr_floor_to_mul(arr, NS_DAY, US_DAY)
                to_dtype = "datetime64[us]"
        # my_unit [us] -> to_unit [D]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
            arr = arr_floor_to_mul(arr, US_DAY)
            to_dtype = "datetime64[us]"
        # my_unit [ms] -> to_unit [D]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
            arr = arr_floor_to_mul(arr, MS_DAY)
            to_dtype = "datetime64[ms]"
        # my_unit [s] -> to_unit [D]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
            arr = arr_floor_to_mul(arr, SS_DAY)
            to_dtype = "datetime64[s]"
        # my_unit [m] -> to_unit [D]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
            arr = arr_floor_to_mul(arr, 1440)
            to_dtype = "datetime64[m]"
        # my_unit [h] -> to_unit [D]
        elif my_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
            arr = arr_floor_to_mul(arr, 24)
            to_dtype = "datetime64[h]"
        # my_unit [D] are lower units, which is covered
        # by fast-path. Other units are not supported.
        else:
            _raise_dt64arr_rcl_unsupport_unit_error("floor", to_unit, my_unit)

    # Invalid unit
    else:
        _raise_dt64arr_rcl_unsupport_unit_error("floor", to_unit, my_unit)

    # Convert to datetime64
    return arr.astype(to_dtype)

# . comparison
cdef inline np.ndarray dt64arr_find_closest(np.ndarray arr1, np.ndarray arr2):
    """For each element in 'arr1', find the closest values in 'arr2' `<'ndarray[int64]'>`."""
    # Validate
    cdef np.npy_int64 arr1_size = arr1.shape[0]
    cdef np.npy_int64 arr2_size = arr2.shape[0]
    if arr1_size <= 0 or arr2_size <= 0:
        raise ValueError("For comparision, array size must be > 0.")
    arr1 = arr_assure_int64_like(arr1)
    arr2 = arr_assure_int64_like(arr2)

    # Sort arr2
    cdef np.ndarray arrS = np.PyArray_Copy(arr2)
    np.PyArray_Sort(arrS, 0, np.NPY_SORTKIND.NPY_QUICKSORT)

    # Prepare result and pointers
    cdef:
        np.ndarray res = np.PyArray_EMPTY(1, [arr1_size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr1_ptr = <np.npy_int64*> np.PyArray_DATA(arr1)
        np.npy_int64* arrS_ptr = <np.npy_int64*> np.PyArray_DATA(arrS)
        np.npy_int64 i, targ, left, right, mid
        np.npy_int64 cur_num, cls_num, min_diff, cur_diff

    # Find closest for elements in 'arr1'
    for i in range(arr1_size):
        targ, left, right = arr1_ptr[i], 0, arr2_size - 1
        cls_num = arrS_ptr[left]
        min_diff = targ - cls_num if cls_num < targ else cls_num - targ
        # Binary search
        while left <= right:
            mid = (left + right) // 2
            cur_num = arrS_ptr[mid]
            cur_diff = targ - cur_num if cur_num < targ else cur_num - targ

            # Update closest number
            if cur_diff < min_diff:
                cls_num, min_diff = cur_num, cur_diff
            elif cur_diff == min_diff:
                # Choose the smaller number in case of a tie
                if cur_num < cls_num:
                    cls_num = cur_num

            if cur_num < targ:
                left = mid + 1
            elif cur_num > targ:
                right = mid - 1
            else:
                # Exact match found
                cls_num = cur_num
                break  # exit loop

        res_ptr[i] = cls_num

    # Return result
    return res

cdef inline np.ndarray dt64arr_find_farthest(np.ndarray arr1, np.ndarray arr2):
    """For each element in 'arr1', find the farthest values in 'arr2' `<'ndarray[int64]'>`."""
    # Validate
    cdef np.npy_int64 arr1_size = arr1.shape[0]
    cdef np.npy_int64 arr2_size = arr2.shape[0]
    if arr1_size <= 0 or arr2_size <= 0:
        raise ValueError("For comparision, array size must be > 0.")
    arr1 = arr_assure_int64_like(arr1)
    arr2 = arr_assure_int64_like(arr2)

    # Prepare result and pointers
    cdef:
        np.ndarray res = np.PyArray_EMPTY(1, [arr1_size], np.NPY_TYPES.NPY_INT64, 0)
        np.npy_int64* res_ptr = <np.npy_int64*> np.PyArray_DATA(res)
        np.npy_int64* arr1_ptr = <np.npy_int64*> np.PyArray_DATA(arr1)
        np.npy_int64* arr2_ptr = <np.npy_int64*> np.PyArray_DATA(arr2)
        np.npy_int64 i, num, min_num, max_num, tie_num
        np.npy_int64 targ, far_num, diff_min, diff_max

    # Find the min & max values in 'arr2'
    min_num = arr2_ptr[0]
    max_num = min_num
    for i in range(1, arr2_size):
        num = arr2_ptr[i]
        if num < min_num:
            min_num = num
        elif num > max_num:
            max_num = num
    tie_num = min(min_num, max_num)

    # Find the furthest for elements in 'arr1'
    for i in range(arr1_size):
        targ = arr1_ptr[i]
        diff_min = targ - min_num if min_num < targ else min_num - targ
        diff_max = targ - max_num if max_num < targ else max_num - targ
        
        if diff_min > diff_max:
            far_num = min_num
        elif diff_min < diff_max:
            far_num = max_num
        else:
            far_num = tie_num
        
        res_ptr[i] = far_num

    # Return result
    return res

# . errors: internal
cdef inline bint _raise_dt64arr_as_int64_unit_error(str reso, int unit, bint is_dt64=True) except -1:
    """(internal) Raise unsupported unit for 'dt64arr_as_int64*()' functions."""
    obj_type = "datetime64" if is_dt64 else "timedelta64"
    try:
        unit_str = map_nptime_unit_int2str(unit)
    except Exception as err:
        raise ValueError(
            "cannot cast ndarray[%s] to int64 under '%s' resolution.\n"
            "Array with datetime unit '%d' is not supported."
            % (obj_type, reso, unit)
        ) from err
    else:
        raise ValueError(
            "cannot cast ndarray[%s[%s]] to int64 under '%s' resolution.\n"
            "Array with dtype of '%s[%s]' is not supported."
            % (obj_type, unit_str, reso, obj_type, unit_str)
        )
        
cdef inline bint _raise_dt64arr_rcl_unsupport_unit_error(str ops, int to_unit, int my_unit) except -1:
    """(internal) Raise unsupported unit for 'dt64arr_round/ceil/floor()' functions."""
    my_unit_str = map_nptime_unit_int2str(my_unit)
    to_unit_str = map_nptime_unit_int2str(to_unit)
    raise ValueError(
        "%s operation on <'ndarray[datetime64[%s]]'> to "
        "'%s' resolution is not supported." % (ops, my_unit_str, to_unit_str)
    )

cdef inline bint _raise_dt64arr_convert_arr_unit_error(np.ndarray arr, str msg) except -1:
    raise ValueError(
        "cannot convert <'ndarray[%s]'> to %s without "
        "specifying the array datetime unit." % (arr.dtype, msg)
    )

# NumPy: ndarray[timedelta64] --------------------------------------------------------------------------
# . type check
cdef inline bint is_td64arr(np.ndarray arr) except -1:
    """Check if the given array is dtype of 'timedelta64' `<'bool'>`.
    
    Equivalent to:
    >>> isinstance(arr.dtype, np.dtypes.TimeDelta64DType)
    """
    return np.PyArray_TYPE(arr) == np.NPY_TYPES.NPY_TIMEDELTA

cdef inline bint validate_td64arr(np.ndarray arr) except -1:
    """Validate if the given array is dtype of 'timedelta64',
    raises `TypeError` if dtype is incorrect.
    """
    if not is_td64arr(arr):
        raise TypeError(
            "expects instance of 'np.ndarray[timedelta64]', "
            "instead got '%s'." % arr.dtype
        )
    return True

# . conversion
cdef inline np.ndarray td64arr_as_int64_us(np.ndarray arr, str arr_unit=None, np.npy_int64 offset=0):
    """Cast np.ndarray[timedelta64] to int64 under 'us' (microsecond)
    resolution `<'ndarray[int64]'>`.
    
    Equivalent to:
    >>> arr.astype("timedelta64[us]").astype("int64") + offset
    """
    # Empty array
    if np.PyArray_SIZE(arr) == 0:
        return arr  # exit: empty array

    # Get time unit
    cdef int unit
    cdef int dtype = np.PyArray_TYPE(arr)
    if dtype != np.NPY_TYPES.NPY_TIMEDELTA:
        if arr_unit is None:
            _raise_dt64arr_convert_arr_unit_error(arr, "int64 under 'us' resolution")
        unit = map_nptime_unit_str2int(arr_unit)
        arr = arr_assure_int64(arr)
    elif arr_unit is not None:
        unit = map_nptime_unit_str2int(arr_unit)
    else:
        unit = get_arr_nptime_unit(arr)
    #: arr can only be only timedelta64 or int64 afterwords

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr_floor_div(arr, NS_MICROSECOND, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr_assure_int64(arr) if offset == 0 else arr_add(arr, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr_mul(arr, US_MILLISECOND, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr_mul(arr, US_SECOND, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr_mul(arr, US_MINUTE, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr_mul(arr, US_HOUR, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr_mul(arr, US_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return _td64arr_W_as_int64_D(arr, US_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _td64arr_M_as_int64_D(arr, US_DAY, offset)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _td64arr_Y_as_int64_D(arr, US_DAY, offset)
    # . unsupported
    _raise_dt64arr_as_int64_unit_error("us", unit, False)

cdef inline np.ndarray _td64arr_Y_as_int64_D(np.ndarray arr, np.npy_int64 factor=1, np.npy_int64 offset=0):
    """(internal) Cast np.ndarray[timedelta64[Y]] to int64
    under 'D' (day) resolution `<'ndarray[int64]'>`.
    """
    # Average number of days in a year: 365.2425
    # We use integer arithmetic by scaling to avoid floating-point inaccuracies.
    # Multiply by 3652425 and divide by 10000 to represent 365.2425 days/year.
    if factor == 1:  # day
        return arr_floor_div(arr_mul(arr, 3_652_425), 10_000, offset) # arr * 365.2425
    if factor == 24:  # hour
        return arr_floor_div(arr_mul(arr, 876_582), 100, offset) # arr * 8765.82 (365.2425 * 24)
    if factor == 1_440:  # minute
        return arr_floor_div(arr_mul(arr, 5_259_492), 10, offset) # arr * 8765.82 (365.2425 * 24)
    if factor == SS_DAY:  # second
        return arr_mul(arr, TD64_YY_SECOND)
    if factor == MS_DAY:  # millisecond
        return arr_mul(arr, TD64_YY_MILLISECOND)
    if factor == US_DAY:  # microsecond
        return arr_mul(arr, TD64_YY_MICROSECOND)
    if factor == NS_DAY:  # nanosecond
        return arr_mul(arr, TD64_YY_NANOSECOND)
    raise AssertionError("unsupported factor '%d' for timedelta unit 'Y' conversion." % factor)

cdef inline np.ndarray _td64arr_M_as_int64_D(np.ndarray arr, np.npy_int64 factor=1, np.npy_int64 offset=0):
    """(internal) Cast np.ndarray[timedelta64[M]] to int64
    under 'D' (day) resolution `<'ndarray[int64]'>`.
    """
    # Average number of days in a month: 30.436875 (365.2425 / 12)
    # We use integer arithmetic by scaling to avoid floating-point inaccuracies.
    # Multiply by 30436875 and divide by 1000000 to represent 30.436875 days/month.
    if factor == 1:  # day
        return arr_floor_div(arr_mul(arr, 30_436_875), 1_000_000, offset) # arr * 30.436875
    if factor == 24:  # hour
        return arr_floor_div(arr_mul(arr, 730_485), 1_000, offset) # arr * 730.485 (30.436875 * 24)
    if factor == 1_440:  # minute
        return arr_floor_div(arr_mul(arr, 438_291), 10, offset) # arr * 43829.1 (30.436875 * 1440)
    if factor == SS_DAY:  # second
        return arr_mul(arr, TD64_MM_SECOND)
    if factor == MS_DAY:  # millisecond
        return arr_mul(arr, TD64_MM_MILLISECOND)
    if factor == US_DAY:  # microsecond
        return arr_mul(arr, TD64_MM_MICROSECOND)
    if factor == NS_DAY:  # nanosecond
        return arr_mul(arr, TD64_MM_NANOSECOND)
    raise AssertionError("unsupported factor '%d' for timedelta unit 'M' conversion." % factor)

cdef inline np.ndarray _td64arr_W_as_int64_D(np.ndarray arr, np.npy_int64 factor=1, np.npy_int64 offset=0):
    """(internal) Cast np.ndarray[timedelta64[W]] to int64
    under 'D' (day) resolution `<'ndarray[int64]'>`.
    """
    return arr_mul(arr, 7 * factor, offset)
