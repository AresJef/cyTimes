# cython: language_level=3

cimport cython
cimport numpy as np
from libc cimport math
from libc.stdlib cimport malloc, free, strtoll
from libc.time cimport (
    strftime, 
    time_t, 
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
    int MONTH_TO_QUARTER[13]
    np.ndarray DAYS_BR_QUARTER_NDARRAY
    # . microseconds
    long long US_DAY
    long long US_HOUR
    # . nanoseconds
    long long NS_DAY
    long long NS_HOUR
    long long NS_MINUTE
    # . date
    int ORDINAL_MAX
    # . datetime
    datetime.tzinfo UTC
    datetime.datetime EPOCH_DT
    long long EPOCH_US
    long long EPOCH_SEC
    int EPOCH_DAY
    long long DT_US_MAX
    long long DT_US_MIN
    long long DT_SEC_MAX
    long long DT_SEC_MIN
    int US_FRAC_CORRECTION[5]
    # . time
    datetime.time TIME_MIN
    datetime.time TIME_MAX

# Struct -----------------------------------------------------------------------------------------------
ctypedef struct ymd:
    int year
    int month
    int day

ctypedef struct hms:
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

# Delta ------------------------------------------------------------------------------------------------
cdef inline int combine_abs_ms_us(int ms, int us) except -2:
    """Combine absolute millisecond and microsecond to microsecond `<'int'>`.

    - If 'ms' is non-negative:
        - Convert to microseconds (ms * 1000)
        - Add the 'us' fraction (up to 1000)
        - Expects value between 0-999999

    - If 'us' is non-negative:
        - Clip microseconds (up to 999999).
        - Expects value between 0-999999

    - If both 'ms' and 'us' are negative:
        - Returns -1.
    """
    cdef int ms2us
    if ms >= 0:
        ms2us = min(ms, 999) * 1_000
        if us > 0:
            ms2us += us % 1_000
        return ms2us
    if us >= 0:
        return min(us, 999_999)
    return -1

# Parser -----------------------------------------------------------------------------------------------
cdef inline Py_ssize_t str_count(str s, str substr) except -1:
    """Get number of occurrences of a 'substr' in an unicode `<'int'>`.

    Equivalent to:
    >>> s.count(substr)
    """
    return PyUnicode_Count(s, substr, 0, PY_SSIZE_T_MAX)

cdef inline bint is_iso_sep(Py_UCS4 ch) except -1:
    """Check if 'ch' is ISO format date/time seperator (" " or "T") `<'bool'>`"""
    return ch in (" ", "t", "T")

cdef inline bint is_isodate_sep(Py_UCS4 ch) except -1:
    """Check if 'ch' is ISO format date values separator ("-" or "/") `<'bool'>`"""
    return ch in ("-", "/")

cdef inline bint is_isoweek_sep(Py_UCS4 ch) except -1:
    """Check if 'ch' is ISO format week separator ("W") `<'bool'>`"""
    return ch in ("w", "W")

cdef inline bint is_isotime_sep(Py_UCS4 ch) except -1:
    """Check if 'ch' is ISO format time values separator (":") `<'bool'>`"""
    return ch == ":"

cdef inline bint is_ascii_digit(Py_UCS4 ch) except -1:
    """Check if 'ch' is an ASCII digit [0-9] `<'bool'>`"""
    return "0" <= ch <= "9"
    
cdef inline bint is_ascii_alpha_upper(Py_UCS4 ch) except -1:
    """Check if 'ch' is an ASCII alpha uppercase [A-Z] `<'bool'>`."""
    return "A" <= ch <= "Z"

cdef inline bint is_ascii_alpha_lower(Py_UCS4 ch) except -1:
    """Check if 'ch' is an ASCII alpha lowercase [a-z] `<'bool'>`."""
    return "a" <= ch <= "z"

cdef inline bint is_ascii_alpha(Py_UCS4 ch) except -1:
    """Check if 'ch' is an ASCII alpha [a-zA-Z] `<'bool'>`."""
    return is_ascii_alpha_lower(ch) or is_ascii_alpha_upper(ch)

cdef inline int parse_isoyear(str data, Py_ssize_t pos, Py_ssize_t size) except -2:
    """Parse ISO format year (YYYY) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO year (YYYY) from.
    :param pos `<'int'>`: The starting position of the ISO year.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format year value.
    """
    # Validate size
    if size <= 0:
        size = str_len(data)
    if size - pos < 4:
        return -1

    # Parse values
    cdef:
        char buffer[5]
        Py_UCS4 ch
        Py_ssize_t i
        int year
    for i in range(4):
        ch = str_read(data, pos + i)
        if not is_ascii_digit(ch):
            return -1
        buffer[i] = ch

    # Convert to integer
    buffer[4] = 0  # null-term
    year = strtoll(buffer, NULL, 10)
    return year if year != 0 else -1

cdef inline int parse_isomonth(str data, Py_ssize_t pos, Py_ssize_t size)  except -2:
    """Parse ISO format month (MM) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO month (MM) from.
    :param pos `<'int'>`: The starting position of the ISO month.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format month value.
    """
    # Validate size
    if size <= 0:
        size = str_len(data)
    if size - pos < 2:
        return -1

    # Parse values
    cdef:
        char buffer[3]
        Py_UCS4 ch1 = str_read(data, pos)
        Py_UCS4 ch2 = str_read(data, pos + 1)
    if ch1 == "0":
        if not "1" <= ch2 <= "9":
            return -1
    elif ch1 == "1":
        if not "0" <= ch2 <= "2":
            return -1
    else:
        return -1

    # Convert to integer
    buffer[0] = ch1
    buffer[1] = ch2
    buffer[2] = 0  # null-term
    return strtoll(buffer, NULL, 10)

cdef inline int parse_isoday(str data, Py_ssize_t pos, Py_ssize_t size) except -2:
    """Parse ISO format day (DD) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO day (DD) from.
    :param pos `<'int'>`: The starting position of the ISO day.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format day value.
    """
    # Validate size
    if size <= 0:
        size = str_len(data)
    if size - pos < 2:
        return -1

    # Parse values
    cdef:
        char buffer[3]
        Py_UCS4 ch1 = str_read(data, pos)
        Py_UCS4 ch2 = str_read(data, pos + 1)
    if ch1 in ("1", "2"):
        if not is_ascii_digit(ch2):
            return -1
    elif ch1 == "0":
        if not "1" <= ch2 <= "9":
            return -1
    elif ch1 == "3":
        if not ch2 in ("0", "1"):
            return -1
    else:
        return -1

    # Convert to integer
    buffer[0] = ch1
    buffer[1] = ch2
    buffer[2] = 0  # null-term
    return strtoll(buffer, NULL, 10)

cdef inline int parse_isoweek(str data, Py_ssize_t pos, Py_ssize_t size) except -2:
    """Parse ISO format week number (WW) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO week number (WW) from.
    :param pos `<'int'>`: The starting position of the ISO week number.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format week number value.
    """
    # Validate size
    if size <= 0:
        size = str_len(data)
    if size - pos < 2:
        return -1

    # Parse values
    cdef:
        char buffer[3]
        Py_UCS4 ch1 = str_read(data, pos)
        Py_UCS4 ch2 = str_read(data, pos + 1)
    if "1" <= ch1 <= "4":
        if not is_ascii_digit(ch2):
            return -1
    elif ch1 == "0":
        if not "1" <= ch2 <= "9":
            return -1
    elif ch1 == "5":
        if not "0" <= ch2 <= "3":
            return -1
    else:
        return -1

    # Convert to integer
    buffer[0] = ch1
    buffer[1] = ch2
    buffer[2] = 0  # null-term
    return strtoll(buffer, NULL, 10)

cdef inline int parse_isoweekday(str data, Py_ssize_t pos, Py_ssize_t size) except -2:
    """Parse ISO format weekday (D) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO weekday (D) from.
    :param pos `<'int'>`: The starting position of the ISO weekday.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format weekday value.
    """
    # Validate size
    if size <= 0:
        size = str_len(data)
    if size - pos < 1:
        return -1

    # Parse values
    cdef Py_UCS4 ch = str_read(data, pos)
    if not "1" <= ch <= "7":
        return -1
    return ord(ch) - 48

cdef inline int parse_isoyearday(str data, Py_ssize_t pos, Py_ssize_t size) except -2:
    """Parse ISO format day of year (DDD) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO day of year (DDD) from.
    :param pos `<'int'>`: The starting position of the ISO day of year.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format day of year value.
    """
    # Validate size
    if size <= 0:
        size = str_len(data)
    if size - pos < 3:
        return -1

    # Parse values
    cdef:
        char buffer[4]
        Py_UCS4 ch
        Py_ssize_t i
    for i in range(3):
        ch = str_read(data, pos + i)
        if not is_ascii_digit(ch):
            return -1
        buffer[i] = ch

    # Convert to integer
    buffer[3] = 0  # null-term
    days = strtoll(buffer, NULL, 10)
    return days if 1 <= days <= 366 else -1

cdef inline int parse_isohour(str data, Py_ssize_t pos, Py_ssize_t size) except -2:
    """Parse ISO format hour (HH) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO hour (HH) from.
    :param pos `<'int'>`: The starting position of the ISO hour.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format hour value.
    """
    # Validate size
    if size <= 0:
        size = str_len(data)
    if size - pos < 2:
        return -1

    # Parse values
    cdef:
        char buffer[3]
        Py_UCS4 ch1 = str_read(data, pos)
        Py_UCS4 ch2 = str_read(data, pos + 1)
    if ch1 in ("0", "1"):
        if not is_ascii_digit(ch2):
            return -1
    elif ch1 == "2":
        if not "0" <= ch2 <= "3":
            return -1
    else:
        return -1
    
    # Convert to integer
    buffer[0] = ch1
    buffer[1] = ch2
    buffer[2] = 0  # null-term
    return strtoll(buffer, NULL, 10)

cdef inline int parse_isominute(str data, Py_ssize_t pos, Py_ssize_t size) except -2:
    """Parse ISO format minute (MM) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO minute (MM) from.
    :param pos `<'int'>`: The starting position of the ISO minute.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format minute value.
    """
    # Validate size
    if size <= 0:
        size = str_len(data)
    if size - pos < 2:
        return -1

    # Parse values
    cdef:
        char buffer[3]
        Py_UCS4 ch1 = str_read(data, pos)
        Py_UCS4 ch2 = str_read(data, pos + 1)
    if not "0" <= ch1 <= "5":
        return -1
    if not is_ascii_digit(ch2):
        return -1
    
    # Convert to integer
    buffer[0] = ch1
    buffer[1] = ch2
    buffer[2] = 0  # null-term
    return strtoll(buffer, NULL, 10)

cdef inline int parse_isosecond(str data, Py_ssize_t pos, Py_ssize_t size) except -2:
    """Parse ISO format second (SS) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO second (SS) from.
    :param pos `<'int'>`: The starting position of the ISO second.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format second value.
    """
    return parse_isominute(data, pos, size)

cdef inline int parse_isofraction(str data, Py_ssize_t pos, Py_ssize_t size) except -2:
    """Parse ISO format fraction (f/us) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO fraction (f/us) from.
    :param pos `<'int'>`: The starting position of the ISO fraction.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format fraction value.
    """
    # Validate size
    if size <= 0:
        size = str_len(data)

    # Parse values
    cdef:
        char buffer[7]
        Py_UCS4 ch
        Py_ssize_t digits = 0
    while pos < size and digits < 6:
        ch = str_read(data, pos)
        if not is_ascii_digit(ch):
            break
        buffer[digits] = ch
        pos += 1
        digits += 1

    # Compensate missing digits
    if digits < 6:
        if digits == 0:
            return -1  # exit: invalid
        ch = "0"
        for i in range(digits, 6):
            buffer[i] = ch
    
    # Convert to integer
    buffer[6] = 0  # null-term
    return strtoll(buffer, NULL, 10)

cdef inline long long slice_to_uint(str data, Py_ssize_t start, Py_ssize_t size) except -2:
    """Slice & convert 'data' to an integer `<'int'>`.

    :param data `<'str'>`: The string to slice to an integer.
    :param start `<'int'>`: The starting position of the integer slice.
    :param size `<'int'>`: Total characters to slice from the starting position.
    :raise `ValueError`: If cannot convert slice of 'data' to an integer.
    """
    # Allocate memory
    cdef char* buffer = <char*>malloc(size + 1)
    if buffer == NULL:
        raise MemoryError("unable to allocate memory for integer slice.")
    
    cdef:
        Py_ssize_t i
        Py_UCS4 ch
    try:
        # Parse value
        for i in range(size):
            ch = str_read(data, start + i)
            if not is_ascii_digit(ch):
                raise ValueError("invalid character '%s' for an integer." % str_chr(ch))
            buffer[i] = ch
        buffer[size] = 0 # null-term

        # Convert integer
        return strtoll(buffer, NULL, 10)
    finally:
        free(buffer)

# Time -------------------------------------------------------------------------------------------------
cdef inline int _raise_from_errno() except -1 with gil:
    """Error handling for localtime_ts function."""
    PyErr_SetFromErrno(RuntimeError)
    return <int> -1  # type: ignore

cdef inline str tm_strftime(tm t, str fmt):
    """Convert struct_time (struct:tm) to string with the given 'fmt' `<'str'>`."""
    # Revert 0-based date values.
    t.tm_year -= 1_900
    t.tm_mon -= 1
    t.tm_wday = (t.tm_wday + 1) % 7
    t.tm_yday -= 1

    # Perform strftime
    cdef:
        char buffer[256]
        Py_ssize_t size = strftime(buffer, sizeof(buffer), PyUnicode_AsUTF8(fmt), &t)
    if size == 0:
        raise OverflowError("The size of the format '%s' is too large.'" % fmt)
    return PyUnicode_DecodeUTF8(buffer, size, NULL)

cdef inline tm tm_gmtime(double ts) except * nogil:
    """Get the struc_time of the 'ts' expressing UTC time `<'struct:tm'>`.

    Equivalent to:
    >>> time.gmtime(ts)
    """
    cdef:
        time_t tic = <time_t> ts
        tm* t
    t = libc_gmtime(&tic)
    if t is NULL:
        _raise_from_errno()
    # Fix 0-based date values (and the 1900-based year).
    # See tmtotuple() in https://github.com/python/cpython/blob/master/Modules/timemodule.c
    t.tm_year += 1900
    t.tm_mon += 1
    t.tm_sec = min(t.tm_sec, 59)  # clamp leap seconds
    t.tm_wday = (t.tm_wday + 6) % 7
    t.tm_yday += 1
    return t[0]

cdef inline tm tm_localtime(double ts) except * nogil:
    """Get struc_time of the 'ts' expressing local time `<'struct:tm'>`.
    
    Equivalent to:
    >>> time.localtime(ts)
    """
    cdef:
        time_t tic = <time_t> ts
        tm* t
    t = libc_localtime(&tic)
    if t is NULL:
        _raise_from_errno()
    # Fix 0-based date values (and the 1900-based year).
    # See tmtotuple() in https://github.com/python/cpython/blob/master/Modules/timemodule.c
    t.tm_year += 1900
    t.tm_mon += 1
    t.tm_sec = min(t.tm_sec, 59)  # clamp leap seconds
    t.tm_wday = (t.tm_wday + 6) % 7
    t.tm_yday += 1
    return t[0]

cdef inline long long ts_gmtime(double ts):
    """Get timestamp of the 'ts' expressing UTC time `<'int'>."""
    cdef:
        tm t = tm_gmtime(ts)
        long long ordinal = ymd_to_ordinal(t.tm_year, t.tm_mon, t.tm_mday)
        long long seconds = ordinal * 86_400 + t.tm_hour * 3_600 + t.tm_min * 60 + t.tm_sec
    return seconds - EPOCH_SEC

cdef inline long long ts_localtime(double ts):
    """Get timestamp of the 'ts' expressing local time `<'int'>`."""
    cdef:
        tm t = tm_localtime(ts)
        long long ordinal = ymd_to_ordinal(t.tm_year, t.tm_mon, t.tm_mday)
        long long seconds = ordinal * 86_400 + t.tm_hour * 3_600 + t.tm_min * 60 + t.tm_sec
    return seconds - EPOCH_SEC

cdef inline tm tm_fr_seconds(double seconds) except *:
    """Convert total seconds since Unix Epoch to `<'struct:tm'>`."""
    # Add back Epoch
    cdef long long ss = int(seconds)
    ss = min(max(ss + EPOCH_SEC, DT_SEC_MIN), DT_SEC_MAX)

    # Calculate ymd & hms
    cdef int ordinal = ss // 86_400
    _ymd = ymd_fr_ordinal(ordinal)
    _hms = hms_fr_seconds(ss)

    # Create struct_time
    return tm(
        _hms.second, _hms.minute, _hms.hour,  # SS, MM, HH
        _ymd.day, _ymd.month, _ymd.year,  # DD, MM, YY
        ymd_weekday(_ymd.year, _ymd.month, _ymd.day),  # wday
        days_of_year(_ymd.year, _ymd.month, _ymd.day),  # yday
        -1,  # isdst
    )

cdef inline tm tm_fr_us(long long us) except *:
    """Convert total microseconds since Unix Epoch to `<'struct:tm'>`."""
    # Add back Epoch
    cdef unsigned long long _us = min(max(us + EPOCH_US, DT_US_MIN), DT_US_MAX)

    # Calculate ymd & hms
    # since '_us' is positive, we can safely divide
    # without checking for negative values.
    cdef int ordinal
    with cython.cdivision(True):
        ordinal = _us // US_DAY
    _ymd = ymd_fr_ordinal(ordinal)
    _hms = hms_fr_us(_us)

    # Create struct_time
    return tm(
        _hms.second, _hms.minute, _hms.hour,  # SS, MM, HH
        _ymd.day, _ymd.month, _ymd.year,  # DD, MM, YY
        ymd_weekday(_ymd.year, _ymd.month, _ymd.day),  # wday
        days_of_year(_ymd.year, _ymd.month, _ymd.day),  # yday
        -1,  # isdst
    )

cdef inline hms hms_fr_seconds(double seconds) except *:
    """Convert total seconds to 'H/M/S' `<'struct:hms'>`."""
    cdef:
        long long ss = int(seconds)
        int hh, mi
        long long us

    if ss <= 0:
        return hms(0, 0, 0, 0)

    ss %= 86_400
    hh = ss // 3_600
    ss = ss % 3_600
    mi = ss // 60
    ss %= 60
    us = int(seconds * 1_000_000)
    us %= 1_000_000
    return hms(hh, mi, ss, us)

cdef inline hms hms_fr_us(long long us) except *:
    """Convert total microseconds to 'H/M/S/F' `<'struct:hms'>`."""
    if us <= 0:
        return hms(0, 0, 0, 0)

    cdef int hh, mi, ss
    # Since 'us' must be positive, we can safely divide
    # without checking for negative values.
    with cython.cdivision(True):
        us = us % US_DAY
        hh = us // US_HOUR
        us = us % US_HOUR
    mi = us // 60_000_000
    us %= 60_000_000
    ss = us // 1_000_000
    us %= 1_000_000
    return hms(hh, mi, ss, us)

# Calendar ---------------------------------------------------------------------------------------------
# . year
cdef inline bint is_leap_year(int year) except -1:
    """Whether the given 'year' is a leap year `<'bool'>`."""
    if year <= 0:
        return False
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

cdef inline bint is_long_year(int year) except -1:
    """Whether the given 'year' is a long year 
    (maximum ISO week number is 53) `<'bool'>`.
    """
    return ymd_isoweek(year, 12, 28) == 53

cdef inline int leap_bt_years(int year1, int year2) except -1:
    """Calculate total leap years between 'year1' and 'year2' `<'int'>`."""
    cdef int y1, y2
    if year1 <= year2:
        y1 = max(year1 - 1, 0)
        y2 = max(year2 - 1, 0)
    else:
        y1 = max(year2 - 1, 0)
        y2 = max(year1 - 1, 0)
    return (
        (y2 // 4 - y1 // 4) 
        - (y2 // 100 - y1 // 100) 
        + (y2 // 400 - y1 // 400)
    )

cdef inline int days_in_year(int year) except -1:
    """Get total days in the given 'year', expects 365 or 366 `<'int'>`."""
    return 366 if is_leap_year(year) else 365

cdef inline int days_bf_year(int year) except -1:
    """Get total days between the 1st day of 1AD 
    and the 1st day of the given 'year' `<'int'>`."""
    if year <= 1:
        return 0

    cdef int y = year - 1
    return y * 365 + y // 4 - y // 100 + y // 400

cdef inline int days_of_year(int year, int month, int day) except -1:
    """Get the days between the 1st day of the 'year' 
    and the given 'Y/M/D' `<'int'>`."""
    return (
        days_bf_month(year, month) 
        + min((max(day, 1)), days_in_month(year, month))
    )

# . quarter
cdef inline int quarter_of_month(int month) except -1:
    """Get the quarter of the given 'month', expects 1-4 `<'int'>`."""
    if month <= 1:
        return 1
    if month >= 12:
        return 4
    return MONTH_TO_QUARTER[month]

cdef inline int days_in_quarter(int year, int month) except -1:
    """Get total days of the quarter for the given 'Y/M' `<'int'>`."""
    cdef:
        int quarter = quarter_of_month(month)
        int days = DAYS_IN_QUARTER[quarter]
    if quarter == 1 and is_leap_year(year):
        days += 1
    return days

cdef inline int days_bf_quarter(int year, int month) except -1:
    """Get total days between the 1st day of the 'year'
    and the 1st day of the quarter for the given 'Y/M' `<'int'>`."""
    cdef:
        int quarter = quarter_of_month(month)
        int days = DAYS_BR_QUARTER[quarter - 1]
    if quarter >= 2 and is_leap_year(year):
        days += 1
    return days

cdef inline int days_of_quarter(int year, int month, int day) except -1:
    """Get the days between the 1st day of the quarter
    and the given 'Y/M/D' `<'int'>`."""
    return days_of_year(year, month, day) - days_bf_quarter(year, month)

cdef inline int quarter_1st_month(int month) except -1:
    """Get the first month of the quarter, expects 1, 4, 7, 10 `<'int'>`."""
    return 3 * quarter_of_month(month) - 2

cdef inline int quarter_lst_month(int month) except -1:
    """Get the last month of the quarter, expects 3, 6, 9, 12 `<'int'>`."""
    return 3 * quarter_of_month(month)

# . month
cdef inline int days_in_month(int year, int month) except -1:
    """Get total days of the 'month' in the given 'year' `<'int'>`."""
    if month <= 1 or month >= 12:
        return 31

    cdef int days = DAYS_IN_MONTH[month]
    if month == 2 and is_leap_year(year):
        days += 1
    return days

cdef inline int days_bf_month(int year, int month) except -1:
    """Get total days between the 1st day of the 'year'
    and the 1st day of the given 'month' `<'int'>`."""
    if month <= 2:
        return 31 if month == 2 else 0

    cdef int days = DAYS_BR_MONTH[min(month, 12) -1]
    if is_leap_year(year):
        days += 1
    return days

# . week
cdef inline int ymd_weekday(int year, int month, int day) except -1:
    """Get the weekday of the given 'Y/M/D', 
    expects 0[Monday]...6[Sunday] `<'int'>`."""
    return (ymd_to_ordinal(year, month, day) + 6) % 7

cdef inline int ymd_isoweekday(int year, int month, int day) except -1:
    """Get the ISO weekday of the given 'Y/M/D', 
    expects 1[Monday]...7[Sunday] `<'int'>`."""
    return ymd_weekday(year, month, day) + 1

cdef inline int ymd_isoweek(int year, int month, int day) except -1:
    """Get the ISO calendar week number of the given 'Y/M/D' `<'int'>`."""
    cdef:
        int ordinal = ymd_to_ordinal(year, month, day)
        int iso_1st = iso_1st_monday(year)
        int delta = ordinal - iso_1st
        int weekday = delta // 7

    if weekday < 0:
        iso_1st = iso_1st_monday(year - 1)
        return (ordinal - iso_1st) // 7 + 1
    elif weekday >= 52 and ordinal >= iso_1st_monday(year + 1):
        return 1
    else:
        return weekday + 1

cdef inline int ymd_isoyear(int year, int month, int day) except -1:
    """Get the ISO calendar year of the given 'Y/M/D' `<'int'>`."""
    cdef:
        int ordinal = ymd_to_ordinal(year, month, day)
        int iso_1st = iso_1st_monday(year)
        int delta = ordinal - iso_1st
        int weekday = delta // 7

    if weekday < 0:
        return year - 1
    elif weekday >= 52 and ordinal >= iso_1st_monday(year + 1):
        return year + 1
    else:
        return year
    
cdef inline iso ymd_isocalendar(int year, int month, int day) except *:
    """Get the ISO calendar of the given 'Y/M/D' `<'struct:iso'>`."""
    cdef:
        int ordinal = ymd_to_ordinal(year, month, day)
        int iso_1st = iso_1st_monday(year)
        int delta = ordinal - iso_1st
        int weekday = delta // 7

    if weekday < 0:
        year -= 1
        iso_1st = iso_1st_monday(year)
        weekday = (ordinal - iso_1st) // 7 + 1
    elif weekday >= 52 and ordinal >= iso_1st_monday(year + 1):
        year += 1
        weekday = 1
    else:
        weekday += 1
    return iso(year, weekday, delta % 7 + 1)

cdef inline int ymd_to_ordinal(int year, int month, int day) except -1:
    """Convert 'Y/M/D' to ordinal days `<'int'>`."""
    return (
        days_bf_year(year) 
        + days_bf_month(year, month) 
        + min(max(day, 1), days_in_month(year, month))
    )

cdef inline ymd ymd_fr_ordinal(int ordinal) except *:
    """Convert ordinal days to 'Y/M/D' `<'stuct:ymd'>`."""
    # n is a 1-based index, starting at 1-Jan-1.  The pattern of leap years
    # repeats exactly every 400 years.  The basic strategy is to find the
    # closest 400-year boundary at or before n, then work with the offset
    # from that boundary to n.  Life is much clearer if we subtract 1 from
    # n first -- then the values of n at 400-year boundaries are exactly
    # those divisible by _DI400Y:
    cdef int n = min(max(ordinal, 1) - 1, ORDINAL_MAX)
    cdef int n400 = n // 146_097
    n %= 146_097
    cdef int year = n400 * 400 + 1

    # Now n is the (non-negative) offset, in days, from January 1 of year, to
    # the desired date.  Now compute how many 100-year cycles precede n.
    # Note that it's possible for n100 to equal 4!  In that case 4 full
    # 100-year cycles precede the desired day, which implies the desired
    # day is December 31 at the end of a 400-year cycle.
    cdef int n100 = n // 36_524
    n %= 36_524

    # Now compute how many 4-year cycles precede it.
    cdef int n4 = n // 1_461
    n %= 1_461

    # And now how many single years.  Again n1 can be 4, and again meaning
    # that the desired day is December 31 at the end of the 4-year cycle.
    cdef int n1 = n // 365
    n %= 365

    # We now know the year and the offset from January 1st.  Leap years are
    # tricky, because they can be century years.  The basic rule is that a
    # leap year is a year divisible by 4, unless it's a century year --
    # unless it's divisible by 400.  So the first thing to determine is
    # whether year is divisible by 4.  If not, then we're done -- the answer
    # is December 31 at the end of the year.
    year += n100 * 100 + n4 * 4 + n1
    if n1 == 4 or n100 == 4:
        return ymd(year - 1, 12, 31)

    # Now the year is correct, and n is the offset from January 1.  We find
    # the month via an estimate that's either exact or one too large.
    cdef int month = (n + 50) >> 5
    cdef int days_bf = days_bf_month(year, month)
    if days_bf > n:
        month -= 1
        days_bf = days_bf_month(year, month)
    return ymd(year, month, n - days_bf + 1)

cdef inline ymd ymd_fr_isocalendar(int year, int week, int weekday) except *:
    """Convert ISO calendar to 'Y/M/D' `<'struct:ymd>`."""
    # Clip year
    year = min(max(year, 1), 9_999)

    # 53th week adjustment
    cdef int day_1st
    if week == 53:
        # ISO years have 53 weeks in them on years starting with a
        # Thursday or leap years starting on a Wednesday. So for
        # invalid weeks, we shift to the 1st week of the next year.
        day_1st = ymd_to_ordinal(year, 1, 1) % 7
        if not (day_1st == 4 or (day_1st == 3 and is_leap_year(year))):
            week = 1
            year += 1
    # Clip week
    else:
        week = min(max(week, 1), 52)

    # Clip weekday
    weekday = min(max(weekday, 1), 7)

    # Calculate ordinal
    cdef int iso_1st = iso_1st_monday(year)
    cdef int offset = (week - 1) * 7 + weekday - 1
    return ymd_fr_ordinal(iso_1st + offset)

cdef inline ymd ymd_fr_days_of_year(int year, int days) except *:
    """Convert days of the year to 'Y/M/D' `<'struct:ymd'>`."""
    # Clip year & days
    year = min(max(year, 1), 9_999)
    days = max(days, 1)

    # January
    if days <= 31:
        return ymd(year, 1, days)

    # February
    cdef int leap = is_leap_year(year)
    if days <= 59 + leap:
        return ymd(year, 2, days - 31)

    # Find month & day
    days -= leap
    cdef int month, days_bf, day
    for month in range(3, 13):
        days_bf = DAYS_BR_MONTH[month]
        if days <= days_bf:
            day = days - DAYS_BR_MONTH[month - 1]
            return ymd(year, month, day)
    return ymd(year, 12, 31)

cdef inline int iso_1st_monday(int year) except -1:
    """Get the ordinal of the 1st Monday of the ISO 'year' `<'int'>`."""
    cdef:
        int day_1st = ymd_to_ordinal(year, 1, 1)
        int weekday_1st = (day_1st + 6) % 7
        int weekmon_1st = day_1st - weekday_1st
    return weekmon_1st + 7 if weekday_1st > 3 else weekmon_1st

# datetime.date ----------------------------------------------------------------------------------------
# . generate
cdef inline datetime.date date_new(int year=1, int month=1, int day=1):
    """Create a new `<'datetime.date'>`.
    
    Equivalent to:
    >>> datetime.date(year, month, day)
    """
    year = min(max(year, 1), 9_999)
    month = min(max(month, 1), 12)
    day = min(max(day, 1), days_in_month(year, month))
    return datetime.date_new(year, month, day)

cdef inline datetime.date date_now(object tz=None):
    """Get the current date `<'datetime.date'>`.
    
    Equivalent to:
    >>> datetime.date.today()
    """
    # With timezone
    if tz is not None:
        return date_fr_dt(datetime.datetime_from_timestamp(unix_time(), tz))

    # With timezone
    _tm = tm_localtime(unix_time())
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

# . conversion
cdef inline tm date_to_tm(datetime.date date) except *:
    """Convert datetime.date to `<'struct:tm'>`.
    
    #### All time values sets to 0.
    """
    cdef:
        int yy = date.year
        int mm = date.month
        int dd = date.day
    return tm(
        0, 0, 0, dd, mm, yy,  # SS, MM, HH, DD, MM, YY
        ymd_weekday(yy, mm, dd), # wday
        days_bf_month(yy, mm) + dd, # yday 
        -1 # isdst
    )

cdef inline str date_to_strformat(datetime.date date, str fmt):
    """Convert datetime.date to str with the specified 'fmt' `<'str'>`.

    Equivalent to:
    >>> date.strftime(fmt)
    """
    cdef:
        list fmt_l = []
        Py_ssize_t idx = 0
        Py_ssize_t size = str_len(fmt)
        Py_UCS4 ch

    # Escape format
    while idx < size:
        ch = str_read(fmt, idx)
        idx += 1
        # Identifier '%'
        if ch == "%":
            if idx < size:
                ch = str_read(fmt, idx)
                idx += 1
                # fraction, utc, and tzname
                if ch in ("f", "z", "Z"):
                    pass
                # rest
                else:
                    fmt_l.append("%")
                    fmt_l.append(str_chr(ch))
            else:
                fmt_l.append("%")
        else:
            fmt_l.append(str_chr(ch))

    # Format to string
    return tm_strftime(date_to_tm(date), "".join(fmt_l))

cdef inline str date_to_isoformat(datetime.date date):
    """Convert datetime.date to ISO format: '%Y-%m-%d' `<'str'>`."""
    return "%04d-%02d-%02d" % (date.year, date.month, date.day)

cdef inline int date_to_ordinal(datetime.date date) except -1:
    """Convert datetime.date to ordinal days `<'int'>`."""
    return ymd_to_ordinal(date.year, date.month, date.day)

cdef inline double date_to_seconds(datetime.date date):
    """Convert datetime.date to total seconds since Unix Epoch `<'float'>`."""
    cdef long long ordinal = date_to_ordinal(date)
    return (ordinal - EPOCH_DAY) * 86_400

cdef inline long long date_to_us(datetime.date date):
    """Convert datetime.date to total microseconds since Unix Epoch `<'int'>`."""
    cdef long long ordinal = date_to_ordinal(date)
    return (ordinal - EPOCH_DAY) * US_DAY

cdef inline double date_to_ts(datetime.date date):
    """Convert datetime.date to timestamp `<'float'>`."""
    cdef:
        long long ts = int(date_to_seconds(date))
        long long offset = ts_localtime(ts) - ts
    return ts - offset

cdef inline datetime.date date_fr_date(datetime.date date):
    """Convert subclass of datetime.date to `<'datetime.date'>`."""
    return datetime.date_new(date.year, date.month, date.day)

cdef inline datetime.date date_fr_dt(datetime.datetime dt):
    """Convert datetime.datetime to `<'datetime.date'>`."""
    return datetime.date_new(dt.year, dt.month, dt.day)

cdef inline datetime.date date_fr_ordinal(int ordinal):
    """Convert ordinal days to `<'datetime.date'>`."""
    _ymd = ymd_fr_ordinal(ordinal)
    return datetime.date_new(_ymd.year, _ymd.month, _ymd.day)

cdef inline datetime.date date_fr_seconds(double seconds):
    """Convert total seconds since Unix Epoch to `<'datetime.date'>`."""
    cdef long long ss = int(seconds)
    return date_fr_ordinal(ss // 86_400 + EPOCH_DAY)

cdef inline datetime.date date_fr_us(long long us):
    """Convert total microseconds since Unix Epoch to `<'datetime.date'>`."""
    # Add back Epoch
    cdef unsigned long long _us = min(max(us + EPOCH_US, DT_US_MIN), DT_US_MAX)

    # Calcuate ordinal days
    # since '_us' is positive, we can safely divide
    # without checking for negative values.
    cdef int ordinal
    with cython.cdivision(True):
        ordinal = _us // US_DAY
    
    # Create date
    return date_fr_ordinal(ordinal)

cdef inline datetime.date date_fr_ts(double ts):
    """Convert timestamp to `<'datetime.date'>`."""
    return datetime.date_from_timestamp(ts)

# . manipulation
cdef inline datetime.date date_replace(datetime.date date, int year=-1, int month=-1, int day=-1):
    """Replace datetime.date values `<'datetime.date'>`.

    #### Default '-1' mean keep the original value.

    Equivalent to:
    >>> date.replace(year, month, day)
    """
    if not 1 <= year <= 9_999:
        year = date.year
    if not 1 <= month <= 12:
        month = date.month
    day = min(day if day > 0 else date.day, days_in_month(year, month))
    return datetime.date_new(year, month, day)

cdef inline datetime.date date_chg_weekday(datetime.date date, int weekday):
    """Change datetime.date 'weekday' within the current week 
    (0[Monday]...6[Sunday]) `<'datetime.date'>`.

    Equivalent to:
    >>> date + datetime.timedelta(weekday - date.weekday())
    """
    cdef int curr_wday = ymd_weekday(date.year, date.month, date.day)
    weekday = min(max(weekday, 0), 6)
    if curr_wday == weekday:
        return date

    cdef int ordinal = date_to_ordinal(date)
    return date_fr_ordinal(ordinal + weekday - curr_wday)

# . arithmetic
cdef inline datetime.date date_add(
    datetime.date date,
    int days=0, int seconds=0, int microseconds=0,
    int milliseconds=0, int minutes=0, int hours=0, int weeks=0,
):
    """Add timedelta to datetime.date `<'datetime.date'>`.
    
    Equivalent to:
    >>> date + datetime.timedelta(
            days, seconds, microseconds, 
            milliseconds, minutes, hours, weeks
        )
    """
    # No change
    if days == seconds == microseconds == milliseconds == minutes == hours == weeks == 0:
        return date

    # Add timedelta
    cdef: 
        long long ordinal = date_to_ordinal(date)
        long long hh = hours
        long long mi = minutes
        long long ss = seconds
        long long us = milliseconds * 1_000 + microseconds
    us += ((ordinal + days + weeks * 7 - EPOCH_DAY) * 86_400 + hh * 3_600 + mi * 60 + ss) * 1_000_000
    return date_fr_us(us)

# datetime.datetime ------------------------------------------------------------------------------------
# . generate
cdef inline datetime.datetime dt_new(
    int year=1, int month=1, int day=1,
    int hour=0, int minute=0, int second=0,
    int microsecond=0, object tz=None, int fold=0,
):
    """Create a new `<'datetime.datetime'>`.
    
    Equivalent to:
    >>> datetime.datetime(year, month, day, hour, minute, second, microsecond, tz, fold)
    """
    year = min(max(year, 1), 9_999)
    month = min(max(month, 1), 12)
    day = min(max(day, 1), days_in_month(year, month))
    hour = min(max(hour, 0), 23)
    minute = min(max(minute, 0), 59)
    second = min(max(second, 0), 59)
    microsecond = min(max(microsecond, 0), 999_999)
    return datetime.datetime_new(
        year, month, day, hour, minute, second, 
        microsecond, tz, 1 if fold == 1 else 0,
    )

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
    _tm = tm_localtime(ts)
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

cdef inline int dt_utcoffset_seconds(datetime.datetime dt) except -200_000:
    """Get the tzinfo 'utcoffset' of the datetime in total seconds `<'int'>`.
    
    #### Returns `-100_000` if utcoffset is None.

    Equivalent to:
    >>> dt.utcoffset().total_seconds()
    """
    return tz_utcoffset_seconds(dt.tzinfo, dt)

cdef inline str dt_utcformat(datetime.datetime dt):
    """Get the tzinfo of the datetime as UTC format '+/-HH:MM' `<'str/None'>`."""
    return tz_utcformat(dt.tzinfo, dt)

# . value check
cdef inline bint dt_is_1st_of_year(datetime.datetime dt) except -1:
    """Check if datetime is the 1st day of the year `<'bool'>`
    
    First day of the year: XXXX-01-01
    """
    return datetime.datetime_month(dt) == 1 and datetime.datetime_day(dt) == 1

cdef inline bint dt_is_lst_of_year(datetime.datetime dt) except -1:
    """Check if datetime is the last day of the year `<'bool'>`
    
    Last day of the year: XXXX-12-31
    """
    return datetime.datetime_month(dt) == 12 and datetime.datetime_day(dt) == 31

cdef inline bint dt_is_1st_of_quarter(datetime.datetime dt) except -1:
    """Check if datetime is the 1st day of the quarter `<'bool'>`.
    
    First day of the quarter: XXXX-(1, 4, 7, 10)-01
    """
    if datetime.datetime_day(dt) != 1:
        return False
    cdef int mm = datetime.datetime_month(dt)
    return mm == quarter_1st_month(mm)

cdef inline bint dt_is_lst_of_quarter(datetime.datetime dt) except -1:
    """Check if datetime is the 1st day of the quarter `<'bool'>`.
    
    Last day of the quarter: XXXX-(3, 6, 9, 12)-(30, 31)
    """
    cdef int mm = datetime.datetime_month(dt)
    if mm != quarter_lst_month(mm):
        return False
    cdef int yy = datetime.datetime_year(dt)
    cdef int dd = datetime.datetime_day(dt)
    return dd == days_in_month(yy, mm)

cdef inline bint dt_is_1st_of_month(datetime.datetime dt) except -1:
    """Check if datetime is the 1st day of the month `<'bool'>`.
    
    First day of the month: XXXX-XX-01
    """
    return datetime.datetime_day(dt) == 1

cdef inline bint dt_is_lst_of_month(datetime.datetime dt) except -1:
    """Check if datetime is the last day of the month `<'bool'>`.
    
    Last day of the month: XXXX-XX-(28, 29, 30, 31)
    """
    cdef:
        int yy = datetime.datetime_year(dt)
        int mm = datetime.datetime_month(dt)
        int dd = datetime.datetime_day(dt)
    return dd == days_in_month(yy, mm)

cdef inline bint dt_is_start_of_time(datetime.datetime dt) except -1:
    """Check if datetime is at start of the time `<'bool'>`.
    
    Start of time: 00:00:00.000000
    """
    return (
        datetime.datetime_hour(dt) == 0
        and datetime.datetime_minute(dt) == 0
        and datetime.datetime_second(dt) == 0
        and datetime.datetime_microsecond(dt) == 0
    )

cdef inline bint dt_is_end_of_time(datetime.datetime dt) except -1:
    """Check if datetime is at end of the time `<'bool'>`.
    
    End of time: 23:59:59.999999
    """
    return (
        datetime.datetime_hour(dt) == 23
        and datetime.datetime_minute(dt) == 59
        and datetime.datetime_second(dt) == 59
        and datetime.datetime_microsecond(dt) == 999_999
    )

# . conversion
cdef inline tm dt_to_tm(datetime.datetime dt, bint utc=False) except *:
    """Convert datetime.datetime to `<'struct:tm'>`."""
    cdef:
        object tz = dt.tzinfo
        int yy, mm, dd, isdst
    
    # No timezone
    if tz is None:
        isdst = 0 if utc else -1
        yy, mm, dd = dt.year, dt.month, dt.day

    # With timezone
    else:
        if utc:
            utc_off = tz_utcoffset(tz, dt)
            if utc_off is not None:
                dt = dt_add(
                    dt,
                    -datetime.timedelta_days(utc_off),
                    -datetime.timedelta_seconds(utc_off),
                    -datetime.timedelta_microseconds(utc_off),
                    0, 0, 0, 0
                )
            isdst = 0
        else:
            dst_off = tz_dst(tz, dt)
            if dst_off is None:
                isdst = -1
            elif dst_off:
                isdst = 1
            else:
                isdst = 0
        yy, mm, dd = dt.year, dt.month, dt.day

    # Create 'struct:tm'
    return tm(
        dt.second, dt.minute, dt.hour,  # SS, MM, HH
        dd, mm, yy,  # DD, MM, YY
        ymd_weekday(yy, mm, dd),  # wday
        days_bf_month(yy, mm) + dd,  # yday 
        isdst,  # isdst
    )
    
cdef inline str dt_to_strformat(datetime.datetime dt, str fmt):
    """Convert datetime.datetime to string with the specified 'fmt' `<'str'>`.
    
    Equivalent to:
    >>> dt.strftime(fmt)
    """
    cdef:
        list fmt_l = []
        Py_ssize_t idx = 0
        Py_ssize_t size = str_len(fmt)
        str frepl = None
        str zrepl = None
        str Zrepl = None
        Py_UCS4 ch

    # Escape format
    while idx < size:
        ch = str_read(fmt, idx)
        idx += 1
        # Identifier '%'
        if ch == "%":
            if idx < size:
                ch = str_read(fmt, idx)
                idx += 1
                # us fraction
                if ch == "f":
                    if frepl is None:
                        frepl = "%06d" % dt.microsecond
                    fmt_l.append(frepl)
                # utc offset
                elif ch == "z":
                    if zrepl is None:
                        utcfmt = dt_utcformat(dt)
                        zrepl = "" if utcfmt is None else utcfmt
                    fmt_l.append(zrepl)
                # tzname
                elif ch == "Z":
                    if Zrepl is None:
                        tzname = dt_tzname(dt)
                        Zrepl = "" if tzname is None else tzname
                    fmt_l.append(Zrepl)
                # rest
                else:
                    fmt_l.append("%")
                    fmt_l.append(str_chr(ch))
            else:
                fmt_l.append("%")
        else:
            fmt_l.append(str_chr(ch))

    # Format to string
    return tm_strftime(dt_to_tm(dt, False), "".join(fmt_l))

cdef inline str dt_to_isoformat(datetime.datetime dt, str sep="T", bint utc=False):
    """Convert datetime.datetime to ISO format `<'str'>`.

    If 'dt' is timezone-aware, setting 'utc=True' 
    adds the UTC(Z) at the end of the ISO format.
    """
    cdef int us = dt.microsecond
    if us == 0:
        if utc:
            utc_fmt = dt_utcformat(dt)
            if utc_fmt is not None:
                return "%04d-%02d-%02d%s%02d:%02d:%02d%s" % (
                    dt.year, dt.month, dt.day, sep,
                    dt.hour, dt.minute, dt.second, utc_fmt,
            )
        return "%04d-%02d-%02d%s%02d:%02d:%02d" % (
            dt.year, dt.month, dt.day, sep,
            dt.hour, dt.minute, dt.second,
        )
    else:
        if utc:
            utc_fmt = dt_utcformat(dt)
            if utc_fmt is not None:
                return "%04d-%02d-%02d%s%02d:%02d:%02d.%06d%s" % (
                    dt.year, dt.month, dt.day, sep,
                    dt.hour, dt.minute, dt.second, us, utc_fmt,
                )    
        return "%04d-%02d-%02d%s%02d:%02d:%02d.%06d" % (
            dt.year, dt.month, dt.day, sep,
            dt.hour, dt.minute, dt.second, us,
        )

cdef inline int dt_to_ordinal(datetime.datetime dt, bint utc=False) except -1:
    """Convert datetime.datetime to ordinal days `<'int'>`.
    
    If 'dt' is timezone-aware, setting 'utc=True' 
    substracts 'utcoffset' from total ordinal days.
    """
    cdef:
        int ordinal = ymd_to_ordinal(dt.year, dt.month, dt.day)
        int seconds

    if utc:
        utc_off = dt_utcoffset(dt)
        if utc_off is not None:
            seconds = dt.hour * 3_600 + dt.minute * 60 + dt.second
            seconds -= datetime.timedelta_days(utc_off) * 86_400
            seconds -= datetime.timedelta_seconds(utc_off)
            if seconds >= 86_400:
                ordinal += 1
            elif seconds < 0:
                ordinal -= 1 
    return ordinal
        
cdef inline double dt_to_seconds(datetime.datetime dt, bint utc=False):
    """Convert datetime.datetime to total seconds since Unix Epoch `<'float'>`.
    
    If 'dt' is timezone-aware, setting 'utc=True' 
    substracts 'utcoffset' from total seconds.
    """
    cdef:
        double ordinal = dt_to_ordinal(dt, False)
        double hh = dt.hour
        double mi = dt.minute
        double ss = dt.second
        double us = dt.microsecond
        double seconds = (
            (ordinal - EPOCH_DAY) * 86_400
            + hh * 3_600 + mi * 60 + ss + us / 1_000_000
        )

    if utc:
        utc_off = dt_utcoffset(dt)
        if utc_off is not None:
            seconds -= td_to_seconds(utc_off)
    return seconds

cdef inline long long dt_to_us(datetime.datetime dt, bint utc=False):
    """Convert datetime.datetime to total microseconds since Unix Epoch `<'int'>`.

    If 'dt' is timezone-aware, setting 'utc=True' 
    substracts 'utcoffset' from total mircroseconds.
    """
    cdef:
        long long ordinal = dt_to_ordinal(dt, False)
        long long hh = dt.hour
        long long mi = dt.minute
        long long ss = dt.second
        long long us = (
            (ordinal - EPOCH_DAY) * 86_400
            + hh * 3_600 + mi * 60 + ss
        ) * 1_000_000 + dt.microsecond
    if utc:
        utc_off = dt_utcoffset(dt)
        if utc_off is not None:
            us -= td_to_us(utc_off)
    return us

cdef inline long long dt_to_posix(datetime.datetime dt):
    """Convert datetime.datetime to POSIX timestamp `<'int'>`.

    This function does not take 'dt.tzinof' into consideration.

    Equivalent to:
    >>> dt._mktime()
    """
    cdef:
        long long t = int(dt_to_seconds(dt, False))
        long long off1 = ts_localtime(t) - t
        long long u1 = t - off1
        long long t1 = ts_localtime(u1)
        long long off2, u2, t2
    
    # Adjustment for Daylight Saving
    if t == t1:
        # We found one solution, but it may not be the one we need.
        # Look for an earlier solution (if `fold` is 0), or a later
        # one (if `fold` is 1).
        u2 = u1 - 86_400 if dt.fold == 0 else u1 + 86_400
        off2 = ts_localtime(u2) - u2
        if off1 == off2:
            return u1
    else:
        off2 = t1 - u1
    
    # Final adjustment
    u2 = t - off2
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
    """Convert datetime.datetime to timestamp `<'float'>`.
    
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
    ts = dt_to_posix(dt)
    cdef double us = dt.microsecond
    return ts + us / 1_000_000

cdef inline datetime.datetime dt_combine(datetime.date date=None, datetime.time time=None, tz: object = None):
    """Combine datetime.date & datetime.time to `<'datetime.datetime'>`.
    
    - If 'date' is None, use current local date.
    - If 'time' is None, all time fields set to 0.
    """
    # Date
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
    """Convert datetime.date to `<'datetime.datetime'>`.
    
    #### All time values sets to 0.
    """
    return datetime.datetime_new(date.year, date.month, date.day, 0, 0, 0, 0, tz, 0)

cdef inline datetime.datetime dt_fr_dt(datetime.datetime dt):
    """Convert subclass of datetime to `<'datetime.datetime'>`."""
    return datetime.datetime_new(
        dt.year, dt.month, dt.day, 
        dt.hour, dt.minute, dt.second, 
        dt.microsecond, dt.tzinfo, dt.fold
    )

cdef inline datetime.datetime dt_fr_time(datetime.time time):
    """Convert datetime.time to `<'datetime.datetime'>`.
    
    #### Date values sets to 1970-01-01.
    """
    return datetime.datetime_new(
        1970, 1, 1, 
        time.hour, time.minute, time.second, 
        time.microsecond, time.tzinfo, time.fold
    )

cdef inline datetime.datetime dt_fr_ordinal(int ordinal, object tz=None):
    """Convert ordinal days to `<'datetime.datetime'>`."""
    _ymd = ymd_fr_ordinal(ordinal)
    return datetime.datetime_new(_ymd.year, _ymd.month, _ymd.day, 0, 0, 0, 0, tz, 0)

cdef inline datetime.datetime dt_fr_seconds(double seconds, object tz=None):
    """Convert total seconds since Unix Epoch to `<'datetime.datetime'>`."""
    cdef long long us = int(seconds * 1_000_000)
    return dt_fr_us(us, tz)
    
cdef inline datetime.datetime dt_fr_us(long long us, object tz=None):
    """Convert total microseconds since Unix Epoch to `<'datetime.datetime'>`."""
    # Add back Epoch
    cdef unsigned long long _us = min(max(us + EPOCH_US, DT_US_MIN), DT_US_MAX)

    # Calculate ymd & hms
    # since '_us' is positive, we can safely divide
    # without checking for negative values.
    cdef int ordinal
    with cython.cdivision(True):
        ordinal = _us // US_DAY
    _ymd = ymd_fr_ordinal(ordinal)
    _hms = hms_fr_us(_us)

    # Create datetime
    return datetime.datetime_new(
        _ymd.year, _ymd.month, _ymd.day, 
        _hms.hour, _hms.minute, _hms.second, 
        _hms.microsecond, tz, 0
    )

cdef inline datetime.datetime dt_fr_ts(double ts, object tz=None):
    """Convert timestamp to `<'datetime.datetime'>`."""
    return datetime.datetime_from_timestamp(ts, tz)

# . manipulation
cdef inline datetime.datetime dt_replace(
    datetime.datetime dt,
    int year=-1, int month=-1, int day=-1,
    int hour=-1, int minute=-1, int second=-1,
    int millisecond=-1, int microsecond=-1,
    object tz=-1, int fold=-1,
):
    """Replace the datetime.datetime values `<'datetime.datetime'>`.

    #### Default '-1' mean keep the original value.
    
    Equivalent to:
    >>> dt.replace(year, month, day, hour, minute, second, microsecond, tz, fold)
    """
    # Current values
    yy: cython.int = datetime.datetime_year(dt)
    mm: cython.int = datetime.datetime_month(dt)
    dd: cython.int = datetime.datetime_day(dt)
    hh: cython.int = datetime.datetime_hour(dt)
    mi: cython.int = datetime.datetime_minute(dt)
    ss: cython.int = datetime.datetime_second(dt)
    us: cython.int = datetime.datetime_microsecond(dt)
    tz_ = datetime.datetime_tzinfo(dt)
    fd_: cython.int = datetime.datetime_fold(dt)

    # New values
    new_yy: cython.int = yy if year < 1 else min(year, 9_999)
    new_mm: cython.int = mm if month < 1 else min(month, 12)
    new_dd: cython.int = dd if day < 1 else min(day, days_in_month(new_yy, new_mm))
    new_hh: cython.int = hh if hour < 0 else min(hour, 23)
    new_mi: cython.int = mi if minute < 0 else min(minute, 59)
    new_ss: cython.int = ss if second < 0 else min(second, 59)
    new_us: cython.int = us if microsecond < 0 else microsecond
    new_us = combine_abs_ms_us(millisecond, new_us)
    new_tz = tz_ if not (tz is None or is_tz(tz)) else tz
    new_fd: cython.int = fd_ if not fold in (0, 1) else fold

    # Same values
    if (
        new_yy == yy and new_mm == mm and new_dd == dd and
        new_hh == hh and new_mi == mi and new_ss == ss and
        new_us == us and new_tz is tz_ and new_fd == fd_
    ):
        return dt  # exit

    # Create new datetime
    return datetime.datetime_new(
        new_yy, new_mm, new_dd, 
        new_hh, new_mi, new_ss, 
        new_us, new_tz, new_fd
    )

cdef inline datetime.datetime dt_replace_date(
    datetime.datetime dt,
    int year=-1, int month=-1, int day=-1,
):
    """Replace datetime.datetime date component values `<'datetime.datetime'>`.

    #### Default '-1' mean keep the original value.
    
    Equivalent to:
    >>> dt.replace(year, month, day)
    """
    # Current values
    yy: cython.int = datetime.datetime_year(dt)
    mm: cython.int = datetime.datetime_month(dt)
    dd: cython.int = datetime.datetime_day(dt)

    # New values
    new_yy: cython.int = yy if year < 1 else min(year, 9_999)
    new_mm: cython.int = mm if month < 1 else min(month, 12)
    new_dd: cython.int = dd if day < 1 else min(day, days_in_month(new_yy, new_mm))

    # Same values
    if new_yy == yy and new_mm == mm and new_dd == dd:
        return dt  # exit

    # Create new datetime
    return datetime.datetime_new(
        new_yy, new_mm, new_dd, 
        datetime.datetime_hour(dt), datetime.datetime_minute(dt), 
        datetime.datetime_second(dt), datetime.datetime_microsecond(dt),
        datetime.datetime_tzinfo(dt), datetime.datetime_fold(dt),
    )


cdef inline datetime.datetime dt_replace_time(
    datetime.datetime dt,
    int hour=-1, int minute=-1, int second=-1,
    int millisecond=-1, int microsecond=-1,
):
    """Replace datetime.datetime time component values `<'datetime.datetime'>`.

    #### Default '-1' mean keep the original value.
    
    Equivalent to:
    >>> dt.replace(hour, minute, second, microsecond)
    """
    # Current values
    hh: cython.int = datetime.datetime_hour(dt)
    mi: cython.int = datetime.datetime_minute(dt)
    ss: cython.int = datetime.datetime_second(dt)
    us: cython.int = datetime.datetime_microsecond(dt)

    # New values
    new_hh: cython.int = hh if hour < 0 else min(hour, 23)
    new_mi: cython.int = mi if minute < 0 else min(minute, 59)
    new_ss: cython.int = ss if second < 0 else min(second, 59)
    new_us: cython.int = us if microsecond < 0 else microsecond
    new_us = combine_abs_ms_us(millisecond, new_us)

    # Same values
    if new_hh == hh and new_mi == mi and new_ss == ss and new_us == us:
        return dt  # exit

    # Create new datetime
    return datetime.datetime_new(
        datetime.datetime_year(dt),
        datetime.datetime_month(dt),
        datetime.datetime_day(dt),
        new_hh, new_mi, new_ss, new_us,
        datetime.datetime_tzinfo(dt),
        datetime.datetime_fold(dt),
    )

cdef inline datetime.datetime dt_replace_tz(datetime.datetime dt, object tz):
    """Replace the datetime.datetime timezone `<'datetime.datetime'>`.

    Equivalent to:
    >>> dt.replace(tzinfo=tz)
    """
    # Same tzinfo
    if tz is datetime.datetime_tzinfo(dt):
        return dt

    # Create new datetime
    return datetime.datetime_new(
        dt.year, dt.month, dt.day, 
        dt.hour, dt.minute, dt.second, 
        dt.microsecond, tz, dt.fold,
    )

cdef inline datetime.datetime dt_replace_fold(datetime.datetime dt, int fold):
    """Replace the datetime.datetime fold `<'datetime.datetime'>`.

    Equivalent to:
    >>> dt.replace(fold=fold)
    """
    # Same fold
    if fold not in (0, 1) or fold == dt.fold:
        return dt

    # Create new datetime
    return datetime.datetime_new(
        dt.year, dt.month, dt.day, 
        dt.hour, dt.minute, dt.second, 
        dt.microsecond, dt.tzinfo, fold,
    )

cdef inline datetime.datetime dt_chg_weekday(datetime.datetime dt, int weekday):
    """Change datetime.datetime 'weekday' within the current week 
    (0[Monday]...6[Sunday]) `<'datetime.date'>`.

    Equivalent to:
    >>> dt + datetime.timedelta(weekday - dt.weekday())
    """
    cdef int curr_wday = ymd_weekday(dt.year, dt.month, dt.day)
    weekday = min(max(weekday, 0), 6)
    if curr_wday == weekday:
        return dt

    cdef int ordinal = dt_to_ordinal(dt, False)
    _ymd = ymd_fr_ordinal(ordinal + weekday - curr_wday)
    return datetime.datetime_new(
        _ymd.year, _ymd.month, _ymd.day, 
        dt.hour, dt.minute, dt.second, 
        dt.microsecond, dt.tzinfo, dt.fold,
    )

cdef inline datetime.datetime dt_astimezone(datetime.datetime dt, object tz=None):
    """Change the timezone for `<'datetime.datetime'>`.
    
    Equivalent to:
    >>> dt.astimezone(tz)
    """
    if tz is None:
        tz = tz_local(dt)
        mytz = datetime.datetime_tzinfo(dt)
        if mytz is None:
            # since 'self' is naive, we simply
            # localize to the local timezone.
            return dt_replace_tz(dt, tz)  # exit
    else:
        mytz = datetime.datetime_tzinfo(dt)

    if mytz is None:
        mytz = tz_local(dt)
        if tz is mytz:
            return dt  # exit
        myoffset = tz_utcoffset(mytz, dt)
    elif tz is mytz:
        return dt  # exit
    else:
        myoffset = tz_utcoffset(mytz, dt)
        if myoffset is None:
            mytz = tz_local(dt_replace_tz(dt, None))
            if tz is mytz:
                return dt  # exit
            myoffset = tz_utcoffset(mytz, dt)

    # Convert to UTC time
    cdef long long us = dt_to_us(dt, False)
    us -= td_to_us(myoffset)
    dt = dt_fr_us(us, tz)

    # Convert from UTC to tz's local time
    return tz.fromutc(dt)

# . arithmetic
cdef inline datetime.datetime dt_add(
    datetime.datetime dt,
    int days=0, int seconds=0, int microseconds=0,
    int milliseconds=0, int minutes=0, int hours=0, int weeks=0,
):
    """Add timedelta to datetime.datetime `<'datetime.datetime'>`.
    
    Equivalent to:
    >>> dt + datetime.timedelta(
            days, seconds, microseconds, 
            milliseconds, minutes, hours, weeks
        )
    """
    # No change
    if days == seconds == microseconds == milliseconds == minutes == hours == weeks == 0:
        return dt

    # Add timedelta
    cdef: 
        long long ordinal = dt_to_ordinal(dt, False)
        long long hh = dt.hour + hours
        long long mi = dt.minute + minutes
        long long ss = dt.second + seconds
        long long us = milliseconds * 1_000 + dt.microsecond + microseconds
    us += ((ordinal + days + weeks * 7 - EPOCH_DAY) * 86_400 + hh * 3_600 + mi * 60 + ss) * 1_000_000
    return dt_fr_us(us, dt.tzinfo)
    
    
# datetime.time ----------------------------------------------------------------------------------------
# . generate
cdef inline datetime.time time_new(
    int hour=0, int minute=0, int second=0,
    int microsecond=0, object tz=None, int fold=0,
):
    """Create a new `<'datetime.time'>`.
    
    Equivalent to:
    >>> datetime.time(hour, minute, second, microsecond, tz, fold)
    """
    hour = min(max(hour, 0), 23)
    minute = min(max(minute, 0), 59)
    second = min(max(second, 0), 59)
    microsecond = min(max(microsecond, 0), 999_999)
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
    _tm = tm_localtime(ts)
    return datetime.time_new(
        _tm.tm_hour, _tm.tm_min, _tm.tm_sec, 
        us % 1_000_000, None, 0
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

cdef inline str time_utcformat(datetime.time time):
    """Get the tzinfo of the datetime.time as UTC format '+/-HH:MM' `<'str/None'>`."""
    return tz_utcformat(time.tzinfo, None)

# . conversion
cdef inline tm time_to_tm(datetime.time time, bint utc=False) except *:
    """Convert datetime.time to `<'struct:tm'>`.

    #### Date values sets to 1970-01-01.

    If 'time' is timezone-aware, setting 'utc=True', checks 'isdst'
    and substracts 'utcoffset' from the time before conversion.
    """
    # Without Timezone
    if time.tzinfo is None:
        return tm(
            time.second, time.minute, time.hour,  # SS, MM, HH
            1, 1, 1970,  # DD, MM, YY
            3, 1, -1, # wday, yday, isdst
        )

    # With Timezone
    return dt_to_tm(dt_fr_time(time), utc)

cdef inline str time_to_strformat(datetime.time time, str fmt):
    """Convert datetime.time to str with the specified 'fmt' `<'str'>`.
    
    Equivalent to:
    >>> time.strftime(fmt)
    """
    cdef:
        list fmt_l = []
        Py_ssize_t idx = 0
        Py_ssize_t size = str_len(fmt)
        str frepl = None
        str zrepl = None
        str Zrepl = None
        Py_UCS4 ch

    # Escape format
    while idx < size:
        ch = str_read(fmt, idx)
        idx += 1
        # Identifier '%'
        if ch == "%":
            if idx < size:
                ch = str_read(fmt, idx)
                idx += 1
                # us fraction
                if ch == "f":
                    if frepl is None:
                        frepl = "%06d" % time.microsecond
                    fmt_l.append(frepl)
                # utc offset
                elif ch == "z":
                    if zrepl is None:
                        utcfmt = time_utcformat(time)
                        zrepl = "" if utcfmt is None else utcfmt
                    fmt_l.append(zrepl)
                # tzname
                elif ch == "Z":
                    if Zrepl is None:
                        tzname = time_tzname(time)
                        Zrepl = "" if tzname is None else tzname
                    fmt_l.append(Zrepl)
                # rest
                else:
                    fmt_l.append("%")
                    fmt_l.append(str_chr(ch))
            else:
                fmt_l.append("%")
        else:
            fmt_l.append(str_chr(ch))

    # Format to string
    return tm_strftime(time_to_tm(time, False), "".join(fmt_l))

cdef inline str time_to_isoformat(datetime.time time, bint utc=False):
    """Convert datetime.time to ISO format `<'str'>`.

    If 'time' is timezone-aware, setting 'utc=True' 
    adds the UTC(Z) at the end of the ISO format.
    """
    cdef:
        int us = time.microsecond
        str utc_fmt

    if us == 0:
        if utc:
            utc_fmt = time_utcformat(time)
            if utc_fmt is not None:
                return "%02d:%02d:%02d%s" % (time.hour, time.minute, time.second, utc_fmt)
        return "%02d:%02d:%02d" % (time.hour, time.minute, time.second)
    else:
        if utc:
            utc_fmt = time_utcformat(time)
            if utc_fmt is not None:
                return "%02d:%02d:%02d.%06d%s" % (time.hour, time.minute, time.second, us, utc_fmt)
        return "%02d:%02d:%02d.%06d" % (time.hour, time.minute, time.second, us)

cdef inline double time_to_seconds(datetime.time time, bint utc=False):
    """Convert datetime.time to total seconds `<'float'>`.
    
    If 'time' is timezone-aware, setting 'utc=True' 
    substracts 'utcoffset' from total seconds.
    """
    cdef:
        double hh = time.hour
        double mi = time.minute
        double ss = time.second
        double us = time.microsecond
        double seconds = hh * 3_600 + mi * 60 + ss + us / 1_000_000

    if utc:
        utc_off = time_utcoffset(time)
        if utc_off is not None:
            seconds -= td_to_seconds(utc_off)
    return seconds

cdef inline long long time_to_us(datetime.time time, bint utc=False):
    """Convert datetime.time to total microseconds `<'int'>`.

    If 'time' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from total mircroseconds.
    """
    cdef:
        long long hh = time.hour
        long long mi = time.minute
        long long ss = time.second
        long long us = (hh * 3_600 + mi * 60 + ss) * 1_000_000 + time.microsecond

    if utc:
        utc_off = time_utcoffset(time)
        if utc_off is not None:
            us -= td_to_us(utc_off)
    return us
    
cdef inline datetime.time time_fr_dt(datetime.datetime dt):
    """Convert datetime.datetime to `<'datetime.time'>`."""
    return datetime.time_new(
        dt.hour, dt.minute, dt.second, 
        dt.microsecond, dt.tzinfo, dt.fold
    )

cdef inline datetime.time time_fr_time(datetime.time time):
    """Convert subclass of datetime.time to `<'datetime.time'>`."""
    return datetime.time_new(
        time.hour, time.minute, time.second, 
        time.microsecond, time.tzinfo, time.fold
    )

cdef inline datetime.time time_fr_seconds(double seconds, object tz=None):
    """Convert total seconds to `<'datetime.time'>`."""
    cdef long long us = int(seconds * 1_000_000)
    return time_fr_us(us, tz)

cdef inline datetime.time time_fr_us(long long us, object tz=None):
    """Convert total microseconds to `<'datetime.time'>`."""
    _hms = hms_fr_us(us)
    return datetime.time_new(
        _hms.hour, _hms.minute, _hms.second, _hms.microsecond, tz, 0
    )

# . manipulation
cdef inline datetime.time time_replace(
    datetime.time time,
    int hour=-1, int minute=-1, int second=-1,
    int microsecond=-1, object tz=-1, int fold=-1,
):
    """Replace the datetime.time values `<'datetime.time'>`.

    #### Default '-1' mean keep the original value.

    Equivalent to:
    >>> time.replace(hour, minute, second, microsecond, tz, fold)  
    """
    if not 0 <= hour <= 23:
        hour = time.hour
    if not 0 <= minute <= 59:
        minute = time.minute
    if not 0 <= second <= 59:
        second = time.second
    if not 0 <= microsecond <= 999_999:
        microsecond = time.microsecond
    if not (is_tz(tz) or tz is None):
        tz = time.tzinfo
    if not fold in (0, 1):
        fold = time.fold
    return datetime.time_new(hour, minute, second, microsecond, tz, fold)

cdef inline datetime.time time_replace_tz(datetime.time time, object tz):
    """Replace the datetime.time timezone `<'datetime.time'>`.

    Equivalent to:
    >>> time.replace(tzinfo=tz)
    """
    # Same tzinfo
    if tz is time.tzinfo:
        return time

    # Create new time
    return datetime.time_new(
        time.hour, time.minute, time.second, 
        time.microsecond, tz, time.fold
    )

cdef inline datetime.time time_replace_fold(datetime.time time, int fold):
    """Replace the datetime.time fold `<'datetime.time'>`.

    Equivalent to:
    >>> time.replace(fold=fold)
    """
    # Same fold
    if fold == time.fold:
        return time

    # Create new time
    return datetime.time_new(
        time.hour, time.minute, time.second, 
        time.microsecond, time.tzinfo, 1 if fold == 1 else 0
    )

# datetime.timedelta -----------------------------------------------------------------------------------
# . generate
cdef inline datetime.timedelta td_new(int days=0, int seconds=0, int microseconds=0):
    """Create a new `<'datetime.timedelta'>`.
    
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
    """Convert datetime.timedelta to ISO format `<'str'>`."""
    cdef:
        long long seconds = td.day * 86_400 + td.second
        long long us = td.microsecond
    
    # Positive timedelta
    if seconds >= 0:
        hours = seconds // 3_600
        seconds %= 3_600
        minutes = seconds // 60
        seconds %= 60
        if us == 0:
            return "%02d:%02d:%02d" % (hours, minutes, seconds)
        else:
            return "%02d:%02d:%02d.%06d" % (hours, minutes, seconds, us)

    # Negative w/o microseconds
    elif us == 0:
        seconds = -seconds
        if seconds >= 3_600:
            hours = seconds // 3_600
            seconds %= 3_600
            minutes = seconds // 60
            seconds %= 60
            return "-%02d:%02d:%02d" % (hours, minutes, seconds)
        elif seconds >= 60:
            minutes = seconds // 60
            seconds %= 60
            return "-00:%02d:%02d" % (minutes, seconds)
        else:
            return "-00:00:%02d" % seconds

    # Negative w/ microseconds
    else:
        us = -(seconds * 1_000_000 + us)
        if us >= US_HOUR:
            with cython.cdivision(True):
                hours = us // US_HOUR
                us = us % US_HOUR
            minutes = us // 60_000_000
            us %= 60_000_000
            seconds = us // 1_000_000
            us %= 1_000_000
            return "-%02d:%02d:%02d.%06d" % (hours, minutes, seconds, us)
        elif us >= 60_000_000:
            minutes = us // 60_000_000
            us %= 60_000_000
            seconds = us // 1_000_000
            us %= 1_000_000
            return "-00:%02d:%02d.%06d" % (minutes, seconds, us)
        elif us >= 1_000_000:
            seconds = us // 1_000_000
            us %= 1_000_000
            return "-00:00:%02d.%06d" % (seconds, us)
        else:
            return "-00:00:00.%06d" % us

cdef inline str td_to_utcformat(datetime.timedelta td):
    """Convert datetime.timedelta to UTC format '+/-HH:MM' `<'str'>`."""
    cdef long long seconds = td.day * 86_400 + td.second
    # Positive timedelta
    if seconds >= 0:
        hours = min(seconds // 3_600, 23)
        minutes = seconds % 3_600 // 60
        return "+%02d:%02d" % (hours, minutes)
    # Negative timedelta
    else:
        seconds = -seconds
        hours = min(seconds // 3_600, 23)
        minutes = seconds % 3_600 // 60
        return "-%02d:%02d" % (hours, minutes)

cdef inline double td_to_seconds(datetime.timedelta td):
    """Convert datetime.timedelta to total seconds `<'float'>`."""
    cdef:
        double days = td.day
        double seconds = td.second
        double us = td.microsecond
    return days * 86_400 + seconds + us / 1_000_000

cdef inline long long td_to_us(datetime.timedelta td):
    """Convert datetime.timedelta to total microseconds `<'int'>`."""
    cdef:
        long long days = td.day
        long long seconds = td.second
        long long us = td.microsecond
    return days * US_DAY + seconds * 1_000_000 + us

cdef inline datetime.timedelta td_fr_td(datetime.timedelta td):
    """Convert subclass of datetime.timedelta to `<'datetime.timedelta'>`."""
    return datetime.timedelta_new(td.day, td.second, td.microsecond)

cdef inline datetime.timedelta td_fr_seconds(double seconds):
    """Convert total seconds to `<'datetime.timedelta'>`."""
    cdef long long us = int(seconds * 1_000_000)
    return td_fr_us(us)

cdef inline datetime.timedelta td_fr_us(long long us):
    """Convert total microseconds to `<'datetime.timedelta'>`."""
    cdef long long days = us // 86_400_000 // 1_000
    us -= days * 86_400_000 * 1_000
    cdef long long seconds = us // 1_000_000
    us %= 1_000_000
    return datetime.timedelta_new(days, seconds, us)

# datetime.tzinfo --------------------------------------------------------------------------------------
# . generate
cdef inline object tz_new(int hours=0, int minites=0, int seconds=0):
    """Create a new `<'datetime.tzinfo'>`.
    
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
    """Get the local `<'datetime.tzinfo'>`."""
    cdef int gmtoff = tz_local_seconds(dt)
    return datetime.timezone_new(datetime.timedelta_new(0, gmtoff, 0), None)

cdef inline int tz_local_seconds(datetime.datetime dt=None) except -200_000:
    """Get the local timezone offset in total seconds `<'int'>`."""
    # Convert to timestamp
    cdef:
        double ts
        long long ts1, ts2
    if dt is not None:
        if dt.tzinfo is None:
            ts1 = dt_to_posix(dt)
            # . detect gap
            ts2 = dt_to_posix(dt_replace_fold(dt, 1-dt.fold))
            if ts2 != ts1 and (ts2 > ts1) == dt.fold:
                ts = ts2
            else:
                ts = ts1
        else:
            ts = dt_to_seconds(dt, True)
    else:
        ts = unix_time()

    # Calculate offset
    loc, gmt = tm_localtime(ts), tm_gmtime(ts)
    return (
        (loc.tm_mday - gmt.tm_mday) * 86_400
        + (loc.tm_hour - gmt.tm_hour) * 3_600 
        + (loc.tm_min - gmt.tm_min) * 60
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
    """Get the name of the tzinfo `<'str/None'>`.
    
    Equivalent to:
    >>> tz.tzname(dt)
    """
    if tz is None:
        return None
    try:
        return tz.tzname(dt)
    except Exception as err:
        if not datetime.PyTZInfo_Check(tz):
            raise TypeError("expects <'datetime.tzinfo'>, got %s." % type(tz)) from err
        raise err

cdef inline datetime.timedelta tz_dst(object tz, datetime.datetime dt=None):
    """Get the 'dst' of the tzinfo `<'datetime.timedelta'>`.

    Equivalent to:
    >>> tz.dst(dt)
    """
    if tz is None:
        return None
    try:
        return tz.dst(dt)
    except Exception as err:
        if not datetime.PyTZInfo_Check(tz):
            raise TypeError("expects <'datetime.tzinfo'>, got %s." % type(tz)) from err
        raise err

cdef inline datetime.timedelta tz_utcoffset(object tz, datetime.datetime dt=None):
    """Get the 'utcoffset' of the tzinfo `<'datetime.timedelta'>`.

    Equivalent to:
    >>> tz.utcoffset(dt)
    """
    if tz is None:
        return None
    try:
        return tz.utcoffset(dt)
    except Exception as err:
        if not datetime.PyTZInfo_Check(tz):
            raise TypeError("expects <'datetime.tzinfo'>, got %s." % type(tz)) from err
        raise err

cdef inline int tz_utcoffset_seconds(object tz, datetime.datetime dt=None) except -200_000:
    """Get the 'utcoffset' of the tzinfo in total seconds `<'int'>`.

    #### Returns `-100_000` if utcoffset is None.

    Equivalent to:
    >>> tz.utcoffset(dt).total_seconds()
    """
    offset = tz_utcoffset(tz, dt)
    # No offset
    if offset is None:
        return -100_000
    
    # Convert to seconds
    return (
        datetime.timedelta_days(offset) * 86_400 
        + datetime.timedelta_seconds(offset)
    )

cdef inline str tz_utcformat(object tz, datetime.datetime dt=None):
    """Access datetime.tzinfo as UTC format '+/-HH:MM' `<'str/None'>`."""
    utc_off = tz_utcoffset(tz, dt)
    return None if utc_off is None else td_to_utcformat(utc_off)

# NumPy: share -----------------------------------------------------------------------------------------
# . time unit
cdef inline str map_nptime_unit_int2str(int unit):
    """Map ndarray[datetime64/timedelta64] unit from integer
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
    raise ValueError("unsupported numpy time unit: %d." % unit)

cdef inline int map_nptime_unit_str2int(str unit):
    """Map ndarray[datetime64/timedelta64] unit from string
    representation to the corresponding integer `<'int'>`."""
    cdef:
        Py_ssize_t size = str_len(unit)
        Py_UCS4 ch

    # Unit: 'ns', 'us', 'ms', 'ps', 'fs', 'as'
    if size == 2:
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

    # Unit: 's', 'm', 'h', 'D', 'Y', 'M', 'W', 'B'
    elif size == 1:
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

    # Unsupported unit
    raise ValueError("unsupported numpy time unit: %s." % unit)

cdef inline int parse_arr_nptime_unit(np.ndarray arr):
    """Parse numpy datetime64/timedelta64 unit from the
    given 'arr', returns the unit in `<'int'>`."""
    cdef:
        str dtype_str = arr.dtype.str
        Py_ssize_t size = str_len(dtype_str)

    if size < 6:
        raise ValueError("unable to parse ndarray time unit from '%s'." % dtype_str)
    dtype_str = str_substr(dtype_str, 4, size - 1)
    try:
        return map_nptime_unit_str2int(dtype_str)
    except ValueError as err:
        raise ValueError("unable to parse ndarray time unit from '%s'." % dtype_str) from err

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
        raise TypeError("expects <'np.datetime64'>, got %s." % type(obj))
    return True

# . conversion
cdef inline tm dt64_to_tm(object dt64) except *:
    """Convert np.datetime64 to `<'struct:tm'>`."""
    return tm_fr_seconds(dt64_to_seconds(dt64))

cdef inline str dt64_to_strformat(object dt64, str fmt, bint strict=True):
    """Convert np.datetime64 to str with the specified 'fmt' `<'str'>`.

    Similar to 'datetime.datetime.strftime(fmt)', 
    but designed work with np.datetime64.

    When 'dt64' resolution is above 'us', such as 'ns', 'ps', etc:
    - If 'strict=True', fraction will be clamp to microseconds.
    - If 'strict=False', fraction will extend to the corresponding unit.

    ### Example:
    >>> dt64 = np.datetime64("1970-01-01T00:00:01.123456789123")  # ps
    >>> dt64_to_strformat(dt64, "%Y/%m/%d %H-%M-%S.%f", True)  # strict=True
    >>> "1970/01/01 00-00-01.123456"
    >>> dt64_to_strformat(dt64, "%Y/%m/%d %H-%M-%S.%f", False)  # strict=False
    >>> "1970/01/01 00-00-01.123456789123"
    """
    # Access unit & value
    validate_dt64(dt64)
    cdef:
        np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(dt64)
        np.npy_datetime value = np.get_datetime64_value(dt64)
        list fmt_l = []
        Py_ssize_t idx = 0
        Py_ssize_t size = str_len(fmt)
        str frepl
        Py_UCS4 ch
        long long us

    # Conversion: common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        us = value // 1_000
        if strict:
            frepl = "%06d" % (us % 1_000_000)
        else:
            frepl = "%09d" % (value % 1_000_000_000)
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        us = value
        frepl = "%06d" % (value % 1_000_000)
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        us = value * 1_000
        frepl = "%06d" % (value % 1_000 * 1_000)
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        us = value * 1_000_000
        frepl = None
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        us = value * 60_000_000
        frepl = None
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        us = value * US_HOUR
        frepl = None
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        us = value * US_DAY
        frepl = None
    # Conversion: uncommon
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        us = value // 1_000_000
        if strict:
            frepl = "%06d" % (us % 1_000_000)
        else:
            frepl = "%012d" % (value
                - (value // 1_000_000_000 // 1_000)
                * 1_000_000_000 * 1_000
            )
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        us = value // 1_000_000_000
        if strict:
            frepl = "%06d" % (us % 1_000_000)
        else:
            frepl = "%015d" % (value
                - (value // 1_000_000_000 // 1_000_000)
                * 1_000_000_000 * 1_000_000
            )
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        us = value // 1_000_000_000 // 1_000
        if strict:
            frepl = "%06d" % (us % 1_000_000)
        else:
            frepl = "%018d" % (value
                - (value // 1_000_000_000 // 1_000_000_000)
                * 1_000_000_000 * 1_000_000_000
            )
    # Unsupported unit
    else:
        raise ValueError(
            "unsupported <'np.datetime64'> unit '%s' for strformat operation."
            % map_nptime_unit_int2str(unit)
        )

    # Escape format
    while idx < size:
        ch = str_read(fmt, idx)
        idx += 1
        # Identifier '%'
        if ch == "%":
            if idx < size:
                ch = str_read(fmt, idx)
                idx += 1
                # us fraction
                if ch == "f":
                    if frepl is not None:
                        fmt_l.append(frepl)
                    else:
                        idx += 1
                # skip utc & tzname
                elif ch in ("z", "Z"):
                    pass
                # rest
                else:
                    fmt_l.append("%")
                    fmt_l.append(str_chr(ch))
            else:
                fmt_l.append("%")
        else:
            fmt_l.append(str_chr(ch))

    # Format to string
    return tm_strftime(tm_fr_us(us), "".join(fmt_l))

cdef inline str dt64_to_isoformat(object dt64, str sep=" ", bint strict=True):
    """Convert np.datetime64 to ISO format `<'str'>`.
    
    When 'dt64' resolution is above 'us', such as 'ns', 'ps', etc:
    - If 'strict=True', fraction will be clamp to microseconds.
    - If 'strict=False', fraction will extend to the corresponding unit.

    ### Example:
    >>> dt64 = np.datetime64("1970-01-01T00:00:01.123456789123")  # ps
    >>> dt64_to_isoformat(dt64, "T", True)  # strict=True
    >>> "1970-01-01T00:00:01.123456"
    >>> dt64_to_isoformat(dt64, "T", False)  # strict=False
    >>> "1970-01-01T00:00:01.123456789123"
    """
    # Access unit & value
    validate_dt64(dt64)
    cdef:
        np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(dt64)
        np.npy_datetime value = np.get_datetime64_value(dt64)
        str frepl
        Py_UCS4 ch
        long long us

    # Conversion: common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        us = value // 1_000
        if strict:
            frepl = "%06d" % (us % 1_000_000)
        else:
            frepl = "%09d" % (value % 1_000_000_000)
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        us = value
        frepl = "%06d" % (value % 1_000_000)
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        us = value * 1_000
        frepl = "%06d" % (value % 1_000 * 1_000)
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        us = value * 1_000_000
        frepl = None
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        us = value * 60_000_000
        frepl = None
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        us = value * US_HOUR
        frepl = None
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        us = value * US_DAY
        frepl = None
    # Conversion: uncommon
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        us = value // 1_000_000
        if strict:
            frepl = "%06d" % (us % 1_000_000)
        else:
            frepl = "%012d" % (value
                - (value // 1_000_000_000 // 1_000)
                * 1_000_000_000 * 1_000
            )
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        us = value // 1_000_000_000
        if strict:
            frepl = "%06d" % (us % 1_000_000)
        else:
            frepl = "%015d" % (value
                - (value // 1_000_000_000 // 1_000_000)
                * 1_000_000_000 * 1_000_000
            )
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        us = value // 1_000_000_000 // 1_000
        if strict:
            frepl = "%06d" % (us % 1_000_000)
        else:
            frepl = "%018d" % (value
                - (value // 1_000_000_000 // 1_000_000_000)
                * 1_000_000_000 * 1_000_000_000
            )
    # Unsupported unit
    else:
        raise ValueError(
            "unsupported <'np.datetime64'> unit '%s' for isoformat operation."
            % map_nptime_unit_int2str(unit)
        )

    # Convert to struct:tm
    _tm = tm_fr_us(us)
    if frepl is None:
        return "%04d-%02d-%02d%s%02d:%02d:%02d" % (
            _tm.tm_year, _tm.tm_mon, _tm.tm_mday, sep,
            _tm.tm_hour, _tm.tm_min, _tm.tm_sec,
        )
    else:
        return "%04d-%02d-%02d%s%02d:%02d:%02d.%s" % (
            _tm.tm_year, _tm.tm_mon, _tm.tm_mday, sep,
            _tm.tm_hour, _tm.tm_min, _tm.tm_sec, frepl,
        )

cdef inline long long dt64_to_int(object dt64, str unit):
    """Convert np.datetime64 to an integer since Unix Epoch 
    based on the given 'unit' `<'int'>`.

    Supported units: 'ns', 'us', 'ms', 's', 'm', 'h', 'D'.

    If 'dt64' resolution is higher than the 'unit',
    returns integer discards the resolution above the time unit.
    """
    cdef:
        Py_ssize_t size = str_len(unit)
        Py_UCS4 ch

    # Unit: 'ns', 'us', 'ms'
    if size == 2:
        ch = str_read(unit, 0)
        if ch == "n":
            return dt64_to_ns(dt64)
        if ch == "u":
            return dt64_to_us(dt64)
        if ch == "m":
            return dt64_to_ms(dt64)

    # Unit: 's', 'm', 'h', 'D'/'d'
    elif size == 1:
        ch = str_read(unit, 0)
        if ch == "s":
            return dt64_to_seconds(dt64)
        if ch == "m":
            return dt64_to_minutes(dt64)
        if ch == "h":
            return dt64_to_hours(dt64)
        if ch in ("D", "d"):
            return dt64_to_days(dt64)
    
    # Unsupported unit
    raise ValueError("conversion for <'np.datetime64'> to unit '%s' is not supported." % unit)

cdef inline long long dt64_to_days(object dt64):
    """Convert np.datetime64 to total days since Unix Epoch `<'int'>`.
    
    If 'dt64' resolution is higher than 'D',
    returns integer discards the resolution above days.
    """
    # Access unit & value
    validate_dt64(dt64)
    cdef np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(dt64)
    cdef np.npy_datetime value = np.get_datetime64_value(dt64)

    # Conversion: common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return value // 86_400_000 // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return value // 86_400_000 // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return value // 86_400_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return value // 86_400
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return value // 1_440
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return value // 24
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return value

    # Conversion: uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return value // 86_400_000 // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return value // 86_400_000 // 1_000_000_000 // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return value // 86_400_000 // 1_000_000_000 // 1_000_000

    # Unsupported unit
    raise ValueError(
        "unsupported <'np.datetime64'> unit '%s' for conversion." 
        % map_nptime_unit_int2str(unit)
    )

cdef inline long long dt64_to_hours(object dt64):
    """Convert np.datetime64 to total hours since Unix Epoch `<'int'>`.

    If 'dt64' resolution is higher than 'h',
    returns integer discards the resolution above hours.
    """
    # Access unit & value
    validate_dt64(dt64)
    cdef np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(dt64)
    cdef np.npy_datetime value = np.get_datetime64_value(dt64)

    # Conversion: common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return value // 3_600_000 // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return value // 3_600_000 // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return value // 3_600_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return value // 3_600
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return value // 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return value
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return value * 24

    # Conversion: uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return value // 3_600_000 // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return value // 3_600_000 // 1_000_000_000 // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return value // 3_600_000 // 1_000_000_000 // 1_000_000

    # Unsupported unit
    raise ValueError(
        "unsupported <'np.datetime64'> unit '%s' for conversion." 
        % map_nptime_unit_int2str(unit)
    )

cdef inline long long dt64_to_minutes(object dt64):
    """Convert np.datetime64 to total minutes since Unix Epoch `<'int'>`.

    If 'dt64' resolution is higher than 'm',
    returns integer discards the resolution above minutes.
    """
    # Access unit & value
    validate_dt64(dt64)
    cdef np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(dt64)
    cdef np.npy_datetime value = np.get_datetime64_value(dt64)

    # Conversion: common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return value // 60_000_000 // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return value // 60_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return value // 60_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return value // 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return value
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return value * 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return value * 1_440

    # Conversion: uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return value // 60_000_000 // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return value // 60_000_000 // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return value // 60_000_000 // 1_000_000_000 // 1_000

    # Unsupported unit
    raise ValueError(
        "unsupported <'np.datetime64'> unit '%s' for conversion." 
        % map_nptime_unit_int2str(unit)
    )

cdef inline long long dt64_to_seconds(object dt64):
    """Convert np.datetime64 to total seconds since Unix Epoch `<'int'>`.

    If 'dt64' resolution is higher than 's',
    returns integer discards the resolution above seconds.
    """
    # Access unit & value
    validate_dt64(dt64)
    cdef np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(dt64)
    cdef np.npy_datetime value = np.get_datetime64_value(dt64)

    # Conversion: common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return value // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return value // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return value // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return value
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return value * 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return value * 3_600
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return value * 86_400

    # Conversion: uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return value // 1_000_000_000 // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return value // 1_000_000_000 // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return value // 1_000_000_000 // 1_000_000_000

    # Unsupported unit
    raise ValueError(
        "unsupported <'np.datetime64'> unit '%s' for conversion." 
        % map_nptime_unit_int2str(unit)
    )

cdef inline long long dt64_to_ms(object dt64):
    """Convert np.datetime64 to total milliseconds since Unix Epoch `<'int'>`.

    If 'dt64' resolution is higher than 'ms',
    returns integer discards the resolution above milliseconds.
    """
    # Access unit & value
    validate_dt64(dt64)
    cdef np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(dt64)
    cdef np.npy_datetime value = np.get_datetime64_value(dt64)

    # Conversion: common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return value // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return value // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return value
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return value * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return value * 60_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return value * 3_600_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return value * 86_400_000

    # Conversion: uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return value // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return value // 1_000_000_000 // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return value // 1_000_000_000 // 1_000_000

    # Unsupported unit
    raise ValueError(
        "unsupported <'np.datetime64'> unit '%s' for conversion." 
        % map_nptime_unit_int2str(unit)
    )

cdef inline long long dt64_to_us(object dt64):
    """Convert np.datetime64 to total microseconds since Unix Epoch `<'int'>`.

    If 'dt64' resolution is higher than 'us',
    returns integer discards the resolution above microseconds.
    """
    # Access unit & value
    validate_dt64(dt64)
    cdef np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(dt64)
    cdef np.npy_datetime value = np.get_datetime64_value(dt64)

    # Conversion: common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return value // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return value
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return value * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return value * 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return value * 60_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return value * US_HOUR
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return value * US_DAY

    # Conversion: uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return value // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return value // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return value // 1_000_000_000 // 1_000

    # Unsupported unit
    raise ValueError(
        "unsupported <'np.datetime64'> unit '%s' for conversion." 
        % map_nptime_unit_int2str(unit)
    )

cdef inline long long dt64_to_ns(object dt64):
    """Convert np.datetime64 to total nanoseconds since Unix Epoch `<'int'>`.

    If 'dt64' resolution is higher than 'ns',
    returns integer discards the resolution above nanoseconds.
    """
    # Access unit & value
    validate_dt64(dt64)
    cdef np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(dt64)
    cdef np.npy_datetime value = np.get_datetime64_value(dt64)

    # Conversion: common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return value
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return value * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return value * 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return value * 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return value * NS_MINUTE
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return value * NS_HOUR
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return value * NS_DAY

    # Conversion: uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return value // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return value // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return value // 1_000_000_000

    # Unsupported unit
    raise ValueError(
        "unsupported <'np.datetime64'> unit '%s' for conversion." 
        % map_nptime_unit_int2str(unit)
    )

cdef inline datetime.date dt64_to_date(object dt64):
    """Convert np.datetime64 to `<'datetime.date'>`.

    If 'dt64' resolution is higher than 'D',
    returns datetime.date discards the resolution above days.
    """
    return date_fr_ordinal(dt64_to_days(dt64) + EPOCH_DAY)

cdef inline datetime.datetime dt64_to_dt(object dt64, tz: object=None):
    """Convert np.datetime64 to `<'datetime.datetime'>`.

    If 'dt64' resolution is higher than 'us',
    returns datetime.datetime discards the resolution above microseconds.
    """
    return dt_fr_us(dt64_to_us(dt64), tz)

cdef inline datetime.time dt64_to_time(object dt64):
    """Convert np.datetime64 to `<'datetime.time'>`.

    If 'dt64' resolution is higher than 'us',
    returns datetime.time discards the resolution above microseconds.
    """
    return time_fr_us(dt64_to_us(dt64) + EPOCH_US, None)

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
        raise TypeError("expects <'np.timedelta64'>, got %s." % type(obj))
    return True

# . conversion
cdef inline long long td64_to_int(object td64, str unit):
    """Convert np.timedelta64 to an integer based on the given 'unit' `<'int'>`.

    Supported units: 'ns', 'us', 'ms', 's', 'm', 'h', 'D'.

    If 'td64' resolution is higher than the 'unit',
    returns integer rounds to the nearest time unit.
    """
    cdef:
        Py_ssize_t size = str_len(unit)
        Py_UCS4 ch

    # Unit: 'ns', 'us', 'ms'
    if size == 2:
        ch = str_read(unit, 0)
        if ch == "n":
            return td64_to_ns(td64)
        if ch == "u":
            return td64_to_us(td64)
        if ch == "m":
            return td64_to_ms(td64)

    # Unit: 's', 'm', 'h', 'D'/'d'
    elif size == 1:
        ch = str_read(unit, 0)
        if ch == "s":
            return td64_to_seconds(td64)
        if ch == "m":
            return td64_to_minutes(td64)
        if ch == "h":
            return td64_to_hours(td64)
        if ch in ("D", "d"):
            return td64_to_days(td64)

    # Unsupported unit
    raise ValueError("conversion for <'np.timedelta64'> to unit '%s' is not supported." % unit)

cdef inline long long td64_to_days(object td64):
    """Convert np.timedelta64 to total days `<'int'>`.
    
    If 'td64' resolution is higher than 'D',
    returns integer rounds to the nearest days.
    """
    # Access unit & value
    validate_td64(td64)
    cdef np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(td64)
    cdef np.npy_datetime value = np.get_timedelta64_value(td64)

    # Conversion: common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return math.llroundl(value / 86_400_000 / 1_000_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return math.llroundl(value / 86_400_000 / 1_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return math.llroundl(value / 86_400_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return math.llroundl(value / 86_400)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return math.llroundl(value / 1_440)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return math.llroundl(value / 24)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return value

    # Conversion: uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return math.llroundl(value / 86_400_000 / 1_000_000_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return math.llroundl(value / 86_400_000 / 1_000_000_000 / 1_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return math.llroundl(value / 86_400_000 / 1_000_000_000 / 1_000_000)

    # Unsupported unit
    raise ValueError(
        "unsupported <'np.timedelta64'> unit '%s' for conversion." 
        % map_nptime_unit_int2str(unit)
    )

cdef inline long long td64_to_hours(object td64):
    """Convert np.timedelta64 to total hours `<'int'>`.

    If 'td64' resolution is higher than 'h',
    returns integer rounds to the nearest hours.
    """
    # Access unit & value
    validate_td64(td64)
    cdef np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(td64)
    cdef np.npy_datetime value = np.get_timedelta64_value(td64)

    # Conversion: common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return math.llroundl(value / 3_600_000 / 1_000_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return math.llroundl(value / 3_600_000 / 1_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return math.llroundl(value / 3_600_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return math.llroundl(value / 3_600)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return math.llroundl(value / 60)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return value
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return value * 24

    # Conversion: uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return math.llroundl(value / 3_600_000 / 1_000_000_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return math.llroundl(value / 3_600_000 / 1_000_000_000 / 1_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return math.llroundl(value / 3_600_000 / 1_000_000_000 / 1_000_000)

    # Unsupported unit
    raise ValueError(
        "unsupported <'np.timedelta64'> unit '%s' for conversion." 
        % map_nptime_unit_int2str(unit)
    )

cdef inline long long td64_to_minutes(object td64):
    """Convert np.timedelta64 to total minutes `<'int'>`.

    If 'td64' resolution is higher than 'm',
    returns integer rounds to the nearest minutes.
    """
    # Access unit & value
    validate_td64(td64)
    cdef np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(td64)
    cdef np.npy_datetime value = np.get_timedelta64_value(td64)

    # Conversion: common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return math.llroundl(value / 60_000_000 / 1_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return math.llroundl(value / 60_000_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return math.llroundl(value / 60_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return math.llroundl(value / 60)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return value
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return value * 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return value * 1_440

    # Conversion: uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return math.llroundl(value / 60_000_000 / 1_000_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return math.llroundl(value / 60_000_000 / 1_000_000_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return math.llroundl(value / 60_000_000 / 1_000_000_000 / 1_000)

    # Unsupported unit
    raise ValueError(
        "unsupported <'np.datetime64'> unit '%s' for conversion." 
        % map_nptime_unit_int2str(unit)
    )

cdef inline long long td64_to_seconds(object td64):
    """Convert np.timedelta64 to total seconds `<'int'>`.

    If 'td64' resolution is higher than 's',
    returns integer rounds to the nearest seconds.
    """
    # Access unit & value
    validate_td64(td64)
    cdef np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(td64)
    cdef np.npy_datetime value = np.get_timedelta64_value(td64)

    # Conversion: common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return math.llroundl(value / 1_000_000_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return math.llroundl(value / 1_000_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return math.llroundl(value / 1_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return value
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return value * 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return value * 3_600
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return value * 86_400

    # Conversion: uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return math.llroundl(value / 1_000_000_000 / 1_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return math.llroundl(value / 1_000_000_000 / 1_000_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return math.llroundl(value / 1_000_000_000 / 1_000_000_000)

    # Unsupported unit
    raise ValueError(
        "unsupported <'np.timedelta64'> unit '%s' for conversion." 
        % map_nptime_unit_int2str(unit)
    )

cdef inline long long td64_to_ms(object td64):
    """Convert np.timedelta64 to total milliseconds `<'int'>`.
    
    If 'td64' resolution is higher than 'ms',
    returns integer rounds to the nearest milliseconds.
    """
    # Access unit & value
    validate_td64(td64)
    cdef np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(td64)
    cdef np.npy_datetime value = np.get_timedelta64_value(td64)

    # Conversion: common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return math.llroundl(value / 1_000_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return math.llroundl(value / 1_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return value
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return value * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return value * 60_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return value * 3_600_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return value * 86_400_000

    # Conversion: uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return math.llroundl(value / 1_000_000_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return math.llroundl(value / 1_000_000_000 / 1_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return math.llroundl(value / 1_000_000_000 / 1_000_000)

    # Unsupported unit
    raise ValueError(
        "unsupported <'np.timedelta64'> unit '%s' for conversion." 
        % map_nptime_unit_int2str(unit)
    )

cdef inline long long td64_to_us(object td64):
    """Convert np.timedelta64 to total microseconds `<'int'>`.

    If 'td64' resolution is higher than 'us',
    returns integer rounds to the nearest microseconds.
    """
    # Access unit & value
    validate_td64(td64)
    cdef np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(td64)
    cdef np.npy_datetime value = np.get_timedelta64_value(td64)

    # Conversion: common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return math.llroundl(value / 1_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return value
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return value * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return value * 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return value * 60_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return value * US_HOUR
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return value * US_DAY

    # Conversion: uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return math.llroundl(value / 1_000_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return math.llroundl(value / 1_000_000_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return math.llroundl(value / 1_000_000_000 / 1_000)

    # Unsupported unit
    raise ValueError(
        "unsupported <'np.timedelta64'> unit '%s' for conversion." 
        % map_nptime_unit_int2str(unit)
    )

cdef inline long long td64_to_ns(object td64):
    """Convert np.timedelta64 to total nanoseconds `<'int'>`.

    If 'td64' resolution is higher than 'ns',
    returns integer rounds to the nearest nanoseconds.
    """
    # Access unit & value
    validate_td64(td64)
    cdef np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(td64)
    cdef np.npy_datetime value = np.get_timedelta64_value(td64)

    # Conversion: common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return value
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return value * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return value * 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return value * 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return value * NS_MINUTE
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return value * NS_HOUR
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return value * NS_DAY

    # Conversion: uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return math.llroundl(value / 1_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return math.llroundl(value / 1_000_000)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return math.llroundl(value / 1_000_000_000)

    # Unsupported unit
    raise ValueError(
        "unsupported <'np.timedelta64'> unit '%s' for conversion." 
        % map_nptime_unit_int2str(unit)
    )

cdef inline datetime.timedelta td64_to_td(object td64):
    """Convert np.timedelta64 to `<'datetime.timedelta'>`.

    If 'td64' resolution is higher than 'us',
    returns datetime.timedelta rounds to the nearest microseconds.
    """
    return td_fr_us(td64_to_us(td64))
