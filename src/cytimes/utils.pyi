import cython
import datetime
import numpy as np

# Constants -----------------------------------------------------------------------------------------
# . date
ORDINAL_MAX: int
# . datetime
#: EPOCH (1970-01-01)
EPOCH_DT: datetime.datetime
EPOCH_YEAR: int
EPOCH_MONTH: int
EPOCH_DAY: int
EPOCH_HOUR: int
EPOCH_MINUTE: int
EPOCH_SECOND: int
EPOCH_MILLISECOND: int
EPOCH_MICROSECOND: int
# . timezone
UTC: datetime.tzinfo
# . conversion for seconds
SS_MINUTE: int
SS_HOUR: int
SS_DAY: int
# . conversion for milliseconds
MS_SECOND: int
MS_MINUTE: int
MS_HOUR: int
MS_DAY: int
# . conversion for microseconds
US_MILLISECOND: int
US_SECOND: int
US_MINUTE: int
US_HOUR: int
US_DAY: int
# . conversion for nanoseconds
NS_MICROSECOND: int
NS_MILLISECOND: int
NS_SECOND: int
NS_MINUTE: int
NS_HOUR: int
NS_DAY: int
# . conversion for timedelta64
TD64_YY_DAY: int
TD64_YY_SECOND: int
TD64_YY_MILLISECOND: int
TD64_YY_MICROSECOND: int
TD64_YY_NANOSECOND: int
TD64_MM_DAY: int
TD64_MM_SECOND: int
TD64_MM_MILLISECOND: int
TD64_MM_MICROSECOND: int
TD64_MM_NANOSECOND: int
# . datetime64 range
#: Minimum datetime64 in nanoseconds (1677-09-21 00:12:43.145224193)
DT64_NS_YY_MIN: int
DT64_NS_MM_MIN: int
DT64_NS_WW_MIN: int
DT64_NS_DD_MIN: int
DT64_NS_HH_MIN: int
DT64_NS_MI_MIN: int
DT64_NS_SS_MIN: int
DT64_NS_MS_MIN: int
DT64_NS_US_MIN: int
DT64_NS_NS_MIN: int
#: Maximum datetime64 in nanoseconds (2262-04-11 23:47:16.854775807)
DT64_NS_YY_MAX: int
DT64_NS_MM_MAX: int
DT64_NS_WW_MAX: int
DT64_NS_DD_MAX: int
DT64_NS_HH_MAX: int
DT64_NS_MI_MAX: int
DT64_NS_SS_MAX: int
DT64_NS_MS_MAX: int
DT64_NS_US_MAX: int
DT64_NS_NS_MAX: int

# Math ----------------------------------------------------------------------------------------------
def math_mod(num: int, factor: int, offset: int = 0) -> int:
    """(cfunc) Computes the modulo of a number by the factor,
    handling negative numbers accoring to Python's modulo
    semantics `<'int'>`.

    Equivalent to:
    >>> (num % factor) + offset
    """

def math_round_div(num: int, factor: int, offset: int = 0) -> int:
    """(cfunc) Divides a number by the factor and rounds the result
    to the nearest integer (half away from zero), handling negative
    numbers accoring to Python's division semantics `<'int'>`.

    Equivalent to:
    >>> round(num / factor, 0) + offset
    """

def math_ceil_div(num: int, factor: int, offset: int = 0) -> int:
    """(cfunc) Divides a number by the factor and rounds
    the result up to the nearest integer, handling negative
    numbers accoring to Python's division semantics `<'int'>`.

    Equivalent to:
    >>> math.ceil(num / factor) + offset
    """

def math_floor_div(num: int, factor: int, offset: int = 0) -> int:
    """(cfunc) Divides a number by the factor and rounds
    the result down to the nearest integer, handling negative
    numbers accoring to Python's division semantics `<'int'>`.

    Equivalent to:
    >>> math.floor(num / factor) + offset
    """

# Parser --------------------------------------------------------------------------------------------
# . check
def str_count(s: str, substr: str) -> int:
    """(cfunc) Get the number of occurrences of a 'substr' in an unicode `<'int'>`.

    Equivalent to:
    >>> s.count(substr)
    """

def is_iso_sep(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check if the passed in 'ch' is an ISO format date/time seperator (" " or "T") `<'bool'>`"""

def is_isodate_sep(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check if the passed in 'ch' is an ISO format date fields separator ("-" or "/") `<'bool'>`"""

def is_isoweek_sep(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check if the passed in 'ch' is an ISO format week number identifier ("W") `<'bool'>`"""

def is_isotime_sep(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check if the passed in 'ch' is an ISO format time fields separator (":") `<'bool'>`"""

def is_ascii_digit(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check if the passed in 'ch' is an ASCII digit [0-9] `<'bool'>`"""

def is_ascii_alpha_upper(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check if the passed in 'ch' is an ASCII alpha in uppercase [A-Z] `<'bool'>`."""

def is_ascii_alpha_lower(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check if the passed in 'ch' is an ASCII alpha in lowercase [a-z] `<'bool'>`."""

def is_ascii_alpha(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check if the passed in 'ch' is an ASCII alpha [a-zA-Z] `<'bool'>`."""

# . parse
def parse_isoyear(data: str, pos: int, size: int) -> int:
    """(cfunc) Parse ISO format year component (YYYY) from a string,
    returns `-1` for invalid ISO years `<'int'>`.

    This function extracts and parses the year component from an ISO date string.
    It reads four characters starting at the specified position and converts them
    into an integer representing the year. The function ensures that the parsed
    year is valid (i.e., between '0001' and '9999'').

    :param data `<'str'>`: The input string containing the ISO year to parse.
    :param pos `<'int'>`: The starting position in the string of the ISO year.
    :param size `<'int'>`: The length of the input 'data' string.
        If 'size <= 0', the function computes the size of the 'data' string internally.
    """

def parse_isomonth(data: str, pos: int, size: int) -> int:
    """(cfunc) Parse ISO format month component (MM) from a string,
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

def parse_isoday(data: str, pos: int, size: int) -> int:
    """(cfunc) Parse ISO format day component (DD) from a string,
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

def parse_isoweek(data: str, pos: int, size: int) -> int:
    """(cfunc) Prase an ISO format week number component (WW) from a string,
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

def parse_isoweekday(data: str, pos: int, size: int) -> int:
    """(cfunc) Prase an ISO format weekday component (D) from a string,
    returns `-1` for invalid ISO weekdays `<'int'>`.

    This function extracts and parses the weekday component from an ISO date string.
    It reads a single character at the specified position and converts it into an
    integer representing the ISO weekday, where Monday is 1 and Sunday is 7.

    :param data `<'str'>`: The input string containing the ISO weekday to parse.
    :param pos `<'int'>`: The starting position in the string of the ISO weekday.
    :param size `<'int'>`: The length of the input 'data' string.
        If 'size <= 0', the function computes the size of the 'data' string internally.
    """

def parse_isoyearday(data: str, pos: int, size: int) -> int:
    """(cfunc) Prase an ISO format day of the year component (DDD) from a string,
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

def parse_isohour(data: str, pos: int, size: int) -> int:
    """(cfunc) Parse an ISO format hour (HH) component from a string,
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

def parse_isominute(data: str, pos: int, size: int) -> int:
    """(cfunc) Parse an ISO format minute (MM) component from a string,
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

def parse_isosecond(data: str, pos: int, size: int) -> int:
    """(cfunc) Parse an ISO format second (SS) component from a string,
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

def parse_isofraction(data: str, pos: int, size: int) -> int:
    """(cfunc) Parse an ISO fractional time component (fractions of a second) from a string,
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

def slice_to_uint(data: str, start: int, size: int) -> int:
    """(cfunc) Slice a substring from a string and convert to an unsigned integer `<'int'>`.

    This function slices a portion of the input string 'data' starting
    at 'start' and spanning 'size' characters. The sliced substring is
    validated to ensure it contains only ASCII digits, before converting
    to unsigned integer.

    :param data `<'str'>`: The input string to slice and convert.
    :param start `<'int'>`: The starting index for slicing the string.
    :param size `<'int'>`: The number of characters to slice from 'start'.
    """

# Time ----------------------------------------------------------------------------------------------
def tm_gmtime(ts: float) -> dict:
    """(cfunc) Convert a timestamp to 'struct:tm' expressing UTC time `<'struct:tm'>`.

    This function takes a Unix timestamp 'ts' and converts it into 'struct:tm'
    representing the UTC time. It is equivalent to 'time.gmtime(ts)'' in Python
    but implemented in Cython for efficiency.
    """

def tm_localtime(ts: float) -> dict:
    """(cfunc) Convert a timestamp to 'struct:tm' expressing local time `<'struct:tm'>`.

    This function takes a Unix timestamp 'ts' and converts it into 'struct:tm'
    representing the local time. It is equivalent to 'time.localtime(ts)' in
    Python but implemented in Cython for efficiency.
    """

def ts_gmtime(ts: float) -> int:
    """(cfunc) Convert a timestamp to UTC seconds since the Unix Epoch `<'int'>`.

    This function converts a Unix timestamp 'ts' to integer in
    seconds since the Unix Epoch, representing the UTC time.
    """

def ts_localtime(ts: float) -> int:
    """(cfunc) Convert a timestamp to local seconds since the Unix Epoch `<'int'>`.

    This function converts a Unix timestamp 'ts' to integer in
    seconds since the Unix Epoch, representing the local time.
    """

# . conversion
def tm_strftime(t: object, fmt: str) -> str:
    """(cfunc) Convert 'struct:tm' to string according to the given format `<'str'>`."""

def tm_fr_us(val: int) -> dict:
    """(cfunc) Create 'struct:tm' from `EPOCH` microseconds (int) `<'struct:tm'>`."""

def tm_fr_seconds(val: float) -> dict:
    """(cfunc) Create 'struct:tm' from `EPOCH` seconds (float) `<'struct:tm'>`."""

def hmsf_fr_us(val: int) -> dict:
    """(cfunc) Create 'struct:hmsf' from microseconds (int) `<'struct:hmsf'>`.

    Notice that the orgin of the microseconds must be 0,
    and `NOT` the Unix Epoch (1970-01-01 00:00:00).
    """

def hmsf_fr_seconds(val: float) -> dict:
    """(cfunc) Create 'struct:hmsf' from seconds (float) `<'struct:hmsf'>`.

    Notice that the orgin of the seconds must be 0,
    and `NOT` the Unix Epoch (1970-01-01 00:00:00).
    """

# Calender ------------------------------------------------------------------------------------------
# . year
def is_leap_year(year: int) -> bool:
    """(cfunc) Check if the passed in 'year' is a leap year `<'bool'>`."""

def is_long_year(year: int) -> bool:
    """(cfunc) Check if the passed in 'year' is a long year `<'bool'>`.

    #### Long year: maximum ISO week number equal 53.
    """

def leap_bt_year(year1: int, year2: int) -> int:
    """(cfunc) Compute the number of leap years between 'year1' & 'year2' `<'int'>`."""

def days_in_year(year: int) -> int:
    """(cfunc) Compute the maximum days (365, 366) in the 'year' `<'int'>`."""

def days_bf_year(year: int) -> int:
    """(cfunc) Compute the number of days between the 1st day of 1AD and the 'year' `<'int'>`."""

def days_of_year(year: int, month: int, day: int) -> int:
    """(cfunc) Compute the number of days between the 1st day of
    the 'year' and the current Y/M/D `<'int'>`.
    """

# . quarter
def quarter_of_month(month: int) -> int:
    """(cfunc) Compute the quarter (1-4) of the passed in 'month' `<'int'>`."""

def days_in_quarter(year: int, month: int) -> int:
    """(cfunc) Compute the maximum days in the quarter `<'int'>`."""

def days_bf_quarter(year: int, month: int) -> int:
    """(cfunc) Compute the number of days between the 1st day
    of the 'year' and the 1st day of the quarter `<'int'>`.
    """

def days_of_quarter(year: int, month: int, day: int) -> int:
    """(cfunc) Compute the number of days between the 1st
    day of the quarter and the current Y/M/D `<'int'>`.
    """

# . month
def days_in_month(year: int, month: int) -> int:
    """(cfunc) Compute the maximum days in the month `<'int'>`."""

def days_bf_month(year: int, month: int) -> int:
    """(cfunc) Compute the number of days between the 1st day
    of the 'year' and the 1st day of the 'month' `<'int'>`.
    """

# . week
def ymd_weekday(year: int, month: int, day: int) -> int:
    """(cfunc) Compute the weekday (0=Mon...6=Sun) `<'int'>`."""

# . iso
def ymd_isocalendar(year: int, month: int, day: int) -> dict:
    """(cfunc) Compute the ISO calendar `<'struct:iso'>`."""

def ymd_isoyear(year: int, month: int, day: int) -> int:
    """(cfunc) Compute the ISO calendar year (0-10000) `<'int'>`."""

def ymd_isoweek(year: int, month: int, day: int) -> int:
    """(cfunc) Compute the ISO calendar week number (1-53) `<'int'>`."""

def ymd_isoweekday(year: int, month: int, day: int) -> int:
    """(cfunc) Compute the ISO weekday (1=Mon...7=Sun) `<'int'>`."""

# . Y/M/D
def ymd_to_ordinal(year: int, month: int, day: int) -> int:
    """(cfunc) Convert 'Y/M/D' to Gregorian ordinal days `<'int'>`."""

def ymd_fr_ordinal(val: int) -> dict:
    """(cfunc) Create 'struct:ymd' from Gregorian ordinal days `<'struct:ymd'>`."""

def ymd_fr_isocalendar(year: int, week: int, weekday: int) -> dict:
    """(cfunc) Create 'struct:ymd' from ISO calendar values `<'struct:ymd'>`."""

def ymd_fr_days_of_year(year: int, days: int) -> dict:
    """(cfunc) Create 'struct:ymd' from the year and days of the year `<'struct:ymd'>`."""

# datetime.date -------------------------------------------------------------------------------------
# . generate
def date_new(year: int = 1, month: int = 1, day: int = 1) -> datetime.date:
    """(cfunc) Create a new `<'datetime.date'>`.

    Equivalent to:
    >>> datetime.date(year, month, day)
    """

def date_now(tz: datetime.tzinfo | None = None) -> datetime.date:
    """(cfunc) Get the current date `<'datetime.date'>`.

    Equivalent to:
    >>> datetime.datetime.now(tz).date()
    """

# . type check
def is_date(obj: object) -> bool:
    """(cfunc) Check if an object is an instance of datetime.date `<'bool'>`.

    Equivalent to:
    >>> isinstance(obj, datetime.date)
    """

def is_date_exact(obj: object) -> bool:
    """(cfunc) Check if an object is the exact datetime.date type `<'bool'>`.

    Equivalent to:
    >>> type(obj) is datetime.date
    """

# . conversion: to
def date_to_tm(date: datetime.date) -> dict:
    """(cfunc) Convert date to `<'struct:tm'>`.

    #### All time fields are set to 0.
    """

def date_to_strformat(date: datetime.date, fmt: str) -> str:
    """(cfunc) Convert date to string according to the given format `<'str'>`.

    Equivalent to:
    >>> date.strftime(fmt)
    """

def date_to_isoformat(date: datetime.date) -> str:
    """Convert date to string in ISO format ('%Y-%m-%d') `<'str'>`."""

def date_to_us(date: datetime.date) -> int:
    """(cfunc) Convert date to `EPOCH` microseconds `<'int'>`."""

def date_to_seconds(date: datetime.date) -> float:
    """(cfunc) Convert date to `EPOCH` seconds `<'float'>`."""

def date_to_ordinal(date: datetime.date) -> int:
    """(cfunc) Convert date to Gregorian ordinal days `<'int'>`."""

def date_to_ts(date: datetime.date) -> float:
    """(cfunc) Convert date to `EPOCH` timestamp `<'float'>`."""

# . conversion: from
def date_fr_us(val: int) -> datetime.date:
    """(cfunc) Create date from `EPOCH` microseconds (int) `<'datetime.date'>`."""

def date_fr_seconds(val: float) -> datetime.date:
    """(cfunc) Create date from `EPOCH` seconds (float) `<'datetime.date'>`."""

def date_fr_ordinal(val: int) -> datetime.date:
    """(cfunc) Create date from Gregorian ordinal days `<'datetime.date'>`."""

def date_fr_ts(val: float) -> datetime.date:
    """(cfunc) Create date from `EPOCH` timestamp (float) `<'datetime.date'>`."""

def date_fr_date(date: datetime.date) -> datetime.date:
    """(cfunc) Create date from another date (include subclass) `<'datetime.date'>`."""

def date_fr_dt(dt: datetime.datetime) -> datetime.date:
    """(cfunc) Create date from datetime (include subclass) `<'datetime.date'>`."""

# datetime.datetime ---------------------------------------------------------------------------------
# . generate
def dt_new(
    year: int = 1,
    month: int = 1,
    day: int = 1,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
    tz: datetime.tzinfo | None = None,
    fold: int = 0,
) -> datetime.datetime:
    """(cfunc) Create a new datetime `<'datetime.datetime'>`.

    Equivalent to:
    >>> datetime.datetime(year, month, day, hour, minute, second, microsecond, tz, fold)
    """

def dt_now(tz: datetime.tzinfo | None = None) -> datetime.datetime:
    """(cfunc) Get the current datetime `<'datetime.datetime'>`.

    Equivalent to:
    >>> datetime.datetime.now(tz)
    """

# . type check
def is_dt(obj: object) -> bool:
    """(cfunc) Check if an object is an instance of datetime.datetime `<'bool'>`.

    Equivalent to:
    >>> isinstance(obj, datetime.datetime)
    """

def is_dt_exact(obj: object) -> bool:
    """(cfunc) Check if an object is the exact datetime.datetime type `<'bool'>`.

    Equivalent to:
    >>> type(obj) is datetime.datetime
    """

# . tzinfo
def dt_tzname(dt: datetime.datetime) -> str | None:
    """(cfunc) Get the tzinfo 'tzname' of the datetime `<'str/None'>`.

    Equivalent to:
    >>> dt.tzname()
    """

def dt_dst(dt: datetime.datetime) -> datetime.timedelta | None:
    """(cfunc) Get the tzinfo 'dst' of the datetime `<'datetime.timedelta/None'>`.

    Equivalent to:
    >>> dt.dst()
    """

def dt_utcoffset(dt: datetime.datetime) -> datetime.timedelta | None:
    """(cfunc) Get the tzinfo 'utcoffset' of the datetime `<'datetime.timedelta/None'>`.

    Equivalent to:
    >>> dt.utcoffset()
    """

def dt_normalize_tz(dt: datetime.datetime) -> datetime.datetime:
    """(cfunc) Normalize the datetime to its tzinfo `<'datetime.datetime'>`.

    This function is designed to handle ambiguous
    datetime by normalizing it to its timezone.
    """

# . conversion: to
def dt_to_tm(dt: datetime.datetime, utc: bool = False) -> dict:
    """(cfunc) Convert datetime to `<'struct:tm'>`.

    If 'dt' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from the result.
    """

def dt_to_ctime(dt: datetime.datetime) -> str:
    """(cfunc) Convert datetime to string in C time format `<'str'>`.

    - ctime format: 'Tue Oct  1 08:19:05 2024'
    """

def dt_to_strformat(dt: datetime.datetime, fmt: str) -> str:
    """(cfunc) Convert datetime to string according to the given format `<'str'>`.

    Equivalent to:
    >>> dt.strftime(fmt)
    """

def dt_to_isoformat(dt: datetime.datetime, sep: str = "T", utc: bool = False) -> str:
    """(cfunc) Convert datetime to string in ISO format `<'str'>`.

    If 'dt' is timezone-aware, setting 'utc=True'
    adds the UTC at the end of the ISO format.
    """

def dt_to_us(dt: datetime.datetime, utc: bool = False) -> int:
    """(cfunc) Convert datetime to `EPOCH` microseconds `<'int'>`.

    If 'dt' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from total mircroseconds.
    """

def dt_to_seconds(dt: datetime.datetime, utc: bool = False) -> float:
    """(cfunc) Convert datetime to `EPOCH` seconds `<'float'>`.

    If 'dt' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from total seconds.
    """

def dt_to_ordinal(dt: datetime.datetime, utc: bool = False) -> int:
    """(cfunc) Convert datetime to Gregorian ordinal days `<'int'>`.

    If 'dt' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from total days.
    """

def dt_to_posix(dt: datetime.date) -> int:
    """(cfunc) Convert datetime to POSIX Time (seconds) `<'int'>`."""

def dt_to_ts(dt: datetime.datetime) -> float:
    """(cfunc) Convert datetime to `EPOCH` timestamp `<'float'>`.

    Equivalent to:
    >>> dt.timestamp()
    """

def dt_as_epoch(dt: datetime.datetime, unit: str, utc: bool = False) -> int:
    """(cfunc) Convert datetime to `EPOCH` integer according to the given unit resolution `<'int'>`.

    Supported units: 'Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us'.

    If 'dt' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from the result.
    """

def dt_as_epoch_iso_W(dt: datetime.datetime, weekday: int, utc: bool = False) -> int:
    """Convert datetime to `EPOCH` integer under 'W' (weeks) resolution `<'int'>`.

    Different from 'dt_as_epoch(dt, "W")', which aligns the weekday
    to Thursday (the weekday of 1970-01-01). This function allows
    specifying the ISO 'weekday' (1=Monday, 7=Sunday) for alignment.

    For example: if 'weekday=1', the result represents the Monday-aligned
    weeks since EPOCH (1970-01-01).

    If 'dt' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from the result.
    """

# . conversion: from
def dt_fr_us(val: int, tz: datetime.tzinfo | None = None) -> datetime.datetime:
    """(cfunc) Create datetime from `EPOCH` microseconds (int) `<'datetime.datetime'>`."""

def dt_fr_seconds(val: float, tz: datetime.tzinfo | None = None) -> datetime.datetime:
    """(cfunc) Create datetime from `EPOCH` seconds (float) `<'datetime.datetime'>`."""

def dt_fr_ordinal(val: int, tz: datetime.tzinfo | None = None) -> datetime.datetime:
    """(cfunc) Create datetime from Gregorian ordinal days (int) `<'datetime.datetime'>`.

    Equivalent to:
    >>> datetime.datetime.fromordinal(val).replace(tzinfo=tz)
    """

def dt_fr_ts(val: float, tz: datetime.tzinfo | None = None) -> datetime.datetime:
    """(cfunc) Create datetime from `EPOCH` timestamp (float) `<'datetime.datetime'>`.

    Equivalent to:
    >>> datetime.datetime.fromtimestamp(val, tz)
    """

def dt_combine(
    date: datetime.date | None = None,
    time: datetime.time | None = None,
    tz: datetime.tzinfo | None = None,
) -> datetime.datetime:
    """ "(cfunc) Create datetime by combining date & time `<'datetime.datetime'>`.

    - If 'date' is None, use current local date.
    - If 'time' is None, all time fields are set to 0.
    """

def dt_fr_date(
    date: datetime.date,
    tz: datetime.tzinfo | None = None,
) -> datetime.datetime:
    """(cfunc) Create datetime from date (include subclass) `<'datetime.datetime'>`.

    #### All time fields are set to 0.
    """

def dt_fr_time(time: datetime.time) -> datetime.datetime:
    """(cfunc) Create datetime from time (include subclass) `<'datetime.datetime'>`.

    #### Date fields are set to 1970-01-01.
    """

def dt_fr_dt(dt: datetime.datetime) -> datetime.datetime:
    """(cfunc) Create datetime from another datetime (include subclass) `<'datetime.datetime'>`."""

# . manipulation
def dt_add(
    dt: datetime.datetime,
    days: int = 0,
    seconds: int = 0,
    microseconds: int = 0,
) -> datetime.datetime:
    """(cfunc) Add delta to datetime `<'datetime.datetime'>`.

    Equivalent to:
    >>> dt + datetime.timedelta(days, seconds, microseconds)
    """

def dt_replace_tz(
    dt: datetime.datetime,
    tz: datetime.tzinfo | None,
) -> datetime.datetime:
    """(cfunc) Replace the datetime timezone `<'datetime.datetime'>`.

    Equivalent to:
    >>> dt.replace(tzinfo=tz)
    """

def dt_replace_fold(dt: datetime.datetime, fold: int) -> datetime.datetime:
    """(cfunc) Replace the datetime fold `<'datetime.datetime'>`.

    Equivalent to:
    >>> dt.replace(fold=fold)
    """

def dt_astimezone(
    dt: datetime.datetime,
    tz: datetime.tzinfo | None = None,
) -> datetime.datetime:
    """(cfunc) Convert the datetime timezone `<'datetime.datetime'>`.

    Equivalent to:
    >>> dt.astimezone(tz)
    """

# datetime.time -------------------------------------------------------------------------------------
# . generate
def time_new(
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
    tz: datetime.tzinfo | None = None,
    fold: int = 0,
) -> datetime.time:
    """(cfunc) Create a new time `<'datetime.time'>`.

    Equivalent to:
    >>> datetime.time(hour, minute, second, microsecond, tz, fold)
    """

def time_now(tz: datetime.tzinfo | None = None) -> datetime.time:
    """(cfunc) Get the current time `<'datetime.time'>`.

    Equivalent to:
    >>> datetime.datetime.now(tz).time()
    """

# . type check
def is_time(obj: object) -> bool:
    """(cfunc) Check if an object is an instance of datetime.time `<'bool'>`.

    Equivalent to:
    >>> isinstance(obj, datetime.time)
    """

def is_time_exact(obj: object) -> bool:
    """(cfunc) Check if an object is the exact datetime.time type `<'bool'>`.

    Equivalent to:
    >>> type(obj) is datetime.time
    """

# . tzinfo
def time_tzname(time: datetime.time) -> str | None:
    """(cfunc) Get the tzinfo 'tzname' of the time `<'str/None'>`.

    Equivalent to:
    >>> time.tzname()
    """

def time_dst(time: datetime.time) -> datetime.timedelta | None:
    """(cfunc) Get the tzinfo 'dst' of the time `<'datetime.timedelta/None'>`.

    Equivalent to:
    >>> time.dst()
    """

def time_utcoffset(time: datetime.time) -> datetime.timedelta | None:
    """(cfunc) Get the tzinfo 'utcoffset' of the time `<'datetime.timedelta/None'>`.

    Equivalent to:
    >>> time.utcoffset()
    """

# . conversion
def time_to_isoformat(time: datetime.time, utc: bool = False) -> str:
    """(cfunc) Convert time to string in ISO format `<'str'>`.

    If 'time' is timezone-aware, setting 'utc=True'
    adds the UTC at the end of the ISO format.
    """

def time_to_us(time: datetime.time, utc: bool = False) -> int:
    """(cfunc) Convert time to microseconds `<'int'>`.

    If 'time' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from total mircroseconds.
    """

def time_to_seconds(time: datetime.time, utc: bool = False) -> float:
    """(cfunc) Convert time to seconds `<'float'>`.

    If 'time' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from total seconds.
    """

def time_fr_us(val: int, tz: datetime.tzinfo | None = None) -> datetime.time:
    """(cfunc) Create time from `EPOCH` microseconds (int) `<'datetime.time'>`."""

def time_fr_seconds(val: float, tz: datetime.tzinfo | None = None) -> datetime.time:
    """(cfunc) Create time from `EPOCH` seconds (float) `<'datetime.time'>`."""

def time_fr_time(time: datetime.time) -> datetime.time:
    """(cfunc) Create time from another time (include subclass) `<'datetime.time'>`."""

def time_fr_dt(dt: datetime.datetime) -> datetime.time:
    """(cfunc) Create time from datetime (include subclass) `<'datetime.time'>`."""

# datetime.timedelta --------------------------------------------------------------------------------
# . generate
def td_new(
    days: int = 0,
    seconds: int = 0,
    microseconds: int = 0,
) -> datetime.timedelta:
    """(cfunc) Create a new timedelta `<'datetime.timedelta'>`.

    Equivalent to:
    >>> datetime.timedelta(days, seconds, microseconds)
    """

# . type check
def is_td(obj: object) -> bool:
    """(cfunc) Check if an object is an instance of datetime.timedelta `<'bool'>`.

    Equivalent to:
    >>> isinstance(obj, datetime.timedelta)
    """

def is_td_exact(obj: object) -> bool:
    """(cfunc) Check if an object is the exact datetime.timedelta type `<'bool'>`.

    Equivalent to:
    >>> type(obj) is datetime.timedelta
    """

# . conversion
def td_to_isoformat(td: datetime.timedelta) -> str:
    """(cfunc) Convert timedelta to string in ISO format `<'str'>`."""

def td_to_utcformat(td: datetime.timedelta) -> str:
    """(cfunc) Convert timedelta to string in UTC format ('+/-HH:MM') `<'str'>`."""

def td_to_us(td: datetime.timedelta) -> int:
    """(cfunc) Convert timedelta to microseconds `<'int'>`."""

def td_to_seconds(td: datetime.timedelta) -> float:
    """(cfunc) Convert timedelta to seconds `<'float'>`."""

def td_fr_us(val: int) -> datetime.timedelta:
    """(cfunc) Create timedelta from microseconds (int) `<'datetime.timedelta'>`."""

def td_fr_seconds(val: float) -> datetime.timedelta:
    """(cfunc) Create timedelta from seconds (float) `<'datetime.timedelta'>`."""

def td_fr_td(td: datetime.timedelta) -> datetime.timedelta:
    """(cfunc) Create timedelta from another timedelta (include subclass) `<'datetime.timedelta'>`."""

# datetime.tzinfo -----------------------------------------------------------------------------------
# . generate
def tz_new(hours: int = 0, minutes: int = 0, seconds: int = 0) -> datetime.timezone:
    """(cfunc) Create a new timezone `<'datetime.timezone'>`.

    Equivalent to:
    >>> datetime.timezone(datetime.timedelta(hours=hours, minutes=minites))
    """

def tz_local(dt: datetime.datetime | None = None) -> datetime.timezone:
    """(cfunc) Get the local timezone `<'datetime.timezone'>`."""

def tz_local_seconds(dt: datetime.datetime | None = None) -> int:
    """Get the local timezone offset in total seconds `<'int'>`."""

def tz_parse(tz: datetime.tzinfo | str) -> datetime.tzinfo | None:
    """(cfunc) Parse 'tz' object into `<'datetime.tzinfo/None'>`.

    :param tz `<'datetime.timezone/Zoneinfo/pytz/str'>`: The timezone object.
        1. If 'tz' is an instance of `<'datetime.timezone'>`, return 'tz' directly.
        2. If 'tz' is a string or timezone from Zoneinfo or pytz,
           use Python 'Zoneinfo' to (re)-create the timezone object.
    """

# . type check
def is_tz(obj: object) -> bool:
    """(cfunc) Check if an object is an instance of datetime.tzinfo `<'bool'>`.

    Equivalent to:
    >>> isinstance(obj, datetime.tzinfo)
    """

def is_tz_exact(obj: object) -> bool:
    """(cfunc) Check if an object is the exact datetime.tzinfo type `<'bool'>`.

    Equivalent to:
    >>> type(obj) is datetime.date
    """

# . access
def tz_name(
    tz: datetime.tzinfo | None,
    dt: datetime.datetime | None = None,
) -> str | None:
    """(cfunc) Access the 'tzname' of the tzinfo `<'str/None'>`.

    Equivalent to:
    >>> tz.tzname(dt)
    """

def tz_dst(
    tz: datetime.tzinfo | None,
    dt: datetime.datetime | None = None,
) -> datetime.timedelta | None:
    """(cfunc) Access the 'dst' of the tzinfo `<'datetime.timedelta/None'>`.

    Equivalent to:
    >>> tz.dst(dt)
    """

def tz_utcoffset(
    tz: datetime.tzinfo | None,
    dt: datetime.datetime | None = None,
) -> datetime.timedelta | None:
    """(cfunc) Access the 'utcoffset' of the tzinfo `<'datetime.timedelta/None'>`.

    Equivalent to:
    >>> tz.utcoffset(dt)
    """

def tz_utcoffset_seconds(
    tz: datetime.tzinfo | None,
    dt: datetime.datetime | None = None,
) -> int:
    """(cfunc) Access the 'utcoffset' of the tzinfo in total seconds `<'int'>`.

    #### Returns `-100_000` if utcoffset is None.

    Equivalent to:
    >>> tz.utcoffset(dt).total_seconds()
    """

def tz_utcformat(
    tz: datetime.tzinfo | None,
    dt: datetime.datetime | None = None,
) -> str | None:
    """(cfunc) Access tzinfo as string in UTC format ('+/-HH:MM') `<'str/None'>`."""

# NumPy: share --------------------------------------------------------------------------------------
def map_nptime_unit_int2str(unit: int) -> str:
    """(cfunc) Map numpy datetime64/timedelta64 unit from integer
    to the corresponding string representation `<'str'>`."""

def map_nptime_unit_str2int(unit: str) -> int:
    """(cfunc) Map numpy datetime64/timedelta64 unit from string
    representation to the corresponding integer `<'int'>`."""

def get_arr_nptime_unit(arr: np.ndarray) -> int:
    """(cfunc) Get ndarray[datetime64/timedelta64] unit from the,
    returns the unit in `<'int'>`."""

def parse_arr_nptime_unit(arr: np.ndarray) -> int:
    """(cfunc) Parse ndarray[datetime64/timedelta64] unit from the,
    returns the unit in `<'int'>`."""

# NumPy: datetime64 ---------------------------------------------------------------------------------
# . type check
def is_dt64(obj: object) -> bool:
    """(cfunc) Check if an object is an instance of np.datetime64 `<'bool'>`.

    Equivalent to:
    >>> isinstance(obj, np.datetime64)
    """

def validate_dt64(obj: object) -> None:
    """(cfunc) Validate if an object is an instance of np.datetime64,
    and raises `TypeError` if not."""

# . conversion
def dt64_as_int64_us(dt64: np.datetime64, offset: int = 0) -> int:
    """(cfunc) Cast np.datetime64 to int64 under 'us' (microsecond) resolution `<'int'>`.

    Equivalent to:
    >>> dt64.astype("datetime64[us]").astype("int64") + offset
    """

def dt64_to_dt(
    dt64: np.datetime64,
    tz: datetime.tzinfo | None = None,
) -> datetime.datetime:
    """(cfunc) Convert np.datetime64 to datetime `<'datetime.datetime'>`."""

# NumPy: timedelta64 --------------------------------------------------------------------------------
# . type check
def is_td64(obj: object) -> bool:
    """(cfunc) Check if an object is an instance of np.timedelta64 `<'bool'>`.

    Equivalent to:
    >>> isinstance(obj, np.timedelta64)
    """

def validate_td64(obj: object) -> None:
    """(cfunc) Validate if an object is an instance of np.timedelta64,
    and raises `TypeError` if not."""

# . conversion
def td64_as_int64_us(td64: np.timedelta64, offset: int = 0) -> int:
    """(cfunc) Cast np.timedelta64 to int64 under 'us' (microsecond) resolution `<'int'>`.

    Equivalent to:
    >>> td64.astype("timedelta64[D]").astype("int64") + offset
    """

def td64_to_td(td64: np.timedelta64) -> datetime.timedelta:
    """(cfunc) Convert np.timedelta64 to timedelta `<'datetime.timedelta'>`.

    Equivalent to:
    >>> us = td64.astype("timedelta64[us]").astype("int64")
    >>> datetime.timedelta(microseconds=int(us))
    """

# NumPy: ndarray ---------------------------------------------------------------------------------------
# . type check
def is_arr(obj: object) -> bool:
    """(cfunc) Check if an object is an instance of np.ndarray `<'bool'>`.

    Equivalent to:
    >>> isinstance(obj, np.ndarray)
    """

# . dtype
def arr_assure_int64(arr: np.ndarray) -> np.ndarray[np.int64]:
    """(cfunc) Assure the given ndarray is dtype of 'int64' `<'ndarray[int64]'>`.

    Automatically cast the 'arr' to 'int64' if not the correct dtype.
    """

def arr_assure_int64_like(arr: np.ndarray) -> np.ndarray:
    """(cfunc) Assure the given ndarray is dtype of [int64/datetime64/timedelta64] `<'ndarray'>`.

    The data of an 'int64-like' array can be directly accessed as 'np.npy_int64*'

    Automatically cast the 'arr' to 'int64' if not the correct dtype.
    """

def arr_assure_float64(arr: np.ndarray) -> np.ndarray[np.float64]:
    """(cfunc) Assure the given ndarray is dtype of 'float64' `<'ndarray[int64]'>`.

    Automatically cast the 'arr' to 'flaot64' if not the correct dtype.
    """

# . create
def arr_zero_int64(size: int) -> np.ndarray[np.int64]:
    """(cfunc) Create an 1-dimensional ndarray[int64]
    filled with zero `<'ndarray[int64]'>`.

    Equivalent to:
    >>> np.zeros(size, dtype="int64")
    """

def arr_full_int64(value: int, size: int) -> np.ndarray[np.int64]:
    """(cfunc) Create an 1-dimensional ndarray[int64]
    filled with 'value' `<'ndarray[int64]'>`.

    Equivalent to:
    >>> np.full(size, value, dtype="int64")
    """

# . range
def arr_clip(
    arr: np.ndarray,
    minimum: int,
    maximum: int,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Clip the values of an ndarray to a given range `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not dtype of [int64/datetime64/timedelta64].

    Equivalent to:
    >>> np.clip(arr, minimum, maximum) + offset
    """

def arr_min(arr: np.ndarray, value: int, offset: int = 0) -> np.ndarray[np.int64]:
    """(cfunc) Get the minimum values between the ndarray and the 'value' `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not dtype of [int64/datetime64/timedelta64].

    Equivalent to:
    >>> np.minimum(arr, value) + offset
    """

def arr_max(arr: np.ndarray, value: int, offset: int = 0) -> np.ndarray[np.int64]:
    """(cfunc) Get the maximum values between the ndarray and the 'value' `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not dtype of [int64/datetime64/timedelta64].

    Equivalent to:
    >>> np.maximum(arr, value) + offset
    """

def arr_min_arr(
    arr1: np.ndarray,
    arr2: np.ndarray,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Get the minimum values between two ndarrays `<'ndarray[int64]'>`.

    Before compute, this function will cast the arrays to 'int64'
    if they are not dtype of [int64/datetime64/timedelta64].

    Equivalent to:
    >>> np.minimum(arr1, arr2) + offset
    """

def arr_max_arr(
    arr1: np.ndarray,
    arr2: np.ndarray,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Get the maximum values between two ndarrays `<'ndarray[int64]'>`.

    Before compute, this function will cast the arrays to 'int64'
    if they are not dtype of [int64/datetime64/timedelta64].

    Equivalent to:
    >>> np.maximum(arr1, arr2) + offset
    """

# . arithmetic
def arr_abs(arr: np.ndarray, offset: int = 0):
    """(cfunc) Compute the absolute values of the ndarray `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not dtype of [int64/datetime64/timedelta64].

    Equivalent to:
    >>> np.abs(arr) + offset
    """

def arr_add(arr: np.ndarray, value: int) -> np.ndarray[np.int64]:
    """(cfunc) Add the value to the ndarray `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> arr + value
    """

def arr_mul(arr: np.ndarray, factor: int, offset: int = 0) -> np.ndarray[np.int64]:
    """(cfunc) Multiply the values of the ndarray by the factor `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> arr * factor + offset
    """

def arr_div(
    arr: np.ndarray,
    factor: int | float,
    offset: int | float = 0,
) -> np.ndarray[np.float64]:
    """(cfunc) Divides the values of the ndarray by the factor, handling negative
    numbers accoring to Python's division semantics `<'ndarray[float64]'>`.

    Before compute, this function will cast the array to 'float64'
    if it is not in 'float64' dtype.

    Equivalent to:
    >>> arr / factor + offset
    """

def arr_mod(
    arr: np.ndarray,
    factor: int,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Computes the modulo of the values of the ndarray by the
    factor, handling negative numbers according to Python's modulo
    semantics `<'ndarray[int64]'>`.

    Before computation, this function will cast the array to 'int64'
    if it is not already in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> arr % factor + offset
    """

def arr_round_div(
    arr: np.ndarray,
    factor: int,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Divides the values of the ndarray by the factor and rounds
    to the nearest integers (half away from zero), handling negative
    numbers accoring to Python's division semantics `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> np.round(arr / factor, 0) + offset
    """

def arr_ceil_div(
    arr: np.ndarray,
    factor: int,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Divides the values of the ndarray by the factor and
    rounds up to the nearest integers, handling negative numbers
    accoring to Python's division semantics `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> np.ceil(arr / factor) + offset
    """

def arr_floor_div(
    arr: np.ndarray,
    factor: int,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Divides the values of the ndarray by the factor and
    rounds down to the nearest integers, handling negative numbers
    accoring to Python's division semantics `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> np.floor(arr / factor) + offset
    """

def arr_round_to_mul(
    arr: np.ndarray,
    factor: int,
    multiple: int = 0,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Round to multiple. Divides the values of the ndarray by the factor
    and rounds to the nearest integers (half away from zero), handling negative
    numbers accoring to Python's division semantics. Finally multiply the the multiple.
    Argument 'multiple' defaults to `0`, which means if not specified, it uses
    factor as the multiple `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> np.round(arr / factor, 0) * multiple + offset
    """

def arr_ceil_to_mul(
    arr: np.ndarray,
    factor: int,
    multiple: int = 0,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Ceil to multiple. Divides the values of the ndarray by the factor
    and rounds up to the nearest integers, handling negative numbers accoring
    to Python's division semantics. Finally multiply the the multiple. Argument
    'multiple' defaults to `0`, which means if not specified, it uses factor
    as the multiple `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> np.ceil(arr / factor) * multiple + offset
    """

def arr_floor_to_mul(
    arr: np.ndarray,
    factor: int,
    multiple: int = 0,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Floor to multiple. Divides the values of the ndarray by the factor
    and rounds down to the nearest integers, handling negative numbers accoring
    to Python's division semantics. Finally multiply the the multiple. Argument
    multiple defaults to '0', which means if not specified, it uses factor as
    the multiple `<'ndarray[int64]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> np.floor(arr / factor) * multiple + offset
    """

def arr_equal_to_arr(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray[bool]:
    """(cfunc) Check if the values of two ndarrays are equal `<'ndarray[bool]'>`

    Equivalent to:
    >>> arr1 == arr2
    """

def arr_add_arr(
    arr1: np.ndarray,
    arr2: np.ndarray,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Addition between two ndarrays `<'ndarray[int64]'>`.

    Before compute, this function will cast the arrays to 'int64'
    if they are not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> arr1 + arr2 + offset
    """

def arr_sub_arr(
    arr1: np.ndarray,
    arr2: np.ndarray,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Substraction between two ndarrays `<'ndarray[int64]'>`.

    Before compute, this function will cast the arrays to 'int64'
    if they are not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> arr1 - arr2 + offset
    """

# . comparison
def arr_equal_to(arr: np.ndarray, value: int) -> np.ndarray[bool]:
    """(cfunc) Check if the values of the ndarray are equal
    to the 'value' `<'ndarray[bool]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> arr == value
    """

def arr_greater_than(arr: np.ndarray, value: int) -> np.ndarray[bool]:
    """(cfunc) Check if the values of the ndarray are greater
    than the 'value' `<'ndarray[bool]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> arr > value
    """

def arr_less_than(arr: np.ndarray, value: int) -> np.ndarray[bool]:
    """(cfunc) Check if the values of the ndarray are less
    than the 'value' `<'ndarray[bool]'>`.

    Before compute, this function will cast the array to 'int64'
    if it is not in 'int64'/'datetime64'/'timedelta64' dtype.

    Equivalent to:
    >>> arr < value
    """

# NumPy: ndarray[datetime64] ---------------------------------------------------------------------------
# . type check
def is_dt64arr(arr: np.ndarray) -> bool:
    """(cfunc) Check if the given array is dtype of 'datetime64' `<'bool'>`.

    Equivalent to:
    >>> isinstance(arr.dtype, np.dtypes.DateTime64DType)
    """

def validate_dt64arr(arr: np.ndarray) -> bool:
    """Validate if the given array is dtype of 'datetime64',
    raises `TypeError` if dtype is incorrect.
    """

# . range check
def is_dt64arr_ns_safe(
    arr: np.ndarray,
    arr_unit: str = None,
    wide: bool = True,
) -> bool:
    """(cfunc) Check if the ndarray[datetime64] is within
    nanosecond conversion range `<'bool'>`.

    - 'wide=True': ragne between '1677-09-22' and '2262-04-10' (+/- 1 day from limits)
    - 'wide=False': range between '1678-01-01' and '2261-01-01' (+/- 1 year from limits)
    """

# . access
def dt64arr_year(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Get the year values of the ndarray[datetime64] `<'ndarray[int64]'>`."""

def dt64arr_quarter(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Get the quarter values of the ndarray[datetime64] `<'ndarray[int64]'>`."""

def dt64arr_month(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Get the month values of the ndarray[datetime64] `<'ndarray[int64]'>`."""

def dt64arr_weekday(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Get the weekday values of the ndarray[datetime64] `<'ndarray[int64]'>`."""

def dt64arr_day(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Get the weekday values of the ndarray[datetime64] `<'ndarray[int64]'>`."""

def dt64arr_hour(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Get the hour values of the ndarray[datetime64] `<'ndarray[int64]'>`."""

def dt64arr_minute(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Get the minute values of the ndarray[datetime64] `<'ndarray[int64]'>`."""

def dt64arr_second(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Get the second values of the ndarray[datetime64] `<'ndarray[int64]'>`."""

def dt64arr_millisecond(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Get the millisecond values of the ndarray[datetime64] `<'ndarray[int64]'>`."""

def dt64arr_microsecond(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """Get the microsecond values of the ndarray[datetime64] `<'ndarray[int64]'>`."""

def dt64arr_nanosecond(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Get the nanosecond values of the ndarray[datetime64] `<'ndarray[int64]'>`."""

def dt64arr_times(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """Get the times values of the ndarray[datetime64] `<'ndarray[int64]'>`."""

# . calendar
def dt64arr_isocalendar(arr: np.ndarray, arr_unit: str = None) -> np.ndarray[np.int64]:
    """(cfunc) Get the ISO calendar values
    of the ndarray[datetime64] `<'ndarray[int64]'>`.

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

def dt64arr_is_leap_year(arr: np.ndarray, arr_unit: str = None) -> np.ndarray[np.bool_]:
    """(cfunc) Check if the ndarray[datetime64] are leap years `<'ndarray[bool]'>`."""

def dt64arr_is_long_year(arr: np.ndarray, arr_unit: str = None) -> np.ndarray[np.bool_]:
    """(cfunc) Check if the ndarray[datetime64] are long years
    (maximum ISO week number equal 53) `<'ndarray[bool]'>`.
    """

def dt64arr_leap_bt_year(
    arr: np.ndarray,
    year: int,
    arr_unit: str = None,
) -> np.ndarray[np.int64]:
    """(cfunc) Calcuate the number of leap years between the ndarray[datetime64]
    and the passed in 'year' value `<'ndarray[int64]'>`.
    """

def dt64arr_days_in_year(arr: np.ndarray, arr_unit: str = None) -> np.ndarray[np.int64]:
    """(cfunc) Get the maximum days (365, 366) in the year
    of the ndarray[datetime64] `<'ndarray[int64]'>`.
    """

def dt64arr_days_bf_year(arr: np.ndarray, arr_unit: str = None) -> np.ndarray[np.int64]:
    """(cfunc) Get the number of days between the np.ndarray[datetime64]
    and the 1st day of the 1AD `<'ndarray[int64]'>`.
    """

def dt64arr_days_of_year(arr: np.ndarray, arr_unit: str = None) -> np.ndarray[np.int64]:
    """(cfunc) Get the number of days between the np.ndarray[datetime64]
    and the 1st day of the array years `<'ndarray[int64]'>`.
    """

def dt64arr_days_in_quarter(
    arr: np.ndarray,
    arr_unit: str = None,
) -> np.ndarray[np.int64]:
    """(cfunc) Get the maximum days in the quarter of the np.npdarray[datetime64] `<'int'>`."""

def dt64arr_days_bf_quarter(
    arr: np.ndarray,
    arr_unit: str = None,
) -> np.ndarray[np.int64]:
    """(cfucn) Get the number of days between the 1st day of the year
    of the np.ndarray[datetime64] and the 1st day of its quarter `<'int'>`.
    """

def dt64arr_days_of_quarter(
    arr: np.ndarray,
    arr_unit: str = None,
) -> np.ndarray[np.int64]:
    """(cfunc) Get the number of days between the 1st day of the quarter
    of the np.ndarray[datetime64] and the its date `<'int'>`."""

def dt64arr_days_in_month(
    arr: np.ndarray,
    arr_unit: str = None,
) -> np.ndarray[np.int64]:
    """(cfunc) Get the maximum days in the month
    of the ndarray[datetime64] <'ndarray[int64]'>.
    """

def dt64arr_days_bf_month(
    arr: np.ndarray,
    arr_unit: str = None,
) -> np.ndarray[np.int64]:
    """(cfunc) Get the number of days between the 1st day of the
    np.ndarray[datetime64] and the 1st day of its month `<'int'>`.
    """

# . conversion: int64
def dt64arr_fr_int64(val: int, unit: str, size: int) -> np.ndarray[np.datetime64]:
    """(cfunc) Create an ndarray[datetime64] from the
    passed in integer and array size `<'ndarray[datetime64]'>`.

    Equivalent to:
    >>> np.array([val for _ in range(size)], dtype="datetime64[%s]" % unit)
    """

def dt64arr_as_int64(
    arr: np.ndarray,
    unit: str,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Cast np.ndarray[datetime64] to int64 according to the given
    'unit' resolution `<'ndarray[int64]'>`.

    Supported units: 'Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'.

    Equivalent to:
    >>> arr.astype(f"datetime64[unit]").astype("int64") + offset
    """

def dt64arr_as_int64_Y(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Cast np.ndarray[datetime64] to int64 under 'Y' (year)
    resolution `<'ndarray[int64]>`.

    Equivalent to:
    >>> arr.astype("datetime64[Y]").astype("int64") + offset
    """

def dt64arr_as_int64_Q(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Cast np.ndarray[datetime64] to int64 under 'Q' (quarter)
    resolution `<'ndarray[int64]'>`.
    """

def dt64arr_as_int64_M(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Cast np.ndarray[datetime64] to int64 under 'M' (month)
    resolution `<'ndarray[int64]'>`.

    Equivalent to:
    >>> arr.astype("datetime64[M]").astype("int64") + offset
    """

def dt64arr_as_int64_W(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Cast np.ndarray[datetime64] to int64 under 'W' (week)
    resolution `<'ndarray[int64]'>`.

    Equivalent to:
    >>> arr.astype("datetime64[W]").astype("int64") + offset
    """

def dt64arr_as_int64_D(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Cast np.ndarray[datetime64] to int64 under 'D' (day)
    resolution `<'ndarray[int64]'>`.

    Equivalent to:
    >>> arr.astype("datetime64[D]").astype("int64") + offset
    """

def dt64arr_as_int64_h(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Cast np.ndarray[datetime64] to int64 under 'h' (hour)
    resolution `<'ndarray[int64]'>`.

    Equivalent to:
    >>> arr.astype("datetime64[h]").astype("int64") + offset
    """

def dt64arr_as_int64_m(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Cast np.ndarray[datetime64] to int64 under 'm' (minute)
    resolution `<'ndarray[int64]'>`.

    Equivalent to:
    >>> arr.astype("datetime64[m]").astype("int64") + offset
    """

def dt64arr_as_int64_s(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Cast np.ndarray[datetime64] to int64 under 's' (second)
    resolution `<'ndarray[int64]'>`.

    Equivalent to:
    >>> arr.astype("datetime64[s]").astype("int64") + offset
    """

def dt64arr_as_int64_ms(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Cast np.ndarray[datetime64] to int64 under 'ms' (millisecond)
    resolution `<'ndarray[int64]'>`.

    Equivalent to:
    >>> arr.astype("datetime64[ms]").astype("int64") + offset
    """

def dt64arr_as_int64_us(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Cast np.ndarray[datetime64] to int64 under 'us' (microsecond)
    resolution `<'ndarray[int64]'>`.

    Equivalent to:
    >>> arr.astype("datetime64[us]").astype("int64") + offset
    """

def dt64arr_as_int64_ns(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Cast np.ndarray[datetime64] to int64 under 'ns' (nanosecond)
    resolution `<'ndarray[int64]'>`.

    Equivalent to:
    >>> arr.astype("datetime64[ns]").astype("int64") + offset
    """

def dt64arr_as_iso_W(
    arr: np.ndarray,
    weekday: int,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Cast np.ndarray[datetime64] to int64 with 'W' (week)
    resolution, aligned to the specified ISO 'weekday' `<'ndarray[int64]'>`.

    NumPy aligns datetime64[W] to Thursday (the weekday of 1970-01-01).
    This function allows specifying the ISO 'weekday' (1=Monday, 7=Sunday)
    for alignment.

    For example: if 'weekday=1', the result represents the Monday-aligned
    weeks since EPOCH (1970-01-01).
    """

def dt64arr_to_ordinal(arr: np.ndarray) -> np.ndarray[np.int64]:
    """(cfunc) Convert np.ndarray[datetime64] to proleptic Gregorian
    ordinals `<'ndarray[int64]'>`.

    '0001-01-01' is day 1 (ordinal=1).
    """

# . conversion: float64
def dt64arr_to_ts(arr: np.ndarray) -> np.ndarray[np.float64]:
    """(cfunc) Convert np.ndarray[datetime64] to timestamps
     since Unix Epoch `<'ndarray[float64]'>`.

    Fractional seconds are rounded to the nearest microsecond.
    """

# . conversion: unit
def dt64arr_as_unit(
    arr: np.ndarray,
    unit: str,
    arr_unit: str = None,
    limit: bool = False,
) -> np.ndarray[np.datetime64]:
    """Convert np.ndarray[datetime64] to the specified unit `<'ndarray[datetime64]'>`.

    - 'limit=False': supports conversion among ['Y', 'M', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'].
    - 'limit=True': supports conversion among ['s', 'ms', 'us', 'ns'].

    Equivalent to:
    >>> arr.astype(f"datetime64[unit]")
    """

# . arithmetic
def dt64arr_round(
    arr: np.ndarray,
    unit: str,
    arr_unit: str = None,
) -> np.ndarray[np.datetime64]:
    """(cfunc) Perform round operation on the np.ndarray[datetime64] to the
    specified unit `<'ndarray[datetime64]'>`.

    - Supported array resolution: 'ns', 'us', 'ms', 's', 'm', 'h', 'D'
    - Supported units: 'ns', 'us', 'ms', 's', 'm', 'h', 'D'.
    """

def dt64arr_ceil(
    arr: np.ndarray,
    unit: str,
    arr_unit: str = None,
) -> np.ndarray[np.datetime64]:
    """(cfunc) Perform ceil operation on the np.ndarray[datetime64] to the
    specified unit `<'ndarray[datetime64]'>`.

    - Supported array resolution: 'ns', 'us', 'ms', 's', 'm', 'h', 'D'
    - Supported units: 'ns', 'us', 'ms', 's', 'm', 'h', 'D'.
    """

def dt64arr_floor(
    arr: np.ndarray,
    unit: str,
    arr_unit: str = None,
) -> np.ndarray[np.datetime64]:
    """(cfunc) Perform floor operation on the np.ndarray[datetime64] to the
    specified unit `<'ndarray[datetime64]'>`.

    - Supported array resolution: 'ns', 'us', 'ms', 's', 'm', 'h', 'D'
    - Supported units: 'ns', 'us', 'ms', 's', 'm', 'h', 'D'.
    """

# . comparison
def dt64arr_find_closest(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray[np.int64]:
    """For each element in 'arr1', find the closest values in 'arr2' `<'ndarray[int64]'>`."""

def dt64arr_find_farthest(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray[np.int64]:
    """For each element in 'arr1', find the farthest values in 'arr2' `<'ndarray[int64]'>`."""

# NumPy: ndarray[timedelta64] --------------------------------------------------------------------------
# . type check
def is_td64arr(arr: np.ndarray) -> bool:
    """(cfunc) Check if the given array is dtype of 'timedelta64' `<'bool'>`.

    Equivalent to:
    >>> isinstance(arr.dtype, np.dtypes.TimeDelta64DType)
    """

def validate_td64arr(arr: np.ndarray) -> bool:
    """(cfunc) Validate if the given array is dtype of 'timedelta64',
    raises `TypeError` if dtype is incorrect.
    """

# . conversion
def td64arr_as_int64_us(
    arr: np.ndarray,
    arr_unit: str = None,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """Cast np.ndarray[timedelta64] to int64 under 'us' (microsecond)
    resolution `<'ndarray[int64]'>`.

    Equivalent to:
    >>> arr.astype("timedelta64[us]").astype("int64") + offset
    """
