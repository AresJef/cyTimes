import cython
import datetime
import numpy as np
from typing import Literal

# Delta ---------------------------------------------------------------------------------------------
def combine_abs_ms_us(ms: int, us: int) -> int:
    """(cfunc) Combine absolute millisecond and microsecond to microsecond `<'int'>`.

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

# Parser --------------------------------------------------------------------------------------------
def str_count(s: str, substr: str) -> int:
    """Get number of occurrences of a 'substr' in an unicode `<'int'>`.

    Equivalent to:
    >>> s.count(substr)
    """

def is_iso_sep(ch: cython.Py_UCS4) -> bool:
    """Check if 'ch' is ISO format date/time seperator (" " or "T") `<'bool'>`"""

def is_isodate_sep(ch: cython.Py_UCS4) -> bool:
    """Check if 'ch' is ISO format date values separator ("-" or "/") `<'bool'>`"""

def is_isoweek_sep(ch: cython.Py_UCS4) -> bool:
    """Check if 'ch' is ISO format week separator ("W") `<'bool'>`"""

def is_isotime_sep(ch: cython.Py_UCS4) -> bool:
    """Check if 'ch' is ISO format time values separator (":") `<'bool'>`"""

def is_ascii_digit(ch: cython.Py_UCS4) -> bool:
    """Check if 'ch' is an ASCII digit [0-9] `<'bool'>`"""

def is_ascii_alpha_upper(ch: cython.Py_UCS4) -> bool:
    """Check if 'ch' is an ASCII alpha uppercase [A-Z] `<'bool'>`."""

def is_ascii_alpha_lower(ch: cython.Py_UCS4) -> bool:
    """Check if 'ch' is an ASCII alpha lowercase [a-z] `<'bool'>`."""

def is_ascii_alpha(ch: cython.Py_UCS4) -> bool:
    """Check if 'ch' is an ASCII alpha [a-zA-Z] `<'bool'>`."""

def parse_isoyear(data: str, pos: int, size: int) -> int:
    """(cfunc) Parse ISO format year (YYYY) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO year (YYYY) from.
    :param pos `<'int'>`: The starting position of the ISO year.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format year value.
    """

def parse_isomonth(data: str, pos: int, size: int) -> int:
    """(cfunc) Parse ISO format month (MM) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO month (MM) from.
    :param pos `<'int'>`: The starting position of the ISO month.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format month value.
    """

def parse_isoday(data: str, pos: int, size: int) -> int:
    """(cfunc) Parse ISO format day (DD) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO day (DD) from.
    :param pos `<'int'>`: The starting position of the ISO day.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format day value.
    """

def parse_isoweek(data: str, pos: int, size: int) -> int:
    """(cfunc) Parse ISO format week number (WW) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO week number (WW) from.
    :param pos `<'int'>`: The starting position of the ISO week number.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format week number value.
    """

def parse_isoweekday(data: str, pos: int, size: int) -> int:
    """(cfunc) Parse ISO format weekday (D) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO weekday (D) from.
    :param pos `<'int'>`: The starting position of the ISO weekday.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format weekday value.
    """

def parse_isoyearday(data: str, pos: int, size: int) -> int:
    """(cfunc) Parse ISO format day of year (DDD) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO day of year (DDD) from.
    :param pos `<'int'>`: The starting position of the ISO day of year.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format day of year value.
    """

def parse_isohour(data: str, pos: int, size: int) -> int:
    """(cfunc) Parse ISO format hour (HH) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO hour (HH) from.
    :param pos `<'int'>`: The starting position of the ISO hour.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format hour value.
    """

def parse_isominute(data: str, pos: int, size: int) -> int:
    """(cfunc) Parse ISO format minute (MM) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO minute (MM) from.
    :param pos `<'int'>`: The starting position of the ISO minute.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format minute value.
    """

def parse_isosecond(data: str, pos: int, size: int) -> int:
    """(cfunc) Parse ISO format second (SS) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO second (SS) from.
    :param pos `<'int'>`: The starting position of the ISO second.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format second value.
    """

def parse_isofraction(data: str, pos: int, size: int) -> int:
    """(cfunc) Parse ISO format fraction (f/us) from 'data' string `<'int'>`.

    :param data `<'str'>`: The string to parse ISO fraction (f/us) from.
    :param pos `<'int'>`: The starting position of the ISO fraction.
    :param size `<'int'>`: The length of the 'data' string.
        - If 'size <= 0', the function measure the size of the 'data' string internal.

    :return `<'int'>`: `-1` for invalid ISO format fraction value.
    """

def slice_to_uint(s: str, start: int, size: int) -> int:
    """(cfunc) Slice & convert 'data' to an integer `<'int'>`.

    :param data `<'str'>`: The string to slice to an integer.
    :param start `<'int'>`: The starting position of the integer slice.
    :param size `<'int'>`: Total characters to slice from the starting position.
    :raise `ValueError`: If cannot convert slice of 'data' to an integer.
    """

# Time ----------------------------------------------------------------------------------------------
def tm_strftime(t: object, fmt: str) -> str:
    """(cfunc) Convert struct_time (struct:tm) to string with the given 'fmt' `<'str'>`."""

def tm_gmtime(ts: float) -> dict:
    """(cfunc) Get the struc_time of the 'ts' expressing UTC time `<'struct:tm'>`.

    Equivalent to:
    >>> time.gmtime(ts)
    """

def tm_localtime(ts: float) -> dict:
    """(cfunc) Get struc_time of the 'ts' expressing local time `<'struct:tm'>`.

    Equivalent to:
    >>> time.localtime(ts)
    """

def ts_gmtime(ts: float) -> int:
    """(cfunc) Get timestamp of the 'ts' expressing UTC time `<'int'>."""

def ts_localtime(ts: float) -> int:
    """(cfunc) Get timestamp of the 'ts' expressing local time `<'int'>`."""

def tm_fr_seconds(seconds: float) -> dict:
    """(cfunc) Convert total seconds since Unix Epoch to `<'struct:tm'>`."""

def tm_fr_us(us: int) -> dict:
    """(cfunc) Convert total microseconds since Unix Epoch to `<'struct:tm'>`."""

def hms_fr_seconds(seconds: float) -> dict:
    """(cfunc) Convert total seconds to 'H/M/S' `<'struct:hms'>`."""

def hms_fr_us(us: int) -> dict:
    """(cfunc) Convert microseconds to 'H/M/S' `<'struct:hms'>`."""

# Calender ------------------------------------------------------------------------------------------
# . year
def is_leap_year(year: int) -> bool:
    """(cfunc) Determines whether the given 'year' is a leap year `<'bool'>`."""

def is_long_year(year: int) -> bool:
    """Whether the given 'year' is a long year
    (maximum ISO week number is 53) `<'bool'>`.
    """

def leap_bt_years(year1: int, year2: int) -> int:
    """(cfunc) Calculate total leap years between 'year1' and 'year2' `<'int'>`."""

def days_in_year(year: int) -> int:
    """(cfunc) Get total days of the given 'year', expects 365 or 366 `<'int'>`."""

def days_bf_year(year: int) -> int:
    """(cfunc) Get total days between the 1st day of 1AD
    and the 1st day of the given 'year' `<'int'>`."""

def days_of_year(year: int, month: int, day: int) -> int:
    """(cfunc) Get the days between the 1st day of the 'year'
    and the given 'Y/M/D' `<'int'>`."""

# . quarter
def quarter_of_month(month: int) -> int:
    """(cfunc) Get the quarter of the given 'month', expects 1-4 `<'int'>`."""

def days_in_quarter(year: int, month: int) -> int:
    """(cfunc) Get total days of the quarter for the given 'Y/M' `<'int'>`."""

def days_bf_quarter(year: int, month: int) -> int:
    """(cfunc) Get total days between the 1st day of the 'year'
    and the 1st day of the quarter for the given 'Y/M' `<'int'>`."""

def days_of_quarter(year: int, month: int, day: int) -> int:
    """(cfunc) Get the days between the 1st day of the quarter
    and the given 'Y/M/D' `<'int'>`."""

def quarter_1st_month(month: int) -> int:
    """(cfunc) Get the first month of the quarter, expects 1, 4, 7, 10 `<'int'>`."""

def quarter_lst_month(month: int) -> int:
    """(cfunc) Get the last month of the quarter, expects 3, 6, 9, 12 `<'int'>`."""

# . month
def days_in_month(year: int, month: int) -> int:
    """(cfunc) Get total days of the 'month' in the given 'year' `<'int'>`."""

def days_bf_month(year: int, month: int) -> int:
    """(cfunc) Get total days between the 1st day of the 'year'
    and the 1st day of the given 'month' `<'int'>`."""

# . week
def ymd_weekday(year: int, month: int, day: int) -> int:
    """(cfunc) Get the weekday of the given 'Y/M/D',
    expects 0[Monday]...6[Sunday] `<'int'>`."""

def ymd_isoweekday(year: int, month: int, day: int) -> int:
    """(cfunc) Get the ISO weekday of the given 'Y/M/D',
    expects 1[Monday]...7[Sunday] `<'int'>`."""

def ymd_isoweek(year: int, month: int, day: int) -> int:
    """(cfunc) Get the ISO calendar week number of the given 'Y/M/D' `<'int'>`."""

def ymd_isoyear(year: int, month: int, day: int) -> int:
    """(cfunc) Get the ISO calendar year of the given 'Y/M/D' `<'int'>`."""

def ymd_isocalendar(year: int, month: int, day: int) -> dict:
    """(cfunc) Get the ISO calendar of the given 'Y/M/D' `<'struct:iso'>`."""

def ymd_to_ordinal(year: int, month: int, day: int) -> int:
    """(cfunc) Convert 'Y/M/D' to ordinal days `<'int'>`."""

def ymd_fr_ordinal(ordinal: int) -> dict:
    """(cfunc) Convert ordinal days to 'Y/M/D' `<'struct:ymd'>`."""

def ymd_fr_isocalendar(year: int, week: int, weekday: int) -> dict:
    """(cfunc) Convert ISO calendar to 'Y/M/D' `<'struct:ymd>`."""

def ymd_fr_days_of_year(year: int, days: int) -> dict:
    """(cfunc) Convert days of the year to 'Y/M/D' `<'struct:ymd'>`."""

def iso_1st_monday(year: int) -> int:
    """(cfunc) Get the ordinal of the 1st Monday of the ISO 'year' `<'int'>`."""

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

# . conversion
def date_to_tm(date: datetime.date) -> dict:
    """(cfunc) Convert datetime.date to `<'struct:tm'>`.

    #### All time values sets to 0.
    """

def date_to_strformat(date: datetime.date, fmt: str) -> str:
    """(cfunc) Convert datetime.date to str with the specified 'fmt' `<'str'>`.

    Equivalent to:
    >>> date.strftime(fmt)
    """

def date_to_isoformat(date: datetime.date) -> str:
    """(cfunc) Convert datetime.date to ISO format: '%Y-%m-%d' `<'str'>`."""

def date_to_ordinal(date: datetime.date) -> int:
    """(cfunc) Convert datetime.date to ordinal days `<'int'>`."""

def date_to_seconds(date: datetime.date) -> float:
    """(cfunc) Convert datetime.date to total seconds since Unix Epoch `<'float'>`."""

def date_to_us(date: datetime.date) -> int:
    """(cfunc) Convert datetime.date to total microseconds since Unix Epoch `<'int'>`."""

def date_to_ts(date: datetime.date) -> float:
    """(cfunc) Convert datetime.date to timestamp `<'float'>`."""

def date_fr_date(date: datetime.date) -> datetime.date:
    """(cfunc) Convert subclass of datetime.date to `<'datetime.date'>`."""

def date_fr_dt(dt: datetime.datetime) -> datetime.date:
    """(cfunc) Convert datetime.datetime to `<'datetime.date'>`."""

def date_fr_ordinal(ordinal: int) -> datetime.date:
    """(cfunc) Convert ordinal days to `<'datetime.date'>`."""

def date_fr_seconds(seconds: float) -> datetime.date:
    """(cfunc) Convert total seconds since Unix Epoch to `<'datetime.date'>`."""

def date_fr_us(us: int) -> datetime.date:
    """(cfunc) Convert total microseconds since Unix Epoch to `<'datetime.date'>`."""

def date_fr_ts(ts: float) -> datetime.date:
    """(cfunc) Convert timestamp to `<'datetime.date'>`."""

# . manipulation
def date_replace(
    date: datetime.date,
    year: int = -1,
    month: int = -1,
    day: int = -1,
) -> datetime.date:
    """(cfunc) Replace datetime.date values `<'datetime.date'>`.

    #### Default '-1' mean keep the original value.

    Equivalent to:
    >>> date.replace(year, month, day)
    """

def date_chg_weekday(date: datetime.datetime, weekday: int) -> datetime.date:
    """(cfunc) Change datetime.date 'weekday' within the current week
    (0[Monday]...6[Sunday]) `<'datetime.date'>`.

    Equivalent to:
    >>> date + datetime.timedelta(weekday - date.weekday())
    """

# . arithmetic
def date_add(
    date: datetime.date,
    days: int = 0,
    seconds: int = 0,
    microseconds: int = 0,
    milliseconds: int = 0,
    minutes: int = 0,
    hours: int = 0,
    weeks: int = 0,
) -> datetime.date:
    """Add timedelta to datetime.date `<'datetime.date'>`.

    Equivalent to:
    >>> date + datetime.timedelta(
            days, seconds, microseconds,
            milliseconds, minutes, hours, weeks
        )
    """

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
    """(cfunc) Create a new `<'datetime.datetime'>`.

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

def dt_utcoffset_seconds(dt: datetime.datetime) -> int:
    """(cfunc) Get the tzinfo 'utcoffset' of the datetime in total seconds `<'int'>`.

    #### Returns `-100_000` if utcoffset is None.

    Equivalent to:
    >>> dt.utcoffset().total_seconds()
    """

def dt_utcformat(dt: datetime.datetime) -> str | None:
    """(cfunc) Get the tzinfo of the datetime as UTC format '+/-HH:MM' `<'str/None'>`."""

def dt_normalize_tz(dt: datetime.datetime) -> datetime.datetime:
    """(cfunc) Normalize the datetime to its tzinfo `<'datetime.datetime'>`."""

# . value check
def dt_is_1st_of_year(dt: datetime.datetime) -> bool:
    """(cfunc) Check if datetime is the 1st day of the year `<'bool'>`

    First day of the year: XXXX-01-01
    """

def dt_is_lst_of_year(dt: datetime.datetime) -> bool:
    """(cfunc) Check if datetime is the last day of the year `<'bool'>`

    Last day of the year: XXXX-12-31
    """

def dt_is_1st_of_quarter(dt: datetime.datetime) -> bool:
    """(cfunc) Check if datetime is the 1st day of the quarter `<'bool'>`.

    First day of the quarter: XXXX-(1, 4, 7, 10)-01
    """

def dt_is_lst_of_quarter(dt: datetime.datetime) -> bool:
    """Check if datetime is the 1st day of the quarter `<'bool'>`.

    Last day of the quarter: XXXX-(3, 6, 9, 12)-(30, 31)
    """

def dt_is_1st_of_month(dt: datetime.datetime) -> bool:
    """(cfunc) Check if datetime is the 1st day of the month `<'bool'>`.

    First day of the month: XXXX-XX-01
    """

def dt_is_lst_of_month(dt: datetime.datetime) -> bool:
    """Check if datetime is the last day of the month `<'bool'>`.

    Last day of the month: XXXX-XX-(28, 29, 30, 31)
    """

def dt_is_start_of_time(dt: datetime.datetime) -> bool:
    """Check if datetime is at start of the time `<'bool'>`.

    Start of time: 00:00:00.000000
    """

def dt_is_end_of_time(dt: datetime.datetime) -> bool:
    """Check if datetime is at end of the time `<'bool'>`.

    End of time: 23:59:59.999999
    """

# . conversion
def dt_to_tm(dt: datetime.datetime, utc: bool = False) -> dict:
    """(cfunc) Convert datetime.datetime to `<'struct:tm'>`."""

def dt_to_strformat(dt: datetime.datetime, fmt: str) -> str:
    """(cfunc) Convert datetime.datetime to string with the specified 'fmt' `<'str'>`.

    Equivalent to:
    >>> dt.strftime(fmt)
    """

def dt_to_isoformat(dt: datetime.datetime, sep: str = "T", utc: bool = False) -> str:
    """(cfunc) Convert datetime.datetime to ISO format `<'str'>`.

    If 'dt' is timezone-aware, setting 'utc=True'
    adds the UTC(Z) at the end of the ISO format.
    """

def dt_to_ordinal(dt: datetime.datetime, utc: bool = False) -> int:
    """(cfunc) Convert datetime.datetime to ordinal days `<'int'>`.

    If 'dt' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from total ordinal days.
    """

def dt_to_seconds(dt: datetime.datetime, utc: bool = False) -> float:
    """(cfunc) Convert datetime.datetime to total seconds since Unix Epoch `<'float'>`.

    If 'dt' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from total seconds.
    """

def dt_to_us(dt: datetime.datetime, utc: bool = False) -> int:
    """(cfunc) Convert datetime.datetime to total microseconds since Unix Epoch `<'int'>`.

    If 'dt' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from total mircroseconds.
    """

def dt_to_posix(dt: datetime.date) -> int:
    """(cfunc) Convert datetime.datetime to POSIX timestamp `<'int'>`.

    This function does not take 'dt.tzinof' into consideration.

    Equivalent to:
    >>> dt._mktime()
    """

def dt_to_ts(dt: datetime.datetime) -> float:
    """(cfunc) Convert datetime.datetime to timestamp `<'float'>`.

    Equivalent to:
    >>> dt.timestamp()
    """

def dt_combine(
    date: datetime.date | None = None,
    time: datetime.time | None = None,
    tz: datetime.tzinfo | None = None,
) -> datetime.datetime:
    """(cfunc) Combine datetime.date & datetime.time to `<'datetime.datetime'>`.

    - If 'date' is None, use current local date.
    - If 'time' is None, all time fields set to 0.
    """

def dt_fr_date(
    date: datetime.date,
    tz: datetime.tzinfo | None = None,
) -> datetime.datetime:
    """(cfunc) Convert datetime.date to `<'datetime.datetime'>`.

    #### All time values sets to 0.
    """

def dt_fr_dt(dt: datetime.datetime) -> datetime.datetime:
    """(cfunc) Convert subclass of datetime to `<'datetime.datetime'>`."""

def dt_fr_time(time: datetime.time) -> datetime.datetime:
    """(cfunc) Convert datetime.time to `<'datetime.datetime'>`.

    #### Date values sets to 1970-01-01.
    """

def dt_fr_ordinal(ordinal: int, tz: datetime.tzinfo | None = None) -> datetime.datetime:
    """(cfunc) Convert ordinal days to `<'datetime.datetime'>`."""

def dt_fr_seconds(
    seconds: float,
    tz: datetime.tzinfo | None = None,
) -> datetime.datetime:
    """(cfunc) Convert total seconds since Unix Epoch to `<'datetime.datetime'>`."""

def dt_fr_us(us: int, tz: datetime.tzinfo | None = None) -> datetime.datetime:
    """(cfunc) Convert total microseconds since Unix Epoch to `<'datetime.datetime'>`."""

def dt_fr_ts(ts: float, tz: datetime.tzinfo | None = None) -> datetime.datetime:
    """(cfunc) Convert timestamp to `<'datetime.datetime'>`."""

# . manipulation
def dt_replace(
    dt: datetime.datetime,
    year: int = -1,
    month: int = -1,
    day: int = -1,
    hour: int = -1,
    minute: int = -1,
    second: int = -1,
    millisecond: int = -1,
    microsecond: int = -1,
    tz: object | datetime.tzinfo | None = -1,
    fold: int = -1,
) -> datetime.datetime:
    """(cfunc) Replace the datetime.datetime values `<'datetime.datetime'>`.

    #### Default '-1' mean keep the original value.

    Equivalent to:
    >>> dt.replace(year, month, day, hour, minute, second, microsecond, tz, fold)
    """

def dt_replace_date(
    dt: datetime.datetime,
    year: int = -1,
    month: int = -1,
    day: int = -1,
) -> datetime.datetime:
    """(cfunc) Replace datetime.datetime date component values `<'datetime.datetime'>`.

    #### Default '-1' mean keep the original value.

    Equivalent to:
    >>> dt.replace(year, month, day)
    """

def dt_replace_time(
    dt: datetime.datetime,
    hour: int = -1,
    minute: int = -1,
    second: int = -1,
    millisecond: int = -1,
    microsecond: int = -1,
) -> datetime.datetime:
    """(cfunc) Replace datetime.datetime time component values `<'datetime.datetime'>`.

    #### Default '-1' mean keep the original value.

    Equivalent to:
    >>> dt.replace(hour, minute, second, microsecond)
    """

def dt_replace_tz(
    dt: datetime.datetime,
    tz: datetime.tzinfo | None,
) -> datetime.datetime:
    """(cfunc) Replace the datetime.datetime timezone `<'datetime.datetime'>`.

    Equivalent to:
    >>> dt.replace(tzinfo=tz)
    """

def dt_replace_fold(dt: datetime.datetime, fold: int) -> datetime.datetime:
    """(cfunc) Replace the datetime.datetime fold `<'datetime.datetime'>`.

    Equivalent to:
    >>> dt.replace(fold=fold)
    """

def dt_chg_weekday(dt: datetime.datetime, weekday: int) -> datetime.datetime:
    """(cfunc) Change datetime.datetime 'weekday' within the current week
    (0[Monday]...6[Sunday]) `<'datetime.date'>`.

    Equivalent to:
    >>> dt + datetime.timedelta(weekday - dt.weekday())
    """

def dt_astimezone(
    dt: datetime.datetime,
    tz: datetime.tzinfo | None = None,
) -> datetime.datetime:
    """(cfunc) Change the timezone for `<'datetime.datetime'>`.

    Equivalent to:
    >>> dt.astimezone(tz)
    """

# . arithmetic
def dt_add(
    dt: datetime.datetime,
    days: int = 0,
    seconds: int = 0,
    microseconds: int = 0,
    milliseconds: int = 0,
    minutes: int = 0,
    hours: int = 0,
    weeks: int = 0,
) -> datetime.datetime:
    """Add timedelta to datetime.datetime `<'datetime.datetime'>`.

    Equivalent to:
    >>> dt + datetime.timedelta(
            days, seconds, microseconds,
            milliseconds, minutes, hours, weeks
        )
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
    """(cfunc) Create a new `<'datetime.time'>`.

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

def time_utcformat(time: datetime.time) -> str | None:
    """(cfunc) Get the tzinfo of the time as UTC format '+/-HH:MM' `<'str/None'>`."""

# . conversion
def time_to_tm(time: datetime.time, utc: bool = False) -> dict:
    """(cfunc) Convert datetime.time to `<'struct:tm'>`.

    #### Date values sets to 1970-01-01.

    If 'time' is timezone-aware, setting 'utc=True', checks 'isdst'
    and substracts 'utcoffset' from the time before conversion.
    """

def time_to_strformat(time: datetime.time, fmt: str) -> str:
    """(cfunc) Convert datetime.time to str with the specified 'fmt' `<'str'>`.

    Equivalent to:
    >>> time.strftime(fmt)
    """

def time_to_isoformat(time: datetime.time, utc: bool = False) -> str:
    """(cfunc) Convert datetime.time to ISO format `<'str'>`.

    If 'time' is timezone-aware, setting 'utc=True'
    adds the UTC(Z) at the end of the ISO format.
    """

def time_to_seconds(time: datetime.time, utc: bool = False) -> float:
    """(cfunc) Convert datetime.time to total seconds `<'float'>`.

    If 'time' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from total seconds.
    """

def time_to_us(time: datetime.time, utc: bool = False) -> int:
    """(cfunc) Convert datetime.time to total microseconds `<'int'>`.

    If 'time' is timezone-aware, setting 'utc=True'
    substracts 'utcoffset' from total microseconds.
    """

def time_fr_dt(dt: datetime.datetime) -> datetime.time:
    """(cfunc) Convert datetime.datetime to `<'datetime.time'>`."""

def time_fr_time(time: datetime.time) -> datetime.time:
    """(cfunc) Convert subclass of datetime.time to `<'datetime.time'>`."""

def time_fr_seconds(seconds: float, tz: datetime.tzinfo | None = None) -> datetime.time:
    """(cfunc) Convert total seconds to `<'datetime.time'>`."""

def time_fr_us(us: int, tz: datetime.tzinfo | None = None) -> datetime.time:
    """(cfunc) Convert total microseconds to `<'datetime.time'>`."""

# . manipulation
def time_replace(
    time: datetime.time,
    hour: int = -1,
    minute: int = -1,
    second: int = -1,
    microsecond: int = -1,
    tz: object | datetime.tzinfo | None = -1,
    fold: int = -1,
) -> datetime.time:
    """(cfunc) Replace the datetime.time values `<'datetime.time'>`.

    #### Default '-1' mean keep the original value.

    Equivalent to:
    >>> time.replace(hour, minute, second, microsecond, tz, fold)
    """

def time_replace_tz(
    time: datetime.time,
    tz: datetime.tzinfo | None,
) -> datetime.time:
    """(cfunc) Replace the datetime.time timezone `<'datetime.time'>`.

    Equivalent to:
    >>> time.replace(tzinfo=tz)
    """

def time_replace_fold(time: datetime.time, fold: int) -> datetime.time:
    """(cfunc) Replace the datetime.time fold `<'datetime.time'>`.

    Equivalent to:
    >>> time.replace(fold=fold)
    """

# datetime.timedelta --------------------------------------------------------------------------------
# . generate
def td_new(
    days: int = 0,
    seconds: int = 0,
    microseconds: int = 0,
) -> datetime.timedelta:
    """(cfunc) Create a new `<'datetime.timedelta'>`.

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
    """(cfunc) Convert datetime.timedelta to ISO format `<'str'>`."""

def td_to_utcformat(td: datetime.timedelta) -> str:
    """(cfunc) Convert datetime.timedelta to UTC format '+/-HH:MM' `<'str'>`."""

def td_to_seconds(td: datetime.timedelta) -> float:
    """(cfunc) Convert datetime.timedelta to total seconds `<'float'>`."""

def td_to_us(td: datetime.timedelta) -> int:
    """(cfunc) Convert datetime.timedelta to total microseconds `<'int'>`."""

def td_fr_td(td: datetime.timedelta) -> datetime.timedelta:
    """(cfunc) Convert subclass of datetime.timedelta to `<'datetime.timedelta'>`."""

def td_fr_seconds(seconds: float) -> datetime.timedelta:
    """(cfunc) Convert total seconds to `<'datetime.timedelta'>`."""

def td_fr_us(us: int) -> datetime.timedelta:
    """(cfunc) Convert total microseconds to `<'datetime.timedelta'>`."""

# datetime.tzinfo -----------------------------------------------------------------------------------
# . generate
def tz_new(hours: int = 0, minutes: int = 0, seconds: int = 0) -> datetime.tzinfo:
    """(cfunc) Create a new `<'datetime.tzinfo'>`.

    Equivalent to:
    >>> datetime.timezone(datetime.timedelta(hours=hours, minutes=minites))
    """

def tz_local(dt: datetime.datetime | None = None) -> datetime.tzinfo:
    """(cfunc) Get the local `<'datetime.tzinfo'>`."""

def tz_local_seconds(dt: datetime.datetime | None = None) -> int:
    """Get the local timezone offset in total seconds `<'int'>`."""

def tz_parse(tz: datetime.tzinfo | str) -> datetime.tzinfo | None:
    """(cfunc) Parse 'tz' object into `<'datetime.tzinfo'>`.

    :param tz `<'datetime.tzinfo/str/None'>`: The timezone object.
        - If 'tz' is an instance of `<'datetime.tzinfo'>`, return 'tz' directly.
        - If 'tz' is a string, use Python 'Zoneinfo' to create the timezone object.
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
    """(cfunc) Get the 'tzname' of the tzinfo `<'str/None'>`.

    Equivalent to:
    >>> tz.tzname(dt)
    """

def tz_dst(
    tz: datetime.tzinfo | None,
    dt: datetime.datetime | None = None,
) -> datetime.timedelta | None:
    """(cfunc) Get the 'dst' of the tzinfo `<'datetime.timedelta/None'>`.

    Equivalent to:
    >>> tz.dst(dt)
    """

def tz_utcoffset(
    tz: datetime.tzinfo | None,
    dt: datetime.datetime | None = None,
) -> datetime.timedelta | None:
    """(cfunc) Get the 'utcoffset' of the tzinfo `<'datetime.timedelta/None'>`.

    Equivalent to:
    >>> tz.utcoffset(dt)
    """

def tz_utcoffset_seconds(
    tz: datetime.tzinfo | None,
    dt: datetime.datetime | None = None,
) -> int:
    """(cfunc) Get the 'utcoffset' of the tzinfo in total seconds `<'int'>`.

    #### Returns `-100_000` if utcoffset is None.

    Equivalent to:
    >>> tz.utcoffset(dt).total_seconds()
    """

def tz_utcformat(
    tz: datetime.tzinfo | None,
    dt: datetime.datetime | None = None,
) -> str | None:
    """(cfunc) Access datetime.tzinfo as UTC format '+/-HH:MM' `<'str/None'>`."""

# NumPy: share --------------------------------------------------------------------------------------
def map_nptime_unit_int2str(unit: int) -> str:
    """(cfunc) Map ndarray[datetime64/timedelta64] unit from integer
    to the corresponding string representation `<'str'>`."""

def map_nptime_unit_str2int(unit: str) -> int:
    """(cfunc) Map ndarray[datetime64/timedelta64] unit from string
    representation to the corresponding integer `<'int'>`."""

def parse_arr_nptime_unit(arr: np.ndarray) -> int:
    """(cfunc) Parse numpy datetime64/timedelta64 unit from the
    given 'arr', returns the unit in `<'int'>`."""

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
def dt64_to_tm(dt64: np.datetime64) -> dict:
    """(cfunc) Convert np.datetime64 to `<'struct:tm'>`."""

def dt64_to_strformat(dt64: np.datetime64, fmt: str, strict: bool = True) -> str:
    """(cfunc) Convert np.datetime64 to str with the specified 'fmt' `<'str'>`.

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

def dt64_to_isoformat(dt64: np.datetime64, sep: str = " ", strict: bool = True) -> str:
    """(cfunc) Convert np.datetime64 to ISO format `<'str'>`.

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

def dt64_to_int(
    dt64: np.datetime64,
    unit: Literal["ns", "us", "ms", "s", "m", "h", "D"],
) -> int:
    """(cfunc) Convert np.datetime64 to an integer since Unix Epoch
    based on the given 'unit' `<'int'>`.

    Supported units: 'ns', 'us', 'ms', 's', 'm', 'h', 'D'.

    If 'dt64' resolution is higher than the 'unit',
    returns integer discards the resolution above the time unit.
    """

def dt64_to_days(dt64: np.datetime64) -> int:
    """(cfunc) Convert np.datetime64 to total days since Unix Epoch `<'int'>`.

    If 'dt64' resolution is higher than 'D',
    returns integer discards the resolution above days.
    """

def dt64_to_hours(dt64: np.datetime64) -> int:
    """(cfunc) Convert np.datetime64 to total hours since Unix Epoch `<'int'>`.

    If 'dt64' resolution is higher than 'h',
    returns integer discards the resolution above hours.
    """

def dt64_to_minites(dt64: np.datetime64) -> int:
    """(cfunc) Convert np.datetime64 to total minutes since Unix Epoch `<'int'>`.

    If 'dt64' resolution is higher than 'm',
    returns integer discards the resolution above minutes.
    """

def dt64_to_seconds(dt64: np.datetime64) -> int:
    """(cfunc) Convert np.datetime64 to total seconds since Unix Epoch `<'int'>`.

    If 'dt64' resolution is higher than 's',
    returns integer discards the resolution above seconds.
    """

def dt64_to_ms(dt64: np.datetime64) -> int:
    """(cfunc) Convert np.datetime64 to total milliseconds since Unix Epoch `<'int'>`.

    If 'dt64' resolution is higher than 'ms',
    returns integer discards the resolution above milliseconds.
    """

def dt64_to_us(dt64: np.datetime64) -> int:
    """(cfunc) Convert np.datetime64 to total microseconds since Unix Epoch `<'int'>`.

    If 'dt64' resolution is higher than 'us',
    returns integer discards the resolution above microseconds.
    """

def dt64_to_ns(dt64: np.datetime64) -> int:
    """(cfunc) Convert np.datetime64 to total nanoseconds since Unix Epoch `<'int'>`.

    If 'dt64' resolution is higher than 'ns',
    returns integer discards the resolution above nanoseconds.
    """

def dt64_to_date(dt64: np.datetime64) -> datetime.date:
    """(cfunc) Convert np.datetime64 to `<'datetime.date'>`.

    If 'dt64' resolution is higher than 'D',
    returns datetime.date discards the resolution above days.
    """

def dt64_to_dt(
    dt64: np.datetime64,
    tz: datetime.tzinfo | None = None,
) -> datetime.datetime:
    """(cfunc) Convert np.datetime64 to `<'datetime.datetime'>`.

    If 'dt64' resolution is higher than 'us',
    returns datetime.datetime discards the resolution above microseconds.
    """

def dt64_to_time(dt64: np.datetime64) -> datetime.time:
    """(cfunc) Convert np.datetime64 to `<'datetime.time'>`.

    If 'dt64' resolution is higher than 'us',
    returns datetime.time discards the resolution above microseconds.
    """

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
def td64_to_int(
    td64: np.timedelta64,
    unit: Literal["ns", "us", "ms", "s", "m", "h", "D"],
) -> int:
    """(cfunc) Convert np.timedelta64 to an integer based on the given 'unit' `<'int'>`.

    Supported units: 'ns', 'us', 'ms', 's', 'm', 'h', 'D'.

    If 'td64' resolution is higher than the 'unit',
    returns integer rounds to the nearest time unit.
    """

def td64_to_days(td64: np.timedelta64) -> int:
    """(cfunc) Convert np.timedelta64 to total days `<'int'>`.

    If 'td64' resolution is higher than 'D',
    returns integer rounds to the nearest days.
    """

def td64_to_hours(td64: np.timedelta64) -> int:
    """(cfunc) Convert np.timedelta64 to total hours `<'int'>`.

    If 'td64' resolution is higher than 'h',
    returns integer rounds to the nearest hours.
    """

def td64_to_minites(td64: np.timedelta64) -> int:
    """(cfunc) Convert np.timedelta64 to total minutes `<'int'>`.

    If 'td64' resolution is higher than 'm',
    returns integer rounds to the nearest minutes.
    """

def td64_to_seconds(td64: np.timedelta64) -> int:
    """(cfunc) Convert np.timedelta64 to total seconds `<'int'>`.

    If 'td64' resolution is higher than 's',
    returns integer rounds to the nearest seconds.
    """

def td64_to_ms(td64: np.timedelta64) -> int:
    """(cfunc) Convert np.timedelta64 to total milliseconds `<'int'>`.

    If 'td64' resolution is higher than 'ms',
    returns integer rounds to the nearest milliseconds.
    """

def td64_to_us(td64: np.timedelta64) -> int:
    """(cfunc) Convert np.timedelta64 to total microseconds `<'int'>`.

    If 'td64' resolution is higher than 'us',
    returns integer rounds to the nearest microseconds.
    """

def td64_to_ns(td64: np.timedelta64) -> int:
    """(cfunc) Convert np.timedelta64 to total nanoseconds `<'int'>`.

    If 'td64' resolution is higher than 'ns',
    returns integer rounds to the nearest nanoseconds.
    """

def td64_to_td(td64: np.timedelta64) -> datetime.timedelta:
    """(cfunc) Convert np.timedelta64 to `<'datetime.timedelta'>`.

    If 'td64' resolution is higher than 'us',
    returns datetime.timedelta rounds to the nearest microseconds.
    """
