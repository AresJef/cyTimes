# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cpython.time import time as unix_time, localtime  # type: ignore
from cython.cimports.cpython.set import PySet_Contains as set_contains  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_READ_CHAR as read_char  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_GET_LENGTH as str_len  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_Substring as str_substr  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_FromOrdinal as str_fr_ucs4  # type: ignore
from cython.cimports.cytimes import typeref  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
from typing import Literal
import datetime, numpy as np
from time import localtime as time_localtime
from pandas import Series
from cytimes import typeref

# Constants --------------------------------------------------------------------------------------------
# . calendar
# fmt: off
DAYS_BR_MONTH: cython.uint[13] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
DAYS_IN_MONTH: cython.uint[13] = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
DAYS_BR_QUARTER: cython.uint[5] = [0, 90, 181, 273, 365]
DAYS_BR_QUARTER_NDARRAY: np.ndarray = np.array([0, 90, 181, 273, 365])
DAYS_IN_QUARTER: cython.uint[5] = [0, 90, 91, 92, 92]
MONTH_TO_QUARTER: cython.uint[13] = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
# fmt: on
# . datetime
UTC: datetime.tzinfo = datetime.get_utc()
EPOCH_UTC: datetime.datetime = datetime.datetime_new(1970, 1, 1, 0, 0, 0, 0, UTC, 0)  # type: ignore
EPOCH_US: cython.longlong = 62_135_683_200_000_000
EPOCH_SEC: cython.longlong = 62_135_683_200
EPOCH_DAY: cython.int = 719_163
DT_MAX_US: cython.longlong = 315_537_983_999_999_999
DT_MIN_US: cython.longlong = 86_400_000_000
# . time
TIME_START: datetime.time = datetime.time(0, 0, 0, 0)
TIME_END: datetime.time = datetime.time(23, 59, 59, 999999)
# . microsecond
US_DAY: cython.longlong = 86_400_000_000
US_HOUR: cython.longlong = 3_600_000_000
# . nanosecond
NS_DAY: cython.longlong = 864_00_000_000_000
NS_HOUR: cython.longlong = 36_00_000_000_000
NS_MINUTE: cython.longlong = 60_000_000_000
# . timedelta unit adjustment
TD_UNIT_ADJUSTMENT: set[str] = {"s", "ms", "us", "ns"}
# . function
FN_TIME_LOCALTIME: object = time_localtime


# Calendar =============================================================================================
# Calender: year ---------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_leapyear(year: cython.uint) -> cython.bint:
    """Determine whether the given 'year' is a leap year `<'bool'>`."""
    if year == 0:
        return False
    else:
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def leap_bt_years(year1: cython.uint, year2: cython.uint) -> cython.uint:
    """Calculate the number of leap years between 'year1' and 'year2' `<'int'>`."""
    if year1 <= year2:
        y1: cython.Py_ssize_t = year1 - 1
        y2: cython.Py_ssize_t = year2 - 1
    else:
        y1: cython.Py_ssize_t = year2 - 1
        y2: cython.Py_ssize_t = year1 - 1
    return (y2 // 4 - y1 // 4) - (y2 // 100 - y1 // 100) + (y2 // 400 - y1 // 400)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_in_year(year: cython.uint) -> cython.uint:
    """Get the maximum number of days in the 'year', expects 365 or 366 `<'int'>`."""
    return 366 if is_leapyear(year) else 365


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_bf_year(year: cython.uint) -> cython.uint:
    """Calculate the number of days between the 1st day
    of the given 'year' and the 1st day of 1AD `<'int'>`."""
    if year <= 1:
        return 0
    y: cython.Py_ssize_t = year - 1
    return y * 365 + y // 4 - y // 100 + y // 400


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_of_year(
    year: cython.uint,
    month: cython.uint,
    day: cython.uint,
) -> cython.uint:
    """Calculate the number of days between the given 'year'
    & 'month' & 'day' and the 1st day of the 'year' `<'int'>`."""
    return days_bf_month(year, month) + min(day, days_in_month(year, month))


# Calendar: quarter ------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def quarter_of_month(month: cython.uint) -> cython.uint:
    """Get the quarter of the 'month', expects 1-4 `<'int'>`."""
    return MONTH_TO_QUARTER[month] if month <= 12 else 4


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_in_quarter(year: cython.uint, month: cython.uint) -> cython.uint:
    """Calculate the maximum number of days in the quarter `<'int'>`."""
    quarter: cython.uint = quarter_of_month(month)
    days: cython.uint = DAYS_IN_QUARTER[quarter]
    if quarter == 1 and is_leapyear(year):
        days += 1
    return days


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_bf_quarter(year: cython.uint, month: cython.uint) -> cython.uint:
    """Calculate the number of days between the 1st day
    of the quarter and the 1st day of the year `<'int'>."""
    quarter: cython.uint = quarter_of_month(month)
    days: cython.uint = DAYS_BR_QUARTER[quarter - 1]
    if quarter >= 2 and is_leapyear(year):
        days += 1
    return days


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_of_quarter(
    year: cython.uint,
    month: cython.uint,
    day: cython.uint,
) -> cython.uint:
    """Calculate the number of days between the given 'year'
    & 'month' & 'day' and the 1st day of the quarter `<'int'>`."""
    return days_of_year(year, month, day) - days_bf_quarter(year, month)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def quarter_1st_month(month: cython.uint) -> cython.uint:
    """Get the first month of the quarter, expects 1, 4, 7, 10 `<'int'>`."""
    return 3 * quarter_of_month(month) - 2


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def quarter_lst_month(month: cython.uint) -> cython.uint:
    """Get the last month of the quarter, expects 3, 6, 9, 12 `<'int'>`."""
    return 3 * quarter_of_month(month)


# Calendar: month --------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_in_month(year: cython.uint, month: cython.uint) -> cython.uint:
    """Get the maximum number of days in the 'month',
    expects 28, 29, 30, 31 `<'int'>`."""
    # Invalid month => 31 days
    if not 1 <= month <= 12:
        return 31
    # Calculate days
    days: cython.uint = DAYS_IN_MONTH[month]
    if month == 2 and is_leapyear(year):
        days += 1
    return days


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_bf_month(year: cython.uint, month: cython.uint) -> cython.uint:
    """Calculate the number of days between the 1st day
    of the given 'year' and the 1st day of the 'month' `<'int'>`."""
    if month <= 2:
        return 31 if month == 2 else 0
    days: cython.uint = DAYS_BR_MONTH[min(month, 12) - 1]
    if is_leapyear(year):
        days += 1
    return days


# Calendar: week ---------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def ymd_weekday(
    year: cython.uint,
    month: cython.uint,
    day: cython.uint,
) -> cython.uint:
    """Get the day of the week, where 0=Monday...6=Sunday `<'int'>`."""
    return (ymd_to_ordinal(year, month, day) + 6) % 7


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def ymd_isoweekday(
    year: cython.uint,
    month: cython.uint,
    day: cython.uint,
) -> cython.uint:
    """Get the ISO calendar day of the week, where 1=Monday...7=Sunday `<'int'>`."""
    return ymd_weekday(year, month, day) + 1


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def ymd_isoweek(year: cython.uint, month: cython.uint, day: cython.uint) -> cython.uint:
    """Get the ISO calendar week number `<'int'>`."""
    ordinal: cython.uint = ymd_to_ordinal(year, month, day)
    iso1st_ord: cython.uint = iso1st_ordinal(year)
    delta: cython.Py_ssize_t = ordinal - iso1st_ord
    isoweek: cython.Py_ssize_t = delta // 7
    if isoweek < 0:
        iso1st_ord = iso1st_ordinal(year - 1)
        return (ordinal - iso1st_ord) // 7 + 1
    elif isoweek >= 52 and ordinal >= iso1st_ordinal(year + 1):
        return 1
    else:
        return isoweek + 1


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def ymd_isoyear(year: cython.uint, month: cython.uint, day: cython.uint) -> cython.uint:
    """Get the ISO calendar year `<'int'>`."""
    ordinal: cython.uint = ymd_to_ordinal(year, month, day)
    iso1st_ord: cython.uint = iso1st_ordinal(year)
    delta: cython.Py_ssize_t = ordinal - iso1st_ord
    isoweek: cython.Py_ssize_t = delta // 7
    if isoweek < 0:
        return year - 1
    elif isoweek >= 52 and ordinal >= iso1st_ordinal(year + 1):
        return year + 1
    else:
        return year


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def ymd_isocalendar(
    year: cython.uint,
    month: cython.uint,
    day: cython.uint,
) -> iso:  # type: ignore
    """Get the ISO calendar of the YMD `<'struct:iso'>`."""
    ordinal: cython.uint = ymd_to_ordinal(year, month, day)
    iso1st_ord: cython.uint = iso1st_ordinal(year)
    delta: cython.Py_ssize_t = ordinal - iso1st_ord
    isoweek: cython.Py_ssize_t = delta // 7
    if isoweek < 0:
        year -= 1
        iso1st_ord = iso1st_ordinal(year)
        delta = ordinal - iso1st_ord
        isoweek = delta // 7
    elif isoweek >= 52 and ordinal >= iso1st_ordinal(year + 1):
        year += 1
        isoweek = 0
    return iso(year, isoweek + 1, delta % 7 + 1)  # type: ignore


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def iso1st_ordinal(year: cython.uint) -> cython.uint:
    """Get the ordinal for the 1st day of ISO calendar year `<'int'>`."""
    day_1st: cython.uint = ymd_to_ordinal(year, 1, 1)
    weekday_1st: cython.uint = (day_1st + 6) % 7
    weekmon_1st: cython.uint = day_1st - weekday_1st
    return weekmon_1st + 7 if weekday_1st > 3 else weekmon_1st


# Calendar: conversion ---------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def ymd_to_ordinal(
    year: cython.uint,
    month: cython.uint,
    day: cython.uint,
) -> cython.uint:
    """Convert 'year' & 'month' & 'day' to ordinal `<'int'>`."""
    return (
        days_bf_year(year)
        + days_bf_month(year, month)
        + min(day, days_in_month(year, month))
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def ordinal_to_ymd(ordinal: cython.int) -> ymd:  # type: ignore
    """Convert ordinal to YMD `<'struct:ymd'>`."""
    # n is a 1-based index, starting at 1-Jan-1.  The pattern of leap years
    # repeats exactly every 400 years.  The basic strategy is to find the
    # closest 400-year boundary at or before n, then work with the offset
    # from that boundary to n.  Life is much clearer if we subtract 1 from
    # n first -- then the values of n at 400-year boundaries are exactly
    # those divisible by _DI400Y:
    n: cython.uint = min(max(ordinal, 1), 3_652_059) - 1
    n400: cython.uint = n // 146_097
    n = n % 146_097
    year: cython.uint = n400 * 400 + 1

    # Now n is the (non-negative) offset, in days, from January 1 of year, to
    # the desired date.  Now compute how many 100-year cycles precede n.
    # Note that it's possible for n100 to equal 4!  In that case 4 full
    # 100-year cycles precede the desired day, which implies the desired
    # day is December 31 at the end of a 400-year cycle.
    n100: cython.uint = n // 36_524
    n = n % 36_524

    # Now compute how many 4-year cycles precede it.
    n4: cython.uint = n // 1_461
    n = n % 1_461

    # And now how many single years.  Again n1 can be 4, and again meaning
    # that the desired day is December 31 at the end of the 4-year cycle.
    n1: cython.uint = n // 365
    n = n % 365

    # We now know the year and the offset from January 1st.  Leap years are
    # tricky, because they can be century years.  The basic rule is that a
    # leap year is a year divisible by 4, unless it's a century year --
    # unless it's divisible by 400.  So the first thing to determine is
    # whether year is divisible by 4.  If not, then we're done -- the answer
    # is December 31 at the end of the year.
    year += n100 * 100 + n4 * 4 + n1
    if n1 == 4 or n100 == 4:
        # Return ymd
        return ymd(year - 1, 12, 31)  # type: ignore

    # Now the year is correct, and n is the offset from January 1.  We find
    # the month via an estimate that's either exact or one too large.
    month: cython.uint = (n + 50) >> 5
    days_bf: cython.uint = days_bf_month(year, month)
    if days_bf > n:
        month -= 1
        days_bf = days_bf_month(year, month)
    n = n - days_bf + 1

    # Return ymd
    return ymd(year, month, n)  # type: ignore


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def isocalendar_to_ymd(
    year: cython.uint,
    week: cython.uint,
    weekday: cython.uint,
) -> ymd:  # type: ignore
    """Convert isocalendar (year, week, weekday) to YMD `<'struct:ymd'>`."""
    # Clip year
    year = min(max(year, 1), 9_999)

    # Adjust the 53rd week
    if week == 53:
        # ISO years have 53 weeks in them on years starting with a
        # Thursday and leap years starting on a Wednesday
        wkd_1st: cython.uint = ymd_to_ordinal(year, 1, 1) % 7
        if wkd_1st == 4 or (wkd_1st == 3 and is_leapyear(year)):
            if year < 9_999:
                week = 1
                year += 1
            else:
                week = 52
    # Clip week
    else:
        week = min(max(week, 1), 52)

    # Clip day
    weekday = min(max(weekday, 1), 7)

    # Calculate ordinal
    iso1st_ord: cython.uint = iso1st_ordinal(year)
    offset: cython.uint = (week - 1) * 7 + weekday - 1
    ordinal: cython.uint = iso1st_ord + offset

    # Convert to ymd
    return ordinal_to_ymd(ordinal)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def days_of_year_to_ymd(year: cython.uint, days: cython.uint) -> ymd:  # type: ignore
    """Convert days of the year to YMD `<'struct:ymd'>`."""
    # Clip year & days
    year = min(max(year, 1), 9_999)
    days = min(max(days, 1), days_in_year(year))

    # Must be January
    if days <= 31:
        return ymd(year, 1, days)  # type: ignore

    # Must be February
    leap: cython.uint = is_leapyear(year)
    if days <= 59 + leap:
        return ymd(year, 2, days - 31)  # type: ignore

    # Find month & calculate day
    days -= leap
    month: cython.uint = 3
    for _ in range(10):
        days_bf_month: cython.uint = DAYS_BR_MONTH[month]
        if days <= days_bf_month:
            day: cython.uint = days - DAYS_BR_MONTH[month - 1]
            return ymd(year, month, day)  # type: ignore
        month += 1
    return ymd(year, 12, 31)  # type: ignore


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
@cython.exceptval(check=False)
def microseconds_to_hms(microseconds: cython.longlong) -> hms:  # type: ignore
    """Convert microseconds to HMS `<'struct:hms'>`."""
    if microseconds <= 0:
        return hms(0, 0, 0, 0)  # type: ignore

    microseconds = microseconds % US_DAY
    hour: cython.uint = microseconds // US_HOUR
    microseconds = microseconds % US_HOUR
    minute: cython.uint = microseconds // 60_000_000
    microseconds = microseconds % 60_000_000
    second: cython.uint = microseconds // 1_000_000
    microsecond: cython.uint = microseconds % 1_000_000
    return hms(hour, minute, second, microsecond)  # type: ignore


# Datetime.date ========================================================================================
# Date: generate ---------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def gen_date(
    year: cython.uint = 1,
    month: cython.uint = 1,
    day: cython.uint = 1,
) -> datetime.date:
    """Generate a new `<'datetime.date'>`."""
    return datetime.date_new(year, month, day)


@cython.cfunc
@cython.inline(True)
def gen_date_now() -> datetime.date:
    """Generate the current local `<'datetime.date'>`.
    Equivalent to `datetime.date.today()`."""
    tms = localtime()
    return datetime.date_new(tms.tm_year, tms.tm_mon, tms.tm_mday)


@cython.cfunc
@cython.inline(True)
def gen_date_now_utc() -> datetime.date:
    """Generate the current `<'datetime.date'>` under UTC.
    Equivalent to `datetime.datetime.now(UTC).date()`."""
    return date_fr_dt(gen_dt_now_utc())


@cython.cfunc
@cython.inline(True)
def gen_date_now_tz(tzinfo: datetime.tzinfo) -> datetime.date:
    """Generate the current `<'datetime.date'>` under specific timezone.
    Equivalent to `datetime.datetime.now(tzinfo).date()`."""
    return date_fr_dt(gen_dt_now_tz(tzinfo))


# Datetime.date: check types ---------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_date(obj: object) -> cython.bint:
    """Check if an object is type of datetime.date `<'bool'>`.
    Equivalent to `isinstance(obj, datetime.date)`."""
    return datetime.PyDate_Check(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_date_exact(obj: object) -> cython.bint:
    """Check if an object is the exact type of datetime.date `<'bool'>`.
    Equivalent to `type(obj) is datetime.date`."""
    return datetime.PyDate_CheckExact(obj)


# Datetime.date: calendar - year -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_is_leapyear(date: datetime.date) -> cython.bint:
    """Whether the 'date' is a leap year `<'bool'>`."""
    return is_leapyear(date.year)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_leap_bt_years(date1: datetime.date, date2: datetime.date) -> cython.uint:
    """Calculate the number of leap years between two 'dates' `<'int'>`."""
    return leap_bt_years(date1.year, date2.year)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_days_in_year(date: datetime.date) -> cython.uint:
    """Get the maximum number of days in the 'year', expects 365 or 366 `<'int'>`."""
    return days_in_year(date.year)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_days_bf_year(date: datetime.date) -> cython.uint:
    """Calculate the number of days between the given 'date'
    and the 1st day of 1AD `<'int'>`."""
    return days_bf_year(date.year)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_days_of_year(date: datetime.date) -> cython.uint:
    """Calculate the number of days between the given 'date'
    and the 1st day of the 'year' `<'int'>`."""
    return days_of_year(date.year, date.month, date.day)


# Datetime.date: calendar - quarter --------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_quarter(date: datetime.date) -> cython.uint:
    """Get the quarter of 'date' `<'int'>`"""
    return quarter_of_month(date.month)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_days_in_quarter(date: datetime.date) -> cython.uint:
    """Calculate the maximum number of days in the
    quarter of the 'date' `<'int'>`."""
    return days_in_quarter(date.year, date.month)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_days_bf_quarter(date: datetime.date) -> cython.uint:
    """Calculate the number of days between the 1st day
    of the quarter and the 1st day of the year `<'int'>."""
    return days_bf_quarter(date.year, date.month)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_days_of_quarter(date: datetime.date) -> cython.uint:
    """Calculate the number of days between the given 'date'
    and the 1st day of the quarter `<'int'>`."""
    return days_of_quarter(date.year, date.month, date.day)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_quarter_1st_month(date: datetime.date) -> cython.uint:
    """Get the first month of the quarter, expects 1, 4, 7, 10 `<'int'>`."""
    return quarter_1st_month(date.month)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_quarter_lst_month(date: datetime.date) -> cython.uint:
    """Get the last month of the quarter, expects 3, 6, 9, 12 `<'int'>`."""
    return quarter_lst_month(date.month)


# Datetime.date: calendar - month ----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_days_in_month(date: datetime.date) -> cython.uint:
    """Get the maximum number of days in the month of the 'date',
    expects 28, 29, 30, 31 `<'int'>`."""
    return days_in_month(date.year, date.month)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_days_bf_month(date: datetime.date) -> cython.uint:
    """Get the number of days between the 1st day of the year
    and the 1st day of the month of the 'date' `<'int'>`."""
    return days_bf_month(date.year, date.month)


# Datetime.date: calendar - week -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_weekday(date: datetime.date) -> cython.uint:
    """Get the day of the week, where 0=Monday...6=Sunday `<'int'>`."""
    return ymd_weekday(date.year, date.month, date.day)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_isoweekday(date: datetime.date) -> cython.uint:
    """Get the ISO calendar day of the week, where 1=Monday...7=Sunday `<'int'>`."""
    return ymd_isoweekday(date.year, date.month, date.day)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_isoweek(date: datetime.date) -> cython.uint:
    """Get the ISO calendar week number `<'int'>`."""
    return ymd_isoweek(date.year, date.month, date.day)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_isoyear(date: datetime.date) -> cython.uint:
    """Get the ISO calendar year `<'int'>`."""
    return ymd_isoyear(date.year, date.month, date.day)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_isocalendar(date: datetime.date) -> iso:  # type: ignore
    """Get the ISO calendar of the 'date' `<'struct:iso'>`."""
    return ymd_isocalendar(date.year, date.month, date.day)


# Datetime.date: conversion ----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def date_to_isoformat(date: datetime.date) -> str:
    """Convert 'date' to ISO format: '%Y-%m-%d' `<'str'>`."""
    return "%04d-%02d-%02d" % (date.year, date.month, date.day)


@cython.cfunc
@cython.inline(True)
def date_to_strformat(date: datetime.date, format: str) -> str:
    """Convert date by string format `<'str'>`.
    Equivalent to `date.strftime(format)`."""
    # fmt: off
    return dt_to_strformat(datetime.datetime_new(
        date.year, date.month, date.day, 0, 0, 0, 0, None, 0), format
    )
    # fmt: on


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_to_ordinal(date: datetime.date) -> cython.uint:
    """Convert 'date' to ordinal `<'int'>`."""
    return ymd_to_ordinal(date.year, date.month, date.day)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_to_seconds(date: datetime.date) -> cython.longlong:
    """Convert 'date' to total seconds after POSIX epoch `<'int'>`."""
    ordinal: cython.longlong = date_to_ordinal(date)
    return (ordinal - EPOCH_DAY) * 86_400


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_to_microseconds(date: datetime.date) -> cython.longlong:
    """Convert 'date' to total microseconds after POSIX epoch `<'int'>`."""
    ordinal: cython.longlong = date_to_ordinal(date)
    return (ordinal - EPOCH_DAY) * US_DAY


@cython.cfunc
@cython.inline(True)
def date_to_timestamp(date: datetime.date) -> cython.longlong:
    """Convert 'date' to timestamp `<'int'>`."""
    base_ts: cython.longlong = date_to_seconds(date)
    offset: cython.longlong = localize_ts(base_ts) - base_ts  # type: ignore
    return base_ts - offset


@cython.cfunc
@cython.inline(True)
def date_fr_dt(dt: datetime.datetime) -> datetime.date:
    """Convert `<'datetime.datetime'>` to `<'datetime.date'>`."""
    return datetime.date_new(dt.year, dt.month, dt.day)


@cython.cfunc
@cython.inline(True)
def date_fr_date(date: datetime.date) -> datetime.date:
    """Convert subclass of `<'datetime.date'>` to `<'datetime.date'>`."""
    return datetime.date_new(date.year, date.month, date.day)


@cython.cfunc
@cython.inline(True)
def date_fr_ordinal(ordinal: cython.int) -> datetime.date:
    """Convert ordinal to `<'datetime.date'>`."""
    ymd = ordinal_to_ymd(ordinal)
    return datetime.date_new(ymd.year, ymd.month, ymd.day)


@cython.cfunc
@cython.inline(True)
def date_fr_seconds(seconds: cython.double) -> datetime.date:
    """Convert total seconds after POSIX epoch to `<'datetime.date'>`."""
    total_sec: cython.longlong = int(seconds)
    return date_fr_ordinal(total_sec // 86_400 + EPOCH_DAY)


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
def date_fr_microseconds(microseconds: cython.longlong) -> datetime.date:
    """Convert total microseconds after POSIX epoch to `<'datetime.date'>`."""
    return date_fr_ordinal(microseconds // US_DAY + EPOCH_DAY)


@cython.cfunc
@cython.inline(True)
def date_fr_timestamp(timestamp: cython.double) -> datetime.date:
    """Convert timestamp to `<'datetime.date'>`."""
    return datetime.date_from_timestamp(timestamp)


# Datetime.date: arithmetic ----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def date_add(
    date: datetime.date,
    days: cython.int = 0,
    seconds: cython.longlong = 0,
    microseconds: cython.longlong = 0,
) -> datetime.date:
    """Add 'days', 'seconds' and 'microseconds' to `<'datetime.date'>`.
    Equivalent to `date + timedelta(days, seconds, microseconds)`."""
    if days == 0 and seconds == 0 and microseconds == 0:
        return date
    return date_fr_microseconds(
        date_to_microseconds(date) + days * US_DAY + seconds * 1_000_000 + microseconds
    )


@cython.cfunc
@cython.inline(True)
def date_add_td(date: datetime.date, td: datetime.timedelta) -> datetime.date:
    """Add `<'datetime.timedelta'>` to `<'datetime.date'>`.
    Equivalent to `date + timedelta`."""
    return date_fr_ordinal(date_to_ordinal(date) + td.day)


@cython.cfunc
@cython.inline(True)
def date_sub_td(date: datetime.date, td: datetime.timedelta) -> datetime.date:
    """Substract `<'datetime.timedelta'>` from `<'datetime.date'>`.
    Equivalent to `date - timedelta`."""
    return date_fr_ordinal(date_to_ordinal(date) - td.day)


@cython.cfunc
@cython.inline(True)
def date_sub_date(date_l: datetime.date, date_r: datetime.date) -> datetime.timedelta:
    """Substruction between datetime.dates and returns `<'datetime.timedelta'>`.
    Equivalent to `date - date`."""
    return datetime.timedelta_new(date_sub_date_days(date_l, date_r), 0, 0)


@cython.cfunc
@cython.inline(True)
def date_sub_date_days(date_l: datetime.date, date_r: datetime.date) -> cython.int:
    """Substruction between datetime.dates and returns `<'int'>`
    (differences in days). Equivalent to `date - date`."""
    return date_to_ordinal(date_l) - date_to_ordinal(date_r)


# Datetime.date: manipulation --------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def date_replace(
    date: datetime.date,
    year: cython.int = -1,
    month: cython.int = -1,
    day: cython.int = -1,
) -> datetime.date:
    """Replace `<'datetime.date'>`. Equivalent to `date.replace()`.
    Value of `-1` indicates preserving the original value."""
    if not 1 <= year <= 9999:
        year = date.year
    if not 1 <= month <= 12:
        month = date.month
    return datetime.date_new(
        year, month, min(day if day > 0 else date.day, days_in_month(year, month))
    )


@cython.cfunc
@cython.inline(True)
def date_adj_weekday(date: datetime.date, weekday: cython.uint) -> datetime.date:
    """Adjust `<'datetime.date'>` to the 'weekday' of
    the current week, where Monday is 0 and Sunday is 6.
    Equivalent to: `date + timedelta(weekday - date.weekday())`."""
    weekday = weekday % 7
    wday: cython.uint = date_weekday(date)
    return date if weekday == wday else date_add(date, days=weekday - wday)


# Datetime.datetime ====================================================================================
# Datetime.datetime: generate --------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def gen_dt(
    year: cython.uint = 1,
    month: cython.uint = 1,
    day: cython.uint = 1,
    hour: cython.uint = 0,
    minute: cython.uint = 0,
    second: cython.uint = 0,
    microsecond: cython.uint = 0,
    tzinfo: datetime.tzinfo = None,
    fold: cython.uint = 0,
) -> datetime.datetime:
    """Generate a new `<'datetime.datetime'>`."""
    # fmt: off
    return datetime.datetime_new(
        year, month, day, hour, minute, second, microsecond, 
        tzinfo, 1 if fold == 1 and tzinfo is not None else 0
    )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def gen_dt_now() -> datetime.datetime:
    """Generate the current local `<'datetime.datetime'>`.
    Equivalent to `datetime.datetime.now()`."""
    microseconds: cython.int = int(unix_time() % 1 * 1_000_000)
    tms = localtime()
    # fmt: off
    return datetime.datetime_new(
        tms.tm_year, tms.tm_mon, tms.tm_mday, 
        tms.tm_hour, tms.tm_min, tms.tm_sec, 
        microseconds, None, 0,
    )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def gen_dt_now_utc() -> datetime.datetime:
    """Generate the current `<'datetime.datetime'>` under UTC.
    Equivalent to `datetime.datetime.now(UTC)`."""
    return dt_fr_timestamp(unix_time(), UTC)


@cython.cfunc
@cython.inline(True)
def gen_dt_now_tz(tzinfo: datetime.tzinfo) -> datetime.datetime:
    """Generate the current `<'datetime.datetime'>` under specific timezone.
    Equivalent to `datetime.datetime.now(tzinfo)`."""
    return dt_fr_timestamp(unix_time(), tzinfo)


# Datetime.datetime: check types -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_dt(obj: object) -> cython.bint:
    """Check if an object is type of datetime.datetime `<'bool'>`.
    Equivalent to `isinstance(obj, datetime.datetime)`"""
    return datetime.PyDateTime_Check(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_dt_exact(obj: object) -> cython.bint:
    """Check if an object is the exact type of datetime.datetime `<'bool'>`.
    Equivalent to `type(obj) is datetime.datetime`."""
    return datetime.PyDateTime_CheckExact(obj)


# Datetime.datetime: tzinfo ----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def dt_tzname(dt: datetime.datetime) -> str:
    """Access the 'tzname' `<'str'>`.
    Equivalent to `dt.tzname()`."""
    return tzinfo_name(dt.tzinfo, dt)


@cython.cfunc
@cython.inline(True)
def dt_dst(dt: datetime.datetime) -> datetime.timedelta:
    """Access the 'dst' `<'datetime.datetime'>`.
    Equivalent to `dt.dst()`."""
    return tzinfo_dst(dt.tzinfo, dt)


@cython.cfunc
@cython.inline(True)
def dt_utcoffset(dt: datetime.datetime) -> datetime.timedelta:
    """Access the 'utcoffset' `<'datetime.timedelta'>`.
    Equivalent to `dt.utcoffset()`."""
    return tzinfo_utcoffset(dt.tzinfo, dt)


# Datetime.datetime: calendar - year -------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def dt_is_leapyear(dt: datetime.datetime) -> cython.bint:
    """Whether the 'dt' is a leap year `<'bool'>`."""
    return is_leapyear(dt.year)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def dt_leap_bt_years(dt1: datetime.datetime, dt2: datetime.datetime) -> cython.uint:
    """Calculate the number of leap years between two 'datetimes' `<'int'>`."""
    return leap_bt_years(dt1.year, dt2.year)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def dt_days_in_year(dt: datetime.datetime) -> cython.uint:
    """Get the maximum number of days in the 'year', expects 365 or 366 `<'int'>`."""
    return days_in_year(dt.year)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def dt_days_bf_year(dt: datetime.datetime) -> cython.uint:
    """Calculate the number of days between the given 'dt'
    and the 1st day of 1AD `<'int'>`."""
    return days_bf_year(dt.year)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def dt_days_of_year(dt: datetime.datetime) -> cython.uint:
    """Calculate the number of days between the given 'dt'
    and the 1st day of the 'year' `<'int'>`."""
    return days_of_year(dt.year, dt.month, dt.day)


# Datetime.datetime: calendar - quarter ----------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def dt_quarter(dt: datetime.datetime) -> cython.uint:
    """Get the quarter of 'dt' `<'int'>`"""
    return quarter_of_month(dt.month)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def dt_days_in_quarter(dt: datetime.datetime) -> cython.uint:
    """Calculate the maximum number of days in the
    quarter of the 'dt' `<'int'>`."""
    return days_in_quarter(dt.year, dt.month)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def dt_days_bf_quarter(dt: datetime.datetime) -> cython.uint:
    """Calculate the number of days between the 1st day
    of the quarter and the 1st day of the year `<'int'>."""
    return days_bf_quarter(dt.year, dt.month)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def dt_days_of_quarter(dt: datetime.datetime) -> cython.uint:
    """Calculate the number of days between the given 'dt'
    and the 1st day of the quarter `<'int'>`."""
    return days_of_quarter(dt.year, dt.month, dt.day)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def dt_quarter_1st_month(dt: datetime.datetime) -> cython.uint:
    """Get the first month of the quarter, expects 1, 4, 7, 10 `<'int'>`."""
    return quarter_1st_month(dt.month)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def dt_quarter_lst_month(dt: datetime.datetime) -> cython.uint:
    """Get the last month of the quarter, expects 3, 6, 9, 12 `<'int'>`."""
    return quarter_lst_month(dt.month)


# Datetime.datetime: calendar - month ------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def dt_days_in_month(dt: datetime.datetime) -> cython.uint:
    """Get the maximum number of days in the month of the 'dt',
    expects 28, 29, 30, 31 `<'int'>`."""
    return days_in_month(dt.year, dt.month)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def dt_days_bf_month(dt: datetime.datetime) -> cython.uint:
    """Get the number of days between the 1st day of the year
    and the 1st day of the month of the 'dt' `<'int'>`."""
    return days_bf_month(dt.year, dt.month)


# Datetime.datetime: calendar - week -------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def dt_weekday(dt: datetime.datetime) -> cython.uint:
    """Get the day of the week, where 0=Monday...6=Sunday `<'int'>`."""
    return ymd_weekday(dt.year, dt.month, dt.day)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def dt_isoweekday(dt: datetime.datetime) -> cython.uint:
    """Get the ISO calendar day of the week, where 1=Monday...7=Sunday `<'int'>`."""
    return ymd_isoweekday(dt.year, dt.month, dt.day)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def dt_isoweek(dt: datetime.datetime) -> cython.uint:
    """Get the ISO calendar week number `<'int'>`."""
    return ymd_isoweek(dt.year, dt.month, dt.day)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def dt_isoyear(dt: datetime.datetime) -> cython.uint:
    """Get the ISO calendar year `<'int'>`."""
    return ymd_isoyear(dt.year, dt.month, dt.day)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_isocalendar(dt: datetime.datetime) -> iso:  # type: ignore
    """Get the ISO calendar of the 'dt' `<'struct:iso'>`."""
    return ymd_isocalendar(dt.year, dt.month, dt.day)


# Datetime.datetime: conversion ------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def dt_to_isoformat(dt: datetime.datetime) -> str:
    """Convert datetime to ISO format: '%Y-%m-%dT%H:%M:%S.f' `<'str'>`."""
    microsecond: cython.int = dt.microsecond
    # fmt: off
    if microsecond == 0:
        return "%04d-%02d-%02dT%02d:%02d:%02d" % (
            dt.year, dt.month, dt.day, 
            dt.hour, dt.minute, dt.second,
        )
    else:
        return "%04d-%02d-%02dT%02d:%02d:%02d.%06d" % (
            dt.year, dt.month, dt.day, 
            dt.hour, dt.minute, dt.second, microsecond,
        )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def dt_to_isoformat_tz(dt: datetime.datetime) -> str:
    """Convert datetime to ISO format with tzinfo:
    '%Y-%m-%dT%H:%M:%S.f.Z' `<'str'>`."""
    fmt: str = dt_to_isoformat(dt)
    offset = dt_utcoffset(dt)
    if offset is not None:
        fmt += td_to_utcformat(offset)
    return fmt


@cython.cfunc
@cython.inline(True)
def dt_to_isospaceformat(dt: datetime.datetime) -> str:
    """Convert datetime to ISO format with space as
    the seperator: '%Y-%m-%d %H:%M:%S.f' `<'str'>`."""
    microsecond: cython.int = dt.microsecond
    # fmt: off
    if microsecond == 0:
        return "%04d-%02d-%02d %02d:%02d:%02d" % (
            dt.year, dt.month, dt.day, 
            dt.hour, dt.minute, dt.second,
        )
    else:
        return "%04d-%02d-%02d %02d:%02d:%02d.%06d" % (
            dt.year, dt.month, dt.day, 
            dt.hour, dt.minute, dt.second, microsecond,
        )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def dt_to_isospaceformat_tz(dt: datetime.datetime) -> str:
    """Convert datetime to ISO format with tzinfo and space
    as the separator: '%Y-%m-%d %H:%M:%S.f.Z' `<'str'>`."""
    fmt: str = dt_to_isospaceformat(dt)
    offset = dt_utcoffset(dt)
    if offset is not None:
        fmt += td_to_utcformat(offset)
    return fmt


@cython.cfunc
@cython.inline(True)
def dt_to_strformat(dt: datetime.datetime, format: str) -> str:
    """Convert datetime by string format `<'str'>`.
    Equivalent to `dt.strftime(format)`."""
    frepl: str = None
    zrepl: str = None
    Zrepl: str = None
    # Scan format for %f %z and %Z escapes, replacing as needed.
    fmt: list = []
    idx: cython.Py_ssize_t = 0
    length: cython.Py_ssize_t = str_len(format)
    ch: cython.Py_UCS4
    while idx < length:
        ch = read_char(format, idx)
        idx += 1
        # Identifier '%'
        if ch == "%":
            if idx < length:
                ch = read_char(format, idx)
                idx += 1
                # Microsecond fraction
                if ch == "f":
                    if frepl is None:
                        frepl = "%06d" % dt.microsecond
                    fmt.append(frepl)
                # UTC offset
                elif ch == "z":
                    if zrepl is None:
                        offset = dt_utcoffset(dt)
                        if offset is None:
                            zrepl = ""
                        else:
                            zrepl = td_to_utcformat(offset)
                    fmt.append(zrepl)
                # Timezone name
                elif ch == "Z":
                    if Zrepl is None:
                        tzname = dt_tzname(dt)
                        if tzname is None:
                            Zrepl = ""
                        else:
                            Zrepl = tzname.replace("%", "%%")
                    fmt.append(Zrepl)
                # Normal escapes
                else:
                    fmt.append("%")
                    fmt.append(str_fr_ucs4(ch))
            else:
                fmt.append("%")
        else:
            fmt.append(str_fr_ucs4(ch))

    # Convert datetime to struct:tm
    # fmt: off
    year: cython.int = dt.year
    month: cython.int = dt.month
    day: cython.int = dt.day
    tms = tm( # type: ignore
        dt.second, dt.minute, dt.hour, day, month, year, 
        ymd_weekday(year, month, day), days_bf_month(year, month) + day, -1
    )  
    # fmt: on
    # Format the datetime
    return stime_to_str(tms, "".join(fmt))  # type: ignore


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def dt_to_ordinal(dt: datetime.datetime) -> cython.uint:
    """Convert the date of 'dt' to ordinal `<'int'>`."""
    return ymd_to_ordinal(dt.year, dt.month, dt.day)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_to_seconds(dt: datetime.datetime) -> cython.double:
    """Convert datetime to total seconds after POSIX epoch `<'float'>`.
    This function ignores the local timezone and tzinfo of the datetime,
    thus should `NOT` be treated as `datetime.timestamp()`.
    """
    days: cython.double = dt_to_ordinal(dt)
    hour: cython.double = dt.hour
    minute: cython.double = dt.minute
    second: cython.double = dt.second
    microsecond: cython.double = dt.microsecond
    return (
        (days - EPOCH_DAY) * 86_400
        + hour * 3_600
        + minute * 60
        + second
        + microsecond / 1_000_000
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_to_seconds_utc(dt: datetime.datetime) -> cython.double:
    """Convert datetime to total seconds after POSIX epoch `<'float'>`.
    - If `dt` is timezone-aware, return total seconds in UTC,
      equivalent to `datetime.timestamp()`.
    - If `dt` is timezone-naive, return total seconds that ignores
      the local timezone and tzinfo of the datetime, requivalent
      to `dt_to_seconds()`.
    """
    sec: cython.double = dt_to_seconds(dt)
    offset = dt_utcoffset(dt)
    if offset is not None:
        sec -= td_to_seconds(offset)
    return sec


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_to_microseconds(dt: datetime.datetime) -> cython.longlong:
    """Convert datetime to total microseconds after POSIX epoch `<'int'>`.
    This function ignores the local timezone and tzinfo of the datetime,
    thus should `NOT` be treated as `datetime.timestamp()`.
    """
    days: cython.longlong = dt_to_ordinal(dt)
    hour: cython.longlong = dt.hour
    minute: cython.longlong = dt.minute
    second: cython.longlong = dt.second
    microsecond: cython.longlong = dt.microsecond
    return (
        (days - EPOCH_DAY) * US_DAY
        + hour * US_HOUR
        + minute * 60_000_000
        + second * 1_000_000
        + microsecond
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_to_microseconds_utc(dt: datetime.datetime) -> cython.longlong:
    """Convert datetime to total microseconds after POSIX epoch `<'int'>`.
    - If `dt` is timezone-aware, return total microseconds in UTC.
    - If `dt` is timezone-naive, return total microseconds that
      ignores the local timezone and tzinfo of the datetime,
      requivalent to `dt_to_microseconds()`.
    """
    us: cython.longlong = dt_to_microseconds(dt)
    offset = dt_utcoffset(dt)
    if offset is not None:
        us -= td_to_microseconds(offset)
    return us


@cython.cfunc
@cython.inline(True)
def dt_to_posixts(dt: datetime.datetime) -> cython.longlong:
    """Convert datetime to POSIX timestamp as `<'int'>`.

    Equivalent to `datetime._mktime()`

    This function takes into account local timezone
    and daylight saving time changes, and converts
    the datetime to POSIX timestamp.
    """
    # Total seconds of the 'dt' since epoch
    t: cython.longlong = int(dt_to_seconds(dt))
    # Adjustment for local time
    adj1: cython.longlong = localize_ts(t) - t  # type: ignore
    adj2: cython.longlong
    u1: cython.longlong = t - adj1
    u2: cython.longlong
    t1: cython.longlong = localize_ts(u1)  # type: ignore
    # Adjustment for Daylight Saving
    if t == t1:
        # We found one solution, but it may not be the one we need.
        # Look for an earlier solution (if `fold` is 0), or a later
        # one (if `fold` is 1).
        u2 = u1 - 86_400 if dt.fold == 0 else u1 + 86_400
        adj2 = localize_ts(u2) - u2  # type: ignore
        if adj1 == adj2:
            return u1
    else:
        adj2 = t1 - u1
        assert adj1 != adj2
    # Final adjustment
    u2 = t - adj2
    t2: cython.longlong = localize_ts(u2)  # type: ignore
    if t == t2:
        return u2
    if t == t1:
        return u1
    # We have found both offsets adj1 and adj2,
    # but neither t - adj1 nor t - adj2 is the
    # solution. This means t is in the gap.
    return max(u1, u2) if dt.fold == 0 else min(u1, u2)


@cython.cfunc
@cython.inline(True)
def dt_to_timestamp(dt: datetime.datetime) -> cython.double:
    """Convert datetime to POSIX timestamp as `<'float'>`.

    Equivalent to `datetime.timestamp()`
    """
    if dt.tzinfo is None:
        ts: cython.double = dt_to_posixts(dt)
        us: cython.double = dt.microsecond
        return ts + us / 1_000_000
    else:
        return td_to_seconds(dt_sub_dt(dt, EPOCH_UTC))


@cython.cfunc
@cython.inline(True)
def dt_to_stime(dt: datetime.datetime) -> tm:  # type: ignore
    """Convert datetime to `<'struct:tm'>`."""
    # Calculate DST
    dst: cython.int
    offset = dt_dst(dt)
    if offset is None:
        dst = -1
    else:
        dst = 1 if offset else 0

    # Generate struct time
    year: cython.int = dt.year
    month: cython.int = dt.month
    day: cython.int = dt.day
    # fmt: off
    return tm( # type: ignore
        dt.second, dt.minute, dt.hour, day, month, year, 
        ymd_weekday(year, month, day), days_bf_month(year, month) + day, dst,
    )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def dt_to_stime_utc(dt: datetime.datetime) -> tm:  # type: ignore
    """Convert datetime to `<'struct:tm'>` under UTC."""
    # Convert to UTC
    offset = dt_utcoffset(dt)
    if offset is not None:
        dt -= offset

    # Generate struct time
    year: cython.int = dt.year
    month: cython.int = dt.month
    day: cython.int = dt.day
    # fmt: off
    return tm( # type: ignore
        dt.second, dt.minute, dt.hour, day, month, year, 
        ymd_weekday(year, month, day), days_bf_month(year, month) + day, 0,
    )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def dt_fr_dt(dt: datetime.datetime) -> datetime.datetime:
    """Convert subclass of datetime.datetime to `<'datetime.datetime'>`."""
    # fmt: off
    return datetime.datetime_new(
        dt.year, dt.month, dt.day, 
        dt.hour, dt.minute, dt.second, 
        dt.microsecond, dt.tzinfo, dt.fold,
    )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def dt_fr_date(
    date: datetime.date,
    tzinfo: datetime.tzinfo = None,
    fold: cython.uint = 0,
) -> datetime.datetime:
    """Convert datetime.date to `<'datetime.datetime'>`."""
    # fmt: off
    return datetime.datetime_new(
        date.year, date.month, date.day, 0, 0, 0, 0, 
        tzinfo, 1 if fold == 1 and tzinfo is not None else 0
    )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def dt_fr_time(time: datetime.time) -> datetime.datetime:
    """Convert datetime.time to `<'datetime.datetime'>`.

    Year, month, day will be auto set to the current date.
    """
    tms = localtime()
    # fmt: off
    return datetime.datetime_new(
        tms.tm_year, tms.tm_mon, tms.tm_mday, 
        time.hour, time.minute, time.second, 
        time.microsecond, time.tzinfo, time.fold,
    )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def dt_fr_date_n_time(date: datetime.date, time: datetime.time) -> datetime.datetime:
    """Convert datetime.date & datetime.time to `<'datetime.datetime'>`."""
    # fmt: off
    return datetime.datetime_new(
        date.year, date.month, date.day, 
        time.hour, time.minute, time.second, 
        time.microsecond, time.tzinfo, time.fold,
    )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def dt_fr_ordinal(
    ordinal: cython.int,
    tzinfo: datetime.tzinfo = None,
    fold: cython.uint = 0,
) -> datetime.datetime:
    """Convert ordinal to `<'datetime.datetime'>`."""
    ymd = ordinal_to_ymd(ordinal)
    # fmt: off
    return datetime.datetime_new(
        ymd.year, ymd.month, ymd.day, 0, 0, 0, 0, 
        tzinfo, 1 if fold == 1 and tzinfo is not None else 0
    )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def dt_fr_seconds(
    seconds: cython.double,
    tzinfo: datetime.tzinfo = None,
    fold: cython.uint = 0,
) -> datetime.datetime:
    """Convert total seconds after POSIX epoch to `<'datetime.datetime'>`."""
    microseconds: cython.longlong = int(seconds * 1_000_000)
    return dt_fr_microseconds(microseconds, tzinfo, fold)


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
def dt_fr_microseconds(
    microseconds: cython.longlong,
    tzinfo: datetime.tzinfo = None,
    fold: cython.uint = 0,
) -> datetime.datetime:
    """Convert total microseconds after POSIX epoch to `<'datetime.datetime'>`."""
    # Add back epoch microseconds
    microseconds += EPOCH_US
    microseconds = min(max(microseconds, DT_MIN_US), DT_MAX_US)
    # Calculate ymd
    ymd = ordinal_to_ymd(microseconds // US_DAY)
    # Calculate hms
    hms = microseconds_to_hms(microseconds)
    # Generate datetime
    # fmt: off
    return datetime.datetime_new(
        ymd.year, ymd.month, ymd.day, 
        hms.hour, hms.minute, hms.second, hms.microsecond, 
        tzinfo, 1 if fold == 1 and tzinfo is not None else 0,
    )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def dt_fr_timestamp(
    timestamp: cython.double,
    tzinfo: datetime.tzinfo = None,
) -> datetime.datetime:
    """Convert timestamp to `<'datetime.datetime'>`."""
    return datetime.datetime_from_timestamp(timestamp, tzinfo)


# Datetime.datetime: arithmetic ------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def dt_add(
    dt: datetime.datetime,
    days: cython.int = 0,
    seconds: cython.longlong = 0,
    microseconds: cython.longlong = 0,
) -> datetime.datetime:
    """Add 'days', 'seconds' and 'microseconds' to `<'datetime.datetime'>`.
    Equivalent to `datetime + timedelta(days, seconds, microseconds)`."""
    if days == 0 and seconds == 0 and microseconds == 0:
        return dt
    return dt_fr_microseconds(
        dt_to_microseconds(dt) + days * US_DAY + seconds * 1_000_000 + microseconds,
        dt.tzinfo,
        dt.fold,
    )


@cython.cfunc
@cython.inline(True)
def dt_add_td(dt: datetime.datetime, td: datetime.timedelta) -> datetime.datetime:
    """Add datetime.timedelta to `<'datetime.datetime'>`.
    Equivalent to `datetime + timedelta)`."""
    return dt + td


@cython.cfunc
@cython.inline(True)
def dt_sub_td(dt: datetime.datetime, td: datetime.timedelta) -> datetime.datetime:
    """Substract datetime.timedelta from `<'datetime.datetime'>`.
    Equivalent to `datetime - timedelta)`."""
    return dt - td


@cython.cfunc
@cython.inline(True)
def dt_sub_dt(dt_l: datetime.datetime, dt_r: datetime.datetime) -> datetime.timedelta:
    """Substraction between datetime.datetimes and returns `<'datetime.timedelta'>`.
    Equivalent to `datetime - datetime`.
    """
    return td_fr_microseconds(dt_sub_dt_us(dt_l, dt_r))


@cython.cfunc
@cython.inline(True)
def dt_sub_dt_us(dt_l: datetime.datetime, dt_r: datetime.datetime) -> cython.longlong:
    """Substraction between datetime.datetimes and returns `<'int'>`
    (difference in microseconds). Equivalent to `datetime - datetime`."""
    delta_us: cython.longlong = dt_to_microseconds(dt_l) - dt_to_microseconds(dt_r)
    tzinfo_l: datetime.tzinfo = dt_l.tzinfo
    tzinfo_r: datetime.tzinfo = dt_r.tzinfo
    # If both are naive or have the same tzinfo
    # return delta directly.
    if tzinfo_l is tzinfo_r:
        return delta_us
    # Calculate left timezone offset
    if tzinfo_l is None:
        offset_l: cython.longlong = 0
    else:
        td = dt_utcoffset(dt_l)
        offset_l: cython.longlong = 0 if td is None else td_to_microseconds(td)
    # Calculate right timezone offset
    if tzinfo_r is None:
        offset_r: cython.longlong = 0
    else:
        td = dt_utcoffset(dt_r)
        offset_r: cython.longlong = 0 if td is None else td_to_microseconds(td)
    # Return the difference
    return delta_us + offset_r - offset_l


# Datetime.datetime: manipulation ----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def dt_replace(
    dt: datetime.datetime,
    year: cython.int = -1,
    month: cython.int = -1,
    day: cython.int = -1,
    hour: cython.int = -1,
    minute: cython.int = -1,
    second: cython.int = -1,
    microsecond: cython.int = -1,
    tzinfo: object = -1,
    fold: cython.int = -1,
) -> datetime.datetime:
    """Replace `<'datetime.datetime'>`. Equivalent to `datetime.replcae()`.
    Value of `-1` indicates preserving the original value."""
    if not 1 <= year <= 9999:
        year = dt.year
    if not 1 <= month <= 12:
        month = dt.month
    return datetime.datetime_new(
        year,
        month,
        min(day if day > 0 else dt.day, days_in_month(year, month)),
        hour if 0 <= hour <= 23 else dt.hour,
        minute if 0 <= minute <= 59 else dt.minute,
        second if 0 <= second <= 59 else dt.second,
        microsecond if 0 <= microsecond <= 999999 else dt.microsecond,
        tzinfo if (datetime.PyTZInfo_Check(tzinfo) or tzinfo is None) else dt.tzinfo,
        0 if tzinfo is None else (fold if 0 <= fold <= 1 else dt.fold),
    )


@cython.cfunc
@cython.inline(True)
def dt_replace_tzinfo(
    dt: datetime.datetime,
    tzinfo: datetime.tzinfo,
) -> datetime.datetime:
    """Replace `<'datetime.datetime'>` timezone information.
    Equivalent to `datetime.replace(tzinfo=tzinfo)."""
    # fmt: off
    return datetime.datetime_new(
        dt.year, dt.month, dt.day, 
        dt.hour, dt.minute, dt.second, 
        dt.microsecond, tzinfo, dt.fold,
    )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def dt_replace_fold(dt: datetime.datetime, fold: cython.uint) -> datetime.datetime:
    """Replace `<'datetime.datetime'>` day light saving 'fold'.
    Equivalent to `datetime.replace(fold=fold)."""
    # fmt: off
    return datetime.datetime_new(
        dt.year, dt.month, dt.day, 
        dt.hour, dt.minute, dt.second, 
        dt.microsecond, dt.tzinfo,
        1 if fold == 1 else 0,
    )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def dt_adj_weekday(dt: datetime.datetime, weekday: cython.uint) -> datetime.datetime:
    """Adjust `<'datetime.datetime'>` to the 'weekday' of
    the current week, where Monday is 0 and Sunday is 6.
    Equivalent to: `datetime + timedelta(days=weekday - dt.weekday())`."""
    weekday = weekday % 7
    wday: cython.uint = dt_weekday(dt)
    return dt if weekday == wday else dt_add(dt, days=weekday - wday)


@cython.cfunc
@cython.inline(True)
def dt_astimezone(dt: datetime.datetime, tzinfo: datetime.tzinfo) -> datetime.datetime:
    """Convert `<'datetime.datetime'>` to the new timezone.
    Equivalent to `datetime.astimezone(tzinfo)`.
    """
    b_tz: datetime.tzinfo = dt.tzinfo
    if tzinfo is None:
        t_tz: datetime.tzinfo = gen_tzinfo_local(None)
        if b_tz is None:
            return dt_replace_tzinfo(dt, t_tz)  # exit: replace tzinfo
    else:
        t_tz: datetime.tzinfo = tzinfo

    if b_tz is None:
        b_tz = gen_tzinfo_local(dt)
        b_offset = tzinfo_utcoffset(b_tz, dt)
    else:
        b_offset = tzinfo_utcoffset(b_tz, dt)
        if b_offset is None:
            b_tz = gen_tzinfo_local(dt_replace_tzinfo(dt, None))
            b_offset = tzinfo_utcoffset(b_tz, dt)

    if t_tz is b_tz:
        return dt

    # Calculate delta in microseconds
    t_delta: cython.longlong = td_to_microseconds(tzinfo_utcoffset(t_tz, dt))
    b_delta: cython.longlong = td_to_microseconds(b_offset)

    # Generate new datetime
    us: cython.longlong = dt_to_microseconds(dt)
    return dt_fr_microseconds(us + t_delta - b_delta, t_tz, 0)


# Datetime.time ========================================================================================
# Datetime.time: generate ------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def gen_time(
    hour: cython.uint = 0,
    minute: cython.uint = 0,
    second: cython.uint = 0,
    microsecond: cython.uint = 0,
    tzinfo: datetime.tzinfo = None,
    fold: cython.uint = 0,
) -> datetime.time:
    """Generate a new `<'datetime.time'>`."""
    # fmt: off
    return datetime.time_new(
        hour, minute, second, microsecond, 
        tzinfo, 1 if fold == 1 and tzinfo is not None else 0
    )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def gen_time_now() -> datetime.time:
    """Generate the current local `<'datetime.time'>`.
    Equivalent to `datetime.datetime.now().time()`."""
    microseconds: cython.int = int(unix_time() % 1 * 1_000_000)
    tms = localtime()
    return datetime.time_new(tms.tm_hour, tms.tm_min, tms.tm_sec, microseconds, None, 0)


@cython.cfunc
@cython.inline(True)
def gen_time_now_utc() -> datetime.time:
    """Generate the current `<'datetime.time'>` under UTC.
    Equivalent to `datetime.datetime.now(UTC).time()`."""
    return time_fr_dt(gen_dt_now_utc())


@cython.cfunc
@cython.inline(True)
def gen_time_now_tz(tzinfo: datetime.tzinfo) -> datetime.time:
    """Generate the current `<'datetime.time'>` under specific timezone.
    Equivalent to `datetime.datetime.now(tzinfo).time()`."""
    return time_fr_dt(gen_dt_now_tz(tzinfo))


# Datetime.time: check types ---------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_time(obj: object) -> cython.bint:
    """Check if an object is type of datetime.time `<'bool'>`.
    Equivalent to `isinstance(obj, datetime.time)`."""
    return datetime.PyTime_Check(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_time_exact(obj: object) -> cython.bint:
    """Check if an object is the exact type of datetime.time `<'bool'>`.
    Equivalent to `type(obj) is datetime.time`."""
    return datetime.PyTime_CheckExact(obj)


# Datetime.time: tzinfo --------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def time_tzname(time: datetime.time) -> str:
    """Access the 'tzname' `<'str'>`.
    Equivalent to `time.tzname()`."""
    return tzinfo_name(time.tzinfo, None)


@cython.cfunc
@cython.inline(True)
def time_dst(time: datetime.time) -> datetime.timedelta:
    """Access the 'dst' `<'datetime.timedelta'>`.
    Equivalent to `time.dst()`."""
    return tzinfo_dst(time.tzinfo, None)


@cython.cfunc
@cython.inline(True)
def time_utcoffset(time: datetime.time) -> datetime.timedelta:
    """Access the 'utcoffset' `<'datetime.timedelta'>`.
    Equivalent to `time.utcoffset()`."""
    return tzinfo_utcoffset(time.tzinfo, None)


# Datetime.time: conversion ----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def time_to_isoformat(time: datetime.time) -> str:
    """Convert time to ISO format: '%H:%M:%S.f' `<'str'>`."""
    microsecond: cython.int = time.microsecond
    if microsecond == 0:
        return "%02d:%02d:%02d" % (time.hour, time.minute, time.second)
    return "%02d:%02d:%02d.%06d" % (time.hour, time.minute, time.second, microsecond)


@cython.cfunc
@cython.inline(True)
def time_to_strformat(time: datetime.time, format: str) -> str:
    """Convert time by string format `<'str'>`.
    Equivalent to `time.strftime(format)`."""
    # fmt: off
    return dt_to_strformat(datetime.datetime_new(
        1900, 1, 1, time.hour, time.minute, time.second, time.microsecond, None, 0), format
    )
    # fmt: on


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_to_seconds(time: datetime.time) -> cython.double:
    """Convert datetime.time to total seconds `<'float'>`."""
    microseconds: cython.double = time_to_microseconds(time)
    return microseconds / 1_000_000


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_to_microseconds(time: datetime.time) -> cython.longlong:
    """Convert datetime.time to total microseconds `<'int'>`."""
    hour: cython.longlong = time.hour
    minute: cython.longlong = time.minute
    second: cython.longlong = time.second
    microsecond: cython.longlong = time.microsecond
    return hour * US_HOUR + minute * 60_000_000 + second * 1_000_000 + microsecond


@cython.cfunc
@cython.inline(True)
def time_fr_dt(dt: datetime.datetime) -> datetime.time:
    """Convert datetime.datetime to `<'datetime.time'>`."""
    # fmt: off
    return datetime.time_new(
        dt.hour, dt.minute, dt.second, 
        dt.microsecond, dt.tzinfo, dt.fold,
    )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def time_fr_seconds(
    seconds: cython.double,
    tzinfo: datetime.tzinfo = None,
    fold: cython.uint = 0,
) -> datetime.time:
    """Convert total seconds to `<'datetime.time'>`."""
    mciroseconds: cython.longlong = int(seconds * 1_000_000)
    return time_fr_microseconds(mciroseconds, tzinfo, fold)


@cython.cfunc
@cython.inline(True)
def time_fr_microseconds(
    microseconds: cython.longlong,
    tzinfo: datetime.tzinfo = None,
    fold: cython.uint = 0,
) -> datetime.time:
    """Convert total microseconds to `<'datetime.time'>`."""
    # Add back epoch microseconds
    if microseconds < 0:
        microseconds += EPOCH_US
    microseconds = min(max(microseconds, 0), DT_MAX_US)
    # Calculate hms
    hms = microseconds_to_hms(microseconds)
    # Generate time
    # fmt: off
    return datetime.time_new(
        hms.hour, hms.minute, hms.second, hms.microsecond, 
        tzinfo, 1 if fold == 1 and tzinfo is not None else 0,
    )
    # fmt: on


# Datetime.time: manipulation --------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def time_replace(
    time: datetime.time,
    hour: cython.int = -1,
    minute: cython.int = -1,
    second: cython.int = -1,
    microsecond: cython.int = -1,
    tzinfo: object = -1,
    fold: cython.int = -1,
) -> datetime.time:
    """Replace `<'datetime.time'>`. Equivalent to `time.replcae()`.
    Value of `-1` indicates preserving the original value."""
    return datetime.time_new(
        hour if 0 <= hour <= 23 else time.hour,
        minute if 0 <= minute <= 59 else time.minute,
        second if 0 <= second <= 59 else time.second,
        microsecond if 0 <= microsecond <= 999999 else time.microsecond,
        tzinfo if (datetime.PyTZInfo_Check(tzinfo) or tzinfo is None) else time.tzinfo,
        0 if tzinfo is None else (fold if 0 <= fold <= 1 else time.fold),
    )


@cython.cfunc
@cython.inline(True)
def time_replace_tzinfo(time: datetime.time, tzinfo: datetime.tzinfo) -> datetime.time:
    """Replace `<'datetime.time'>` timezone information.
    Equivalent to `time.replace(tzinfo=tzinfo)."""
    # fmt: off
    return datetime.time_new(
        time.hour, time.minute, time.second, 
        time.microsecond, tzinfo, time.fold,
    )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def time_replace_fold(time: datetime.time, fold: cython.uint) -> datetime.time:
    """Replace `<'datetime.time'>` day light saving 'fold'.
    Equivalent to `time.replace(fold=fold)."""
    # fmt: off
    return datetime.time_new(
        time.hour, time.minute, time.second, 
        time.microsecond, time.tzinfo, 1 if fold == 1 else 0,
    )
    # fmt: on


# Datetime.timedelta ===================================================================================
# Datetime.timedelta: generate -------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def gen_td(
    days: cython.int = 0,
    seconds: cython.int = 0,
    microseconds: cython.int = 0,
) -> datetime.timedelta:
    """Generate a new `<'datetime.timedelta'>`."""
    return datetime.timedelta_new(days, seconds, microseconds)


# Datetime.timedelta: check types ----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_td(obj: object) -> cython.bint:
    """Check if an object is type of datetime.timedelta `<'bool'>`.
    Equivalent to `isinstance(obj, datetime.timedelta)`."""
    return datetime.PyDelta_Check(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_td_exact(obj: object) -> cython.bint:
    """Check if an object is the exact type of datetime.timedelta `<'bool'>`.
    Equivalent to `type(obj) is datetime.timedelta`."""
    return datetime.PyDelta_CheckExact(obj)


# Datetime.timedelta: conversion -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def td_to_isoformat(td: datetime.timedelta) -> str:
    """Convert timedelta to ISO format: '%H:%M:%S.f' `<'str'>`."""
    days: cython.int = td.day
    secs: cython.int = td.second
    hours: cython.int = secs // 3_600 % 24 + days * 24
    minutes: cython.int = secs // 60 % 60
    seconds: cython.int = secs % 60
    microseconds: cython.int = td.microsecond
    if microseconds == 0:
        return "%02d:%02d:%02d" % (hours, minutes, seconds)
    else:
        return "%02d:%02d:%02d.%06d" % (hours, minutes, seconds, microseconds)


@cython.cfunc
@cython.inline(True)
def td_to_utcformat(td: datetime.timedelta) -> str:
    """Convert timedelta to UTC format: '+HH:MM' `<'str'>`."""
    days: cython.int = td.day
    secs: cython.int = td.second
    hours: cython.int = secs // 3_600 % 24 + days * 24
    if hours < 0:
        hours = -hours
        sign = "-"
    else:
        sign = "+"
    minutes: cython.int = secs // 60 % 60
    seconds: cython.int = secs % 60
    if seconds:
        return "%s%02d:%02d:%02d" % (sign, hours, minutes, seconds)
    else:
        return "%s%02d:%02d" % (sign, hours, minutes)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def td_to_seconds(td: datetime.timedelta) -> cython.double:
    """Convert timedelta to total seconds `<'float'>`."""
    microseconds: cython.double = td_to_microseconds(td)
    return microseconds / 1_000_000


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def td_to_microseconds(td: datetime.timedelta) -> cython.longlong:
    """Convert datetime.timedelta to total microseconds `<'int'>`."""
    days: cython.longlong = td.day
    seconds: cython.longlong = td.second
    microseconds: cython.longlong = td.microsecond
    return days * US_DAY + seconds * 1_000_000 + microseconds


@cython.cfunc
@cython.inline(True)
def td_fr_td(td: datetime.timedelta) -> datetime.timedelta:
    """Convert subclass of datetime.timedelta to `<'datetime.timedelta'>`."""
    return datetime.timedelta_new(td.day, td.second, td.microsecond)


@cython.cfunc
@cython.inline(True)
def td_fr_seconds(seconds: cython.double) -> datetime.timedelta:
    """Convert total seconds to `<'datetime.timedelta'>`."""
    microseconds: cython.longlong = int(seconds * 1_000_000)
    return td_fr_microseconds(microseconds)


@cython.cfunc
@cython.inline(True)
def td_fr_microseconds(microseconds: cython.longlong) -> datetime.timedelta:
    """Convert total microseconds to `<'datetime.timedelta'>`."""
    # Calculate days, seconds and microseconds
    days: cython.int = microseconds // US_DAY
    microseconds = microseconds % US_DAY
    seconds: cython.int = microseconds // 1_000_000
    microseconds = microseconds % 1_000_000
    # Generate timedelta
    return datetime.timedelta_new(days, seconds, microseconds)


# Datetime.timedelta: arithmetic -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def td_add(
    td: datetime.timedelta,
    days: cython.int = 0,
    seconds: cython.int = 0,
    microseconds: cython.int = 0,
) -> datetime.timedelta:
    """Add 'days', 'seconds' and 'microseconds' to `<'datetime.timedelta'>`.
    Equivalent to `timedelta + timedelta(d, s, us)`."""
    if days == 0 and seconds == 0 and microseconds == 0:
        return td
    return datetime.timedelta_new(
        td.day + days,
        td.second + seconds,
        td.microsecond + microseconds,
    )


@cython.cfunc
@cython.inline(True)
def td_add_td(
    td_l: datetime.timedelta,
    td_r: datetime.timedelta,
) -> datetime.timedelta:
    """Addition between datetime.timedeltas.
    Equivalent to `timedelta + timedelta`."""
    return datetime.timedelta_new(
        td_l.day + td_r.day,
        td_l.second + td_r.second,
        td_l.microsecond + td_r.microsecond,
    )


@cython.cfunc
@cython.inline(True)
def td_sub_td(
    td_l: datetime.timedelta,
    td_r: datetime.timedelta,
) -> datetime.timedelta:
    """Substraction between datetime.timedeltas.
    Equivalent to `timedelta - timedelta`."""
    return datetime.timedelta_new(
        td_l.day - td_r.day,
        td_l.second - td_r.second,
        td_l.microsecond - td_r.microsecond,
    )


# Datetime.tzinfo ======================================================================================
# Datetime.tzinfo: generate ----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def gen_tzinfo(offset: cython.int) -> datetime.tzinfo:
    """Generate a new `<'datetime.tzinfo'>` from offset in seconds."""
    # Validate offset
    if not -86_340 <= offset <= 86_340:
        raise ValueError(
            "Timezone expects offset -86,340...86,340 "
            "in seconds, instead got: %d." % offset
        )
    # Generate tzinfo
    return datetime.PyTimeZone_FromOffset(datetime.timedelta_new(0, offset, 0))


@cython.cfunc
@cython.inline(True)
def gen_tzinfo_local(dt: datetime.datetime = None) -> datetime.tzinfo:
    """Generate the local `<'datetime.tzinfo'>`.

    If 'dt' is not specified, defaults to localtime.
    """
    # Get local timestamp
    if datetime.PyDateTime_Check(dt):
        if dt.tzinfo is None:
            ts = dt_to_posixts(dt)
        else:
            ts = td_to_seconds(dt_sub_dt(dt, EPOCH_UTC))
    else:
        ts = unix_time()
    # Generate tzinfo
    return gen_tzinfo(FN_TIME_LOCALTIME(ts).tm_gmtoff)  # type: ignore


# Datetime.tzinfo: check types -------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_tzinfo(obj: object) -> cython.bint:
    """Check if an object is type of datetime.tzinfo `<'bool'>`.
    Equivalent to `isinstance(obj, datetime.tzinfo)`."""
    return datetime.PyTZInfo_Check(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_tzinfo_exact(obj: object) -> cython.bint:
    """Check if an object is the exact type of datetime.tzinfo `<'bool'>`.
    Equivalent to `type(obj) is datetime.tzinfo`."""
    return datetime.PyTZInfo_CheckExact(obj)


# Datetime.tzinfo: access ------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def tzinfo_name(tzinfo: object, dt: datetime.datetime) -> str:
    """Access the 'tzname' `<'str'>`.
    Equivalent to `tzinfo.tzname(dt)`."""
    if tzinfo is None:
        return None
    try:
        name = tzinfo.tzname(dt)
    except Exception as err:
        if not datetime.PyTZInfo_Check(tzinfo):
            raise TypeError(
                "Expects <'datetime.tzinfo'>, instead got: %s" % type(tzinfo)
            ) from err
        raise err
    if name is None:
        return None
    try:
        res: str = name
    except Exception as err:
        raise TypeError(
            "<'datetime.tzinfo'> tzname() must return 'None' "
            "or <'str'>, instead got: %s." % type(name)
        ) from err
    return res


@cython.cfunc
@cython.inline(True)
def tzinfo_dst(tzinfo: object, dt: datetime.datetime) -> datetime.timedelta:
    """Access the 'dst' `<'datetime.timedelta'>`.
    Equivalent to `tzinfo.dst(dt)`."""
    if tzinfo is None:
        return None
    try:
        offset = tzinfo.dst(dt)
    except Exception as err:
        if not datetime.PyTZInfo_Check(tzinfo):
            raise TypeError(
                "Expects <'datetime.tzinfo'>, instead got: %s" % type(tzinfo)
            ) from err
        raise err
    if offset is None:
        return None
    try:
        res: datetime.timedelta = offset
    except Exception as err:
        raise TypeError(
            "<'datetime.tzinfo'> dst() must return 'None' "
            "or <'timedelta'>, instead got: %s." % type(offset)
        ) from err
    return res


@cython.cfunc
@cython.inline(True)
def tzinfo_utcoffset(tzinfo: object, dt: datetime.datetime) -> datetime.timedelta:
    """Access the 'utcoffset' `<'datetime.timedelta'>`.
    Equivalent to `tzinfo.utcoffset(dt)`."""
    if tzinfo is None:
        return None
    try:
        offset = tzinfo.utcoffset(dt)
    except Exception as err:
        if not datetime.PyTZInfo_Check(tzinfo):
            raise TypeError(
                "Expects <'datetime.tzinfo'>, instead got: %s" % type(tzinfo)
            ) from err
        raise err
    if offset is None:
        return None
    try:
        res: datetime.timedelta = offset
    except Exception as err:
        raise TypeError(
            "<'datetime.tzinfo'> utcoffset() must return 'None' "
            "or <'timedelta'>, instead got: %s." % type(offset)
        ) from err
    return res


# numpy.datetime64 =====================================================================================
# numpy.datetime64: check types ------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_dt64(obj: object) -> cython.bint:
    """Check if an object is type of numpy.datetime64 `<'bool'>`.
    Equivalent to `isinstance(obj, numpy.datetime64)`."""
    return np.is_datetime64_object(obj)


@cython.cfunc
@cython.inline(True)
def validate_dt64(obj: object):
    """Validate if an object is type of `<'numpy.datetime64'>`.
    Raise `TypeError` if object type is incorrect."""
    if not np.is_datetime64_object(obj):
        raise TypeError(
            "Expects <'numpy.datetime64'>, "
            "instead got: %s %s." % (type(obj), repr(obj))
        )


# numpy.datetime64: conversion -------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
def dt64_to_isoformat(dt64: object) -> str:
    """Convert numpy.datetime64 to ISO format:
    '%Y-%m-%dT%H:%M:%S.f' `<'str'>`."""
    # Add back epoch seconds
    microseconds: cython.longlong = dt64_to_microseconds(dt64) + EPOCH_US
    microseconds = min(max(microseconds, DT_MIN_US), DT_MAX_US)
    # Calculate ymd
    ymd = ordinal_to_ymd(microseconds // US_DAY)
    # Calculate hms
    hms = microseconds_to_hms(microseconds)
    # Return isoformat
    # fmt: off
    microsecond: cython.uint = hms.microsecond
    if microsecond == 0:
        return "%04d-%02d-%02dT%02d:%02d:%02d" % (
            ymd.year, ymd.month, ymd.day, 
            hms.hour, hms.minute, hms.second,
        )
    else:
        return "%04d-%02d-%02dT%02d:%02d:%02d.%06d" % (
            ymd.year, ymd.month, ymd.day, 
            hms.hour, hms.minute, hms.second, microsecond,
        )
    # fmt: on


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
def dt64_to_isospaceformat(dt64: object) -> str:
    """Convert numpy.datetime64 to ISO format with space
    as the seperator: '%Y-%m-%d %H:%M:%S.f' `<'str'>`."""
    # Add back epoch seconds
    microseconds: cython.longlong = dt64_to_microseconds(dt64) + EPOCH_US
    microseconds = min(max(microseconds, DT_MIN_US), DT_MAX_US)
    # Calculate ymd
    ymd = ordinal_to_ymd(microseconds // US_DAY)
    # Calculate hms
    hms = microseconds_to_hms(microseconds)
    # Return isoformat
    # fmt: off
    microsecond: cython.uint = hms.microsecond
    if microsecond == 0:
        return "%04d-%02d-%02d %02d:%02d:%02d" % (
            ymd.year, ymd.month, ymd.day, 
            hms.hour, hms.minute, hms.second,
        )
    else:
        return "%04d-%02d-%02d %02d:%02d:%02d.%06d" % (
            ymd.year, ymd.month, ymd.day, 
            hms.hour, hms.minute, hms.second, microsecond,
        )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def dt64_to_int(
    dt64: object,
    unit: Literal["D", "h", "m", "s", "ms", "us", "ns"],
) -> cython.longlong:
    """Convert numpy.datetime64 to intger based on
    the target time 'unit' `<'int'>`.

    ### Notice
    Percision will be lost if the datetime64 unit
    is smaller than the target time 'unit'.
    """
    if unit == "ns":
        return dt64_to_nanoseconds(dt64)
    elif unit == "us":
        return dt64_to_microseconds(dt64)
    elif unit == "ms":
        return dt64_to_milliseconds(dt64)
    elif unit == "s":
        return dt64_to_seconds(dt64)
    elif unit == "m":
        return dt64_to_minutes(dt64)
    elif unit == "h":
        return dt64_to_hours(dt64)
    elif unit == "D" or unit == "d":
        return dt64_to_days(dt64)
    else:
        raise ValueError(
            "Does not support conversion of "
            "<'numpy.datetime64'> to time unit: %s." % repr(unit)
        )


@cython.cfunc
@cython.inline(True)
def dt64_to_days(dt64: object) -> cython.longlong:
    """Convert numpy.datetime64 to total days `<'int'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'day (D)'.
    """
    # Access value & unit
    validate_dt64(dt64)
    val: np.npy_datetime = np.get_datetime64_value(dt64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(dt64)

    # Converstion
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // NS_DAY
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // US_DAY
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val // 86_400_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val // 86_400
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val // 1_440
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val // 24
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return val // NS_DAY // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return val // NS_DAY // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return val // NS_DAY // 1_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'numpy.datetime64'> time unit "
        "to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def dt64_to_hours(dt64: object) -> cython.longlong:
    """Convert numpy.datetime64 to total hours `<'int'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'hour (h)'.
    """
    # Access value & unit
    validate_dt64(dt64)
    val: np.npy_datetime = np.get_datetime64_value(dt64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(dt64)

    # Converstion
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // NS_HOUR
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // US_HOUR
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val // 3_600_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val // 3_600
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val // 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * 24
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return val // NS_HOUR // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return val // NS_HOUR // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return val // NS_HOUR // 1_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'numpy.datetime64'> time unit "
        "to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def dt64_to_minutes(dt64: object) -> cython.longlong:
    """Convert numpy.datetime64 to total minutes `<'int'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'minute (m)'.
    """
    # Access value & unit
    validate_dt64(dt64)
    val: np.npy_datetime = np.get_datetime64_value(dt64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(dt64)

    # Converstion
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // NS_MINUTE
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // 60_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val // 60_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val // 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * 1_440
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return val // NS_MINUTE // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return val // NS_MINUTE // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return val // NS_MINUTE // 1_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'numpy.datetime64'> time unit "
        "to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def dt64_to_seconds(dt64: object) -> cython.longlong:
    """Convert numpy.datetime64 to total seconds `<'int'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'second (s)'.
    """
    # Access value & unit
    validate_dt64(dt64)
    val: np.npy_datetime = np.get_datetime64_value(dt64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(dt64)

    # Converstion
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * 3_600
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * 86_400
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return val // 1_000_000_000 // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return val // 1_000_000_000 // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return val // 1_000_000_000 // 1_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'numpy.datetime64'> time unit "
        "to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def dt64_to_milliseconds(dt64: object) -> cython.longlong:
    """Convert numpy.datetime64 to total milliseconds `<'int'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'millisecond (ms)'.
    """
    # Access value & unit
    validate_dt64(dt64)
    val: np.npy_datetime = np.get_datetime64_value(dt64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(dt64)

    # Converstion
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * 60_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * 3_600_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * 86_400_000
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return val // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return val // 1_000_000_000 // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return val // 1_000_000_000 // 1_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'numpy.datetime64'> time unit "
        "to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def dt64_to_microseconds(dt64: object) -> cython.longlong:
    """Convert numpy.datetime64 to total microseconds `<'int'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'microsecond (us)'.
    """
    # Access value & unit
    validate_dt64(dt64)
    val: np.npy_datetime = np.get_datetime64_value(dt64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(dt64)

    # Converstion
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val * 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * 60_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * US_HOUR
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * US_DAY
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return val // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return val // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return val // 1_000_000_000 // 1_000
    # . unsupported
    raise ValueError(
        "Unsupported <'numpy.datetime64'> time unit "
        "to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def dt64_to_nanoseconds(dt64: object) -> cython.longlong:
    """Convert numpy.datetime64 to total nanoseconds `<'int'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'nanosecond (ns)'.
    """
    # Access value & unit
    validate_dt64(dt64)
    val: np.npy_datetime = np.get_datetime64_value(dt64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(dt64)

    # Converstion
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val * 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val * 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * NS_MINUTE
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * NS_HOUR
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * NS_DAY
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return val // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return val // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return val // 1_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'numpy.datetime64'> time unit "
        "to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def dt64_to_date(dt64: object) -> datetime.date:
    """Convert numpy.datetime64 to to `<'datetime.date'>`.

    Value out of datetime.date range will be clipped by:
    - Upper limit: `9999-12-31`
    - Lower limit: `0001-01-01`

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'day (D)'.
    """
    return date_fr_ordinal(dt64_to_days(dt64) + EPOCH_DAY)


@cython.cfunc
@cython.inline(True)
def dt64_to_dt(dt64: object) -> datetime.datetime:
    """Convert numpy.datetime64 to `<'datetime.datetime'>`.

    Value out of datetime.datetime range will be clipped by:
    - Upper limit: `9999-12-31 23:59:59.999999`.
    - Lower limit: `0001-01-01 00:00:00.000000`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'microsecond (us)'.
    """
    return dt_fr_microseconds(dt64_to_microseconds(dt64), None, 0)


@cython.cfunc
@cython.inline(True)
def dt64_to_time(dt64: object) -> datetime.time:
    """Convert numpy.datetime64 to `<'datetime.time'>`.

    Value out of datetime.time range will be clipped by:
    - Upper limit: `23:59:59.999999`.
    - Lower limit: `00:00:00.000000`.

    ### Notice
    Only the HMS information will be extracted from the
    datetime64, all other information will be discarded.
    """
    return time_fr_microseconds(dt64_to_microseconds(dt64), None, 0)


# numpy.timedelta64 ====================================================================================
# numpy.timedelta64: check types -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_td64(obj: object) -> cython.bint:
    """Check if an object is type of numpy.timedelta64 `<'bool'>`.
    Equivalent to `isinstance(obj, numpy.timedelta64)`.
    """
    return np.is_timedelta64_object(obj)


@cython.cfunc
@cython.inline(True)
def validate_td64(obj: object):
    """Validate if an object is type of `<'numpy.timedelta64'>`.
    Raise `TypeError` if the object type is incorrect."""
    if not np.is_timedelta64_object(obj):
        raise TypeError(
            "Expects <'numpy.timedelta64'>, "
            "instead got: %s %s." % (type(obj), repr(obj))
        )


# numpy.timedelta64: conversion ------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def td64_to_isoformat(td64: object) -> str:
    """Convert numpy.timedelta64 to ISO format: '%H:%M:%S.f' `<'str'>`."""
    us: cython.longlong = td64_to_microseconds(td64)
    days: cython.longlong = us // US_DAY
    secs: cython.longlong = us // 1_000_000
    hours: cython.longlong = secs // 3_600 % 24 + days * 24
    minutes: cython.longlong = secs // 60 % 60
    seconds: cython.longlong = secs % 60
    microseconds: cython.longlong = us % 1_000_000
    if microseconds == 0:
        return "%02d:%02d:%02d" % (hours, minutes, seconds)
    else:
        return "%02d:%02d:%02d.%06d" % (hours, minutes, seconds, microseconds)


@cython.cfunc
@cython.inline(True)
def td64_to_int(
    td64: object,
    unit: Literal["D", "h", "m", "s", "ms", "us", "ns"],
) -> cython.longlong:
    """Convert numpy.timedelta64 to intger based on
    the target time unit `<'int'>`.

    ### Notice
    Percision will be lost if the datetime64 unit
    is smaller than the target time 'unit'.
    """
    if unit == "ns":
        return td64_to_nanoseconds(td64)
    elif unit == "us":
        return td64_to_microseconds(td64)
    elif unit == "ms":
        return td64_to_milliseconds(td64)
    elif unit == "s":
        return td64_to_seconds(td64)
    elif unit == "m":
        return td64_to_minutes(td64)
    elif unit == "h":
        return td64_to_hours(td64)
    elif unit == "D" or unit == "d":
        return td64_to_days(td64)
    else:
        raise ValueError(
            "Does not support conversion of "
            "<'numpy.timedelta64'> to time unit: %s." % repr(unit)
        )


@cython.cfunc
@cython.inline(True)
def td64_to_days(td64: object) -> cython.longlong:
    """Convert numpy.timedelta64 to total days `<'int'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'day (D)'.
    """
    # Access value & unit
    validate_td64(td64)
    val: np.npy_timedelta = np.get_timedelta64_value(td64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(td64)

    # Converstion
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // NS_DAY
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // US_DAY
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val // 86_400_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val // 86_400
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val // 1_440
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val // 24
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return val // NS_DAY // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return val // NS_DAY // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return val // NS_DAY // 1_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'numpy.timedelta64'> time unit "
        "to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def td64_to_hours(td64: object) -> cython.longlong:
    """Convert numpy.timedelta64 to total hours `<'int'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'hour (h)'.
    """
    # Access value & unit
    validate_td64(td64)
    val: np.npy_timedelta = np.get_timedelta64_value(td64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(td64)

    # Converstion
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // NS_HOUR
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // US_HOUR
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val // 3_600_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val // 3_600
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val // 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * 24
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return val // NS_HOUR // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return val // NS_HOUR // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return val // NS_HOUR // 1_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'numpy.timedelta64'> time unit "
        "to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def td64_to_minutes(td64: object) -> cython.longlong:
    """Convert numpy.timedelta64 to total minutes `<'int'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'minute (m)'.
    """
    # Access value & unit
    validate_td64(td64)
    val: np.npy_timedelta = np.get_timedelta64_value(td64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(td64)

    # Converstion
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // NS_MINUTE
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // 60_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val // 60_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val // 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * 1_440
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return val // NS_MINUTE // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return val // NS_MINUTE // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return val // NS_MINUTE // 1_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'numpy.timedelta64'> time unit "
        "to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def td64_to_seconds(td64: object) -> cython.longlong:
    """Convert numpy.timedelta64 to total seconds `<'int'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'second (s)'.
    """
    # Access value & unit
    validate_td64(td64)
    val: np.npy_timedelta = np.get_timedelta64_value(td64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(td64)

    # Converstion
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * 3_600
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * 86_400
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return val // 1_000_000_000 // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return val // 1_000_000_000 // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return val // 1_000_000_000 // 1_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'numpy.timedelta64'> time unit "
        "to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def td64_to_milliseconds(td64: object) -> cython.longlong:
    """Convert numpy.timedelta64 to total milliseconds `<'int'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'millisecond (ms)'.
    """
    # Access value & unit
    validate_td64(td64)
    val: np.npy_timedelta = np.get_timedelta64_value(td64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(td64)

    # Converstion
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * 60_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * 3_600_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * 86_400_000
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return val // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return val // 1_000_000_000 // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return val // 1_000_000_000 // 1_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'numpy.timedelta64'> time unit "
        "to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def td64_to_microseconds(td64: object) -> cython.longlong:
    """Convert numpy.timedelta64 to total microseconds `<'int'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'microsecond (us)'.
    """
    # Access value & unit
    validate_td64(td64)
    val: np.npy_timedelta = np.get_timedelta64_value(td64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(td64)

    # Converstion
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val * 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * 60_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * US_HOUR
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * US_DAY
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return val // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return val // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return val // 1_000_000_000 // 1_000
    # . unsupported
    raise ValueError(
        "Unsupported <'numpy.timedelta64'> time unit "
        "to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def td64_to_nanoseconds(td64: object) -> cython.longlong:
    """Convert numpy.timedelta64 to total nanoseconds `<'int'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'nanosecond (ns)'.
    """
    # Access value & unit
    validate_td64(td64)
    val: np.npy_timedelta = np.get_timedelta64_value(td64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(td64)

    # Converstion
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val * 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val * 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * NS_MINUTE
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * NS_HOUR
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * NS_DAY
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return val // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return val // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return val // 1_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'numpy.timedelta64'> time unit "
        "to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def td64_to_td(td64: object) -> datetime.timedelta:
    """Convert numpy.timedelta64 to `<'datetime.timedelta'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'microsecond (us)'."""
    return td_fr_microseconds(td64_to_microseconds(td64))


# numpy.ndarray[dateimte64] ============================================================================
# numpy.ndarray[dateimte64]: type checks ---------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_dt64array(arr: np.ndarray) -> cython.bint:
    """Check if a ndarray is dtype of datetime64 `<'bool'>`.
    Equivalent to `isinstance(arr.dtype, np.dtypes.DateTime64DType)`."""
    return np.PyArray_TYPE(arr) == np.NPY_TYPES.NPY_DATETIME


@cython.cfunc
@cython.inline(True)
def validate_dt64array(arr: np.ndarray):
    """Validate if a ndarray is dtype of datetime64.
    Raise `TypeError` if dtype is incorrect."""
    if not is_dt64array(arr):
        raise TypeError(
            "Expects <'ndarray[datetime64]'>, "
            "instead got: <'ndarray[%s]'>." % (arr.dtype)
        )


@cython.cfunc
@cython.inline(True)
def get_dt64array_unit(arr: np.ndarray) -> str:
    """Get the time unit of the `<'ndarray[datetime64]'>`,
    such as: 'D', 's', 'ms', 'us', 'ns', etc `<'str'>`."""
    # Validate array
    validate_dt64array(arr)

    # Get time unit
    unit: np.NPY_DATETIMEUNIT
    ndim: cython.Py_ssize_t = arr.ndim
    if ndim == 1:
        if arr.shape[0] == 0:
            return parse_dt64array_unit(arr)
        unit = np.get_datetime64_unit(arr[0])
    elif ndim == 2:
        if arr.shape[1] == 0:
            return parse_dt64array_unit(arr)
        unit = np.get_datetime64_unit(arr[0, 0])
    elif ndim == 3:
        if arr.shape[2] == 0:
            return parse_dt64array_unit(arr)
        unit = np.get_datetime64_unit(arr[0, 0, 0])
    elif ndim == 4:
        if arr.shape[3] == 0:
            return parse_dt64array_unit(arr)
        unit = np.get_datetime64_unit(arr[0, 0, 0, 0])
    else:
        raise ValueError(
            "Only support <'ndarray'> with up to 4 "
            "dimensions, instead got: %d." % ndim
        )

    # Match unit: common
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
    # Match unit: uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return "ps"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return "fs"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return "as"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:
        return "Y"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:
        return "M"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:
        return "W"
    # if unit == np.NPY_DATETIMEUNIT.NPY_FR_B:
    #     return "B"
    raise ValueError("Unknown <'ndarray[datetime64]'> time unit: %d." % unit)


@cython.cfunc
@cython.inline(True)
def parse_dt64array_unit(arr: np.ndarray) -> str:
    """Prase the time unit of the `<'ndarray[datetime64]'>`
    from 'ndarray.dtype.str' `<'str'>`."""
    dtype_str = arr.dtype.str
    unit: str = str_substr(dtype_str, 4, str_len(dtype_str) - 1)
    if not 1 <= str_len(unit) <= 2:
        raise ValueError(
            "Unable to parse <'ndarray[datetime64]'> "
            "time unit from: '%s'." % dtype_str
        )
    return unit


# numpy.ndarray[dateimte64]: conversion ----------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def dt64array_to_int(
    arr: np.ndarray,
    unit: Literal["D", "h", "m", "s", "ms", "us", "ns"],
) -> np.ndarray:
    """Convert ndarray[datetime64] to integer based
    on the target time unit `<'ndarray[int64]'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than the target time 'unit'.
    """
    if unit == "ns":
        return dt64array_to_nanoseconds(arr)
    elif unit == "us":
        return dt64array_to_microseconds(arr)
    elif unit == "ms":
        return dt64array_to_milliseconds(arr)
    elif unit == "s":
        return dt64array_to_seconds(arr)
    elif unit == "m":
        return dt64array_to_minutes(arr)
    elif unit == "h":
        return dt64array_to_hours(arr)
    elif unit == "D" or unit == "d":
        return dt64array_to_days(arr)
    else:
        raise ValueError(
            "Does not support conversion of "
            "<'ndarray[datetime64]'> to time unit: %s." % repr(unit)
        )


@cython.cfunc
@cython.inline(True)
def dt64array_to_days(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[datetime64] to total days `<'ndarray[int64]'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'day (D)'.
    """
    # Validate array
    validate_dt64array(arr)

    # Get time unit
    unit: np.NPY_DATETIMEUNIT
    ndim: cython.Py_ssize_t = arr.ndim
    if ndim == 1:
        if arr.shape[0] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0])
    elif ndim == 2:
        if arr.shape[1] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0])
    elif ndim == 3:
        if arr.shape[2] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0])
    elif ndim == 4:
        if arr.shape[3] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0, 0])
    else:
        raise ValueError(
            "Only support <'ndarray'> with up to 4 "
            "dimensions, instead got: %d." % ndim
        )

    # Conversion
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // 86_400_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // 86_400_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr // 86_400_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr // 86_400
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr // 1_440
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr // 24
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return arr // 86_400_000_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return arr // 86_400_000_000_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return arr // 86_400_000_000_000_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'ndarray[datetime64]'> "
        "time unit to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def dt64array_to_hours(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[datetime64] to total hours `<'ndarray[int64]'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'hour (h)'.
    """
    # Validate array
    validate_dt64array(arr)

    # Get time unit
    unit: np.NPY_DATETIMEUNIT
    ndim: cython.Py_ssize_t = arr.ndim
    if ndim == 1:
        if arr.shape[0] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0])
    elif ndim == 2:
        if arr.shape[1] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0])
    elif ndim == 3:
        if arr.shape[2] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0])
    elif ndim == 4:
        if arr.shape[3] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0, 0])
    else:
        raise ValueError(
            "Only support <'ndarray'> with up to 4 "
            "dimensions, instead got: %d." % ndim
        )

    # Conversion
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // 3_600_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // 3_600_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr // 3_600_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr // 3_600
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr // 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 24
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return arr // 3_600_000_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return arr // 3_600_000_000_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return arr // 3_600_000_000_000_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'ndarray[datetime64]'> "
        "time unit to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def dt64array_to_minutes(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[datetime64] to total minutes `<'ndarray[int64]'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'minute (m)'.
    """
    # Validate array
    validate_dt64array(arr)

    # Get time unit
    unit: np.NPY_DATETIMEUNIT
    ndim: cython.Py_ssize_t = arr.ndim
    if ndim == 1:
        if arr.shape[0] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0])
    elif ndim == 2:
        if arr.shape[1] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0])
    elif ndim == 3:
        if arr.shape[2] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0])
    elif ndim == 4:
        if arr.shape[3] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0, 0])
    else:
        raise ValueError(
            "Only support <'ndarray'> with up to 4 "
            "dimensions, instead got: %d." % ndim
        )

    # Conversion
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // 60_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // 60_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr // 60_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr // 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 1_440
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return arr // 60_000_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return arr // 60_000_000_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return arr // 60_000_000_000_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'ndarray[datetime64]'> "
        "time unit to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def dt64array_to_seconds(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[datetime64] to total seconds `<'ndarray[int64]'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'second (s)'.
    """
    # Validate array
    validate_dt64array(arr)

    # Get time unit
    unit: np.NPY_DATETIMEUNIT
    ndim: cython.Py_ssize_t = arr.ndim
    if ndim == 1:
        if arr.shape[0] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0])
    elif ndim == 2:
        if arr.shape[1] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0])
    elif ndim == 3:
        if arr.shape[2] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0])
    elif ndim == 4:
        if arr.shape[3] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0, 0])
    else:
        raise ValueError(
            "Only support <'ndarray'> with up to 4 "
            "dimensions, instead got: %d." % ndim
        )

    # Conversion
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr * 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * 3_600
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 86_400
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return arr // 1_000_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return arr // 1_000_000_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return arr // 1_000_000_000_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'ndarray[datetime64]'> "
        "time unit to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def dt64array_to_milliseconds(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[datetime64] to total milliseconds `<'ndarray[int64]'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'millisecond (ms)'.
    """
    # Validate array
    validate_dt64array(arr)

    # Get time unit
    unit: np.NPY_DATETIMEUNIT
    ndim: cython.Py_ssize_t = arr.ndim
    if ndim == 1:
        if arr.shape[0] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0])
    elif ndim == 2:
        if arr.shape[1] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0])
    elif ndim == 3:
        if arr.shape[2] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0])
    elif ndim == 4:
        if arr.shape[3] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0, 0])
    else:
        raise ValueError(
            "Only support <'ndarray'> with up to 4 "
            "dimensions, instead got: %d." % ndim
        )

    # Conversion
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr * 60_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * 3_600_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 86_400_000
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return arr // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return arr // 1_000_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return arr // 1_000_000_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'ndarray[datetime64]'> "
        "time unit to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def dt64array_to_microseconds(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[datetime64] to total microseconds `<'ndarray[int64]'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'microsecond (us)'.
    """
    # Validate array
    validate_dt64array(arr)

    # Get time unit
    unit: np.NPY_DATETIMEUNIT
    ndim: cython.Py_ssize_t = arr.ndim
    if ndim == 1:
        if arr.shape[0] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0])
    elif ndim == 2:
        if arr.shape[1] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0])
    elif ndim == 3:
        if arr.shape[2] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0])
    elif ndim == 4:
        if arr.shape[3] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0, 0])
    else:
        raise ValueError(
            "Only support <'ndarray'> with up to 4 "
            "dimensions, instead got: %d." % ndim
        )

    # Conversion
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr * 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr * 60_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * 3_600_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 86_400_000_000
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return arr // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return arr // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return arr // 1_000_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'ndarray[datetime64]'> "
        "time unit to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def dt64array_to_nanoseconds(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[datetime64] to total nanoseconds `<'ndarray[int64]'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'nanosecond (ns)'.
    """
    # Validate array
    validate_dt64array(arr)

    # Get time unit
    unit: np.NPY_DATETIMEUNIT
    ndim: cython.Py_ssize_t = arr.ndim
    if ndim == 1:
        if arr.shape[0] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0])
    elif ndim == 2:
        if arr.shape[1] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0])
    elif ndim == 3:
        if arr.shape[2] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0])
    elif ndim == 4:
        if arr.shape[3] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0, 0])
    else:
        raise ValueError(
            "Only support <'ndarray'> with up to 4 "
            "dimensions, instead got: %d." % ndim
        )

    # Conversion
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr * 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr * 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr * 60_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * 3_600_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 86_400_000_000_000
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return arr // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return arr // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return arr // 1_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'ndarray[datetime64]'> "
        "time unit to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def dt64array_to_ordinals(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[datetime64] to ordinal `<'ndarray[int64]'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'day (D)'.
    """
    return dt64array_to_days(arr) + 719_163


@cython.cfunc
@cython.inline(True)
def dt64array_to_timestamps(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[datetime64] to timestamp `<ndarray[float64]>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'microseconds (us)'.
    """
    return dt64array_to_microseconds(arr) / 1_000_000


# numpy.ndarray[timedelta64] ===========================================================================
# numpy.ndarray[timedelta64]: check types --------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_td64array(arr: np.ndarray) -> cython.bint:
    """Check if a ndarray is dtype of timedelta64 `<'bool'>`.
    Equivalent to `isinstance(arr.dtype, np.dtypes.TimeDelta64DType)`.
    """
    return np.PyArray_TYPE(arr) == np.NPY_TYPES.NPY_TIMEDELTA


@cython.cfunc
@cython.inline(True)
def validate_td64array(arr: np.ndarray):
    """Validate if a ndarray is dtype of timedelta64.
    Raise `TypeError` if dtype is incorrect."""
    if not is_td64array(arr):
        raise TypeError(
            "Expects <'ndarray[timedelta64]'>, "
            "instead got: <'ndarray[%s]'>." % (arr.dtype)
        )


@cython.cfunc
@cython.inline(True)
def get_td64array_unit(arr: np.ndarray) -> str:
    """Get the time unit of the `<'ndarray[timedelta64]'>`,
    such as: 'D', 's', 'ms', 'us', 'ns', etc `<'str'>`."""
    # Validate array
    validate_td64array(arr)

    # Get time unit
    unit: np.NPY_DATETIMEUNIT
    ndim: cython.Py_ssize_t = arr.ndim
    if ndim == 1:
        if arr.shape[0] == 0:
            return parse_td64array_unit(arr)
        unit = np.get_datetime64_unit(arr[0])
    elif ndim == 2:
        if arr.shape[1] == 0:
            return parse_td64array_unit(arr)
        unit = np.get_datetime64_unit(arr[0, 0])
    elif ndim == 3:
        if arr.shape[2] == 0:
            return parse_td64array_unit(arr)
        unit = np.get_datetime64_unit(arr[0, 0, 0])
    elif ndim == 4:
        if arr.shape[3] == 0:
            return parse_td64array_unit(arr)
        unit = np.get_datetime64_unit(arr[0, 0, 0, 0])
    else:
        raise ValueError(
            "Only support <'ndarray'> with up to 4 "
            "dimensions, instead got: %d." % ndim
        )

    # Match unit: common
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
    # Match unit: uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return "ps"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return "fs"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return "as"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:
        return "Y"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:
        return "M"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:
        return "W"
    # if unit == np.NPY_DATETIMEUNIT.NPY_FR_B:
    #     return "B"
    raise ValueError("Unknown <'ndarray[timedelta64]'> time unit: %d." % unit)


@cython.cfunc
@cython.inline(True)
def parse_td64array_unit(arr: np.ndarray) -> str:
    """Prase the time unit of the `<'ndarray[timedelta64]'>`
    from 'ndarray.dtype.str' `<'str'>`."""
    dtype_str = arr.dtype.str
    unit: str = str_substr(dtype_str, 4, str_len(dtype_str) - 1)
    if not 1 <= str_len(unit) <= 2:
        raise ValueError(
            "Unable to parse <'ndarray[timedelta64]'> "
            "time unit from: '%s'." % dtype_str
        )
    return unit


# numpy.ndarray[timedelta64]: conversion ---------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def td64array_to_int(
    arr: np.ndarray,
    unit: Literal["D", "h", "m", "s", "ms", "us", "ns"],
) -> np.ndarray:
    """Convert ndarray[timedelta64] to integer based
    on the target time unit `<'ndarray[int64]'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than the target time 'unit'.
    """
    if unit == "ns":
        return td64array_to_nanoseconds(arr)
    elif unit == "us":
        return td64array_to_microseconds(arr)
    elif unit == "ms":
        return td64array_to_milliseconds(arr)
    elif unit == "s":
        return td64array_to_seconds(arr)
    elif unit == "m":
        return td64array_to_minutes(arr)
    elif unit == "h":
        return td64array_to_hours(arr)
    elif unit == "D" or unit == "d":
        return td64array_to_days(arr)
    else:
        raise ValueError(
            "Does not support conversion of "
            "<'ndarray[timedelta64]'> to time unit: %s." % repr(unit)
        )


@cython.cfunc
@cython.inline(True)
def td64array_to_days(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[timedelta64] to total days `<'ndarray[int64]'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'day (D)'.
    """
    # Validate array
    validate_td64array(arr)

    # Get time unit
    unit: np.NPY_DATETIMEUNIT
    ndim: cython.Py_ssize_t = arr.ndim
    if ndim == 1:
        if arr.shape[0] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0])
    elif ndim == 2:
        if arr.shape[1] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0])
    elif ndim == 3:
        if arr.shape[2] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0])
    elif ndim == 4:
        if arr.shape[3] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0, 0])
    else:
        raise ValueError(
            "Only support <'ndarray'> with up to 4 "
            "dimensions, instead got: %d." % ndim
        )

    # Conversion
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // 86_400_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // 86_400_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr // 86_400_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr // 86_400
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr // 1_440
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr // 24
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return arr // 86_400_000_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return arr // 86_400_000_000_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return arr // 86_400_000_000_000_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'ndarray[timedelta64]'> "
        "time unit to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def td64array_to_hours(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[timedelta64] to total hours `<'ndarray[int64]'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'hour (h)'.
    """
    # Validate array
    validate_td64array(arr)

    # Get time unit
    unit: np.NPY_DATETIMEUNIT
    ndim: cython.Py_ssize_t = arr.ndim
    if ndim == 1:
        if arr.shape[0] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0])
    elif ndim == 2:
        if arr.shape[1] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0])
    elif ndim == 3:
        if arr.shape[2] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0])
    elif ndim == 4:
        if arr.shape[3] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0, 0])
    else:
        raise ValueError(
            "Only support <'ndarray'> with up to 4 "
            "dimensions, instead got: %d." % ndim
        )

    # Conversion
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // 3_600_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // 3_600_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr // 3_600_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr // 3_600
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr // 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 24
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return arr // 3_600_000_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return arr // 3_600_000_000_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return arr // 3_600_000_000_000_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'ndarray[timedelta64]'> "
        "time unit to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def td64array_to_minutes(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[timedelta64] to total minutes `<'ndarray[int64]'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'minute (m)'.
    """
    # Validate array
    validate_td64array(arr)

    # Get time unit
    unit: np.NPY_DATETIMEUNIT
    ndim: cython.Py_ssize_t = arr.ndim
    if ndim == 1:
        if arr.shape[0] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0])
    elif ndim == 2:
        if arr.shape[1] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0])
    elif ndim == 3:
        if arr.shape[2] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0])
    elif ndim == 4:
        if arr.shape[3] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0, 0])
    else:
        raise ValueError(
            "Only support <'ndarray'> with up to 4 "
            "dimensions, instead got: %d." % ndim
        )

    # Conversion
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // 60_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // 60_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr // 60_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr // 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 1_440
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return arr // 60_000_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return arr // 60_000_000_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return arr // 60_000_000_000_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'ndarray[timedelta64]'> "
        "time unit to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def td64array_to_seconds(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[timedelta64] to total seconds `<'ndarray[int64]'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'second (s)'.
    """
    # Validate array
    validate_td64array(arr)

    # Get time unit
    unit: np.NPY_DATETIMEUNIT
    ndim: cython.Py_ssize_t = arr.ndim
    if ndim == 1:
        if arr.shape[0] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0])
    elif ndim == 2:
        if arr.shape[1] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0])
    elif ndim == 3:
        if arr.shape[2] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0])
    elif ndim == 4:
        if arr.shape[3] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0, 0])
    else:
        raise ValueError(
            "Only support <'ndarray'> with up to 4 "
            "dimensions, instead got: %d." % ndim
        )

    # Conversion
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr * 60
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * 3_600
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 86_400
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return arr // 1_000_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return arr // 1_000_000_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return arr // 1_000_000_000_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'ndarray[timedelta64]'> "
        "time unit to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def td64array_to_milliseconds(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[timedelta64] to total milliseconds `<'ndarray[int64]'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'millisecond (ms)'.
    """
    # Validate array
    validate_td64array(arr)

    # Get time unit
    unit: np.NPY_DATETIMEUNIT
    ndim: cython.Py_ssize_t = arr.ndim
    if ndim == 1:
        if arr.shape[0] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0])
    elif ndim == 2:
        if arr.shape[1] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0])
    elif ndim == 3:
        if arr.shape[2] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0])
    elif ndim == 4:
        if arr.shape[3] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0, 0])
    else:
        raise ValueError(
            "Only support <'ndarray'> with up to 4 "
            "dimensions, instead got: %d." % ndim
        )

    # Conversion
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr * 60_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * 3_600_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 86_400_000
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return arr // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return arr // 1_000_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return arr // 1_000_000_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'ndarray[timedelta64]'> "
        "time unit to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def td64array_to_microseconds(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[timedelta64] to total microseconds `<'ndarray[int64]'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'microsecond (us)'.
    """
    # Validate array
    validate_td64array(arr)

    # Get time unit
    unit: np.NPY_DATETIMEUNIT
    ndim: cython.Py_ssize_t = arr.ndim
    if ndim == 1:
        if arr.shape[0] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0])
    elif ndim == 2:
        if arr.shape[1] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0])
    elif ndim == 3:
        if arr.shape[2] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0])
    elif ndim == 4:
        if arr.shape[3] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0, 0])
    else:
        raise ValueError(
            "Only support <'ndarray'> with up to 4 "
            "dimensions, instead got: %d." % ndim
        )

    # Conversion
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr * 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr * 60_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * 3_600_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 86_400_000_000
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return arr // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return arr // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return arr // 1_000_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'ndarray[timedelta64]'> "
        "time unit to perform conversion: %d." % unit
    )


@cython.cfunc
@cython.inline(True)
def td64array_to_nanoseconds(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[timedelta64] to total nanoseconds `<'ndarray[int64]'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'nanosecond (ns)'.
    """
    # Validate array
    validate_td64array(arr)

    # Get time unit
    unit: np.NPY_DATETIMEUNIT
    ndim: cython.Py_ssize_t = arr.ndim
    if ndim == 1:
        if arr.shape[0] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0])
    elif ndim == 2:
        if arr.shape[1] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0])
    elif ndim == 3:
        if arr.shape[2] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0])
    elif ndim == 4:
        if arr.shape[3] == 0:
            return arr
        unit = np.get_datetime64_unit(arr[0, 0, 0, 0])
    else:
        raise ValueError(
            "Only support <'ndarray'> with up to 4 "
            "dimensions, instead got: %d." % ndim
        )

    # Conversion
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    # . common
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr * 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr * 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr * 60_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * 3_600_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 86_400_000_000_000
    # . uncommon
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return arr // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return arr // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return arr // 1_000_000_000
    # . unsupported
    raise ValueError(
        "Unsupported <'ndarray[timedelta64]'> "
        "time unit to perform conversion: %d." % unit
    )


# pandas.Series[datetime64] ============================================================================
# pandas.Series[datetime64] check types ----------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_dt64series(obj: object) -> cython.bint:
    """Check if an object is dtype of <'Series[datetime64]'> `<'bool'>`.

    Equivalent to `isinstance(s.dtype, (np.dtypes.DateTime64DType, pandas.DatetimeTZDtype))`.
    """
    try:
        return isinstance(obj, typeref.SERIES) and isinstance(
            obj.dtype, (typeref.DT64_ARRAY, typeref.DT64TZ_ARRAY)
        )
    except Exception:
        return False


@cython.cfunc
@cython.inline(True)
def validate_dt64series(obj: object):
    """Validate if an object is dtype of <'Series[datetime64]'>.
    Raise `TypeError` if dtype is incorrect."""
    if not is_dt64series(obj):
        raise TypeError(
            "Expects <'Series[datetime64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (obj.dtype)
                if isinstance(obj, typeref.SERIES)
                else type(obj)
            )
        )


@cython.cfunc
@cython.inline(True)
def get_dt64series_arr(s: Series) -> np.ndarray:
    """Get `<'Series[datetime64]'>` underlying data `<'ndarray'>`."""
    try:
        arr: np.ndarray = s.values
    except Exception as err:
        raise TypeError(
            "Expects <'Series[datetime64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err
    return arr


@cython.cfunc
@cython.inline(True)
def get_dt64series_unit(s: Series) -> str:
    """Get the time unit of the `<'Series[datetime64]'>`,
    such as: 'D', 's', 'ms', 'us', 'ns', etc `<'str'>`."""
    arr: np.ndarray = get_dt64series_arr(s)
    try:
        return get_dt64array_unit(arr)
    except TypeError as err:
        raise TypeError(
            "Expects <'Series[datetime64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err


# pandas.Series[datetime64] conversion -----------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def dt64series_to_int(
    s: Series,
    unit: Literal["D", "h", "m", "s", "ms", "us", "ns"],
) -> object:
    """Convert Series[datetime64] to integer based
    on the target time unit `<'Series[int64]'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than the target time 'unit'.
    """
    if unit == "ns":
        return dt64series_to_nanoseconds(s)
    elif unit == "us":
        return dt64series_to_microseconds(s)
    elif unit == "ms":
        return dt64series_to_milliseconds(s)
    elif unit == "s":
        return dt64series_to_seconds(s)
    elif unit == "m":
        return dt64series_to_minutes(s)
    elif unit == "h":
        return dt64series_to_hours(s)
    elif unit == "D" or unit == "d":
        return dt64series_to_days(s)
    else:
        raise ValueError(
            "Does not support conversion of "
            "<'Series[datetime64]'> to time unit: %s." % repr(unit)
        )


@cython.cfunc
@cython.inline(True)
def dt64series_to_days(s: Series) -> object:
    """Convert Series[datetime64] to total days `<'Series[int64]'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'day (D)'.
    """
    arr: np.ndarray = get_dt64series_arr(s)
    try:
        arr = dt64array_to_days(arr)
    except TypeError as err:
        raise TypeError(
            "Expects <'Series[datetime64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err
    return typeref.SERIES(arr, index=s.index, name=s.name)


@cython.cfunc
@cython.inline(True)
def dt64series_to_hours(s: Series) -> object:
    """Convert Series[datetime64] to total hours `<'Series[int64]'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'hour (H)'.
    """
    arr: np.ndarray = get_dt64series_arr(s)
    try:
        arr = dt64array_to_hours(arr)
    except TypeError as err:
        raise TypeError(
            "Expects <'Series[datetime64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err
    return typeref.SERIES(arr, index=s.index, name=s.name)


@cython.cfunc
@cython.inline(True)
def dt64series_to_minutes(s: Series) -> object:
    """Convert Series[datetime64] to total minutes `<'Series[int64]'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'minute (m)'.
    """
    arr: np.ndarray = get_dt64series_arr(s)
    try:
        arr = dt64array_to_minutes(arr)
    except TypeError as err:
        raise TypeError(
            "Expects <'Series[datetime64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err
    return typeref.SERIES(arr, index=s.index, name=s.name)


@cython.cfunc
@cython.inline(True)
def dt64series_to_seconds(s: Series) -> object:
    """Convert Series[datetime64] to total seconds `<'Series[int64]'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'second (s)'.
    """
    arr: np.ndarray = get_dt64series_arr(s)
    try:
        arr = dt64array_to_seconds(arr)
    except TypeError as err:
        raise TypeError(
            "Expects <'Series[datetime64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err
    return typeref.SERIES(arr, index=s.index, name=s.name)


@cython.cfunc
@cython.inline(True)
def dt64series_to_milliseconds(s: Series) -> object:
    """Convert Series[datetime64] to total milliseconds `<'Series[int64]'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'millisecond (ms)'.
    """
    arr: np.ndarray = get_dt64series_arr(s)
    try:
        arr = dt64array_to_milliseconds(arr)
    except TypeError as err:
        raise TypeError(
            "Expects <'Series[datetime64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err
    return typeref.SERIES(arr, index=s.index, name=s.name)


@cython.cfunc
@cython.inline(True)
def dt64series_to_microseconds(s: Series) -> object:
    """Convert Series[datetime64] to total microseconds `<'Series[int64]'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'microsecond (us)'.
    """
    arr: np.ndarray = get_dt64series_arr(s)
    try:
        arr = dt64array_to_microseconds(arr)
    except TypeError as err:
        raise TypeError(
            "Expects <'Series[datetime64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err
    return typeref.SERIES(arr, index=s.index, name=s.name)


@cython.cfunc
@cython.inline(True)
def dt64series_to_nanoseconds(s: Series) -> object:
    """Convert Series[datetime64] to total nanoseconds `<'Series[int64]'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'nanosecond (ns)'.
    """
    arr: np.ndarray = get_dt64series_arr(s)
    try:
        arr = dt64array_to_nanoseconds(arr)
    except TypeError as err:
        raise TypeError(
            "Expects <'Series[datetime64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err
    return typeref.SERIES(arr, index=s.index, name=s.name)


@cython.cfunc
@cython.inline(True)
def dt64series_to_ordinals(s: Series) -> object:
    """Convert Series[datetime64] to ordinals `<'Series[int64]'>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'day (D)'.
    """
    arr: np.ndarray = get_dt64series_arr(s)
    try:
        arr = dt64array_to_ordinals(arr)
    except TypeError as err:
        raise TypeError(
            "Expects <'Series[datetime64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err
    return typeref.SERIES(arr, index=s.index, name=s.name)


@cython.cfunc
@cython.inline(True)
def dt64series_to_timestamps(s: Series) -> object:
    """Convert Series[datetime64] to timestamps `<Series[float64]>`.

    ### Notice
    Percision will be lost if the datetime64
    unit is smaller than 'microseconds (us)'.
    """
    arr: np.ndarray = get_dt64series_arr(s)
    try:
        arr = dt64array_to_timestamps(arr)
    except TypeError as err:
        raise TypeError(
            "Expects <'Series[datetime64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err
    return typeref.SERIES(arr, index=s.index, name=s.name)


# pandas.Series[timedelta64] ============================================================================
# pandas.Series[timedelta64]: check types ---------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_td64series(obj: Series) -> cython.bint:
    """Check if an object is dtype of <'Series[timedelta64]'> `<'bool'>`.
    Equivalent to `isinstance(s.dtype, np.dtypes.TimeDelta64DType)`."""
    try:
        return isinstance(obj, typeref.SERIES) and isinstance(
            obj.dtype, typeref.TD64_ARRAY
        )
    except Exception:
        return False


@cython.cfunc
@cython.inline(True)
def validate_td64series(obj: Series):
    """Validate if an object is dtype of <'Series[timedelta64]'>.
    Raise `TypeError` if dtype is incorrect."""
    if not is_td64series(obj):
        raise TypeError(
            "Expects <'Series[timedelta64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (obj.dtype)
                if isinstance(obj, typeref.SERIES)
                else type(obj)
            )
        )


@cython.cfunc
@cython.inline(True)
def get_td64series_arr(s: Series) -> np.ndarray:
    """Get `<'Series[timedelta64]'>` underlying data `<'ndarray'>`."""
    try:
        arr: np.ndarray = s.values
    except Exception as err:
        raise TypeError(
            "Expects <'Series[timedelta64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err
    return arr


@cython.cfunc
@cython.inline(True)
def get_td64series_unit(s: Series) -> str:
    """Get the time unit of the `<'Series[timedelta64]'>`,
    such as: 'D', 's', 'ms', 'us', 'ns', etc `<'str'>`."""
    arr: np.ndarray = get_td64series_arr(s)
    try:
        return get_td64array_unit(arr)
    except TypeError as err:
        raise TypeError(
            "Expects <'Series[timedelta64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err


# pandas.Series[timedelta64]: conversion ----------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def td64series_to_int(
    s: Series,
    unit: Literal["D", "h", "m", "s", "ms", "us", "ns"],
) -> object:
    """Convert Series[timedelta64] to integer based
    on the target time unit `<'Series[int64]'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than the target time 'unit'.
    """
    if unit == "ns":
        return td64series_to_nanoseconds(s)
    elif unit == "us":
        return td64series_to_microseconds(s)
    elif unit == "ms":
        return td64series_to_milliseconds(s)
    elif unit == "s":
        return td64series_to_seconds(s)
    elif unit == "m":
        return td64series_to_minutes(s)
    elif unit == "h":
        return td64series_to_hours(s)
    elif unit == "D" or unit == "d":
        return td64series_to_days(s)
    else:
        raise ValueError(
            "Does not support conversion of "
            "<'Series[timedelta64]'> to time unit: %s." % repr(unit)
        )


@cython.cfunc
@cython.inline(True)
def td64series_to_days(s: Series) -> object:
    """Convert Series[timedelta64] to total days `<'Series[int64]'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'day (D)'.
    """
    arr: np.ndarray = get_dt64series_arr(s)
    try:
        arr = td64array_to_days(arr)
    except TypeError as err:
        raise TypeError(
            "Expects <'Series[timedelta64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err
    return typeref.SERIES(arr, index=s.index, name=s.name)


@cython.cfunc
@cython.inline(True)
def td64series_to_hours(s: Series) -> object:
    """Convert Series[timedelta64] to total hours `<'Series[int64]'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'hour (h)'.
    """
    arr: np.ndarray = get_dt64series_arr(s)
    try:
        arr = td64array_to_hours(arr)
    except TypeError as err:
        raise TypeError(
            "Expects <'Series[timedelta64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err
    return typeref.SERIES(arr, index=s.index, name=s.name)


@cython.cfunc
@cython.inline(True)
def td64series_to_minutes(s: Series) -> object:
    """Convert Series[timedelta64] to total minutes `<'Series[int64]'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'minute (m)'.
    """
    arr: np.ndarray = get_dt64series_arr(s)
    try:
        arr = td64array_to_minutes(arr)
    except TypeError as err:
        raise TypeError(
            "Expects <'Series[timedelta64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err
    return typeref.SERIES(arr, index=s.index, name=s.name)


@cython.cfunc
@cython.inline(True)
def td64series_to_seconds(s: Series) -> object:
    """Convert Series[timedelta64] to total seconds `<'Series[int64]'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'second (s)'.
    """
    arr: np.ndarray = get_dt64series_arr(s)
    try:
        arr = td64array_to_seconds(arr)
    except TypeError as err:
        raise TypeError(
            "Expects <'Series[timedelta64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err
    return typeref.SERIES(arr, index=s.index, name=s.name)


@cython.cfunc
@cython.inline(True)
def td64series_to_milliseconds(s: Series) -> object:
    """Convert Series[timedelta64] to total milliseconds `<'Series[int64]'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'millisecond (ms)'.
    """
    arr: np.ndarray = get_dt64series_arr(s)
    try:
        arr = td64array_to_milliseconds(arr)
    except TypeError as err:
        raise TypeError(
            "Expects <'Series[timedelta64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err
    return typeref.SERIES(arr, index=s.index, name=s.name)


@cython.cfunc
@cython.inline(True)
def td64series_to_microseconds(s: Series) -> object:
    """Convert Series[timedelta64] to total microseconds `<'Series[int64]'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'microsecond (us)'.
    """
    arr: np.ndarray = get_dt64series_arr(s)
    try:
        arr = td64array_to_microseconds(arr)
    except TypeError as err:
        raise TypeError(
            "Expects <'Series[timedelta64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err
    return typeref.SERIES(arr, index=s.index, name=s.name)


@cython.cfunc
@cython.inline(True)
def td64series_to_nanoseconds(s: Series) -> object:
    """Convert Series[timedelta64] to total nanoseconds `<'Series[int64]'>`.

    ### Notice
    Percision will be lost if the timedelta64
    unit is smaller than 'nanosecond (ns)'.
    """
    arr: np.ndarray = get_dt64series_arr(s)
    try:
        arr = td64array_to_nanoseconds(arr)
    except TypeError as err:
        raise TypeError(
            "Expects <'Series[timedelta64]'>, instead got: %s."
            % (
                "<'Series[%s]'>" % (s.dtype)
                if isinstance(s, typeref.SERIES)
                else type(s)
            )
        ) from err
    return typeref.SERIES(arr, index=s.index, name=s.name)


# pandas.Series[timedelta64]: adjustment ----------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def td64series_adjust_unit(s: Series, unit: Literal["s", "ms", "us", "ns"]) -> object:
    """Adjust Series[timedelta64] to the target time 'unit' `<'Series[timedelta64]'>`."""
    # Validate target unit
    if not set_contains(TD_UNIT_ADJUSTMENT, unit):
        raise ValueError(
            "Cannot adjust <'Series[timedelta64]'> to unit: %s. "
            "Supported units are <'str'>: ['s', 'ms', 'us', 'ns']." % repr(unit)
        )
    # Already is target unit
    if get_td64series_unit(s) == unit:
        return s
    # Convert to target unit
    if isinstance(s, typeref.TIMEDELTAINDEX):
        return s.as_unit(unit)
    else:
        return s.dt.as_unit(unit)
