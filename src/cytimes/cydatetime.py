# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cpython.time import time as unix_time  # type: ignore
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cytimes import cytime  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
from typing import Literal
import datetime, numpy as np
from time import localtime as time_localtime
from pandas import Series, DatetimeIndex, TimedeltaIndex

# Constants --------------------------------------------------------------------------------------------
# fmt: off
DAYS_BR_MONTH: cython.uint[13] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
DAYS_IN_MONTH: cython.uint[13] = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
DAYS_BR_QUARTER: cython.uint[5] = [0, 90, 181, 273, 365]
DAYS_IN_QUARTER: cython.uint[5] = [0, 90, 91, 92, 92]
# fmt: on
# . datetime EPOCH
UTC: datetime.tzinfo = datetime.get_utc()
EPOCH_NAI: datetime.datetime = datetime.datetime_new(1970, 1, 1, 0, 0, 0, 0, None, 0)  # type: ignore
EPOCH_UTC: datetime.datetime = datetime.datetime_new(1970, 1, 1, 0, 0, 0, 0, UTC, 0)  # type: ignore
EPOCH_US: cython.longlong = 62_135_683_200_000_000
EPOCH_SEC: cython.longlong = 62_135_683_200
EPOCH_DAY: cython.int = 719_163
# . max & min datetime seconds
DT_MIN_US: cython.longlong = 86_400_000_000
DT_MAX_US: cython.longlong = 315_537_983_999_999_999
# . nanoseconds conversion
NS_DAY: cython.longlong = 864_00_000_000_000
NS_HOUR: cython.longlong = 36_00_000_000_000
NS_MINUTE: cython.longlong = 60_000_000_000
# . microseconds conversion
US_DAY: cython.longlong = 86_400_000_000
US_HOUR: cython.longlong = 3_600_000_000
# . numpy dtype
DT64ARRAY_DTYPE: object = np.dtypes.DateTime64DType
DELTA64ARRAY_DTYPE: object = np.dtypes.TimeDelta64DType
PDSERIES_DTYPE: object = Series


# Struct -----------------------------------------------------------------------------------------------
ymd = cython.struct(
    year=cython.uint,
    month=cython.uint,
    day=cython.uint,
)
hms = cython.struct(
    hour=cython.uint,
    minute=cython.uint,
    second=cython.uint,
    microsecond=cython.uint,
)
iso = cython.struct(
    year=cython.uint,
    week=cython.uint,
    weekday=cython.uint,
)


# Calendar =============================================================================================
# Calender: year ---------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_leapyear(year: cython.uint) -> cython.bint:
    """Whether the given year is a leap year `<bool>`."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def leap_bt_years(year1: cython.uint, year2: cython.uint) -> cython.uint:
    """Get the number of leap years between
    year1 and year2 `<int>`."""
    y1: cython.int
    y2: cython.int
    if year1 <= year2:
        y1, y2 = year1 - 1, year2 - 1
    else:
        y1, y2 = year2 - 1, year1 - 1
    return (y2 // 4 - y1 // 4) - (y2 // 100 - y1 // 100) + (y2 // 400 - y1 // 400)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_in_year(year: cython.uint) -> cython.uint:
    """Get the maximum number of days in the year.
    Expect 365 or 366 (leapyear) `<int>`."""
    return 366 if is_leapyear(year) else 365


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_bf_year(year: cython.uint) -> cython.uint:
    """Get the number of days betweem the 1st day
    of 1AD and the 1st day of the given year `<int>`."""
    if year <= 1:
        return 0
    y: cython.int = year - 1
    return y * 365 + y // 4 - y // 100 + y // 400


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_of_year(
    year: cython.uint,
    month: cython.uint,
    day: cython.uint,
) -> cython.uint:
    """Get the number of days between the 1st day of the
    year and the given year, month, and day `<int>`."""
    return days_bf_month(year, month) + min(day, days_in_month(year, month))


# Calendar: quarter ------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def quarter_of_month(month: cython.uint) -> cython.uint:
    """Get the quarter of the month
    Expect 1, 2, 3 and 4 `<int>`."""
    if not 1 <= month <= 12:
        raise ValueError(
            "Expect 'month' value between 1 and 12, instead got: %s" % month
        )
    return (month - 1) // 3 + 1


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_in_quarter(year: cython.uint, month: cython.uint) -> cython.uint:
    """Get the maximum number of days
    in the quarter `<int>`."""
    quarter: cython.uint = quarter_of_month(month)
    days: cython.uint = DAYS_IN_QUARTER[quarter]
    if quarter == 1 and is_leapyear(year):
        days += 1
    return days


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_bf_quarter(year: cython.uint, month: cython.uint) -> cython.uint:
    """Get the number of days between the 1st day of
    the year and the 1st day of the quarter `<int>."""
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
    """Get the number of days between the 1st day of the
    quarter and the given year, month, and day `<int>`."""
    return days_of_year(year, month, day) - days_bf_quarter(year, month)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def quarter_1st_month(month: cython.uint) -> cython.uint:
    """Get the first month of the quarter.
    Expect 1, 4, 7, 10 `<int>`."""
    return 3 * quarter_of_month(month) - 2


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def quarter_lst_month(month: cython.uint) -> cython.uint:
    """Get the last month of the quarter.
    Expect 3, 6, 9, 12 `<int>`."""
    return 3 * quarter_of_month(month)


# Calendar: month --------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_in_month(year: cython.uint, month: cython.uint) -> cython.uint:
    """Get the maximum number of days in the month `<int>`."""
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
    """Get the number of days between the 1st day of the
    year and the 1st day of the month `<int>`."""
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
    """Get the day of the week, where
    Monday is 0 ... Sunday is 6 `<int>`."""
    return (ymd_to_ordinal(year, month, day) + 6) % 7


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def ymd_isoweekday(
    year: cython.uint,
    month: cython.uint,
    day: cython.uint,
) -> cython.uint:
    """Get the ISO calendar day of the week, where
    Monday is 1 ... Sunday is 7 `<int>`."""
    return ymd_weekday(year, month, day) + 1


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def ymd_isoweek(year: cython.uint, month: cython.uint, day: cython.uint) -> cython.uint:
    """Get the ISO calendar week number `<int>`."""
    ordinal: cython.uint = ymd_to_ordinal(year, month, day)
    iso1st_ord: cython.uint = iso1st_ordinal(year)
    delta: cython.int = ordinal - iso1st_ord
    isoweek: cython.int = delta // 7
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
    """Get the ISO calendar year `<int>`."""
    ordinal: cython.uint = ymd_to_ordinal(year, month, day)
    iso1st_ord: cython.uint = iso1st_ordinal(year)
    delta: cython.int = ordinal - iso1st_ord
    isoweek: cython.int = delta // 7
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
) -> iso:
    """Get the ISO calendar of the YMD `<struct:iso>`."""
    ordinal: cython.uint = ymd_to_ordinal(year, month, day)
    iso1st_ord: cython.uint = iso1st_ordinal(year)
    delta: cython.int = ordinal - iso1st_ord
    isoweek: cython.int = delta // 7
    if isoweek < 0:
        year -= 1
        iso1st_ord = iso1st_ordinal(year)
        delta = ordinal - iso1st_ord
        isoweek = delta // 7
    elif isoweek >= 52 and ordinal >= iso1st_ordinal(year + 1):
        year += 1
        isoweek = 0
    return iso(year, isoweek + 1, delta % 7 + 1)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def iso1st_ordinal(year: cython.uint) -> cython.uint:
    """Get the ordinal of the 1st day of ISO calendar year `<int>`."""
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
    """Convert year, month, day to ordinal `<int>`."""
    return (
        days_bf_year(year)
        + days_bf_month(year, month)
        + min(day, days_in_month(year, month))
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def ordinal_to_ymd(ordinal: cython.int) -> ymd:
    """Convert ordinal to YMD `<struct:ymd>`."""
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
        return ymd(year - 1, 12, 31)

    # Now the year is correct, and n is the offset from January 1.  We find
    # the month via an estimate that's either exact or one too large.
    month: cython.uint = (n + 50) >> 5
    days_bf: cython.uint = days_bf_month(year, month)
    if days_bf > n:
        month -= 1
        days_bf = days_bf_month(year, month)
    n = n - days_bf + 1

    # Return ymd
    return ymd(year, month, n)


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
@cython.exceptval(check=False)
def microseconds_to_hms(microseconds: cython.longlong) -> hms:
    """Convert microseconds to HMS `<struct:hms>`."""
    hour: cython.uint
    minute: cython.uint
    second: cython.uint
    microsecond: cython.uint

    # Convert
    if microseconds > 0:
        microseconds = microseconds % US_DAY
        hour = microseconds // US_HOUR
        microseconds = microseconds % US_HOUR
        minute = microseconds // 60_000_000
        microseconds = microseconds % 60_000_000
        second = microseconds // 1_000_000
        microsecond = microseconds % 1_000_000
    else:
        hour, minute, second, microsecond = 0, 0, 0, 0

    # Return hms
    return hms(hour, minute, second, microsecond)


# Time =================================================================================================
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time() -> cython.double:
    """Get the current time in seconds since the
    Epoch `<float>`. Equivalent to `time.time()`."""
    return unix_time()


@cython.cfunc
@cython.inline(True)
def localtime() -> cytime.tm:
    """Get current local time as a `<struct_time>`.
    Equivalent to `time.localtime()`."""
    return cytime.localtime()


@cython.cfunc
@cython.inline(True)
def localize_time(timestamp: cython.double) -> cytime.tm:
    """Convert timestamp (seconds since the Epoch)
    to a `<struct_time>` expressed in local time.
    Equivalent to `time.localtime(timestamp)`."""
    return cytime.localize_time(timestamp)


@cython.cfunc
@cython.inline(True)
def localize_ts(timestamp: cython.double) -> cython.longlong:
    """Convert timestamp (seconds since the Epoch) to a
    local timestamp (adjusted by timezone offset) `<int>`."""
    # Localize timestamp
    tms = cytime.localize_time(timestamp)
    # Calculate total seconds
    sec: cython.longlong = (
        ymd_to_ordinal(tms.tm_year, tms.tm_mon, tms.tm_mday) * 86_400
        + tms.tm_hour * 3_600
        + tms.tm_min * 60
        + tms.tm_sec
    )
    # Return seconds since epoch
    return sec - EPOCH_SEC


# Datetime.date ========================================================================================
# Date: generate ---------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def gen_date(
    year: cython.uint = 1,
    month: cython.uint = 1,
    day: cython.uint = 1,
) -> datetime.date:
    """Generate a new `<datetime.date>`."""
    return datetime.date_new(year, month, day)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def gen_date_now() -> datetime.date:
    """Generate the current local '<datetime.date>'.
    Equivalent to `datetime.date.today()`."""
    tms = cytime.localtime()
    return datetime.date_new(tms.tm_year, tms.tm_mon, tms.tm_mday)


# Datetime.date: check types ---------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_date(obj: object) -> cython.bint:
    """Check if an object is type of datetime.date `<bool>`. Equivalent
    to `isinstance(obj, datetime.date)`, includes all subclasses."""
    return datetime.PyDate_Check(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_date_exact(obj: object) -> cython.bint:
    """Check if an object is the exact type of datetime.date `<bool>`.
    Equivalent to `type(obj) is datetime.date`."""
    return datetime.PyDate_CheckExact(obj)


# Datetime.date: attributes ----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def access_year(date: datetime.date) -> cython.uint:
    """Access the 'year' attribute of the date `<int>`.
    (Supports subclasses such as `datetime.datetime` and
    `pandas.Timestamp`)."""
    return datetime.PyDateTime_GET_YEAR(date)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def access_month(date: datetime.date) -> cython.uint:
    """Access the 'month' attribute of the date `<int>`.
    (Supports subclasses such as `datetime.datetime` and
    `pandas.Timestamp`)."""
    return datetime.PyDateTime_GET_MONTH(date)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def access_day(date: datetime.date) -> cython.uint:
    """Access the 'day' attribute of the date `<int>`.
    (Supports subclasses such as `datetime.datetime` and
    `pandas.Timestamp`)."""
    return datetime.PyDateTime_GET_DAY(date)


# Datetime.date: calendar - yeal -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_is_leapyear(date: datetime.date) -> cython.bint:
    """Whether the year of the `datetime.date` is a leap year `<bool>`.
    (Supports subclasses such as `datetime.datetime` and `pandas.Timestamp`)."""
    return is_leapyear(access_year(date))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_leap_bt_years(date1: datetime.date, date2: datetime.date) -> cython.uint:
    """Get the number of leap years between two dates `<int>`.
    (Supports subclasses such as `datetime.datetime` and `pandas.Timestamp`)."""
    return leap_bt_years(access_year(date1), access_year(date2))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_days_in_year(date: datetime.date) -> cython.uint:
    """Get the maximum number of days in the year of the date `<int>`.
    (Supports subclasses such as `datetime.datetime` and `pandas.Timestamp`)."""
    return days_in_year(access_year(date))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_days_bf_year(date: datetime.date) -> cython.uint:
    """Get the number of days betweem the 1st day of 1AD and
    the 1st day of the given date `<int>`. (Supports subclasses
    such as `datetime.datetime` and `pandas.Timestamp`)."""
    return days_bf_year(access_year(date))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_days_of_year(date: datetime.date) -> cython.uint:
    """Get the number of days between the 1st day of the
    year and the date `<int>`. (Supports subclasses such
    as `datetime.datetime` and `pandas.Timestamp`)."""
    return days_of_year(access_year(date), access_month(date), access_day(date))


# Datetime.date: calendar - quarter --------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_quarter(date: datetime.date) -> cython.uint:
    """Get the quarter of date `<int>` (Supports subclasses
    such as `datetime.datetime` and `pandas.Timestamp`)."""
    return quarter_of_month(access_month(date))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_days_in_quarter(date: datetime.date) -> cython.uint:
    """Get the maximum number of days in the quarter of the date `<int>`.
    (Supports subclasses such as `datetime.datetime` and `pandas.Timestamp`)."""
    return days_in_quarter(access_year(date), access_month(date))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_days_bf_quarter(date: datetime.date) -> cython.uint:
    """Get the number of days between the 1st day of the year
    and the 1st day of the quarter of the date `<int>. (Supports
    subclasses such as `datetime.datetime` and `pandas.Timestamp`)."""
    return days_bf_quarter(access_year(date), access_month(date))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_days_of_quarter(date: datetime.date) -> cython.uint:
    """Get the number of days between the 1st day of the
    quarter and the given date `<int>`. (Supports subclasses
    such as `datetime.datetime` and `pandas.Timestamp`)."""
    return days_of_quarter(access_year(date), access_month(date), access_day(date))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_quarter_1st_month(date: datetime.date) -> cython.uint:
    """Get the first month of the quarter. Expect 1, 4, 7, 10 `<int>`.
    (Supports subclasses such as `datetime.datetime` and `pandas.Timestamp`)."""
    return quarter_1st_month(access_month(date))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_quarter_lst_month(date: datetime.date) -> cython.uint:
    """Get the last month of the quarter. Expect 3, 6, 9, 12 `<int>`.
    (Supports subclasses such as `datetime.datetime` and `pandas.Timestamp`)."""
    return quarter_lst_month(access_month(date))


# Datetime.date: calendar - month ----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_days_in_month(date: datetime.date) -> cython.uint:
    """Get the maximum number of days in the month of the date `<int>`.
    (Supports subclasses such as `datetime.datetime` and `pandas.Timestamp`)."""
    return days_in_month(access_year(date), access_month(date))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_days_bf_month(date: datetime.date) -> cython.uint:
    """Get the number of days between the 1st day of the year and the
    1st day of the month of the date `<int>`. (Supports subclasses
    such as `datetime.datetime` and `pandas.Timestamp`)."""
    return days_bf_month(access_year(date), access_month(date))


# Datetime.date: calendar - week -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_weekday(date: datetime.date) -> cython.uint:
    """Get the day of the week for the date, where Monday is 0 ... Sunday
    is 6 `<int>`. (Supports subclasses such as `datetime.datetime` and
    `pandas.Timestamp`)."""
    return ymd_weekday(access_year(date), access_month(date), access_day(date))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_isoweekday(date: datetime.date) -> cython.uint:
    """Get the ISO calendar day of the week for the date, where Monday
    is 1 ... Sunday is 7 `<int>`. (Supports subclasses such as
    `datetime.datetime` and `pandas.Timestamp`)."""
    return ymd_isoweekday(access_year(date), access_month(date), access_day(date))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_isoweek(date: datetime.date) -> cython.uint:
    """Get the ISO calendar week number of the date `<int>`.
    (Supports subclasses such as `datetime.datetime` and `pandas.Timestamp`)."""
    return ymd_isoweek(access_year(date), access_month(date), access_day(date))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_isoyear(date: datetime.date) -> cython.uint:
    """Get the ISO calendar year of the date `<int>`.
    (Supports subclasses such as `datetime.datetime` and `pandas.Timestamp`)."""
    return ymd_isoyear(access_year(date), access_month(date), access_day(date))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_iso1st_ordinal(date: datetime.date) -> cython.uint:
    """Get the ordinal of the 1st day of ISO calendar year of the date `<int>`.
    (Supports subclasses such as `datetime.datetime` and `pandas.Timestamp`)."""
    return iso1st_ordinal(access_year(date))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_isocalendar(date: datetime.date) -> iso:
    """Get the ISO calendar of the date `<struct:iso>`.
    (Supports subclasses such as `datetime.datetime` and `pandas.Timestamp`)."""
    return ymd_isocalendar(access_year(date), access_month(date), access_day(date))


# Datetime.date: conversion ----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_to_isoformat(date: datetime.date) -> str:
    """Convert date to ISO format: '%Y-%m-%d' `<str>`.
    (Supports subclasses such as `datetime.datetime` and `pandas.Timestamp`)."""
    return "%04d-%02d-%02d" % (access_year(date), access_month(date), access_day(date))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def date_to_ordinal(date: datetime.date) -> cython.uint:
    """Convert date to ordinal `<int>`. (Supports subclasses such as
    `datetime.datetime` and `pandas.Timestamp`)."""
    return ymd_to_ordinal(access_year(date), access_month(date), access_day(date))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_to_seconds(date: datetime.date) -> cython.longlong:
    """Convert date to total seconds after POSIX epoch `<int>`.
    (Supports subclasses such as `datetime.datetime` and `pandas.Timestamp`)."""
    return (date_to_ordinal(date) - EPOCH_DAY) * 86_400


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_to_microseconds(date: datetime.date) -> cython.longlong:
    """Convert date to total microseconds after POSIX epoch `<int>`.
    (Supports subclasses such as `datetime.datetime` and `pandas.Timestamp`)."""
    return (date_to_ordinal(date) - EPOCH_DAY) * US_DAY


@cython.cfunc
@cython.inline(True)
def date_to_timestamp(date: datetime.date) -> cython.longlong:
    """Convert date to timestamp `<int>`. (Supports subclasses such as
    `datetime.datetime` and `pandas.Timestamp`)."""
    base_ts: cython.longlong = date_to_seconds(date)
    offset: cython.longlong = localize_ts(base_ts) - base_ts
    return base_ts - offset


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_fr_date(date: datetime.date) -> datetime.date:
    """Convert subclass of datetime.date to `<datetime.date>`.
    (Support subclasses such as `datetime.datetime` and
    `pandas.Timestamp`.)
    """
    return datetime.date_new(access_year(date), access_month(date), access_day(date))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_fr_ordinal(ordinal: cython.int) -> datetime.date:
    """Convert ordinal to `<datetime.date>`.
    Equivalent to `datetime.date.fromordinal(ordinal)`."""
    ymd = ordinal_to_ymd(ordinal)
    return datetime.date_new(ymd.year, ymd.month, ymd.day)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_fr_seconds(seconds: cython.double) -> datetime.date:
    """Convert total seconds after POSIX epoch to `<datetime.date>`."""
    total_sec: cython.longlong = int(seconds)
    return date_fr_ordinal(total_sec + EPOCH_SEC // 86_400)


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
@cython.exceptval(check=False)
def date_fr_microseconds(microseconds: cython.longlong) -> datetime.date:
    """Convert total microseconds after POSIX epoch to `<datetime.date>`."""
    return date_fr_ordinal(microseconds // US_DAY + EPOCH_DAY)


@cython.cfunc
@cython.inline(True)
def date_fr_timestamp(timestamp: cython.double) -> datetime.date:
    """Convert timestamp to `<datetime.date>`."""
    return datetime.date_from_timestamp(timestamp)


# Datetime.date: arithmetic ----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_add(
    date: datetime.date,
    days: cython.int = 0,
    seconds: cython.longlong = 0,
    microseconds: cython.longlong = 0,
) -> datetime.date:
    """Add days, seconds and microseconds to `<datetime.date>`.
    Equivalent to `date + timedelta(d, s, us)`."""
    return date_fr_microseconds(
        date_to_microseconds(date) + days * US_DAY + seconds * 1_000_000 + microseconds
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_add_delta(date: datetime.date, delta: datetime.timedelta) -> datetime.date:
    """Add datetime.timedelta to `<datetime.date>`.
    Equivalent to `date + timedelta(instance)`."""
    return date_fr_microseconds(
        date_to_microseconds(date) + delta_to_microseconds(delta)
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_sub_delta(date: datetime.date, delta: datetime.timedelta) -> datetime.date:
    """Substract datetime.timedelta from `<datetime.date>`.
    Equivalent to `date - timedelta(instance)`."""
    return date_fr_microseconds(
        date_to_microseconds(date) - delta_to_microseconds(delta)
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_sub_date(date_l: datetime.date, date_r: datetime.date) -> datetime.timedelta:
    """Substruction between `datetime.date`. Equivalent
    to `date - date`, and ruturns the `<timedelta>`."""
    return datetime.timedelta_new(date_sub_date_days(date_l, date_r), 0, 0)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_sub_date_days(date_l: datetime.date, date_r: datetime.date) -> cython.int:
    """Substruction between `datetime.date`. Equivalent to
    `date - date`, but ruturns the difference in days `<int>`."""
    return date_to_ordinal(date_l) - date_to_ordinal(date_r)


# Datetime.date: manipulation --------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_replace(
    date: datetime.date,
    year: cython.int = -1,
    month: cython.int = -1,
    day: cython.int = -1,
) -> datetime.date:
    """Replace `<datetime.date>`. Equivalent to `date.replace()`.
    Value of `-1` indicates preserving the original value."""
    if not 1 <= year <= 9999:
        year = access_year(date)
    if not 1 <= month <= 12:
        month = access_month(date)
    return datetime.date_new(
        year,
        month,
        min(day if day > 0 else access_day(date), days_in_month(year, month)),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_adj_weekday(date: datetime.date, weekday: cython.uint) -> datetime.date:
    """Adjust `<datetime.date>` to the nearest weekday,
    where Monday is 0 and Sunday is 6. Equivalent to:
    `date + timedelta(days=weekday - date.weekday())`."""
    weekday = weekday % 7
    date_wday: cython.uint = date_weekday(date)
    if weekday == date_wday:
        return date
    return date_add(date, days=weekday - date_wday)


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
    """Generate a new `<datetime.datetime>`."""
    return datetime.datetime_new(
        year, month, day, hour, minute, second, microsecond, tzinfo, fold
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def gen_dt_now() -> datetime.datetime:
    """Generate the current local `<datetime.datetime>`.
    Equivalent to `datetime.datetime.now()`."""
    microseconds: cython.int = int(time() % 1 * 1_000_000)
    tms = cytime.localtime()
    return datetime.datetime_new(
        tms.tm_year,
        tms.tm_mon,
        tms.tm_mday,
        tms.tm_hour,
        tms.tm_min,
        tms.tm_sec,
        microseconds,
        None,
        0,
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def gen_dt_now_utc() -> datetime.datetime:
    """Generate current `<datetime.datetime>` under UTC.
    Equivalent to `datetime.datetime.now(UTC)`."""
    return dt_fr_timestamp(time(), UTC)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def gen_dt_now_tz(tzinfo: datetime.tzinfo) -> datetime.datetime:
    """Generate current `<datetime.datetime>` under specific timezone.
    Equivalent to `datetime.datetime.now(tzinfo)`."""
    return dt_fr_timestamp(time(), tzinfo)


# Datetime.datetime: check types -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_dt(obj: object) -> cython.bint:
    """Check if an object is type of datetime.datetime `<bool>`. Equivalent
    to `isinstance(obj, datetime.datetime)`, includes all subclasses."""
    return datetime.PyDateTime_Check(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_dt_exact(obj: object) -> cython.bint:
    """Check if an object is the exact type of datetime.datetime `<bool>`.
    Equivalent to `type(obj) is datetime.datetime`."""
    return datetime.PyDateTime_CheckExact(obj)


# Datetime.datetime: attributes ------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def access_dt_year(dt: datetime.datetime) -> cython.uint:
    """Access the 'year' attribute of the datetime `<int>`.
    (Supports subclasses such as `pandas.Timestamp`)."""
    return datetime.PyDateTime_GET_YEAR(dt)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def access_dt_month(dt: datetime.datetime) -> cython.uint:
    """Access the 'month' attribute of the datetime `<int>`.
    (Supports subclasses such as `pandas.Timestamp`)."""
    return datetime.PyDateTime_GET_MONTH(dt)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def access_dt_day(dt: datetime.datetime) -> cython.uint:
    """Access the 'day' attribute of the datetime `<int>`.
    (Supports subclasses such as `pandas.Timestamp`)."""
    return datetime.PyDateTime_GET_DAY(dt)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def access_dt_hour(dt: datetime.datetime) -> cython.uint:
    """Access the 'hour' attribute of the datetime `<int>`.
    (Supports subclasses such as `pandas.Timestamp`.)"""
    return datetime.PyDateTime_DATE_GET_HOUR(dt)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def access_dt_minute(dt: datetime.datetime) -> cython.uint:
    """Access the 'minute' attribute of the datetime `<int>`.
    (Supports subclasses such as `pandas.Timestamp`.)"""
    return datetime.PyDateTime_DATE_GET_MINUTE(dt)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def access_dt_second(dt: datetime.datetime) -> cython.uint:
    """Access the 'second' attribute of the datetime `<int>`.
    (Supports subclasses such as `pandas.Timestamp`.)"""
    return datetime.PyDateTime_DATE_GET_SECOND(dt)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def access_dt_microsecond(dt: datetime.datetime) -> cython.uint:
    """Access the 'microsecond' attribute of the datetime `<int>`.
    (Supports subclasses such as `pandas.Timestamp`.)"""
    return datetime.PyDateTime_DATE_GET_MICROSECOND(dt)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def access_dt_tzinfo(dt: datetime.datetime) -> datetime.tzinfo:
    """Access the 'tzinfo' attribute of the datetime `<tzinfo>`.
    (Supports subclasses such as `pandas.Timestamp`.)"""
    return datetime.datetime_tzinfo(dt)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def access_dt_fold(dt: datetime.datetime) -> cython.uint:
    """Access the 'fold' attribute of the datetime `<int>`.
    (Supports subclasses such as `pandas.Timestamp`.)"""
    return datetime.PyDateTime_DATE_GET_FOLD(dt)


# Datetime.datetime: conversion ------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_to_isoformat(dt: datetime.datetime) -> str:
    """Convert datetime to ISO format: '%Y-%m-%dT%H:%M:%S.f' `<str>`.
    (Supports subclasses such as `pandas.Timestamp`)."""
    microsecond = access_dt_microsecond(dt)
    if microsecond > 0:
        return "%04d-%02d-%02dT%02d:%02d:%02d.%06d" % (
            access_dt_year(dt),
            access_dt_month(dt),
            access_dt_day(dt),
            access_dt_hour(dt),
            access_dt_minute(dt),
            access_dt_second(dt),
            microsecond,
        )
    else:
        return "%04d-%02d-%02dT%02d:%02d:%02d" % (
            access_dt_year(dt),
            access_dt_month(dt),
            access_dt_day(dt),
            access_dt_hour(dt),
            access_dt_minute(dt),
            access_dt_second(dt),
        )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_to_isoformat_tz(dt: datetime.datetime) -> str:
    """Convert datetime to ISO format with tzinfo:
    '%Y-%m-%dT%H:%M:%S.f.Z' `<str>`. (Supports
    subclasses such as `pandas.Timestamp`)."""
    fmt: str = dt_to_isoformat(dt)
    tzinfo = access_dt_tzinfo(dt)
    if tzinfo is not None:
        delta: datetime.timedelta = tzinfo.utcoffset(dt)
        if delta is not None:
            fmt += delta_to_utcformat(delta)
    return fmt


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_to_seconds(dt: datetime.datetime) -> cython.double:
    """Convert datetime to total seconds after POSIX epoch `<float>`.
    This function ignores the local timezone and tzinfo of the datetime.
    (Supports subclasses such as `pandas.Timestamp`).

    ### Notice
    This should `NOT` be treated as `datetime.timestamp()`.
    """
    days: cython.double = date_to_ordinal(dt)
    hour: cython.double = access_dt_hour(dt)
    minute: cython.double = access_dt_minute(dt)
    second: cython.double = access_dt_second(dt)
    microsecond: cython.double = access_dt_microsecond(dt)
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
    """Convert datetime to total seconds after POSIX epoch `<float>`.
    (Supports subclasses such as `pandas.Timestamp`).
    - If `dt` is timezone-aware, return total seconds in UTC,
      equivalent to `datetime.timestamp()`.
    - If `dt` is timezone-naive, return total seconds that ignores
      the local timezone and tzinfo of the datetime, requivalent
      to `dt_to_seconds()`.

    ### Notice
    This should `NOT` be treated as `datetime.timestamp()`.
    """
    sec: cython.double = dt_to_seconds(dt)
    tzinfo = access_dt_tzinfo(dt)
    if tzinfo is not None:
        delta: datetime.timedelta = tzinfo.utcoffset(dt)
        if delta is not None:
            sec -= delta_to_seconds(delta)
    return sec


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_to_microseconds(dt: datetime.datetime) -> cython.longlong:
    """Convert datetime to total microseconds after POSIX epoch `<int>`.
    This function ignores the local timezone and tzinfo of the datetime.
    (Supports subclasses such as `pandas.Timestamp`).
    """
    days: cython.longlong = date_to_ordinal(dt)
    hour: cython.longlong = access_dt_hour(dt)
    minute: cython.longlong = access_dt_minute(dt)
    second: cython.longlong = access_dt_second(dt)
    microsecond: cython.longlong = access_dt_microsecond(dt)
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
    """Convert datetime to total microseconds after POSIX epoch `<int>`.
    (Supports subclasses such as `pandas.Timestamp`).
    - If `dt` is timezone-aware, return total microseconds in UTC.
    - If `dt` is timezone-naive, return total microseconds that
      ignores the local timezone and tzinfo of the datetime,
      requivalent to `dt_to_microseconds()`.
    """
    us: cython.longlong = dt_to_microseconds(dt)
    tzinfo = access_dt_tzinfo(dt)
    if tzinfo is not None:
        delta: datetime.timedelta = tzinfo.utcoffset(dt)
        if delta is not None:
            us -= delta_to_microseconds(delta)
    return us


@cython.cfunc
@cython.inline(True)
def dt_to_posixts(dt: datetime.datetime) -> cython.longlong:
    """Convert datetime to POSIX timestamp as `<int>`.

    Equivalent to `datetime._mktime()`

    This function takes into account local timezone
    and daylight saving time changes, and converts
    the datetime to POSIX timestamp.
    """
    # Total seconds of the 'dt' since epoch
    t: cython.longlong = int(dt_to_seconds(dt))
    # Adjustment for local time
    adj1: cython.longlong = localize_ts(t) - t
    adj2: cython.longlong
    u1: cython.longlong = t - adj1
    u2: cython.longlong
    t1: cython.longlong = localize_ts(u1)
    # Adjustment for Daylight Saving
    if t == t1:
        # We found one solution, but it may not be the one we need.
        # Look for an earlier solution (if `fold` is 0), or a later
        # one (if `fold` is 1).
        u2 = u1 - 86_400 if access_dt_fold(dt) == 0 else u1 + 86_400
        adj2 = localize_ts(u2) - u2
        if adj1 == adj2:
            return u1
    else:
        adj2 = t1 - u1
        if adj1 == adj2:
            raise ValueError(
                "<dt_to_posix_ts> Got unexpect result, "
                "where 'adj1 == adj2': %d == %d" % (adj1, adj2)
            )
    # Final adjustment
    u2 = t - adj2
    t2: cython.longlong = localize_ts(u2)
    if t == t2:
        return u2
    if t == t1:
        return u1
    # We have found both offsets adj1 and adj2,
    # but neither t - adj1 nor t - adj2 is the
    # solution. This means t is in the gap.
    return max(u1, u2) if access_dt_fold(dt) == 0 else min(u1, u2)


@cython.cfunc
@cython.inline(True)
def dt_to_timestamp(dt: datetime.datetime) -> cython.double:
    """Convert datetime to POSIX timestamp as `<float>`.

    Equivalent to `datetime.timestamp()`
    """
    if access_dt_tzinfo(dt) is None:
        ts: cython.double = dt_to_posixts(dt)
        us: cython.double = access_dt_microsecond(dt)
        return ts + us / 1_000_000
    else:
        return delta_to_seconds(dt_sub_dt(dt, EPOCH_UTC))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_fr_dt(dt: datetime.datetime) -> datetime.datetime:
    """Convert subclass of datetime.datetime to `<datetime.datetime>`.
    (Support subclasses such as `pandas.Timestamp`.)
    """
    return datetime.datetime_new(
        access_dt_year(dt),
        access_dt_month(dt),
        access_dt_day(dt),
        access_dt_hour(dt),
        access_dt_minute(dt),
        access_dt_second(dt),
        access_dt_microsecond(dt),
        access_dt_tzinfo(dt),
        access_dt_fold(dt),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_fr_date(
    date: datetime.date,
    tzinfo: datetime.tzinfo = None,
) -> datetime.datetime:
    """Convert datetime.date to `<datetime.datetime>`."""
    return datetime.datetime_new(
        access_year(date), access_month(date), access_day(date), 0, 0, 0, 0, tzinfo, 0
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_fr_time(time: datetime.time) -> datetime.datetime:
    """Convert datetime.time to `<datetime.datetime>`.

    Year, month, day will be auto set to the current date.
    """
    tms = cytime.localtime()
    return datetime.datetime_new(
        tms.tm_year,
        tms.tm_mon,
        tms.tm_mday,
        access_time_hour(time),
        access_time_minute(time),
        access_time_second(time),
        access_time_microsecond(time),
        access_time_tzinfo(time),
        access_time_fold(time),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_fr_date_n_time(date: datetime.date, time: datetime.time) -> datetime.datetime:
    """Convert datetime.date & datetime.time to `<datetime.datetime>`."""
    return datetime.datetime_new(
        access_year(date),
        access_month(date),
        access_day(date),
        access_time_hour(time),
        access_time_minute(time),
        access_time_second(time),
        access_time_microsecond(time),
        access_time_tzinfo(time),
        access_time_fold(time),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_fr_ordinal(
    ordinal: cython.int,
    tzinfo: datetime.tzinfo = None,
) -> datetime.datetime:
    """Convert ordinal to `<datetime.datetime>`."""
    ymd = ordinal_to_ymd(ordinal)
    return datetime.datetime_new(ymd.year, ymd.month, ymd.day, 0, 0, 0, 0, tzinfo, 0)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_fr_seconds(
    seconds: cython.double,
    tzinfo: datetime.tzinfo = None,
) -> datetime.datetime:
    """Convert total seconds after POSIX epoch to `<datetime.datetime>`."""
    microseconds: cython.longlong = int(seconds * 1_000_000)
    return dt_fr_microseconds(microseconds, tzinfo)


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
@cython.exceptval(check=False)
def dt_fr_microseconds(
    microseconds: cython.longlong,
    tzinfo: datetime.tzinfo = None,
) -> datetime.datetime:
    """Convert total microseconds after POSIX epoch to `<datetime.datetime>`."""
    # Add back epoch microseconds
    microseconds += EPOCH_US
    microseconds = min(max(microseconds, DT_MIN_US), DT_MAX_US)
    # Calculate ymd
    ymd = ordinal_to_ymd(microseconds // US_DAY)
    # Calculate hms
    hms = microseconds_to_hms(microseconds)
    # Generate datetime
    return datetime.datetime_new(
        ymd.year,
        ymd.month,
        ymd.day,
        hms.hour,
        hms.minute,
        hms.second,
        hms.microsecond,
        tzinfo,
        0,
    )


@cython.cfunc
@cython.inline(True)
def dt_fr_timestamp(
    timestamp: cython.double,
    tzinfo: datetime.tzinfo = None,
) -> datetime.datetime:
    """Convert timestamp to `<datetime.datetime>`."""
    return datetime.datetime_from_timestamp(timestamp, tzinfo)


# Datetime.datetime: arithmetic ------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_add(
    dt: datetime.datetime,
    days: cython.int = 0,
    seconds: cython.longlong = 0,
    microseconds: cython.longlong = 0,
) -> datetime.datetime:
    """Add days, seconds and microseconds to `<datetime.datetime>`.
    Equivalent to `datetime + timedelta(d, s, us)`)."""
    return dt_fr_microseconds(
        dt_to_microseconds(dt) + days * US_DAY + seconds * 1_000_000 + microseconds,
        access_dt_tzinfo(dt),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_add_delta(dt: datetime.datetime, delta: datetime.timedelta) -> datetime.datetime:
    """Add datetime.timedelta to `<datetime.datetime>`.
    Equivalent to `datetime + timedelta(instance)`)."""
    return dt_fr_microseconds(
        dt_to_microseconds(dt) + delta_to_microseconds(delta),
        access_dt_tzinfo(dt),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_sub_delta(dt: datetime.datetime, delta: datetime.timedelta) -> datetime.datetime:
    """Substract datetime.timedelta from `<datetime.datetime>`.
    Equivalent to `datetime - timedelta(instance)`)."""
    return dt_fr_microseconds(
        dt_to_microseconds(dt) - delta_to_microseconds(delta),
        access_dt_tzinfo(dt),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_sub_dt(dt_l: datetime.datetime, dt_r: datetime.datetime) -> datetime.timedelta:
    """Substraction between `datetime.datetime`. Equivalent to
    `datetime - datetime`, and returns `<datetime.timedelta>`.
    """
    return delta_fr_microseconds(dt_sub_dt_us(dt_l, dt_r))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_sub_dt_us(dt_l: datetime.datetime, dt_r: datetime.datetime) -> cython.longlong:
    """Substraction between `datetime.datetime`. Equivalent to
    `datetime - datetime`, but returns the difference in microseconds `<int>`."""
    delta_us: cython.longlong = dt_to_microseconds(dt_l) - dt_to_microseconds(dt_r)
    tzinfo_l: datetime.tzinfo = access_dt_tzinfo(dt_l)
    tzinfo_r: datetime.tzinfo = access_dt_tzinfo(dt_r)
    # If both are naive or have the same tzinfo
    # return delta directly.
    if tzinfo_l is tzinfo_r:
        return delta_us
    # Calculate left timezone offset
    offset_l: cython.int
    if tzinfo_l is None:
        offset_l = 0
    else:
        delta: datetime.timedelta = dt_l.utcoffset()
        offset_l = 0 if delta is None else delta_to_microseconds(delta)
    # Calculate right timezone offset
    offset_r: cython.int
    if tzinfo_r is None:
        offset_r = 0
    else:
        delta: datetime.timedelta = dt_r.utcoffset()
        offset_r = 0 if delta is None else delta_to_microseconds(delta)
    # Return the difference
    return delta_us + offset_r - offset_l


# Datetime.datetime: manipulation ----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
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
    """Replace `<datetime.datetime>`. Equivalent to `datetime.replcae()`.
    Value of `-1` indicates preserving the original value."""
    if not 1 <= year <= 9999:
        year = access_dt_year(dt)
    if not 1 <= month <= 12:
        month = access_dt_month(dt)
    return datetime.datetime_new(
        year,
        month,
        min(day if day > 0 else access_dt_day(dt), days_in_month(year, month)),
        hour if 0 <= hour <= 23 else access_dt_hour(dt),
        minute if 0 <= minute <= 59 else access_dt_minute(dt),
        second if 0 <= second <= 59 else access_dt_second(dt),
        microsecond if 0 <= microsecond <= 999999 else access_dt_microsecond(dt),
        tzinfo if (is_tzinfo(tzinfo) or tzinfo is None) else access_dt_tzinfo(dt),
        fold if 0 <= fold <= 1 else access_dt_fold(dt),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_replace_tzinfo(
    dt: datetime.datetime,
    tzinfo: datetime.tzinfo,
) -> datetime.datetime:
    """Replace `<datetime.datetime>` timezone information.
    Equivalent to `datetime.replace(tzinfo=tzinfo)."""
    return datetime.datetime_new(
        access_dt_year(dt),
        access_dt_month(dt),
        access_dt_day(dt),
        access_dt_hour(dt),
        access_dt_minute(dt),
        access_dt_second(dt),
        access_dt_microsecond(dt),
        tzinfo,
        access_dt_fold(dt),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_replace_fold(dt: datetime.datetime, fold: cython.uint) -> datetime.datetime:
    """Replace `<datetime.datetime>` day light saving 'fold'.
    Equivalent to `datetime.replace(fold=fold)."""
    return datetime.datetime_new(
        access_dt_year(dt),
        access_dt_month(dt),
        access_dt_day(dt),
        access_dt_hour(dt),
        access_dt_minute(dt),
        access_dt_second(dt),
        access_dt_microsecond(dt),
        access_dt_tzinfo(dt),
        1 if fold > 0 else 0,
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_adj_weekday(dt: datetime.datetime, weekday: cython.uint) -> datetime.datetime:
    """Adjust `<datetime.datetime>` to the nearest weekday,
    where Monday is 0 and Sunday is 6. Equivalent to:
    `datetime + timedelta(days=weekday - dt.weekday())`."""
    weekday = weekday % 7
    dt_wday: cython.uint = date_weekday(dt)
    if weekday == dt_wday:
        return dt
    return dt_add(dt, days=weekday - dt_wday)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_astimezone(
    dt: datetime.datetime,
    tzinfo: datetime.tzinfo = None,
) -> datetime.datetime:
    """Convert `<datetime.datetime>` to the new timezone.
    Equivalent to `datetime.astimezone(tzinfo)`.
    """
    t_tz: datetime.tzinfo
    b_tz: datetime.tzinfo
    if tzinfo is None:
        t_tz = gen_tzinfo_local(None)
        b_tz = access_dt_tzinfo(dt)
        if b_tz is None:
            return dt_replace_tzinfo(dt, t_tz)  # exit: replace tzinfo
    else:
        t_tz = tzinfo
        b_tz = access_dt_tzinfo(dt)

    if b_tz is None:
        b_tz = gen_tzinfo_local(dt)
        b_offset: datetime.timedelta = b_tz.utcoffset(dt)
    else:
        b_offset: datetime.timedelta = b_tz.utcoffset(dt)
        if b_offset is None:
            b_tz = gen_tzinfo_local(dt_replace_tzinfo(dt, None))
            b_offset = b_tz.utcoffset(dt)

    if t_tz is b_tz:
        return dt

    # Calculate delta in microseconds
    t_delta: cython.longlong = delta_to_microseconds(t_tz.utcoffset(dt))
    b_delta: cython.longlong = delta_to_microseconds(b_offset)

    # Generate new datetime
    us: cython.longlong = dt_to_microseconds(dt)
    return dt_fr_microseconds(us + t_delta - b_delta, t_tz)


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
    """Generate a new `<datetime.time>`."""
    return datetime.time_new(hour, minute, second, microsecond, tzinfo, fold)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def gen_time_now() -> datetime.time:
    """Generate the current local `<datetime.time>`.
    Equivalent to `datetime.datetime.now().time()`."""
    microseconds: cython.int = int(time() % 1 * 1_000_000)
    tms = cytime.localtime()
    return datetime.time_new(tms.tm_hour, tms.tm_min, tms.tm_sec, microseconds, None, 0)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def gen_time_now_utc() -> datetime.time:
    """Generate current `<datetime.time>` under UTC.
    Equivalent to `datetime.datetime.now(UTC).time()`."""
    return time_fr_dt(gen_dt_now_utc())


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def gen_time_now_tz(tzinfo: datetime.tzinfo) -> datetime.time:
    """Generate current `<datetime.time>` under specific timezone.
    Equivalent to `datetime.datetime.now(tzinfo).time()`."""
    return time_fr_dt(gen_dt_now_tz(tzinfo))


# Datetime.time: check types ---------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_time(obj: object) -> cython.bint:
    """Check if an object is type of datetime.time `<bool>`. Equivalent
    to `isinstance(obj, datetime.time)`, includes all subclasses."""
    return datetime.PyTime_Check(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_time_exact(obj: object) -> cython.bint:
    """Check if an object is the exact type of datetime.time `<bool>`.
    Equivalent to `type(obj) is datetime.time`."""
    return datetime.PyTime_CheckExact(obj)


# Datetime.time: attribute -----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def access_time_hour(time: datetime.time) -> cython.uint:
    """Access the 'hour' attribute of the time `<int>`."""
    return datetime.PyDateTime_TIME_GET_HOUR(time)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def access_time_minute(time: datetime.time) -> cython.uint:
    """Access the 'minute' attribute of the time `<int>`."""
    return datetime.PyDateTime_TIME_GET_MINUTE(time)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def access_time_second(time: datetime.time) -> cython.uint:
    """Access the 'second' attribute of the time `<int>`."""
    return datetime.PyDateTime_TIME_GET_SECOND(time)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def access_time_microsecond(time: datetime.time) -> cython.uint:
    """Access the 'microsecond' attribute of the time `<int>`."""
    return datetime.PyDateTime_TIME_GET_MICROSECOND(time)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def access_time_tzinfo(time: datetime.time) -> datetime.tzinfo:
    """Access the 'tzinfo' attribute of the time `<tzinfo>`."""
    return datetime.time_tzinfo(time)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def access_time_fold(time: datetime.time) -> cython.uint:
    """Access the 'fold' attribute of the time `<int>`."""
    return datetime.PyDateTime_TIME_GET_FOLD(time)


# Datetime.time: conversion ----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_to_isoformat(time: datetime.time) -> str:
    """Convert time to ISO format: '%H:%M:%S.f' `<str>`."""
    microsecond: cython.int = access_time_microsecond(time)
    if microsecond > 0:
        return "%02d:%02d:%02d.%06d" % (
            access_time_hour(time),
            access_time_minute(time),
            access_time_second(time),
            microsecond,
        )
    else:
        return "%02d:%02d:%02d" % (
            access_time_hour(time),
            access_time_minute(time),
            access_time_second(time),
        )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_to_seconds(time: datetime.time) -> cython.double:
    """Convert datetime.time to total seconds `<float>`."""
    microseconds: cython.double = time_to_microseconds(time)
    return microseconds / 1_000_000


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_to_microseconds(time: datetime.time) -> cython.longlong:
    """Convert datetime.time to total microseconds `<int>`."""
    hour: cython.longlong = access_time_hour(time)
    minute: cython.longlong = access_time_minute(time)
    second: cython.longlong = access_time_second(time)
    microsecond: cython.longlong = access_time_microsecond(time)
    return hour * US_HOUR + minute * 60_000_000 + second * 1_000_000 + microsecond


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_fr_dt(dt: datetime.datetime) -> datetime.time:
    """Convert datetime.datetime to `<datetime.time>`."""
    return datetime.time_new(
        access_dt_hour(dt),
        access_dt_minute(dt),
        access_dt_second(dt),
        access_dt_microsecond(dt),
        access_dt_tzinfo(dt),
        access_dt_fold(dt),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_fr_seconds(
    seconds: cython.double,
    tzinfo: datetime.tzinfo = None,
) -> datetime.time:
    """Convert total seconds to `<datetime.time>`."""
    mciroseconds: cython.longlong = int(seconds * 1_000_000)
    return time_fr_microseconds(mciroseconds, tzinfo)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_fr_microseconds(
    microseconds: cython.longlong,
    tzinfo: datetime.tzinfo = None,
) -> datetime.time:
    """Convert total microseconds to `<datetime.time>`."""
    # Add back epoch microseconds
    if microseconds < 0:
        microseconds += EPOCH_US
    microseconds = min(max(microseconds, 0), DT_MAX_US)
    # Calculate hms
    hms = microseconds_to_hms(microseconds)
    # Generate time
    return datetime.time_new(
        hms.hour,
        hms.minute,
        hms.second,
        hms.microsecond,
        tzinfo,
        0,
    )


# Datetime.time: manipulation --------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_replace(
    time: datetime.time,
    hour: cython.int = -1,
    minute: cython.int = -1,
    second: cython.int = -1,
    microsecond: cython.int = -1,
    tzinfo: object = -1,
    fold: cython.int = -1,
) -> datetime.time:
    """Replace `<datetime.time>`. Equivalent to `time.replcae()`.
    Value of `-1` indicates preserving the original value."""
    return datetime.time_new(
        hour if 0 <= hour <= 23 else access_time_hour(time),
        minute if 0 <= minute <= 59 else access_time_minute(time),
        second if 0 <= second <= 59 else access_time_second(time),
        microsecond if 0 <= microsecond <= 999999 else access_time_microsecond(time),
        tzinfo if (is_tzinfo(tzinfo) or tzinfo is None) else access_time_tzinfo(time),
        fold if 0 <= fold <= 1 else access_time_fold(time),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_replace_tzinfo(time: datetime.time, tzinfo: datetime.tzinfo) -> datetime.time:
    """Replace `<datetime.time>` timezone information.
    Equivalent to `time.replace(tzinfo=tzinfo)."""
    return datetime.time_new(
        access_time_hour(time),
        access_time_minute(time),
        access_time_second(time),
        access_time_microsecond(time),
        tzinfo,
        access_time_fold(time),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_replace_fold(time: datetime.time, fold: cython.uint) -> datetime.time:
    """Replace `<datetime.time>` day light saving 'fold'.
    Equivalent to `time.replace(fold=fold)."""
    return datetime.time_new(
        access_time_hour(time),
        access_time_minute(time),
        access_time_second(time),
        access_time_microsecond(time),
        access_time_tzinfo(time),
        1 if fold > 0 else 0,
    )


# Datetime.timedelta ===================================================================================
# Datetime.timedelta: generate -------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def gen_delta(
    days: cython.int = 0,
    seconds: cython.int = 0,
    microseconds: cython.int = 0,
) -> datetime.timedelta:
    """Generate a new `<datetime.timedelta>`."""
    return datetime.timedelta_new(days, seconds, microseconds)


# Datetime.timedelta: check types ----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_delta(obj: object) -> cython.bint:
    """Check if an object is type of datetime.timedelta `<bool>`. Equivalent
    to `isinstance(obj, datetime.timedelta)`, includes all subclasses."""
    return datetime.PyDelta_Check(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_delta_exact(obj: object) -> cython.bint:
    """Check if an object is the exact type of datetime.timedelta `<bool>`.
    Equivalent to `type(obj) is datetime.timedelta`."""
    return datetime.PyDelta_CheckExact(obj)


# Datetime.timedelta: attributes -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def access_delta_days(delta: datetime.timedelta) -> cython.int:
    """Access the 'days' attribute of the timedelta `<int>`.
    (Supports subclasses such as `pandas.Timedelta`.)"""
    return datetime.PyDateTime_DELTA_GET_DAYS(delta)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def access_delta_seconds(delta: datetime.timedelta) -> cython.int:
    """Access the 'seconds' attribute of the timedelta `<int>`.
    (Supports subclasses such as `pandas.Timedelta`.)"""
    return datetime.PyDateTime_DELTA_GET_SECONDS(delta)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def access_delta_microseconds(delta: datetime.timedelta) -> cython.int:
    """Access the 'microseconds' attribute of the timedelta `<int>`.
    (Supports subclasses such as `pandas.Timedelta`.)"""
    return datetime.PyDateTime_DELTA_GET_MICROSECONDS(delta)


# Datetime.timedelta: conversion -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def delta_to_isoformat(delta: datetime.timedelta) -> str:
    """Convert timedelta to ISO format: '%H:%M:%S.f' `<str>`.
    (Supports subclasses such as `pandas.Timestamp`)."""
    days: cython.int = access_delta_days(delta)
    secs: cython.int = access_delta_seconds(delta)
    hours: cython.int = secs // 3_600 % 24 + days * 24
    minutes: cython.int = secs // 60 % 60
    seconds: cython.int = secs % 60
    microseconds: cython.int = access_delta_microseconds(delta)
    if microseconds:
        return "%02d:%02d:%02d.%06d" % (hours, minutes, seconds, microseconds)
    else:
        return "%02d:%02d:%02d" % (hours, minutes, seconds)


@cython.cfunc
@cython.inline(True)
def delta_to_utcformat(delta: datetime.timedelta) -> str:
    """Convert timedelta to UTC format: '+HH:MM' `<str>`."""
    days: cython.int = access_delta_days(delta)
    secs: cython.int = access_delta_seconds(delta)
    hours: cython.int = secs // 3_600 % 24 + days * 24
    if hours < 0:
        hours = -hours
        sign = "-"
    else:
        sign = "+"
    minutes: cython.int = secs // 60 % 60
    return "%s%02d:%02d" % (sign, hours, minutes)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def delta_to_seconds(delta: datetime.timedelta) -> cython.double:
    """Convert timedelta to total seconds `<float>`.
    (Supports subclasses such as `pandas.Timestamp`)."""
    microseconds: cython.double = delta_to_microseconds(delta)
    return microseconds / 1_000_000


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def delta_to_microseconds(delta: datetime.timedelta) -> cython.longlong:
    """Convert datetime.timedelta to total microseconds `<int>`.
    (Supports subclasses such as `pandas.Timestamp`)."""
    days: cython.longlong = access_delta_days(delta)
    seconds: cython.longlong = access_delta_seconds(delta)
    microseconds: cython.longlong = access_delta_microseconds(delta)
    return days * US_DAY + seconds * 1_000_000 + microseconds


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def delta_fr_delta(delta: datetime.timedelta) -> datetime.timedelta:
    """Convert subclass of datetime.timedelta to `<datetime.timedelta>`.
    (Supports subclasses such as `pandas.Timedelta`.)"""
    return datetime.timedelta_new(
        access_delta_days(delta),
        access_delta_seconds(delta),
        access_delta_microseconds(delta),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def delta_fr_seconds(seconds: cython.double) -> datetime.timedelta:
    """Convert total seconds to `<datetime.timedelta>`."""
    microseconds: cython.longlong = int(seconds * 1_000_000)
    return delta_fr_microseconds(microseconds)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def delta_fr_microseconds(microseconds: cython.longlong) -> datetime.timedelta:
    """Convert total microseconds to `<datetime.timedelta>`."""
    # Calculate days, seconds and microseconds
    days: cython.longlong = microseconds // US_DAY
    microseconds = microseconds % US_DAY
    seconds: cython.longlong = microseconds // 1_000_000
    microseconds = microseconds % 1_000_000
    # Generate timedelta
    return datetime.timedelta_new(days, seconds, microseconds)


# Datetime.timedelta: arithmetic -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def delta_add(
    delta: datetime.timedelta,
    days: cython.int = 0,
    seconds: cython.int = 0,
    microseconds: cython.int = 0,
) -> datetime.timedelta:
    """Add days, seconds and microseconds to `<datetime.timedelta>`.
    Equivalent to `timedelta + timedelta(d, s, us)`)."""
    return datetime.timedelta_new(
        access_delta_days(delta) + days,
        access_delta_seconds(delta) + seconds,
        access_delta_microseconds(delta) + microseconds,
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def delta_add_delta(
    delta_l: datetime.timedelta,
    delta_r: datetime.timedelta,
) -> datetime.timedelta:
    """Addition between datetime.timedelta.
    Equivalent to `timedelta + timedelta(instance)`."""
    return datetime.timedelta_new(
        access_delta_days(delta_l) + access_delta_days(delta_r),
        access_delta_seconds(delta_l) + access_delta_seconds(delta_r),
        access_delta_microseconds(delta_l) + access_delta_microseconds(delta_r),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def delta_sub_delta(
    delta_l: datetime.timedelta,
    delta_r: datetime.timedelta,
) -> datetime.timedelta:
    """Substraction between datetime.timedelta.
    Equivalent to `timedelta - timedelta(instance)`."""
    return datetime.timedelta_new(
        access_delta_days(delta_l) - access_delta_days(delta_r),
        access_delta_seconds(delta_l) - access_delta_seconds(delta_r),
        access_delta_microseconds(delta_l) - access_delta_microseconds(delta_r),
    )


# Datetime.tzinfo ======================================================================================
# Datetime.tzinfo: generate ----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def gen_tzinfo(offset: cython.int, tzname: str = None) -> datetime.tzinfo:
    """Generate a new `<datetime.tzinfo>` from
    offset (seconds) and tzname (optional)."""
    # Validate offset
    if not -86_340 <= offset <= 86_340:
        raise ValueError(
            "Timezone expected offset between -86,340 and 86,340 (seconds), got %s"
            % offset
        )
    # Generate tzinfo
    delta: datetime.timedelta = datetime.timedelta_new(0, offset, 0)
    if tzname is not None:
        return datetime.PyTimeZone_FromOffsetAndName(delta, tzname)
    else:
        return datetime.PyTimeZone_FromOffset(delta)


@cython.cfunc
@cython.inline(True)
def gen_tzinfo_local(dt: datetime.datetime = None) -> datetime.tzinfo:
    """Generate the local `<datetime.tzinfo>`.
    If `dt` is not specified, use the current local time.
    """
    # Get local timestamp
    if is_dt(dt):
        if access_dt_tzinfo(dt) is None:
            ts = dt_to_posixts(dt)
        else:
            ts = delta_to_seconds(dt_sub_dt(dt, EPOCH_UTC))
    else:
        ts = time()
    # Generate tzinfo
    return gen_tzinfo(time_localtime(ts).tm_gmtoff, None)


# Datetime.tzinfo: check types -------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_tzinfo(obj: object) -> cython.bint:
    """Check if an object is type of datetime.tzinfo `<bool>`. Equivalent
    to `isinstance(obj, datetime.tzinfo)`, includes all subclasses."""
    return datetime.PyTZInfo_Check(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_tzinfo_exact(obj: object) -> cython.bint:
    """Check if an object is the exact type of datetime.tzinfo `<bool>`.
    Equivalent to `type(obj) is datetime.tzinfo`."""
    return datetime.PyTZInfo_CheckExact(obj)


# numpy.datetime64 =====================================================================================
# numpy.datetime64: check types ------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_dt64(obj: object) -> cython.bint:
    """Check if an object is type of numpy.datetime64 `<bool>`.
    Equivalent to `isinstance(obj, numpy.datetime64)`.
    """
    return np.is_datetime64_object(obj)


@cython.cfunc
@cython.inline(True)
def validate_dt64(obj: object):
    """Validate if an object is type of `<numpy.datetime64>`.
    Raise `TypeError` if the object type is incorrect."""
    if not is_dt64(obj):
        raise TypeError(
            "Expect type of `numpy.datetime64`, "
            "instead got: %s %s" % (type(obj), repr(obj))
        )


# numpy.datetime64: conversion -------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
def dt64_to_isoformat(dt64: object) -> str:
    """Convert `numpy.datetime64` to ISO format:
    '%Y-%m-%dT%H:%M:%S.f' `<str>`."""
    # Add back epoch seconds
    microseconds: cython.longlong = dt64_to_microseconds(dt64) + EPOCH_US
    microseconds = min(max(microseconds, DT_MIN_US), DT_MAX_US)
    # Calculate ymd
    ymd = ordinal_to_ymd(microseconds // US_DAY)
    # Calculate hms
    hms = microseconds_to_hms(microseconds)
    # Return isoformat
    if hms.microsecond > 0:
        return "%04d-%02d-%02dT%02d:%02d:%02d.%06d" % (
            ymd.year,
            ymd.month,
            ymd.day,
            hms.hour,
            hms.minute,
            hms.second,
            hms.microsecond,
        )
    else:
        return "%04d-%02d-%02dT%02d:%02d:%02d" % (
            ymd.year,
            ymd.month,
            ymd.day,
            hms.hour,
            hms.minute,
            hms.second,
        )


@cython.cfunc
@cython.inline(True)
def dt64_to_int(
    dt64: object,
    unit: Literal["D", "h", "m", "s", "ms", "us", "ns"],
) -> cython.longlong:
    """Convert `numpy.datetime64` to intger based on the time unit `<int>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than the desired time 'unit'.
    """
    if unit == "D":
        return dt64_to_days(dt64)
    elif unit == "h":
        return dt64_to_hours(dt64)
    elif unit == "m":
        return dt64_to_minutes(dt64)
    elif unit == "s":
        return dt64_to_seconds(dt64)
    elif unit == "ms":
        return dt64_to_miliseconds(dt64)
    elif unit == "us":
        return dt64_to_microseconds(dt64)
    elif unit == "ns":
        return dt64_to_nanoseconds(dt64)
    else:
        raise ValueError(
            "Does not support conversion of "
            "`numpy.datetime64` to time unit: %s." % repr(unit)
        )


@cython.cfunc
@cython.inline(True)
def dt64_to_days(dt64: object) -> cython.longlong:
    """Convert `numpy.datetime64` to total days `<int>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'day (D)'.
    """
    # Access value & unit
    validate_dt64(dt64)
    val: np.npy_datetime = np.get_datetime64_value(dt64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(dt64)

    # Converstion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // NS_DAY
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // US_DAY
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val // 86_400_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val // 86_400
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val // 1_440
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val // 24
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val
    else:
        raise ValueError("Unsupported `numpy.datetime64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def dt64_to_hours(dt64: object) -> cython.longlong:
    """Convert `numpy.datetime64` to total hours `<int>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'hour (h)'.
    """
    # Access value & unit
    validate_dt64(dt64)
    val: np.npy_datetime = np.get_datetime64_value(dt64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(dt64)

    # Converstion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // NS_HOUR
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // US_HOUR
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val // 3_600_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val // 3_600
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val // 60
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * 24
    else:
        raise ValueError("Unsupported `numpy.datetime64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def dt64_to_minutes(dt64: object) -> cython.longlong:
    """Convert `numpy.datetime64` to total minutes `<int>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'minute (m)'.
    """
    # Access value & unit
    validate_dt64(dt64)
    val: np.npy_datetime = np.get_datetime64_value(dt64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(dt64)

    # Converstion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // NS_MINUTE
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // 60_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val // 60_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val // 60
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * 60
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * 1_440
    else:
        raise ValueError("Unsupported `numpy.datetime64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def dt64_to_seconds(dt64: object) -> cython.longlong:
    """Convert `numpy.datetime64` to total seconds `<int>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'second (s)'.
    """
    # Access value & unit
    validate_dt64(dt64)
    val: np.npy_datetime = np.get_datetime64_value(dt64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(dt64)

    # Converstion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // 1_000_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // 1_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val // 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * 60
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * 3_600
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * 86_400
    else:
        raise ValueError("Unsupported `numpy.datetime64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def dt64_to_miliseconds(dt64: object) -> cython.longlong:
    """Convert `numpy.datetime64` to total miliseconds `<int>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'milisecond (ms)'.
    """
    # Access value & unit
    validate_dt64(dt64)
    val: np.npy_datetime = np.get_datetime64_value(dt64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(dt64)

    # Converstion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // 1_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val * 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * 60_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * 3_600_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * 86_400_000
    else:
        raise ValueError("Unsupported `numpy.datetime64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def dt64_to_microseconds(dt64: object) -> cython.longlong:
    """Convert `numpy.datetime64` to total microseconds `<int>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'microsecond (us)'.
    """
    # Access value & unit
    validate_dt64(dt64)
    val: np.npy_datetime = np.get_datetime64_value(dt64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(dt64)

    # Converstion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val * 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val * 1_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * 60_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * US_HOUR
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * US_DAY
    else:
        raise ValueError("Unsupported `numpy.datetime64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def dt64_to_nanoseconds(dt64: object) -> cython.longlong:
    """Convert `numpy.datetime64` to total nanoseconds `<int>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'nanosecond (ns)'.
    """
    # Access value & unit
    validate_dt64(dt64)
    val: np.npy_datetime = np.get_datetime64_value(dt64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(dt64)

    # Converstion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val * 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val * 1_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val * 1_000_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * NS_MINUTE
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * NS_HOUR
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * NS_DAY
    else:
        raise ValueError("Unsupported `numpy.datetime64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def dt64_to_date(dt64: object) -> datetime.date:
    """Convert `numpy.datetime64` to to `<datetime.date>`.

    For value out of `datetime.date` range, result will be clipped to:
    - Upper limit: `<9999-12-31>`
    - Lower limit: `<0001-01-01>`
    """
    return date_fr_ordinal(dt64_to_days(dt64) + EPOCH_DAY)


@cython.cfunc
@cython.inline(True)
def dt64_to_dt(dt64: object) -> datetime.datetime:
    """Convert `numpy.datetime64` to `<datetime.datetime>`.

    For value out of `datetime.datetime` range, result will be clipped to:
    - Upper limit: `<9999-12-31 23:59:59.999999>`.
    - Lower limit: `<0001-01-01 00:00:00.000000>`.
    """
    return dt_fr_microseconds(dt64_to_microseconds(dt64), None)


@cython.cfunc
@cython.inline(True)
def dt64_to_time(dt64: object) -> datetime.time:
    """Convert `numpy.datetime64` to `<datetime.time>`."""
    return time_fr_microseconds(dt64_to_microseconds(dt64), None)


# numpy.timedelta64 ====================================================================================
# numpy.timedelta64: check types -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_delta64(obj: object) -> cython.bint:
    """Check if an object is type of numpy.timedelta64 `<bool>`.
    Equivalent to `isinstance(obj, numpy.timedelta64)`.
    """
    return np.is_timedelta64_object(obj)


@cython.cfunc
@cython.inline(True)
def validate_delta64(obj: object):
    """Validate if an object is type of `<numpy.timedelta64>`.
    Raise `TypeError` if the object type is incorrect."""
    if not is_delta64(obj):
        raise TypeError(
            "Expect type of `numpy.timedelta64`, "
            "instead got: %s %s" % (type(obj), repr(obj))
        )


# numpy.timedelta64: conversion ------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def delta64_to_isoformat(delta64: object) -> str:
    us: cython.longlong = delta64_to_microseconds(delta64)
    days: cython.longlong = us // US_DAY
    secs: cython.longlong = us // 1_000_000
    hours: cython.longlong = secs // 3_600 % 24 + days * 24
    minutes: cython.longlong = secs // 60 % 60
    seconds: cython.longlong = secs % 60
    microseconds: cython.longlong = us % 1_000_000
    if microseconds:
        return "%02d:%02d:%02d.%06d" % (hours, minutes, seconds, microseconds)
    else:
        return "%02d:%02d:%02d" % (hours, minutes, seconds)


@cython.cfunc
@cython.inline(True)
def delta64_to_int(
    delta64: object,
    unit: Literal["D", "h", "m", "s", "ms", "us", "ns"],
) -> cython.longlong:
    """Convert `numpy.timedelta64` to intger based on the time unit `<int>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than the desired time 'unit'.
    """
    if unit == "D":
        return delta64_to_days(delta64)
    elif unit == "h":
        return delta64_to_hours(delta64)
    elif unit == "m":
        return delta64_to_minutes(delta64)
    elif unit == "s":
        return delta64_to_seconds(delta64)
    elif unit == "ms":
        return delta64_to_miliseconds(delta64)
    elif unit == "us":
        return delta64_to_microseconds(delta64)
    elif unit == "ns":
        return delta64_to_nanoseconds(delta64)
    else:
        raise ValueError(
            "Does not support conversion of "
            "`numpy.timedelta64` to time unit: %s." % repr(unit)
        )


@cython.cfunc
@cython.inline(True)
def delta64_to_days(delta64: object) -> cython.longlong:
    """Convert `numpy.timedelta64` to total days `<int>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'day (D)'.
    """
    # Access value & unit
    validate_delta64(delta64)
    val: np.npy_timedelta = np.get_timedelta64_value(delta64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(delta64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // NS_DAY
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // US_DAY
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val // 86_400_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val // 86_400
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val // 1_440
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val // 24
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val
    else:
        raise ValueError("Unsupported `numpy.timedelta64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def delta64_to_hours(delta64: object) -> cython.longlong:
    """Convert `numpy.timedelta64` to total hours `<int>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'hour (h)'.
    """
    # Access value & unit
    validate_delta64(delta64)
    val: np.npy_timedelta = np.get_timedelta64_value(delta64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(delta64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // NS_HOUR
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // US_HOUR
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val // 3_600_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val // 3_600
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val // 60
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * 24
    else:
        raise ValueError("Unsupported `numpy.timedelta64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def delta64_to_minutes(delta64: object) -> cython.longlong:
    """Convert `numpy.timedelta64` to total minutes `<int>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'minute (m)'.
    """
    # Access value & unit
    validate_delta64(delta64)
    val: np.npy_timedelta = np.get_timedelta64_value(delta64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(delta64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // NS_MINUTE
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // 60_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val // 60_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val // 60
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * 60
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * 1_440
    else:
        raise ValueError("Unsupported `numpy.timedelta64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def delta64_to_seconds(delta64: object) -> cython.longlong:
    """Convert `numpy.timedelta64` to total seconds `<int>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'second (s)'.
    """
    # Access value & unit
    validate_delta64(delta64)
    val: np.npy_timedelta = np.get_timedelta64_value(delta64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(delta64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // 1_000_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // 1_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val // 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * 60
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * 3_600
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * 86_400
    else:
        raise ValueError("Unsupported `numpy.timedelta64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def delta64_to_miliseconds(delta64: object) -> cython.longlong:
    """Convert `numpy.timedelta64` to total miliseconds `<int>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'milisecond (ms)'.
    """
    # Access value & unit
    validate_delta64(delta64)
    val: np.npy_timedelta = np.get_timedelta64_value(delta64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(delta64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // 1_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val // 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val * 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * 60_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * 3_600_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * 86_400_000
    else:
        raise ValueError("Unsupported `numpy.timedelta64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def delta64_to_microseconds(delta64: object) -> cython.longlong:
    """Convert `numpy.timedelta64` to total microseconds `<int>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'microsecond (us)'.
    """
    # Access value & unit
    validate_delta64(delta64)
    val: np.npy_timedelta = np.get_timedelta64_value(delta64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(delta64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val * 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val * 1_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * 60_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * US_HOUR
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * US_DAY
    else:
        raise ValueError("Unsupported `numpy.timedelta64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def delta64_to_nanoseconds(delta64: object) -> cython.longlong:
    """Convert `numpy.timedelta64` to total nanoseconds `<int>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'nanosecond (ns)'.
    """
    # Access value & unit
    validate_delta64(delta64)
    val: np.npy_timedelta = np.get_timedelta64_value(delta64)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(delta64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val * 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val * 1_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val * 1_000_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * NS_MINUTE
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * NS_HOUR
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * NS_DAY
    else:
        raise ValueError("Unsupported `numpy.timedelta64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def delta64_to_delta(delta64: object) -> datetime.timedelta:
    """Convert `numpy.timedelta64` to `<datetime.timedelta>`."""
    return delta_fr_microseconds(delta64_to_microseconds(delta64))


# numpy.ndarray[dateimte64] ============================================================================
# numpy.ndarray[dateimte64]: type checks ---------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_dt64array(arr: np.ndarray) -> cython.bint:
    """Check if numpy.ndarray is dtype of ndarray[datetime64] `<bool>`.
    Equivalent to `isinstance(arr.dtype, np.dtypes.DateTime64DType)`.
    """
    return isinstance(arr.dtype, DT64ARRAY_DTYPE)


@cython.cfunc
@cython.inline(True)
def validate_dt64array(arr: np.ndarray):
    """Validate if the numpy.ndarray is dtype of `<ndarray[datetime64]>`.
    Raise `TypeError` if dtype is incorrect."""
    if not is_dt64array(arr):
        raise TypeError(
            "Expect type of `numpy.ndarray[datetime64]`, "
            "instead got: `numpy.ndarray[%s]`." % (arr.dtype)
        )


# numpy.ndarray[dateimte64]: conversion ----------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def dt64array_to_int(
    arr: np.ndarray,
    unit: Literal["D", "h", "m", "s", "ms", "us", "ns"],
) -> np.ndarray:
    """Convert `numpy.ndarray[datetime64]` to integer
    based on the time unit `<numpy.ndarray[int64]>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than the desired time 'unit'.
    """
    if unit == "D":
        return dt64array_to_days(arr)
    elif unit == "h":
        return dt64array_to_hours(arr)
    elif unit == "m":
        return dt64array_to_minutes(arr)
    elif unit == "s":
        return dt64array_to_seconds(arr)
    elif unit == "ms":
        return dt64array_to_miliseconds(arr)
    elif unit == "us":
        return dt64array_to_microseconds(arr)
    elif unit == "ns":
        return dt64array_to_nanoseconds(arr)
    else:
        raise ValueError(
            "Does not support conversion of "
            "`numpy.ndarray[datetime64]` to time unit: %s." % repr(unit)
        )


@cython.cfunc
@cython.inline(True)
def dt64array_to_days(arr: np.ndarray) -> np.ndarray:
    """Convert `numpy.ndarray[datetime64]` to
    total days `<numpy.ndarray[int64]>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'day (D)'.
    """
    # Validate array
    if arr.shape[0] == 0:
        return arr
    validate_dt64array(arr)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0])
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // NS_DAY
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // US_DAY
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr // 86_400_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr // 86_400
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr // 1_440
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr // 24
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr
    else:
        raise ValueError("Unsupported `numpy.datetime64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def dt64array_to_hours(arr: np.ndarray) -> np.ndarray:
    """Convert `numpy.ndarray[datetime64]` to
    total hours `<numpy.ndarray[int64]>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'hour (h)'.
    """
    # Validate array
    if arr.shape[0] == 0:
        return arr
    validate_dt64array(arr)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0])
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // NS_HOUR
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // US_HOUR
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr // 3_600_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr // 3_600
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr // 60
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 24
    else:
        raise ValueError("Unsupported `numpy.datetime64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def dt64array_to_minutes(arr: np.ndarray) -> np.ndarray:
    """Convert `numpy.ndarray[datetime64]` to
    total minutes `<numpy.ndarray[int64]>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'minute (m)'.
    """
    # Validate array
    if arr.shape[0] == 0:
        return arr
    validate_dt64array(arr)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0])
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // NS_MINUTE
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // 60_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr // 60_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr // 60
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * 60
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 1_440
    else:
        raise ValueError("Unsupported `numpy.datetime64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def dt64array_to_seconds(arr: np.ndarray) -> np.ndarray:
    """Convert `numpy.ndarray[datetime64]` to
    total seconds `<numpy.ndarray[int64]>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'second (s)'.
    """
    # Validate array
    if arr.shape[0] == 0:
        return arr
    validate_dt64array(arr)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0])
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // 1_000_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // 1_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr // 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr * 60
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * 3_600
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 86_400
    else:
        raise ValueError("Unsupported `numpy.datetime64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def dt64array_to_miliseconds(arr: np.ndarray) -> np.ndarray:
    """Convert `numpy.ndarray[datetime64]` to
    total miliseconds `<numpy.ndarray[int64]>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'milisecond (ms)'.
    """
    # Validate array
    if arr.shape[0] == 0:
        return arr
    validate_dt64array(arr)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0])
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // 1_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr * 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr * 60_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * 3_600_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 86_400_000
    else:
        raise ValueError("Unsupported `numpy.datetime64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def dt64array_to_microseconds(arr: np.ndarray) -> np.ndarray:
    """Convert `numpy.ndarray[datetime64]` to
    total microseconds `<numpy.ndarray[int64]>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'microsecond (us)'.
    """
    # Validate array
    if arr.shape[0] == 0:
        return arr
    validate_dt64array(arr)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0])
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr * 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr * 1_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr * 60_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * US_HOUR
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * US_DAY
    else:
        raise ValueError("Unsupported `numpy.datetime64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def dt64array_to_nanoseconds(arr: np.ndarray) -> np.ndarray:
    """Convert `numpy.ndarray[datetime64]` to
    total nanoseconds `<numpy.ndarray[int64]>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'nanosecond (ns)'.
    """
    # Validate array
    if arr.shape[0] == 0:
        return arr
    validate_dt64array(arr)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0])
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr * 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr * 1_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr * 1_000_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr * NS_MINUTE
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * NS_HOUR
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * NS_DAY
    else:
        raise ValueError("Unsupported `numpy.datetime64` unit: %s" % unit)


# numpy.ndarray[timedelta64] ===========================================================================
# numpy.ndarray[timedelta64]: check types --------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_delta64array(arr: np.ndarray) -> cython.bint:
    """Check if numpy.ndarray is dtype of ndarray[timedelta64] `<bool>`.
    Equivalent to `isinstance(arr.dtype, np.dtypes.TimeDelta64DType)`.
    """
    return isinstance(arr.dtype, DELTA64ARRAY_DTYPE)


@cython.cfunc
@cython.inline(True)
def validate_delta64array(arr: np.ndarray):
    """Validate if the numpy.ndarray is dtype of `<ndarray[timedelta64]>`.
    Raise `TypeError` if dtype is incorrect."""
    if not is_delta64array(arr):
        raise TypeError(
            "Expect type of `numpy.ndarray[timedelta64]`, "
            "instead got: `numpy.ndarray[%s]`." % (arr.dtype)
        )


# numpy.ndarray[timedelta64]: conversion ---------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def delta64array_to_int(
    arr: np.ndarray,
    unit: Literal["D", "h", "m", "s", "ms", "us", "ns"],
) -> np.ndarray:
    """Convert `numpy.ndarray[timedelta64]` to integer
    based on the time unit `<numpy.ndarray[int64]>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than the desired time 'unit'.
    """
    if unit == "D":
        return delta64array_to_days(arr)
    elif unit == "h":
        return delta64array_to_hours(arr)
    elif unit == "m":
        return delta64array_to_minutes(arr)
    elif unit == "s":
        return delta64array_to_seconds(arr)
    elif unit == "ms":
        return delta64array_to_miliseconds(arr)
    elif unit == "us":
        return delta64array_to_microseconds(arr)
    elif unit == "ns":
        return delta64array_to_nanoseconds(arr)
    else:
        raise ValueError(
            "Does not support conversion of "
            "`numpy.ndarray[timedelta64]` to time unit: %s." % repr(unit)
        )


@cython.cfunc
@cython.inline(True)
def delta64array_to_days(arr: np.ndarray) -> np.ndarray:
    """Convert `numpy.ndarray[timedelta64]` to
    total days `<numpy.ndarray[int64]>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'day (D)'.
    """
    # Validate array
    if arr.shape[0] == 0:
        return arr
    validate_delta64array(arr)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0])
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // NS_DAY
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // US_DAY
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr // 86_400_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr // 86_400
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr // 1_440
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr // 24
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr
    else:
        raise ValueError("Unsupported `numpy.timedelta64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def delta64array_to_hours(arr: np.ndarray) -> np.ndarray:
    """Convert `numpy.ndarray[timedelta64]` to
    total hours `<numpy.ndarray[int64]>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'hour (h)'.
    """
    # Validate array
    if arr.shape[0] == 0:
        return arr
    validate_delta64array(arr)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0])
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // NS_HOUR
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // US_HOUR
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr // 3_600_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr // 3_600
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr // 60
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 24
    else:
        raise ValueError("Unsupported `numpy.timedelta64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def delta64array_to_minutes(arr: np.ndarray) -> np.ndarray:
    """Convert `numpy.ndarray[timedelta64]` to
    total minutes `<numpy.ndarray[int64]>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'minute (m)'.
    """
    # Validate array
    if arr.shape[0] == 0:
        return arr
    validate_delta64array(arr)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0])
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // NS_MINUTE
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // 60_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr // 60_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr // 60
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * 60
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 1_440
    else:
        raise ValueError("Unsupported `numpy.timedelta64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def delta64array_to_seconds(arr: np.ndarray) -> np.ndarray:
    """Convert `numpy.ndarray[timedelta64]` to
    total seconds `<numpy.ndarray[int64]>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'second (s)'.
    """
    # Validate array
    if arr.shape[0] == 0:
        return arr
    validate_delta64array(arr)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0])
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // 1_000_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // 1_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr // 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr * 60
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * 3_600
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 86_400
    else:
        raise ValueError("Unsupported `numpy.timedelta64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def delta64array_to_miliseconds(arr: np.ndarray) -> np.ndarray:
    """Convert `numpy.ndarray[timedelta64]` to
    total miliseconds `<numpy.ndarray[int64]>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'milisecond (ms)'.
    """
    # Validate array
    if arr.shape[0] == 0:
        return arr
    validate_delta64array(arr)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0])
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // 1_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr // 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr * 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr * 60_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * 3_600_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * 86_400_000
    else:
        raise ValueError("Unsupported `numpy.timedelta64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def delta64array_to_microseconds(arr: np.ndarray) -> np.ndarray:
    """Convert `numpy.ndarray[timedelta64]` to
    total microseconds `<numpy.ndarray[int64]>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'microsecond (us)'.
    """
    # Validate array
    if arr.shape[0] == 0:
        return arr
    validate_delta64array(arr)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0])
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr // 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr * 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr * 1_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr * 60_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * US_HOUR
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * US_DAY
    else:
        raise ValueError("Unsupported `numpy.timedelta64` unit: %s" % unit)


@cython.cfunc
@cython.inline(True)
def delta64array_to_nanoseconds(arr: np.ndarray) -> np.ndarray:
    """Convert `numpy.ndarray[timedelta64]` to
    total nanoseconds `<numpy.ndarray[int64]>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'nanosecond (ns)'.
    """
    # Validate array
    if arr.shape[0] == 0:
        return arr
    validate_delta64array(arr)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0])
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)

    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return arr
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return arr * 1_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return arr * 1_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return arr * 1_000_000_000
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return arr * NS_MINUTE
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return arr * NS_HOUR
    elif unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return arr * NS_DAY
    else:
        raise ValueError("Unsupported `numpy.timedelta64` unit: %s" % unit)


# pandas.Series ========================================================================================
# pandas.Series type checks ----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_pdseries(obj: object) -> cython.bint:
    """Check if an object is type of pandas.Series `<bool>`."""
    return isinstance(obj, PDSERIES_DTYPE)


@cython.cfunc
@cython.inline(True)
def validate_pdseries(obj: object):
    if not is_pdseries(obj):
        raise TypeError(
            "Expect type of `pandas.Series`, instead got: %s." % (type(obj))
        )


# pandas.Series[datetime64] ============================================================================
# pandas.Series[datetime64] conversion -----------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def dt64series_to_int(
    s: Series,
    unit: Literal["D", "h", "m", "s", "ms", "us", "ns"],
) -> object:
    """Convert `pandas.Series[datetime64]` to integer
    based on the time unit `<pandas.Series[int64]>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than the desired time 'unit'.
    """
    if unit == "D":
        return dt64series_to_days(s)
    elif unit == "h":
        return dt64series_to_hours(s)
    elif unit == "m":
        return dt64series_to_minutes(s)
    elif unit == "s":
        return dt64series_to_seconds(s)
    elif unit == "ms":
        return dt64series_to_miliseconds(s)
    elif unit == "us":
        return dt64series_to_microseconds(s)
    elif unit == "ns":
        return dt64series_to_nanoseconds(s)
    else:
        raise ValueError(
            "Does not support conversion of "
            "`pandas.Series[datetime64]` to time unit: %s." % repr(unit)
        )


@cython.cfunc
@cython.inline(True)
def dt64series_to_days(s: Series) -> object:
    """Convert `pandas.Series[datetime64]` to
    total days `<pandas.Series[int64]>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'day (D)'.
    """
    # Validate series
    validate_pdseries(s)
    # Conversion
    arr = dt64array_to_days(s.values)
    # Reconstruction
    return Series(arr, index=s.index)


@cython.cfunc
@cython.inline(True)
def dt64series_to_hours(s: Series) -> object:
    """Convert `pandas.Series[datetime64]` to
    total hours `<pandas.Series[int64]>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'hour (H)'.
    """
    # Validate series
    validate_pdseries(s)
    # Conversion
    arr = dt64array_to_hours(s.values)
    # Reconstruction
    return Series(arr, index=s.index)


@cython.cfunc
@cython.inline(True)
def dt64series_to_minutes(s: Series) -> object:
    """Convert `pandas.Series[datetime64]` to
    total minutes `<pandas.Series[int64]>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'minute (m)'.
    """
    # Validate series
    validate_pdseries(s)
    # Conversion
    arr = dt64array_to_minutes(s.values)
    # Reconstruction
    return Series(arr, index=s.index)


@cython.cfunc
@cython.inline(True)
def dt64series_to_seconds(s: Series) -> object:
    """Convert `pandas.Series[datetime64]` to
    total seconds `<pandas.Series[int64]>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'second (s)'.
    """
    # Validate series
    validate_pdseries(s)
    # Conversion
    arr = dt64array_to_seconds(s.values)
    # Reconstruction
    return Series(arr, index=s.index)


@cython.cfunc
@cython.inline(True)
def dt64series_to_miliseconds(s: Series) -> object:
    """Convert `pandas.Series[datetime64]` to
    total miliseconds `<pandas.Series[int64]>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'milisecond (ms)'.
    """
    # Validate series
    validate_pdseries(s)
    # Conversion
    arr = dt64array_to_miliseconds(s.values)
    # Reconstruction
    return Series(arr, index=s.index)


@cython.cfunc
@cython.inline(True)
def dt64series_to_microseconds(s: Series) -> object:
    """Convert `pandas.Series[datetime64]` to
    total microseconds `<pandas.Series[int64]>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'microsecond (us)'.
    """
    # Validate series
    validate_pdseries(s)
    # Conversion
    arr = dt64array_to_microseconds(s.values)
    # Reconstruction
    return Series(arr, index=s.index)


@cython.cfunc
@cython.inline(True)
def dt64series_to_nanoseconds(s: Series) -> object:
    """Convert `pandas.Series[datetime64]` to
    total nanoseconds `<pandas.Series[int64]>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'nanosecond (ns)'.
    """
    # Validate series
    validate_pdseries(s)
    # Conversion
    arr = dt64array_to_nanoseconds(s.values)
    # Reconstruction
    return Series(arr, index=s.index)


@cython.cfunc
@cython.inline(True)
def dt64series_to_ordinals(s: Series) -> object:
    """Convert `pandas.Series[datetime64]` to
    ordinals `<pandas.Series[int64]>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'day (D)'.
    """
    # Validate series
    validate_pdseries(s)
    # Conversion
    arr = dt64array_to_days(s.values) + 719_163
    # Reconstruction
    return Series(arr, index=s.index)


@cython.cfunc
@cython.inline(True)
def dt64series_to_timestamps(s: Series) -> object:
    """Convert `pandas.Series[datetime64]` to
    timestamps `<pandas.Series[float64]>`.

    ### Notice
    Percision will be lost if the original datetime64
    unit is smaller than 'microseconds (us)'.
    """
    # Validate series
    validate_pdseries(s)
    # Conversion
    arr = dt64array_to_microseconds(s.values) / 1_000_000
    # Reconstruction
    return Series(arr, index=s.index)


# pandas.Series[datetime64] adjustment -----------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def dt64series_adj_to_ns(s: Series) -> object:
    """Adjust `pandas.Series[datetime64]` to `datetime64[ns]`.

    This function tries to adjust any non-`datetime64[ns]` Series
    to `datetime64[ns]` by converting the values to nanoseconds.
    Support both timezone-naive and timezone-aware Series.
    """
    # Validate series
    validate_pdseries(s)
    # Check dtype
    dtype: str = s.dtype.str
    if dtype == "<M8[ns]" or dtype == "|M8[ns]":
        return s  # exit: already datetime64[us]
    # Adjust to nanoseconds
    arr = dt64array_to_nanoseconds(s.values)
    # Reconstruction
    return Series(DatetimeIndex(arr, tz=s.dt.tz), index=s.index)


# pandas.Series[timedelta64] ===========================================================================
# pandas.Series[timedelta64]: conversion ---------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def delta64series_to_int(
    s: Series,
    unit: Literal["D", "h", "m", "s", "ms", "us", "ns"],
) -> object:
    """Convert `pandas.Series[timedelta64]` to integer
    based on the time unit `<pandas.Series[int64]>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than the desired time 'unit'.
    """
    if unit == "D":
        return delta64series_to_days(s)
    elif unit == "h":
        return delta64series_to_hours(s)
    elif unit == "m":
        return delta64series_to_minutes(s)
    elif unit == "s":
        return delta64series_to_seconds(s)
    elif unit == "ms":
        return delta64series_to_miliseconds(s)
    elif unit == "us":
        return delta64series_to_microseconds(s)
    elif unit == "ns":
        return delta64series_to_nanoseconds(s)
    else:
        raise ValueError(
            "Does not support conversion of "
            "`pandas.Series[timedelta64]` to time unit: %s." % repr(unit)
        )


@cython.cfunc
@cython.inline(True)
def delta64series_to_days(s: Series) -> object:
    """Convert `pandas.Series[timedelta64]` to
    total days `<pandas.Series[int64]>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'day (D)'.
    """
    # Validate series
    validate_pdseries(s)
    # Conversion
    arr = delta64array_to_days(s.values)
    # Reconstruction
    return Series(arr, index=s.index)


@cython.cfunc
@cython.inline(True)
def delta64series_to_hours(s: Series) -> object:
    """Convert `pandas.Series[timedelta64]` to
    total hours `<pandas.Series[int64]>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'hour (h)'.
    """
    # Validate series
    validate_pdseries(s)
    # Conversion
    arr = delta64array_to_hours(s.values)
    # Reconstruction
    return Series(arr, index=s.index)


@cython.cfunc
@cython.inline(True)
def delta64series_to_minutes(s: Series) -> object:
    """Convert `pandas.Series[timedelta64]` to
    total minutes `<pandas.Series[int64]>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'minute (m)'.
    """
    # Validate series
    validate_pdseries(s)
    # Conversion
    arr = delta64array_to_minutes(s.values)
    # Reconstruction
    return Series(arr, index=s.index)


@cython.cfunc
@cython.inline(True)
def delta64series_to_seconds(s: Series) -> object:
    """Convert `pandas.Series[timedelta64]` to
    total seconds `<pandas.Series[int64]>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'second (s)'.
    """
    # Validate series
    validate_pdseries(s)
    # Conversion
    arr = delta64array_to_seconds(s.values)
    # Reconstruction
    return Series(arr, index=s.index)


@cython.cfunc
@cython.inline(True)
def delta64series_to_miliseconds(s: Series) -> object:
    """Convert `pandas.Series[timedelta64]` to
    total miliseconds `<pandas.Series[int64]>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'milisecond (ms)'.
    """
    # Validate series
    validate_pdseries(s)
    # Conversion
    arr = delta64array_to_miliseconds(s.values)
    # Reconstruction
    return Series(arr, index=s.index)


@cython.cfunc
@cython.inline(True)
def delta64series_to_microseconds(s: Series) -> object:
    """Convert `pandas.Series[timedelta64]` to
    total microseconds `<pandas.Series[int64]>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'microsecond (us)'.
    """
    # Validate series
    validate_pdseries(s)
    # Conversion
    arr = delta64array_to_microseconds(s.values)
    # Reconstruction
    return Series(arr, index=s.index)


@cython.cfunc
@cython.inline(True)
def delta64series_to_nanoseconds(s: Series) -> object:
    """Convert `pandas.Series[timedelta64]` to
    total nanoseconds `<pandas.Series[int64]>`.

    ### Notice
    Percision will be lost if the original timedelta64
    unit is smaller than 'nanosecond (ns)'.
    """
    # Validate series
    validate_pdseries(s)
    # Conversion
    arr = delta64array_to_nanoseconds(s.values)
    # Reconstruction
    return Series(arr, index=s.index)


# pandas.Series[timedelta64]: adjustment ---------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def delta64series_adj_to_ns(s: Series) -> object:
    """Adjust `pandas.Series[timedelta64]` to `timedelta64[ns]`.

    This function tries to adjust any non-`timedelta64[ns]` Series
    to `timedelta64[ns]` by converting the values to nanoseconds.
    """
    # Validate series
    validate_pdseries(s)
    # Check dtype
    dtype: str = s.dtype.str
    if dtype == "<m8[ns]" or dtype == "|m8[ns]":
        return s  # exit: already timedelta64[ns]
    # Adjust to nanoseconds
    arr = delta64array_to_nanoseconds(s.values)
    # Reconstruction
    return Series(TimedeltaIndex(arr), index=s.index)
