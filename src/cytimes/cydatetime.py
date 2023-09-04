# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports.cpython.time import time as _time  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cpython.datetime import date_new as _date_new  # type: ignore
from cython.cimports.cpython.datetime import date_from_timestamp as _date_from_timestamp  # type: ignore
from cython.cimports.cpython.datetime import datetime_new as _datetime_new  # type: ignore
from cython.cimports.cpython.datetime import datetime_tzinfo as _datetime_tzinfo  # type: ignore
from cython.cimports.cpython.datetime import datetime_from_timestamp as _datetime_from_timestamp  # type: ignore
from cython.cimports.cpython.datetime import time_new as _time_new  # type: ignore
from cython.cimports.cpython.datetime import time_tzinfo as _time_tzinfo  # type: ignore
from cython.cimports.cpython.datetime import timedelta_new as _timedelta_new  # type: ignore
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.numpy import is_datetime64_object as _is_datetime64_object  # type: ignore
from cython.cimports.numpy import get_datetime64_unit as _get_datetime64_unit  # type: ignore
from cython.cimports.numpy import get_datetime64_value as _get_datetime64_value  # type: ignore
from cython.cimports.numpy import is_timedelta64_object as _is_timedelta64_object  # type: ignore
from cython.cimports.numpy import get_timedelta64_value as _get_timedelta64_value  # type: ignore
from cython.cimports.cytimes import cymath  # type: ignore
from cython.cimports.cytimes import cytime  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
from cytimes import cymath
from time import localtime as _localtime
import datetime, numpy as np, pandas as pd


# Constants --------------------------------------------------------------------------------------------
# fmt: off
MAX_ORD: cython.int = 3652059
DI400Y: cython.int = 146097
DI100Y: cython.int = 36524
DI4Y: cython.int = 1461
DI1Y: cython.int = 365

DAYS_BR_MONTH: cython.int[13] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
DAYS_BR_QUARTER: cython.int[5] = [0, 90, 181, 273, 365]

UTC: datetime.timezone = datetime.get_utc()
EPOCH_NAI: datetime.datetime = _datetime_new(1970, 1, 1, 0, 0, 0, 0, None, 0)  # type: ignore
EPOCH_UTC: datetime.datetime = _datetime_new(1970, 1, 1, 0, 0, 0, 0, UTC, 0)  # type: ignore
EPOCH_US: cython.longlong = 62135683200000000
EPOCH_SEC: cython.longlong = 62135683200
EPOCH_DAY: cython.longlong = 719163

DT_MIN_US: cython.longlong = 86400000000
DT_MAX_US: cython.longlong = 315537983999999999

NS_DAY: cython.longlong = 86400000000000
NS_HOUR: cython.longlong = 3600000000000
NS_MINUTE: cython.longlong = 60000000000
NS_SECOND: cython.longlong = 1000000000
NS_MILLISECOND: cython.longlong = 1000000
NS_MICROSECOND: cython.longlong = 1000
NS_NANOSECOND: cython.longlong = 1

US_DAY: cython.longlong = 86400000000
US_HOUR: cython.longlong = 3600000000
US_MINUTE: cython.longlong = 60000000
US_SECOND: cython.longlong = 1000000
US_MILLISECOND: cython.longlong = 1000
US_MICROSECOND: cython.longlong = 1

MS_DAY: cython.longlong = 86400000
MS_HOUR: cython.longlong = 3600000
MS_MINUTE: cython.longlong = 60000
MS_SECOND: cython.longlong = 1000
MS_MILLISECOND: cython.longlong = 1

SEC_DAY: cython.longlong = 86400
SEC_HOUR: cython.longlong = 3600
SEC_MINUTE: cython.longlong = 60
SEC_SECOND: cython.longlong = 1

MIN_DAY: cython.longlong = 1440
MIN_HOUR: cython.longlong = 60
MIN_MINUTE: cython.longlong = 1

HOUR_DAY: cython.longlong = 24
HOUR_HOUR: cython.longlong = 1
# fmt: on

# Struct -----------------------------------------------------------------------------------------------
ymd = cython.struct(
    year=cython.int,
    month=cython.int,
    day=cython.int,
)
hms = cython.struct(
    hour=cython.int,
    minute=cython.int,
    second=cython.int,
    microsecond=cython.int,
)
iso = cython.struct(
    year=cython.int,
    week=cython.int,
    weekday=cython.int,
)


# Calendar Year ----------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_leapyear(year: cython.int) -> cython.bint:
    "Determine whether is a leap year."
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def leap_bt_years(year1: cython.int, year2: cython.int) -> cython.int:
    "Calculate number of leap years between year1 and year2."
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
def days_in_year(year: cython.int) -> cython.int:
    """Number of days in the year. Expect 365 or 366 (leapyear)."""
    return 366 if is_leapyear(year) else 365


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_bf_year(year: cython.int) -> cython.int:
    "Number of days before January 1st of the year."
    y: cython.int = year - 1
    return y * 365 + y // 4 - y // 100 + y // 400


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_of_year(year: cython.int, month: cython.int, day: cython.int) -> cython.int:
    "Number of days into the year.."
    return days_bf_month(year, month) + min(days_in_month(year, month), day)


# Calendar Quarter -------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def quarter_of_month(month: cython.int) -> cython.int:
    "The quarter of the month."
    return cymath.clip((month - 1) // 3 + 1, 1, 4)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_in_quarter(year: cython.int, month: cython.int) -> cython.int:
    "Number of days in the quarter of the year."
    quarter: cython.int = quarter_of_month(month)
    days: cython.int = DAYS_BR_QUARTER[quarter] - DAYS_BR_QUARTER[quarter - 1]

    if quarter == 1 and is_leapyear(year):
        days += 1
    return days


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_bf_quarter(year: cython.int, month: cython.int) -> cython.int:
    "Number of days in the year preceding first day of the quarter."
    quarter: cython.int = quarter_of_month(month)
    days: cython.int = DAYS_BR_QUARTER[quarter - 1]

    if quarter >= 2 and is_leapyear(year):
        days += 1
    return days


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_of_quarter(year: cython.int, month: cython.int, day: cython.int) -> cython.int:
    "Number of days into the quarter."
    return days_of_year(year, month, day) - days_bf_quarter(year, month)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def quarter_1st_month(month: cython.int) -> cython.int:
    "First month of the quarter."
    return 3 * quarter_of_month(month) - 2


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def quarter_lst_month(month: cython.int) -> cython.int:
    "Last month of the quarter."
    return 3 * quarter_of_month(month)


# Calendar Month ---------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_in_month(year: cython.int, month: cython.int) -> cython.int:
    "Number of days in the month of the year."
    # Adjust year
    if not 1 <= year <= 9999:
        year = 2000 + year % 400
    # Jan to Jul
    if 1 <= month <= 7:
        if month == 2:
            return 29 if is_leapyear(year) else 28
        else:
            return 31 if month % 2 == 1 else 30
    # Aug to Dec
    elif 7 < month <= 12:
        return 30 if month % 2 == 1 else 31
    # Clip invalid month
    else:
        return 31


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def days_bf_month(year: cython.int, month: cython.int) -> cython.int:
    "Number of days in the year preceding first day of the month."
    if month <= 2:
        if month <= 1:
            return 0
        else:
            return 31

    if month > 12:
        month = 12
    extra: cython.int = 1 if is_leapyear(year) else 0
    return DAYS_BR_MONTH[month - 1] + extra


# Calendar Week ----------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def ymd_weekday(year: cython.int, month: cython.int, day: cython.int) -> cython.int:
    "The day of the week, where Monday == 0 ... Sunday == 6."
    return (ymd_to_ordinal(year, month, day) + 6) % 7


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def ymd_isoweekday(year: cython.int, month: cython.int, day: cython.int) -> cython.int:
    "The ISO day of the week, where Monday == 1 ... Sunday == 7."
    return ymd_to_ordinal(year, month, day) % 7 or 7


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def ymd_isoweek(year: cython.int, month: cython.int, day: cython.int) -> cython.int:
    "The ISO calendar week number of the YMD."
    ordinal: cython.int = ymd_to_ordinal(year, month, day)
    week1st: cython.int = isoweek_1st_ordinal(year)
    week: cython.int = (ordinal - week1st) // 7

    if week < 0:
        week1st = isoweek_1st_ordinal(year - 1)
        return (ordinal - week1st) // 7 + 1
    elif week >= 52 and ordinal >= isoweek_1st_ordinal(year + 1):
        return 1
    else:
        return week + 1


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def ymd_isoyear(year: cython.int, month: cython.int, day: cython.int) -> cython.int:
    "The ISO calendar year of the YMD."
    ordinal: cython.int = ymd_to_ordinal(year, month, day)
    week1st: cython.int = isoweek_1st_ordinal(year)
    week: cython.int = (ordinal - week1st) // 7

    if week < 0:
        return year - 1
    elif week >= 52 and ordinal >= isoweek_1st_ordinal(year + 1):
        return year + 1
    else:
        return year


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def ymd_isocalendar(
    year: cython.int,
    month: cython.int,
    day: cython.int,
) -> iso:
    "The ISO calendar of the YMD."
    ordinal: cython.int = ymd_to_ordinal(year, month, day)
    week1st: cython.int = isoweek_1st_ordinal(year)
    delta: cython.int = ordinal - week1st
    week: cython.int = delta // 7

    if week < 0:
        year -= 1
        week1st = isoweek_1st_ordinal(year)
        delta = ordinal - week1st
        week = delta // 7
    elif week >= 52 and ordinal >= isoweek_1st_ordinal(year + 1):
        year += 1
        week = 0

    return iso(year, week + 1, delta % 7 + 1)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def isoweek_1st_ordinal(year: cython.int) -> cython.int:
    "The ordinal of the 1st iso week's monday of the year."
    day_1st: cython.int = ymd_to_ordinal(year, 1, 1)
    weekday_1st: cython.int = (day_1st + 6) % 7
    week_1st_mon: cython.int = day_1st - weekday_1st

    return week_1st_mon + 7 if weekday_1st > 3 else week_1st_mon


# Calendar Conversion ----------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def ymd_to_ordinal(year: cython.int, month: cython.int, day: cython.int) -> cython.int:
    "Convert year, month, day to ordinal."
    return (
        days_bf_year(year)
        + days_bf_month(year, month)
        + min(day, days_in_month(year, month))
    )


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
@cython.exceptval(check=False)
def ordinal_to_ymd(ordinal: cython.int) -> ymd:
    "Convert ordinal to (year, month, day)."
    # n is a 1-based index, starting at 1-Jan-1.  The pattern of leap years
    # repeats exactly every 400 years.  The basic strategy is to find the
    # closest 400-year boundary at or before n, then work with the offset
    # from that boundary to n.  Life is much clearer if we subtract 1 from
    # n first -- then the values of n at 400-year boundaries are exactly
    # those divisible by _DI400Y:
    n: cython.int = cymath.clip(ordinal, 1, MAX_ORD) - 1
    n400: cython.int = n // DI400Y
    n = n % DI400Y
    year: cython.int = n400 * 400 + 1

    # Now n is the (non-negative) offset, in days, from January 1 of year, to
    # the desired date.  Now compute how many 100-year cycles precede n.
    # Note that it's possible for n100 to equal 4!  In that case 4 full
    # 100-year cycles precede the desired day, which implies the desired
    # day is December 31 at the end of a 400-year cycle.
    n100: cython.int = n // DI100Y
    n = n % DI100Y

    # Now compute how many 4-year cycles precede it.
    n4: cython.int = n // DI4Y
    n = n % DI4Y

    # And now how many single years.  Again n1 can be 4, and again meaning
    # that the desired day is December 31 at the end of the 4-year cycle.
    n1: cython.int = n // DI1Y
    n = n % DI1Y

    year += n100 * 100 + n4 * 4 + n1
    if n1 == 4 or n100 == 4:
        # Return ymd
        return ymd(year - 1, 12, 31)

    # Now the year is correct, and n is the offset from January 1.  We find
    # the month via an estimate that's either exact or one too large.
    month: cython.int = (n + 50) >> 5
    days_bf: cython.int = days_bf_month(year, month)
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
    "Convert microseconds to (hour, minute, second, microsecond)."
    # Pre binding
    hour: cython.int
    minute: cython.int
    second: cython.int
    microsecond: cython.int

    # Convert
    if microseconds > 0:
        microseconds = microseconds % US_DAY
        hour = microseconds // US_HOUR
        microseconds = microseconds % US_HOUR
        minute = microseconds // US_MINUTE
        microseconds = microseconds % US_MINUTE
        second = microseconds // US_SECOND
        microsecond = microseconds % US_SECOND
    else:
        hour, minute, second, microsecond = 0, 0, 0, 0

    # Return hms
    return hms(hour, minute, second, microsecond)


# Time -------------------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time() -> cython.double:
    "Equivalent to python `time.time()`"
    return _time()


@cython.cfunc
@cython.inline(True)
def localtime() -> cytime.tm:
    "Equivalent to `time.localtime()`."
    return cytime.localtime()


@cython.cfunc
@cython.inline(True)
def localize_time(timestamp: cython.double) -> cytime.tm:
    "Equivalent to `time.localtime(timestamp)`."
    return cytime.localize_time(timestamp)


@cython.cfunc
@cython.inline(True)
def localize_timestamp(timestamp: cython.double) -> cython.longlong:
    "Equivalent to `datetime.mktime(timestamp)`."
    # Localize timestamp
    tms = cytime.localize_time(timestamp)
    # Seconds since epoch
    total_seconds: cython.longlong = (
        ymd_to_ordinal(tms.tm_year, tms.tm_mon, tms.tm_mday) * SEC_DAY
    )
    total_seconds += tms.tm_hour * SEC_HOUR + tms.tm_min * SEC_MINUTE + tms.tm_sec
    # Return local timestamp
    return total_seconds - EPOCH_SEC


# Date: Generate ---------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def gen_date(
    year: cython.int = 1,
    month: cython.int = 1,
    day: cython.int = 1,
) -> datetime.date:
    "Generate `datetime.date`."
    return _date_new(year, month, day)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def gen_date_local() -> datetime.date:
    "Generate datetime.date from local time. Equivalent to `datetime.date.today()`."
    tms = cytime.localtime()
    return _date_new(tms.tm_year, tms.tm_mon, tms.tm_mday)


@cython.cfunc
@cython.inline(True)
def date_mktime(date: datetime.date) -> cython.longlong:
    "Generate timestamp from datetime.date."
    base_ts: cython.longlong = date_to_seconds(date)  # Seconds after POSIX epoch.
    offset: cython.longlong = localize_timestamp(base_ts) - base_ts
    return base_ts - offset


# Date: Check Types ------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_date(obj: object) -> cython.bint:
    "Check if an obj is `datetime.date` (include subclass such as `datetime.datetime`)."
    return datetime.PyDate_Check(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_date_exact(obj: object) -> cython.bint:
    "Check if an obj is exactly `datetime.date`."
    return datetime.PyDate_CheckExact(obj)


# Date: Get Attribute ----------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_year(obj: object) -> cython.int:
    "Get the 'year' attribute of `datetime.date` or `datetime.datetime`."
    return datetime.PyDateTime_GET_YEAR(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_month(obj: object) -> cython.int:
    "Get the 'month' attribute of `datetime.date` or `datetime.datetime`."
    return datetime.PyDateTime_GET_MONTH(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_day(obj: object) -> cython.int:
    "Get the 'day' attribute of `datetime.date` or `datetime.datetime`."
    return datetime.PyDateTime_GET_DAY(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_weekday(obj: object) -> cython.int:
    "Get the 'weekday' of `datetime.date` or `datetime.datetime`, where Monday == 0 ... Sunday == 6."
    return ymd_weekday(get_year(obj), get_month(obj), get_day(obj))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_isoweekday(obj: object) -> cython.int:
    "Get the 'isoweekday' of `datetime.date` or `datetime.datetime`, where Monday == 1 ... Sunday == 7."
    return ymd_isoweekday(get_year(obj), get_month(obj), get_day(obj))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_isoweek(obj: object) -> cython.int:
    "Get the 'week' of the ISO calendar for `datetime.date` or `datetime.datetime`."
    return ymd_isoweek(get_year(obj), get_month(obj), get_day(obj))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_isoyear(obj: object) -> cython.int:
    "Get the 'year' of the ISO calendar for `datetime.date` or `datetime.datetime`."
    return ymd_isoyear(get_year(obj), get_month(obj), get_day(obj))


@cython.cfunc
@cython.inline(True)
def get_isocalendar(obj: object) -> iso:
    "Get the ISO calendar of `datetime.date` or `datetime.datetime`."
    return ymd_isocalendar(get_year(obj), get_month(obj), get_day(obj))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_is_leapyear(obj: object) -> cython.bint:
    "Get whether the `datetime.date` or `datetime.datetime` is a leap year."
    return is_leapyear(get_year(obj))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_days_in_year(obj: object) -> cython.int:
    "Get the number of days in the year of `datetime.date` or `datetime.datetime`."
    return days_in_year(get_year(obj))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_days_bf_year(obj: object) -> cython.int:
    "Get the number of days before the year (Jan 1st) of `datetime.date` or `datetime.datetime`."
    return days_bf_year(get_year(obj))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_days_of_year(obj: object) -> cython.int:
    "Get the number of days into the year of `datetime.date` or `datetime.datetime`."
    return days_of_year(get_year(obj), get_month(obj), get_day(obj))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_quarter(obj: object) -> cython.int:
    "Get the quarter of `datetime.date` or `datetime.datetime`."
    return quarter_of_month(get_month(obj))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_days_in_quarter(obj: object) -> cython.int:
    "Get the number of days in the quarter of `datetime.date` or `datetime.datetime`."
    return days_in_quarter(get_year(obj), get_month(obj))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_days_bf_quarter(obj: object) -> cython.int:
    "Get the number of days in the year preceding first day of the quarter of `datetime.date` or `datetime.datetime`."
    return days_bf_quarter(get_year(obj), get_month(obj))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_days_of_quarter(obj: object) -> cython.int:
    "Get the number of days into the quarter of `datetime.date` or `datetime.datetime`."
    return days_of_quarter(get_year(obj), get_month(obj), get_day(obj))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_quarter_1st_month(obj: object) -> cython.int:
    "Get the first month of the quarter of `datetime.date` or `datetime.datetime`."
    return quarter_1st_month(get_month(obj))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_quarter_lst_month(obj: object) -> cython.int:
    "Get the last month of the quarter of `datetime.date` or `datetime.datetime`."
    return quarter_lst_month(get_month(obj))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_days_in_month(obj: object) -> cython.int:
    "Get the number of days in the month of `datetime.date` or `datetime.datetime`."
    return days_in_month(get_year(obj), get_month(obj))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_days_bf_month(obj: object) -> cython.int:
    "Get the number of days in the year preceding 1st day of the month of `datetime.date` or `datetime.datetime`."
    return days_bf_month(get_year(obj), get_month(obj))


# Date: Conversion -------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def to_ordinal(obj: object) -> cython.int:
    "Convert `datetime.date` or `datetime.datetime` to ordinal."
    return ymd_to_ordinal(get_year(obj), get_month(obj), get_day(obj))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_to_isoformat(date: datetime.date) -> str:
    "Convert `datetime.date` to ISO format."
    return "%04d-%02d-%02d" % (get_year(date), get_month(date), get_day(date))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_to_seconds(date: datetime.date) -> cython.longlong:
    "Convert `datetime.date` to total seconds after POSIX epoch."
    return (to_ordinal(date) - EPOCH_DAY) * SEC_DAY


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_to_microseconds(date: datetime.date) -> cython.longlong:
    "Convert `datetime.date` to total microseconds after POSIX epoch."
    return (to_ordinal(date) - EPOCH_DAY) * US_DAY


@cython.cfunc
@cython.inline(True)
def date_to_timestamp(date: datetime.date) -> cython.longlong:
    return date_mktime(date)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_fr_dt(dt: datetime.datetime) -> datetime.date:
    "Create `datetime.date` from `datetime.datetime`."
    return _date_new(get_year(dt), get_month(dt), get_day(dt))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_fr_ordinal(ordinal: cython.int) -> datetime.date:
    "Create `datetime.date` from ordinal."
    # Convert ordinal to ymd
    ymd = ordinal_to_ymd(ordinal)
    # Generate date
    return _date_new(ymd.year, ymd.month, ymd.day)


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
@cython.exceptval(check=False)
def date_fr_seconds(seconds: cython.double) -> datetime.date:
    "Create `datetime.date` from total seconds after POSIX epoch."
    total_sec: cython.longlong = int(seconds)
    return date_fr_ordinal((total_sec + EPOCH_SEC) // SEC_DAY)


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
@cython.exceptval(check=False)
def date_fr_microseconds(microseconds: cython.longlong) -> datetime.date:
    "Create `datetime.date` from total microseconds after POSIX epoch."
    return date_fr_ordinal((microseconds + EPOCH_US) // US_DAY)


@cython.cfunc
@cython.inline(True)
def date_fr_timestamp(timestamp: cython.double) -> datetime.date:
    "Create `datetime.date` from timestamp."
    return _date_from_timestamp(timestamp)


# Date: Arithmetic -------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_add(
    date: datetime.date,
    days: cython.int = 0,
    seconds: cython.longlong = 0,
    microseconds: cython.longlong = 0,
) -> datetime.date:
    "Add days, seconds and microseconds to `datetime.date`. Equivalent to `date + timedelta(d, s, us)`."
    return date_fr_microseconds(
        date_to_microseconds(date) + days * US_DAY + seconds * US_SECOND + microseconds
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_add_delta(date: datetime.date, delta: datetime.timedelta) -> datetime.date:
    "Add `datetime.timedelta` to `datetime.date`. Equivalent to `date + timedelta(instance)`."
    return date_fr_microseconds(
        date_to_microseconds(date) + delta_to_microseconds(delta)
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_sub_date_days(date_l: datetime.date, date_r: datetime.date) -> cython.int:
    "Sub `datetime.date`. Equivalent to `date - date` but ruturn total days `<int>`."
    return to_ordinal(date_l) - to_ordinal(date_r)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_sub_date(date_l: datetime.date, date_r: datetime.date) -> datetime.timedelta:
    "Sub `datetime.date`. Equivalent to `date - date`."
    return _timedelta_new(date_sub_date_days(date_l, date_r), 0, 0)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def date_replace(
    date: datetime.date,
    year: cython.int = -1,
    month: cython.int = -1,
    day: cython.int = -1,
) -> datetime.date:
    "Replace `datetime.date`. Equivalent to `date.replace()`. -1 means no change."
    year = year if 1 <= year <= 9999 else get_year(date)
    month = month if 1 <= month <= 12 else get_month(date)
    day = min(day if day > 0 else get_day(date), days_in_month(year, month))
    return _date_new(year, month, day)


# Datetime: Generate -----------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def gen_dt(
    year: cython.int = 1,
    month: cython.int = 1,
    day: cython.int = 1,
    hour: cython.int = 0,
    minute: cython.int = 0,
    second: cython.int = 0,
    microsecond: cython.int = 0,
    tzinfo: object = None,
    fold: cython.int = 0,
) -> datetime.datetime:
    "Generate `datetime.datetime`."
    return _datetime_new(
        year, month, day, hour, minute, second, microsecond, tzinfo, fold
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def gen_dt_local() -> datetime.datetime:
    "Generate datetime.datetime from local time. Equivalent to `datetime.now()`."
    microseconds: cython.longlong = int(_time() % 1 * US_SECOND)
    tms = cytime.localtime()
    return _datetime_new(
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
def dt_mktime(dt: datetime.datetime) -> cython.longlong:
    "Generate timestamp (Integer) from `datetime.datetime`."
    base_t: cython.longlong = int(dt_to_seconds(dt))  # seconds after POSIX epoch
    off1: cython.longlong = localize_timestamp(base_t) - base_t
    off2: cython.longlong
    ts1: cython.longlong = base_t - off1
    ts2: cython.longlong
    t1: cython.longlong = localize_timestamp(ts1)
    t2: cython.longlong
    if base_t == t1:
        # We found one solution, but it may not be the one we need.
        # Look for an earlier solution (if `fold` is 0), or a
        # later one (if `fold` is 1).
        if get_dt_fold(dt) == 0:
            ts2 = ts1 - SEC_DAY
        else:
            ts2 = ts1 + SEC_DAY
        off2 = localize_timestamp(ts2) - ts2
        if off1 == off2:
            return ts1
    else:
        off2 = t1 - ts1
        if off1 == off2:
            raise ValueError("mktime: off1 == off2: ")
    ts2 = base_t - off2
    t2 = localize_timestamp(ts2)
    if base_t == t2:
        return ts2
    if base_t == t1:
        return ts1
    # We have found both offsets a and b, but neither t - a nor t - b is
    # a solution.  This means t is in the gap.
    if get_dt_fold(dt) == 0:
        return max(ts1, ts2)
    else:
        return min(ts1, ts2)


# Datetime: Check Types --------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_dt(obj: object) -> cython.bint:
    "Check if an obj is `datetime.datetime` (include subclass such as pandas.Timestamp)."
    return datetime.PyDateTime_Check(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_dt_exact(obj: object) -> cython.bint:
    "Check if an obj is exactly `datetime.datetime`."
    return datetime.PyDateTime_CheckExact(obj)


# Datetime: Get Attribute ------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_dt_hour(dt: datetime.datetime) -> cython.int:
    "Get the 'hour' attribute of `datetime.datetime`."
    return datetime.PyDateTime_DATE_GET_HOUR(dt)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_dt_minute(dt: datetime.datetime) -> cython.int:
    "Get the 'minute' attribute of `datetime.datetime`."
    return datetime.PyDateTime_DATE_GET_MINUTE(dt)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_dt_second(dt: datetime.datetime) -> cython.int:
    "Get the 'second' attribute of `datetime.datetime`."
    return datetime.PyDateTime_DATE_GET_SECOND(dt)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_dt_microsecond(dt: datetime.datetime) -> cython.int:
    "Get the 'microsecond' attribute of `datetime.datetime`."
    return datetime.PyDateTime_DATE_GET_MICROSECOND(dt)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def get_dt_tzinfo(dt: datetime.datetime) -> datetime.tzinfo:
    "Get the 'tzinfo' attribute of `datetime.datetime`."
    try:
        return _datetime_tzinfo(dt)
    except Exception:
        return None


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_dt_fold(dt: datetime.datetime) -> cython.int:
    "Get the 'fold' attribute of `datetime.datetime`."
    return datetime.PyDateTime_DATE_GET_FOLD(dt)


# Datetime: Conversion ---------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_to_isoformat(dt: datetime.datetime) -> str:
    "Convert `datetime.datetime` to ISO format."
    microsecond: cython.int = get_dt_microsecond(dt)
    if microsecond:
        return "%04d-%02d-%02dT%02d:%02d:%02d.%06d" % (
            get_year(dt),
            get_month(dt),
            get_day(dt),
            get_dt_hour(dt),
            get_dt_minute(dt),
            get_dt_second(dt),
            microsecond,
        )
    else:
        return "%04d-%02d-%02dT%02d:%02d:%02d" % (
            get_year(dt),
            get_month(dt),
            get_day(dt),
            get_dt_hour(dt),
            get_dt_minute(dt),
            get_dt_second(dt),
        )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_to_isoformat_tz(dt: datetime.datetime) -> str:
    "Convert `datetime.datetime` to ISO format with timezone."
    fmt: str = dt_to_isoformat(dt)
    tzinfo: datetime.tzinfo = get_dt_tzinfo(dt)
    if tzinfo is not None:
        fmt += format_utcoffset(tzinfo.utcoffset(dt))
    return fmt


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
@cython.exceptval(check=False)
def dt_to_seconds(dt: datetime.datetime) -> cython.double:
    "Convert `datetime.datetime` to total seconds after POSIX epoch."
    days: cython.double = to_ordinal(dt)
    hour: cython.double = get_dt_hour(dt)
    minute: cython.double = get_dt_minute(dt)
    second: cython.double = get_dt_second(dt)
    microsecond: cython.double = get_dt_microsecond(dt)
    return (
        (days - EPOCH_DAY) * SEC_DAY
        + hour * SEC_HOUR
        + minute * SEC_MINUTE
        + second
        + microsecond / US_SECOND
    )


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
@cython.exceptval(check=False)
def dt_to_seconds_utc(dt: datetime.datetime) -> cython.double:
    """Convert `datetime.datetime` to total seconds after POSIX epoch.
    - If `dt` is timezone-aware, return total seconds in UTC.
    - If `dt` is timezone-naive, requivalent to `dt_to_seconds()`.

    #### Notice
    This should `NOT` be treated as timestamp, but rather adjustment of the
    total seconds of the datetime from utcoffset.
    """
    sec: cython.double = dt_to_seconds(dt)
    tzinfo: datetime.tzinfo = get_dt_tzinfo(dt)
    if tzinfo is not None:
        sec -= delta_to_seconds(tzinfo.utcoffset(dt))
    return sec


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_to_microseconds(dt: datetime.datetime) -> cython.longlong:
    "Convert `datetime.datetime` to total microseconds after POSIX epoch."
    days: cython.longlong = to_ordinal(dt)
    hour: cython.longlong = get_dt_hour(dt)
    minute: cython.longlong = get_dt_minute(dt)
    second: cython.longlong = get_dt_second(dt)
    microsecond: cython.longlong = get_dt_microsecond(dt)
    return (
        (days - EPOCH_DAY) * US_DAY
        + hour * US_HOUR
        + minute * US_MINUTE
        + second * US_SECOND
        + microsecond
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_to_microseconds_utc(dt: datetime.datetime) -> cython.longlong:
    """Convert `datetime.datetime` to total microseconds after POSIX epoch.
    - If `dt` is timezone-aware, return total microseconds in UTC.
    - If `dt` is timezone-naive, requivalent to `dt_to_microseconds()`.

    #### Notice
    This should `NOT` be treated as timestamp, but rather adjustment of the
    total microseconds of the datetime from utcoffset.
    """
    us: cython.longlong = dt_to_microseconds(dt)
    tzinfo: datetime.tzinfo = get_dt_tzinfo(dt)
    if tzinfo is not None:
        us -= delta_to_microseconds(tzinfo.utcoffset(dt))
    return us


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
def dt_to_timestamp(dt: datetime.datetime) -> cython.double:
    "Convert `datetime.datetime` to timestamp."
    if get_dt_tzinfo(dt) is None:
        microseconds: cython.double = get_dt_microsecond(dt)
        microseconds = microseconds / US_SECOND
        return dt_mktime(dt) + microseconds
    else:
        return delta_to_seconds(dt_sub_dt(dt, EPOCH_UTC))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_fr_dt(dt: datetime.datetime) -> datetime.datetime:
    "Create `datetime.datetime` from `datetime.datetime` subclass such as `pandas.Timestamp`."
    return _datetime_new(
        get_year(dt),
        get_month(dt),
        get_day(dt),
        get_dt_hour(dt),
        get_dt_minute(dt),
        get_dt_second(dt),
        get_dt_microsecond(dt),
        get_dt_tzinfo(dt),
        get_dt_fold(dt),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_fr_date(date: datetime.date) -> datetime.datetime:
    "Create `datetime.datetime` from `datetime.date`."
    return _datetime_new(
        get_year(date), get_month(date), get_day(date), 0, 0, 0, 0, None, 0
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_fr_time(time: datetime.time) -> datetime.datetime:
    "Create `datetime.datetime` from local date + `datetime.time`."
    tms = cytime.localtime()
    return _datetime_new(
        tms.tm_year,
        tms.tm_mon,
        tms.tm_mday,
        get_time_hour(time),
        get_time_minute(time),
        get_time_second(time),
        get_time_microsecond(time),
        get_time_tzinfo(time),
        get_time_fold(time),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_fr_date_time(date: datetime.date, time: datetime.time) -> datetime.datetime:
    "Create `datetime.datetime` from `datetime.date` and `datetime.time`."
    return _datetime_new(
        get_year(date),
        get_month(date),
        get_day(date),
        get_time_hour(time),
        get_time_minute(time),
        get_time_second(time),
        get_time_microsecond(time),
        get_time_tzinfo(time),
        get_time_fold(time),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_fr_ordinal(ordinal: cython.int) -> datetime.datetime:
    "Create `datetime.datetime` from ordinal."
    # Convert ordinal to ymd
    ymd = ordinal_to_ymd(ordinal)
    # Generate datetime
    return _datetime_new(ymd.year, ymd.month, ymd.day, 0, 0, 0, 0, None, 0)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_fr_seconds(
    seconds: cython.double,
    tzinfo: object = None,
    fold: cython.int = 0,
) -> datetime.datetime:
    "Create `datetime.datetime` from total seconds after POSIX epoch."
    microseconds: cython.longlong = int(seconds * US_SECOND)
    return dt_fr_microseconds(microseconds, tzinfo, fold)


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
@cython.exceptval(check=False)
def dt_fr_microseconds(
    microseconds: cython.longlong,
    tzinfo: object = None,
    fold: cython.int = 0,
) -> datetime.datetime:
    "Create `datetime.datetime` from total microseconds after POSIX epoch."
    # Add back epoch seconds
    microseconds += EPOCH_US
    # Clip microseconds
    microseconds = cymath.clip(microseconds, DT_MIN_US, DT_MAX_US)
    # Calculate ymd
    ymd = ordinal_to_ymd(microseconds // US_DAY)
    # Calculate hms
    hms = microseconds_to_hms(microseconds)
    # Validate tzinfo
    if tzinfo is not None and not isinstance(tzinfo, datetime.tzinfo):
        tzinfo = None
    # Validate fold
    if fold != 1:
        fold = 0
    # Generate datetime
    return _datetime_new(
        ymd.year,
        ymd.month,
        ymd.day,
        hms.hour,
        hms.minute,
        hms.second,
        hms.microsecond,
        tzinfo,
        fold,
    )


@cython.cfunc
@cython.inline(True)
def dt_fr_timestamp(
    timestamp: cython.double,
    tzinfo: object = None,
) -> datetime.datetime:
    "Create `datetime.datetime` from timestamp."
    return _datetime_from_timestamp(timestamp, tzinfo)


# Datetime: Arithmetic ---------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_add(
    dt: datetime.datetime,
    days: cython.int = 0,
    seconds: cython.longlong = 0,
    microseconds: cython.longlong = 0,
) -> datetime.datetime:
    "Add days, seconds and microseconds to `datetime.datetime`. Equivalent to `datetime + timedelta(d, s, us)`)."
    return dt_fr_microseconds(
        dt_to_microseconds(dt) + days * US_DAY + seconds * US_SECOND + microseconds,
        get_dt_tzinfo(dt),
        get_dt_fold(dt),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_add_delta(dt: datetime.datetime, delta: datetime.timedelta) -> datetime.datetime:
    "Add `datetime.timedelta` to `datetime.datetime`. Equivalent to `datetime + timedelta(instance)`)."
    return dt_fr_microseconds(
        dt_to_microseconds(dt) + delta_to_microseconds(delta),
        get_dt_tzinfo(dt),
        get_dt_fold(dt),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_sub_dt(dt_l: datetime.datetime, dt_r: datetime.datetime) -> datetime.timedelta:
    "Sub `datetime.datetime`. Equivalent to `datetime - datetime`."
    return delta_fr_microseconds(dt_sub_dt_microseconds(dt_l, dt_r))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_sub_dt_microseconds(
    dt_l: datetime.datetime,
    dt_r: datetime.datetime,
) -> cython.longlong:
    "Sub `datetime.datetime`. Equivalent to `datetime - datetime` but return total microseconds `<int>`."
    delta_us: cython.longlong = dt_to_microseconds(dt_l) - dt_to_microseconds(dt_r)
    tzinfo_l: object = get_dt_tzinfo(dt_l)
    tzinfo_r: object = get_dt_tzinfo(dt_r)

    # If both are naive, return delta
    if tzinfo_l is tzinfo_r:
        return delta_us

    # Calculate offset
    offset_us_l: cython.longlong = (
        0 if tzinfo_l is None else delta_to_microseconds(dt_l.utcoffset())
    )
    offset_us_r: cython.longlong = (
        0 if tzinfo_r is None else delta_to_microseconds(dt_r.utcoffset())
    )
    # Return delta with offset
    return delta_us + offset_us_r - offset_us_l


# Datetime: Manipulation -------------------------------------------------------------------------------
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
    "Replace `datetime.datetime`. Equivalent to `datetime.replace(). -1 means no change`."
    year = year if 1 <= year <= 9999 else get_year(dt)
    month = month if 1 <= month <= 12 else get_month(dt)
    day = min(day if day > 0 else get_day(dt), days_in_month(year, month))
    return _datetime_new(
        year,
        month,
        day,
        hour if 0 <= hour <= 23 else get_dt_hour(dt),
        minute if 0 <= minute <= 59 else get_dt_minute(dt),
        second if 0 <= second <= 59 else get_dt_second(dt),
        microsecond if 0 <= microsecond <= 999999 else get_dt_microsecond(dt),
        tzinfo if is_tzinfo(tzinfo) or tzinfo is None else get_dt_tzinfo(dt),
        fold if 0 <= fold <= 1 else get_dt_fold(dt),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_replace_tzinfo(dt: datetime.datetime, tzinfo: object = -1) -> datetime.datetime:
    "Replace `datetime.datetime` tzinfo. Equivalent to `datetime.replace(tzinfo=tzinfo)`. -1 means no change`."
    return _datetime_new(
        get_year(dt),
        get_month(dt),
        get_day(dt),
        get_dt_hour(dt),
        get_dt_minute(dt),
        get_dt_second(dt),
        get_dt_microsecond(dt),
        tzinfo if is_tzinfo(tzinfo) or tzinfo is None else get_dt_tzinfo(dt),
        get_dt_fold(dt),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def dt_replace_fold(dt: datetime.datetime, fold: cython.int = -1) -> datetime.datetime:
    "Replace `datetime.datetime` fold. Equivalent to `datetime.replace(fold=fold)`. -1 means no change."
    return _datetime_new(
        get_year(dt),
        get_month(dt),
        get_day(dt),
        get_dt_hour(dt),
        get_dt_minute(dt),
        get_dt_second(dt),
        get_dt_microsecond(dt),
        get_dt_tzinfo(dt),
        fold if 0 <= fold <= 1 else get_dt_fold(dt),
    )


# Datetime.Time: Generate ------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def gen_time(
    hour: cython.int = 0,
    minute: cython.int = 0,
    second: cython.int = 0,
    microsecond: cython.int = 0,
    tzinfo: object = None,
    fold: cython.int = 0,
) -> datetime.time:
    "Generate `datetime.time`."
    return _time_new(hour, minute, second, microsecond, tzinfo, fold)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def gen_time_local() -> datetime.time:
    "Generate datetime.time from local time. Equivalent to `datetime.time.now()`."
    tms = cytime.localtime()
    microseconds: cython.int = int(_time() % 1 * US_SECOND)
    return _time_new(tms.tm_hour, tms.tm_min, tms.tm_sec, microseconds, None, 0)


# Datetime.Time: Check Types ---------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_time(obj: object) -> cython.bint:
    "Check if an obj is `datetime.time` (include subclass)."
    return datetime.PyTime_Check(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_time_exact(obj: object) -> cython.bint:
    "Check if an obj is exactly `datetime.time`."
    return datetime.PyTime_CheckExact(obj)


# Datetime.Time: Get Attribute -------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_time_hour(time: datetime.time) -> cython.int:
    "Get the 'hour' attribute of `datetime.time`."
    return datetime.PyDateTime_TIME_GET_HOUR(time)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_time_minute(time: datetime.time) -> cython.int:
    "Get the 'minute' attribute of `datetime.time`."
    return datetime.PyDateTime_TIME_GET_MINUTE(time)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_time_second(time: datetime.time) -> cython.int:
    "Get the 'second' attribute of `datetime.time`."
    return datetime.PyDateTime_TIME_GET_SECOND(time)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_time_microsecond(time: datetime.time) -> cython.int:
    "Get the 'microsecond' attribute of `datetime.time`."
    return datetime.PyDateTime_TIME_GET_MICROSECOND(time)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def get_time_tzinfo(time: datetime.time) -> datetime.tzinfo:
    "Get the 'tzinfo' attribute of `datetime.time`."
    try:
        return _time_tzinfo(time)
    except Exception:
        return None


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def get_time_fold(time: datetime.time) -> cython.int:
    "Get the 'fold' attribute of `datetime.time`."
    return datetime.PyDateTime_TIME_GET_FOLD(time)


# Datetime.Time: Conversion ----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_to_isoformat(time: datetime.time) -> str:
    "Convert `datetime.time` to ISO format."
    microsecond: cython.int = get_time_microsecond(time)
    if microsecond:
        return "%02d:%02d:%02d.%06d" % (
            get_time_hour(time),
            get_time_minute(time),
            get_time_second(time),
            microsecond,
        )
    else:
        return "%02d:%02d:%02d" % (
            get_time_hour(time),
            get_time_minute(time),
            get_time_second(time),
        )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_to_isoformat_tz(time: datetime.time) -> str:
    "Convert `datetime.time` to ISO format with timezone."
    fmt: str = time_to_isoformat(time)
    tzinfo: datetime.tzinfo = get_time_tzinfo(time)
    if tzinfo is not None:
        fmt += format_utcoffset(tzinfo.utcoffset(None))
    return fmt


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
@cython.exceptval(check=False)
def time_to_seconds(time: datetime.time) -> cython.double:
    "Convert `datetime.time` to total seconds."
    microseconds: cython.double = time_to_microseconds(time)
    return microseconds / US_SECOND


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_to_microseconds(time: datetime.time) -> cython.longlong:
    "Convert `datetime.time` to total microseconds."
    hour: cython.longlong = get_time_hour(time)
    minute: cython.longlong = get_time_minute(time)
    second: cython.longlong = get_time_second(time)
    microsecond: cython.longlong = get_time_microsecond(time)
    return hour * US_HOUR + minute * US_MINUTE + second * US_SECOND + microsecond


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_fr_dt(dt: datetime.datetime) -> datetime.time:
    "Create `datetime.time` from `datetime.datetime`."
    return _time_new(
        get_dt_hour(dt),
        get_dt_minute(dt),
        get_dt_second(dt),
        get_dt_microsecond(dt),
        None,
        0,
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_fr_dt_tz(dt: datetime.datetime) -> datetime.time:
    "Create `datetime.time` from `datetime.datetime` with timezone."
    return _time_new(
        get_dt_hour(dt),
        get_dt_minute(dt),
        get_dt_second(dt),
        get_dt_microsecond(dt),
        get_dt_tzinfo(dt),
        get_dt_fold(dt),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_fr_seconds(
    seconds: cython.double,
    tzinfo: object = None,
    fold: cython.int = 0,
) -> datetime.time:
    "Create `datetime.time` from total seconds."
    mciroseconds: cython.longlong = int(seconds * US_SECOND)
    return time_fr_microseconds(mciroseconds, tzinfo, fold)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_fr_microseconds(
    microseconds: cython.longlong,
    tzinfo: object = None,
    fold: cython.int = 0,
) -> datetime.time:
    # Add back epoch seconds
    if microseconds < 0:
        microseconds += EPOCH_US
    # Clip microseconds
    microseconds = cymath.clip(microseconds, 0, DT_MAX_US)
    # Calculate hms
    hms = microseconds_to_hms(microseconds)
    # Validate tzinfo
    if tzinfo is not None and not isinstance(tzinfo, datetime.tzinfo):
        tzinfo = None
    # Validate fold
    if fold != 1:
        fold = 0
    # Generate the time object
    return _time_new(hms.hour, hms.minute, hms.second, hms.microsecond, tzinfo, fold)


# Datetime.Time: Manipulation --------------------------------------------------------------------------
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
    "Replace `datetime.time`. Equivalent to `time.replace()`. -1 means no change."
    return _time_new(
        hour if 0 <= hour <= 23 else get_time_hour(time),
        minute if 0 <= minute <= 59 else get_time_minute(time),
        second if 0 <= second <= 59 else get_time_second(time),
        microsecond if 0 <= microsecond <= 999999 else get_time_microsecond(time),
        tzinfo if is_tzinfo(tzinfo) or tzinfo is None else get_time_tzinfo(time),
        fold if 0 <= fold <= 1 else get_time_fold(time),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_replace_tzinfo(time: datetime.time, tzinfo: object = -1) -> datetime.time:
    "Replace `datetime.time` tzinfo. Equivalent to `time.replace(tzinfo=tzinfo)`. -1 means no change."
    return _time_new(
        get_time_hour(time),
        get_time_minute(time),
        get_time_second(time),
        get_time_microsecond(time),
        tzinfo if is_tzinfo(tzinfo) or tzinfo is None else get_time_tzinfo(time),
        get_time_fold(time),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def time_replace_fold(time: datetime.time, fold: cython.int = -1) -> datetime.time:
    "Replace `datetime.time` fold. Equivalent to `time.replace(fold=fold)`. -1 means no change."
    return _time_new(
        get_time_hour(time),
        get_time_minute(time),
        get_time_second(time),
        get_time_microsecond(time),
        get_time_tzinfo(time),
        fold if 0 <= fold <= 1 else get_time_fold(time),
    )


# Timedelta: Generate ----------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def gen_delta(
    days: cython.int = 0,
    seconds: cython.longlong = 0,
    microseconds: cython.longlong = 0,
) -> datetime.timedelta:
    "Generate `datetime.timedelta`."
    return _timedelta_new(days, seconds, microseconds)


# Timedelta: Check Types -------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_delta(obj: object) -> cython.bint:
    "Check if an obj is `datetime.timedelta` (include subclass)."
    return datetime.PyDelta_Check(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_delta_exact(obj: object) -> cython.bint:
    "Check if an obj is exactly `datetime.timedelta`."
    return datetime.PyDelta_CheckExact(obj)


# Timedelta: Get Attribute -----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def get_delta_days(delta: datetime.timedelta) -> cython.int:
    "Get the 'days' attribute of `datetime.timedelta`."
    return datetime.PyDateTime_DELTA_GET_DAYS(delta)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def get_delta_seconds(delta: datetime.timedelta) -> cython.int:
    "Get the 'seconds' attribute of `datetime.timedelta`."
    return datetime.PyDateTime_DELTA_GET_SECONDS(delta)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def get_delta_microseconds(delta: datetime.timedelta) -> cython.int:
    "Get the 'microseconds' attribute of `datetime.timedelta`."
    return datetime.PyDateTime_DELTA_GET_MICROSECONDS(delta)


# Timedelta: Conversion --------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
@cython.exceptval(check=False)
def delta_to_isoformat(delta: datetime.timedelta) -> str:
    days: cython.int = get_delta_days(delta)
    secs: cython.int = get_delta_seconds(delta)
    hours: cython.int = secs // SEC_HOUR % 24 + days * 24
    minutes: cython.int = secs // SEC_MINUTE % 60
    seconds: cython.int = secs % 60
    microseconds: cython.int = get_delta_microseconds(delta)
    if microseconds:
        return "%02d:%02d:%02d.%06d" % (hours, minutes, seconds, microseconds)
    else:
        return "%02d:%02d:%02d" % (hours, minutes, seconds)


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
@cython.exceptval(check=False)
def delta_to_seconds(delta: datetime.timedelta) -> cython.double:
    "Convert `datetime.timedelta` to total seconds."
    microseconds: cython.double = delta_to_microseconds(delta)
    return microseconds / US_SECOND


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def delta_to_microseconds(delta: datetime.timedelta) -> cython.longlong:
    "Convert `datetime.timedelta` to total microseconds."
    days: cython.longlong = get_delta_days(delta)
    seconds: cython.longlong = get_delta_seconds(delta)
    microseconds: cython.longlong = get_delta_microseconds(delta)
    return days * US_DAY + seconds * US_SECOND + microseconds


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def delta_fr_seconds(seconds: cython.double) -> datetime.timedelta:
    "Create `datetime.timedelta` from total seconds."
    microseconds: cython.longlong = int(seconds * US_SECOND)
    return delta_fr_microseconds(microseconds)


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
@cython.exceptval(check=False)
def delta_fr_microseconds(microseconds: cython.longlong) -> datetime.timedelta:
    "Create `datetime.timedelta` from total microseconds."
    # Pre binding
    days: cython.int
    seconds: cython.longlong
    # Negative microseconds
    if microseconds < 0:
        # Calculate days & seconds
        microseconds = -microseconds
        days = (microseconds // US_DAY) * -1
        seconds = (microseconds % US_DAY // US_SECOND) * -1
        microseconds = (microseconds % US_SECOND) * -1
    # Positive microseconds
    else:
        # Calculate days & seconds
        days = microseconds // US_DAY
        seconds = microseconds % US_DAY // US_SECOND
        microseconds = microseconds % US_SECOND
    # Generate timedelta
    return _timedelta_new(days, seconds, microseconds)


# Timedelta: Arithmetic --------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def delta_add(
    delta: datetime.timedelta,
    days: cython.int = 0,
    seconds: cython.longlong = 0,
    microseconds: cython.longlong = 0,
) -> datetime.timedelta:
    "Add days, seconds and microseconds to `datetime.timedelta`. Equivalent to `timedelta + timedelta(d, s, us)`)."
    return _timedelta_new(
        get_delta_days(delta) + days,
        get_delta_seconds(delta) + seconds,
        get_delta_microseconds(delta) + microseconds,
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def delta_add_delta(
    delta1: datetime.timedelta,
    delta2: datetime.timedelta,
) -> datetime.timedelta:
    "Add `timedelta`. Equivalent to `timedelta + timedelta(instance)`."
    return _timedelta_new(
        get_delta_days(delta1) + get_delta_days(delta2),
        get_delta_seconds(delta1) + get_delta_seconds(delta2),
        get_delta_microseconds(delta1) + get_delta_microseconds(delta2),
    )


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def delta_sub_delta(
    delta_l: datetime.timedelta,
    delta_r: datetime.timedelta,
) -> datetime.timedelta:
    "Sub `timedelta`. Equivalent to `timedelta - timedelta(instance)`."
    return _timedelta_new(
        get_delta_days(delta_l) - get_delta_days(delta_r),
        get_delta_seconds(delta_l) - get_delta_seconds(delta_r),
        get_delta_microseconds(delta_l) - get_delta_microseconds(delta_r),
    )


# Timezone: Generate -----------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def gen_timezone(offset_seconds: cython.int, tzname: str = None) -> datetime.tzinfo:
    "Generate `datetime.timezone` from tzname & offset (seconds)."
    if not -86340 <= offset_seconds <= 86340:
        raise ValueError(
            "Timezone expected offset between -86340 and 86340 (seconds), got %s"
            % offset_seconds
        )
    delta: datetime.timedelta = _timedelta_new(0, offset_seconds, 0)
    if tzname:
        return datetime.PyTimeZone_FromOffsetAndName(delta, tzname)
    else:
        return datetime.PyTimeZone_FromOffset(delta)


@cython.cfunc
@cython.inline(True)
def gen_timezone_local(dt: datetime.datetime = None) -> datetime.tzinfo:
    "Generate local timezone. If `dt` is given, use `dt` to get local timezone, else `None` use localtime."
    # Get localized timestamp
    ts: cython.double
    if is_dt(dt):
        if get_dt_tzinfo(dt) is None:
            ts = dt_mktime(dt)
        else:
            ts = delta_to_seconds(dt_sub_dt(dt, EPOCH_UTC))
    else:
        ts = _time()
    # Get local time
    tms = _localtime(ts)
    # Generate timezone
    return gen_timezone(tms.tm_gmtoff, tms.tm_zone)


# Timezone: Check Types --------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_tzinfo(obj: object) -> cython.bint:
    "Check if an obj is `datetime.tzinfo` (include subclass)."
    return datetime.PyTZInfo_Check(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_tzinfo_exact(obj: object) -> cython.bint:
    "Check if an obj is exactly `datetime.tzinfo`."
    return datetime.PyTZInfo_CheckExact(obj)


# Timezone: Conversion ---------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
def format_utcoffset(utcoffset: datetime.timedelta) -> str:
    "Format utcoffset (timedelta) to e.g.: +03:00."
    if utcoffset is None:
        return ""

    sign: str
    time_us: cython.longlong = delta_to_microseconds(utcoffset)
    if time_us < 0:
        sign = "-"
        time_us = -time_us
    else:
        sign = "+"
    hours: cython.int = time_us // US_HOUR
    time_us = time_us % US_HOUR
    minutes: cython.int = time_us // US_MINUTE
    time_us = time_us % US_MINUTE
    seconds: cython.int = time_us // US_SECOND
    microseconds: cython.int = time_us % US_SECOND
    if microseconds:
        return "%s%02d:%02d:%02d.%06d" % (sign, hours, minutes, seconds, microseconds)
    elif seconds:
        return "%s%02d:%02d:%02d" % (sign, hours, minutes, seconds)
    else:
        return "%s%02d:%02d" % (sign, hours, minutes)


# Datetime64 -------------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_dt64(obj: object) -> cython.bint:
    "Check if an obj is `numpy.datetime64`."
    return _is_datetime64_object(obj)


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
def dt64_to_isoformat(dt64: object) -> str:
    "Convert `numpy.datetime64` to ISO format. Only support units from `'D'` to `'ms'`."
    # Add back epoch seconds
    microseconds: cython.longlong = dt64_to_microseconds(dt64) + EPOCH_US
    # Clip microseconds
    microseconds = cymath.clip(microseconds, DT_MIN_US, DT_MAX_US)
    # Calculate ymd
    ymd = ordinal_to_ymd(microseconds // US_DAY)
    # Calculate hms
    hms = microseconds_to_hms(microseconds)
    # Return isoformat
    if hms.microsecond:
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
def dt64_to_int(dt64: object, unit: str) -> cython.longlong:
    """Convert numpy.datetime64 to integer based on time unit.

    - Support units from `'D'` to `'ns'`.
    - Percision will be lost if the original datetime64 unit is smaller than 'unit'.
    """
    # Validate
    if not _is_datetime64_object(dt64):
        raise TypeError("Expected a `numpy.datetime64` object, got '%s'" % type(dt64))

    # Get val & unit
    dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(dt64)
    value: np.npy_datetime = _get_datetime64_value(dt64)

    # Convert from [ns] to desired [unit]
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        if unit == "ns":
            return value
        elif unit == "us":
            return value // NS_MICROSECOND
        elif unit == "ms":
            return value // NS_MILLISECOND
        elif unit == "s":
            return value // NS_SECOND
        elif unit == "m":
            return value // NS_MINUTE
        elif unit == "h":
            return value // NS_HOUR
        elif unit == "D":
            return value // NS_DAY

    # Convert from [us] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        if unit == "ns":
            return value * NS_MICROSECOND
        elif unit == "us":
            return value
        elif unit == "ms":
            return value // US_MILLISECOND
        elif unit == "s":
            return value // US_SECOND
        elif unit == "m":
            return value // US_MINUTE
        elif unit == "h":
            return value // US_HOUR
        elif unit == "D":
            return value // US_DAY

    # Convert from [ms] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        if unit == "ns":
            return value * NS_MILLISECOND
        elif unit == "us":
            return value * US_MILLISECOND
        elif unit == "ms":
            return value
        elif unit == "s":
            return value // MS_SECOND
        elif unit == "m":
            return value // MS_MINUTE
        elif unit == "h":
            return value // MS_HOUR
        elif unit == "D":
            return value // MS_DAY

    # Convert from [s] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        if unit == "ns":
            return value * NS_SECOND
        elif unit == "us":
            return value * US_SECOND
        elif unit == "ms":
            return value * MS_SECOND
        elif unit == "s":
            return value
        elif unit == "m":
            return value // SEC_MINUTE
        elif unit == "h":
            return value // SEC_HOUR
        elif unit == "D":
            return value // SEC_DAY

    # Convert from [m] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        if unit == "ns":
            return value * NS_MINUTE
        elif unit == "us":
            return value * US_MINUTE
        elif unit == "ms":
            return value * MS_MINUTE
        elif unit == "s":
            return value * SEC_MINUTE
        elif unit == "m":
            return value
        elif unit == "h":
            return value // MIN_HOUR
        elif unit == "D":
            return value // MIN_DAY

    # Convert from [h] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        if unit == "ns":
            return value * NS_HOUR
        elif unit == "us":
            return value * US_HOUR
        elif unit == "ms":
            return value * MS_HOUR
        elif unit == "s":
            return value * SEC_HOUR
        elif unit == "m":
            return value * MIN_HOUR
        elif unit == "h":
            return value
        elif unit == "D":
            return value // HOUR_DAY

    # Convert from [D] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        if unit == "ns":
            return value * NS_DAY
        elif unit == "us":
            return value * US_DAY
        elif unit == "ms":
            return value * MS_DAY
        elif unit == "s":
            return value * SEC_DAY
        elif unit == "m":
            return value * MIN_DAY
        elif unit == "h":
            return value * HOUR_DAY
        elif unit == "D":
            return value

    # Unsupported unit
    raise ValueError(
        "Unsupported datetime64 '%s' - (val: %s, unit: %s)" % (dt64, value, unit)
    )


@cython.cfunc
@cython.inline(True)
def dt64_to_days(dt64: object) -> cython.longlong:
    """Convert `numpy.datetime64` to integer (days).

    - Percision will be lost if the original datetime64 unit is smaller than 'days'.
    """
    # Validate
    if not _is_datetime64_object(dt64):
        raise TypeError("Expected a `numpy.datetime64` object, got '%s'" % type(dt64))

    # Get val & unit
    dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(dt64)
    value: np.npy_datetime = _get_datetime64_value(dt64)

    # Converstion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return value // NS_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return value // US_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return value // MS_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return value // SEC_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return value // MIN_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return value // HOUR_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return value
    else:
        raise ValueError(
            "Unsupported datetime64 '%s' - (val: %s, unit: %s)" % (dt64, value, dt_unit)
        )


@cython.cfunc
@cython.inline(True)
def dt64_to_hours(dt64: object) -> cython.longlong:
    """Convert `numpy.datetime64` to integer (hours).

    - Percision will be lost if the original datetime64 unit is smaller than 'hours'.
    """
    # Validate
    if not _is_datetime64_object(dt64):
        raise TypeError("Expected a `numpy.datetime64` object, got '%s'" % type(dt64))

    # Get val & unit
    dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(dt64)
    value: np.npy_datetime = _get_datetime64_value(dt64)

    # Converstion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return value // NS_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return value // US_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return value // MS_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return value // SEC_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return value // MIN_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return value
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return value * HOUR_DAY
    else:
        raise ValueError(
            "Unsupported datetime64 '%s' - (val: %s, unit: %s)" % (dt64, value, dt_unit)
        )


@cython.cfunc
@cython.inline(True)
def dt64_to_minutes(dt64: object) -> cython.longlong:
    """Convert `numpy.datetime64` to integer (minutes).

    - Percision will be lost if the original datetime64 unit is smaller than 'minutes'.
    """
    # Validate
    if not _is_datetime64_object(dt64):
        raise TypeError("Expected a `numpy.datetime64` object, got '%s'" % type(dt64))

    # Get val & unit
    dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(dt64)
    value: np.npy_datetime = _get_datetime64_value(dt64)

    # Converstion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return value // NS_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return value // US_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return value // MS_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return value // SEC_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return value
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return value * MIN_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return value * MIN_DAY
    else:
        raise ValueError(
            "Unsupported datetime64 '%s' - (val: %s, unit: %s)" % (dt64, value, dt_unit)
        )


@cython.cfunc
@cython.inline(True)
def dt64_to_seconds(dt64: object) -> cython.longlong:
    """Convert `numpy.datetime64` to integer (seconds).

    - Percision will be lost if the original datetime64 unit is smaller than 'seconds'.
    """
    # Validate
    if not _is_datetime64_object(dt64):
        raise TypeError("Expected a `numpy.datetime64` object, got '%s'" % type(dt64))

    # Get val & unit
    dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(dt64)
    value: np.npy_datetime = _get_datetime64_value(dt64)

    # Converstion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return value // NS_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return value // US_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return value // MS_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return value
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return value * SEC_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return value * SEC_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return value * SEC_DAY
    else:
        raise ValueError(
            "Unsupported datetime64 '%s' - (val: %s, unit: %s)" % (dt64, value, dt_unit)
        )


@cython.cfunc
@cython.inline(True)
def dt64_to_miliseconds(dt64: object) -> cython.longlong:
    """Convert `numpy.datetime64` to integer (miliseconds).

    - Percision will be lost if the original datetime64 unit is smaller than 'miliseconds'.
    """
    # Validate
    if not _is_datetime64_object(dt64):
        raise TypeError("Expected a `numpy.datetime64` object, got '%s'" % type(dt64))

    # Get val & unit
    dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(dt64)
    value: np.npy_datetime = _get_datetime64_value(dt64)

    # Converstion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return value // NS_MILLISECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return value // US_MILLISECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return value
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return value * MS_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return value * MS_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return value * MS_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return value * MS_DAY
    else:
        raise ValueError(
            "Unsupported datetime64 '%s' - (val: %s, unit: %s)" % (dt64, value, dt_unit)
        )


@cython.cfunc
@cython.inline(True)
def dt64_to_microseconds(dt64: object) -> cython.longlong:
    """Convert `numpy.datetime64` to integer (microseconds).

    - Percision will be lost if the original datetime64 unit is smaller than 'microseconds'.
    """
    # Validate
    if not _is_datetime64_object(dt64):
        raise TypeError("Expected a `numpy.datetime64` object, got '%s'" % type(dt64))

    # Get val & unit
    dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(dt64)
    value: np.npy_datetime = _get_datetime64_value(dt64)

    # Converstion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return value // NS_MICROSECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return value
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return value * US_MILLISECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return value * US_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return value * US_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return value * US_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return value * US_DAY
    else:
        raise ValueError(
            "Unsupported datetime64 '%s' - (val: %s, unit: %s)" % (dt64, value, dt_unit)
        )


@cython.cfunc
@cython.inline(True)
def dt64_to_nanoseconds(dt64: object) -> cython.longlong:
    """Convert `numpy.datetime64` to integer (nanoseconds).

    - Percision will be lost if the original datetime64 unit is smaller than 'nanoseconds'.
    """
    # Validate
    if not _is_datetime64_object(dt64):
        raise TypeError("Expected a `numpy.datetime64` object, got '%s'" % type(dt64))

    # Get val & unit
    dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(dt64)
    value: np.npy_datetime = _get_datetime64_value(dt64)

    # Converstion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return value
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return value * NS_MICROSECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return value * NS_MILLISECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return value * NS_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return value * NS_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return value * NS_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return value * NS_DAY
    else:
        raise ValueError(
            "Unsupported datetime64 '%s' - (val: %s, unit: %s)" % (dt64, value, dt_unit)
        )


@cython.cfunc
@cython.inline(True)
def dt64_to_date(dt64: object) -> datetime.date:
    """Convert `numpy.datetime64` to to `datetime.date`. Only support units from `'D'` to `'ms'`.

    For value out of `datetime.date` range, will be clipped to supported value. For example:
    - Upper limit: numpy.datetime64 `<10000-01-01T00:00:00>`
      will clip to date `<9999-12-31>`.
    - Lower limit: numpy.datetime64 `<0000-01-01T00:00:00>`
      will clip to date `<0001-01-01>`.
    """
    return date_fr_ordinal(dt64_to_days(dt64) + EPOCH_DAY)


@cython.cfunc
@cython.inline(True)
def dt64_to_dt(dt64: object) -> datetime.datetime:
    """Convert `numpy.datetime64` to `datetime.datetime`. Only support units from `'D'` to `'ms'`.

    For value out of `datetime.datetime` range, will be clipped to supported value. For example:
    - Upper limit: numpy.datetime64 `<10000-01-01T00:00:00>`
      will clip to datetime `<9999-12-31 23:59:59.999999>`.
    - Lower limit: numpy.datetime64 `<0000-01-01T00:00:00>`
      will clip to datetime `<0001-01-01 00:00:00.000000>`.
    """
    return dt_fr_microseconds(dt64_to_microseconds(dt64), None, 0)


@cython.cfunc
@cython.inline(True)
def dt64_to_time(dt64: object) -> datetime.time:
    "Convert `numpy.datetime64` to `datetime.time`. Only support units from `'D'` to `'ms'`."
    return time_fr_microseconds(dt64_to_microseconds(dt64), None, 0)


# Timedelta64 ------------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_delta64(obj: object) -> cython.bint:
    "Check if an obj is `numpy.timedelta64`."
    return _is_timedelta64_object(obj)


@cython.cfunc
@cython.inline(True)
def delta64_to_isoformat(delta64: object) -> str:
    us: cython.longlong = delta64_to_microseconds(delta64)
    days: cython.longlong = us // US_DAY
    secs: cython.longlong = us // US_SECOND
    hours: cython.longlong = secs // SEC_HOUR % 24 + days * 24
    minutes: cython.longlong = secs // SEC_MINUTE % 60
    seconds: cython.longlong = secs % 60
    microseconds: cython.longlong = us % US_SECOND
    if microseconds:
        return "%02d:%02d:%02d.%06d" % (hours, minutes, seconds, microseconds)
    else:
        return "%02d:%02d:%02d" % (hours, minutes, seconds)


@cython.cfunc
@cython.inline(True)
def delta64_to_int(delta64: object, unit: str) -> cython.longlong:
    """Convert `numpy.timedelta64` to integer based on time unit.

    - Support units from `'D'` to `'ns'`.
    - Percision will be lost if the original timedelta64 unit is smaller than 'unit'.
    """
    # Validate
    if not _is_timedelta64_object(delta64):
        raise TypeError(
            "Expected a `numpy.timedelta64` object, got '%s'" % type(delta64)
        )

    # Get val & unit
    dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(delta64)
    value: np.npy_timedelta = _get_timedelta64_value(delta64)

    # Convert from [ns] to desired [unit]
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        if unit == "ns":
            return value
        elif unit == "us":
            return value // NS_MICROSECOND
        elif unit == "ms":
            return value // NS_MILLISECOND
        elif unit == "s":
            return value // NS_SECOND
        elif unit == "m":
            return value // NS_MINUTE
        elif unit == "h":
            return value // NS_HOUR
        elif unit == "D":
            return value // NS_DAY

    # Convert from [us] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        if unit == "ns":
            return value * NS_MICROSECOND
        elif unit == "us":
            return value
        elif unit == "ms":
            return value // US_MILLISECOND
        elif unit == "s":
            return value // US_SECOND
        elif unit == "m":
            return value // US_MINUTE
        elif unit == "h":
            return value // US_HOUR
        elif unit == "D":
            return value // US_DAY

    # Convert from [ms] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        if unit == "ns":
            return value * NS_MILLISECOND
        elif unit == "us":
            return value * US_MILLISECOND
        elif unit == "ms":
            return value
        elif unit == "s":
            return value // MS_SECOND
        elif unit == "m":
            return value // MS_MINUTE
        elif unit == "h":
            return value // MS_HOUR
        elif unit == "D":
            return value // MS_DAY

    # Convert from [s] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        if unit == "ns":
            return value * NS_SECOND
        elif unit == "us":
            return value * US_SECOND
        elif unit == "ms":
            return value * MS_SECOND
        elif unit == "s":
            return value
        elif unit == "m":
            return value // SEC_MINUTE
        elif unit == "h":
            return value // SEC_HOUR
        elif unit == "D":
            return value // SEC_DAY

    # Convert from [m] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        if unit == "ns":
            return value * NS_MINUTE
        elif unit == "us":
            return value * US_MINUTE
        elif unit == "ms":
            return value * MS_MINUTE
        elif unit == "s":
            return value * SEC_MINUTE
        elif unit == "m":
            return value
        elif unit == "h":
            return value // MIN_HOUR
        elif unit == "D":
            return value // MIN_DAY

    # Convert from [h] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        if unit == "ns":
            return value * NS_HOUR
        elif unit == "us":
            return value * US_HOUR
        elif unit == "ms":
            return value * MS_HOUR
        elif unit == "s":
            return value * SEC_HOUR
        elif unit == "m":
            return value * MIN_HOUR
        elif unit == "h":
            return value
        elif unit == "D":
            return value // HOUR_DAY

    # Convert from [D] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        if unit == "ns":
            return value * NS_DAY
        elif unit == "us":
            return value * US_DAY
        elif unit == "ms":
            return value * MS_DAY
        elif unit == "s":
            return value * SEC_DAY
        elif unit == "m":
            return value * MIN_DAY
        elif unit == "h":
            return value * HOUR_DAY
        elif unit == "D":
            return value

    # Unsupported unit
    raise ValueError(
        "Unsupported timedelta64 '%s' - (val: %s, unit: %s)" % (delta64, value, unit)
    )


@cython.cfunc
@cython.inline(True)
def delta64_to_days(delta64: object) -> cython.longlong:
    """Convert `numpy.timedelta64` to integer (days).

    - Support units from `'D'` to `'ns'`.
    - Percision will be lost if the original timedelta64 unit is smaller than 'days'.
    """
    # Validate
    if not _is_timedelta64_object(delta64):
        raise TypeError(
            "Expected a `numpy.timedelta64` object, got '%s'" % type(delta64)
        )

    # Get val & unit
    dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(delta64)
    value: np.npy_timedelta = _get_timedelta64_value(delta64)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return value // NS_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return value // US_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return value // MS_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return value // SEC_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return value // MIN_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return value // HOUR_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return value
    else:
        raise ValueError(
            "Unsupported timedelta64 '%s' - (val: %s, unit: %s)"
            % (delta64, value, dt_unit)
        )


@cython.cfunc
@cython.inline(True)
def delta64_to_hours(delta64: object) -> cython.longlong:
    """Convert `numpy.timedelta64` to integer (hours).

    - Support units from `'D'` to `'ns'`.
    - Percision will be lost if the original timedelta64 unit is smaller than 'hours'.
    """
    # Validate
    if not _is_timedelta64_object(delta64):
        raise TypeError(
            "Expected a `numpy.timedelta64` object, got '%s'" % type(delta64)
        )

    # Get val & unit
    dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(delta64)
    value: np.npy_timedelta = _get_timedelta64_value(delta64)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return value // NS_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return value // US_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return value // MS_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return value // SEC_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return value // MIN_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return value
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return value * HOUR_DAY
    else:
        raise ValueError(
            "Unsupported timedelta64 '%s' - (val: %s, unit: %s)"
            % (delta64, value, dt_unit)
        )


@cython.cfunc
@cython.inline(True)
def delta64_to_minutes(delta64: object) -> cython.longlong:
    """Convert `numpy.timedelta64` to integer (minutes).

    - Support units from `'D'` to `'ns'`.
    - Percision will be lost if the original timedelta64 unit is smaller than 'minutes'.
    """
    # Validate
    if not _is_timedelta64_object(delta64):
        raise TypeError(
            "Expected a `numpy.timedelta64` object, got '%s'" % type(delta64)
        )

    # Get val & unit
    dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(delta64)
    value: np.npy_timedelta = _get_timedelta64_value(delta64)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return value // NS_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return value // US_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return value // MS_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return value // SEC_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return value
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return value * MIN_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return value * MIN_DAY
    else:
        raise ValueError(
            "Unsupported timedelta64 '%s' - (val: %s, unit: %s)"
            % (delta64, value, dt_unit)
        )


@cython.cfunc
@cython.inline(True)
def delta64_to_seconds(delta64: object) -> cython.longlong:
    """Convert `numpy.timedelta64` to integer (seconds).

    - Support units from `'D'` to `'ns'`.
    - Percision will be lost if the original timedelta64 unit is smaller than 'seconds'.
    """
    # Validate
    if not _is_timedelta64_object(delta64):
        raise TypeError(
            "Expected a `numpy.timedelta64` object, got '%s'" % type(delta64)
        )

    # Get val & unit
    dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(delta64)
    value: np.npy_timedelta = _get_timedelta64_value(delta64)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return value // NS_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return value // US_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return value // MS_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return value
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return value * SEC_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return value * SEC_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return value * SEC_DAY
    else:
        raise ValueError(
            "Unsupported timedelta64 '%s' - (val: %s, unit: %s)"
            % (delta64, value, dt_unit)
        )


@cython.cfunc
@cython.inline(True)
def delta64_to_miliseconds(delta64: object) -> cython.longlong:
    """Convert `numpy.timedelta64` to integer (miliseconds).

    - Support units from `'D'` to `'ns'`.
    - Percision will be lost if the original timedelta64 unit is smaller than 'miliseconds'.
    """
    # Validate
    if not _is_timedelta64_object(delta64):
        raise TypeError(
            "Expected a `numpy.timedelta64` object, got '%s'" % type(delta64)
        )

    # Get val & unit
    dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(delta64)
    value: np.npy_timedelta = _get_timedelta64_value(delta64)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return value // NS_MILLISECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return value // US_MILLISECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return value
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return value * MS_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return value * MS_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return value * MS_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return value * MS_DAY
    else:
        raise ValueError(
            "Unsupported timedelta64 '%s' - (val: %s, unit: %s)"
            % (delta64, value, dt_unit)
        )


@cython.cfunc
@cython.inline(True)
def delta64_to_microseconds(delta64: object) -> cython.longlong:
    """Convert `numpy.timedelta64` to integer (microseconds).

    - Support units from `'D'` to `'ns'`.
    - Percision will be lost if the original timedelta64 unit is smaller than 'microseconds'.
    """
    # Validate
    if not _is_timedelta64_object(delta64):
        raise TypeError(
            "Expected a `numpy.timedelta64` object, got '%s'" % type(delta64)
        )

    # Get val & unit
    dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(delta64)
    value: np.npy_timedelta = _get_timedelta64_value(delta64)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return value // NS_MICROSECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return value
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return value * US_MILLISECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return value * US_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return value * US_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return value * US_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return value * US_DAY
    else:
        raise ValueError(
            "Unsupported timedelta64 '%s' - (val: %s, unit: %s)"
            % (delta64, value, dt_unit)
        )


@cython.cfunc
@cython.inline(True)
def delta64_to_nanoseconds(delta64: object) -> cython.longlong:
    """Convert `numpy.timedelta64` to integer (nanoseconds).

    - Support units from `'D'` to `'ns'`.
    - Percision will be lost if the original timedelta64 unit is smaller than 'nanoseconds'.
    """
    # Validate
    if not _is_timedelta64_object(delta64):
        raise TypeError(
            "Expected a `numpy.timedelta64` object, got '%s'" % type(delta64)
        )

    # Get val & unit
    dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(delta64)
    value: np.npy_timedelta = _get_timedelta64_value(delta64)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return value
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return value * NS_MICROSECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return value * NS_MILLISECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return value * NS_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return value * NS_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return value * NS_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return value * NS_DAY
    else:
        raise ValueError(
            "Unsupported timedelta64 '%s' - (val: %s, unit: %s)"
            % (delta64, value, dt_unit)
        )


@cython.cfunc
@cython.inline(True)
def delta64_to_delta(delta64: object) -> datetime.timedelta:
    "Convert `numpy.timedelta64` to `datetime.timedelta`. Only support units from `'D'` to `'ms'`."
    return delta_fr_microseconds(delta64_to_microseconds(delta64))


# ndarray[dateimte64] ----------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def arraydt64_to_arrayint(arr: np.ndarray, unit: str) -> np.ndarray:
    """Convert ndarray[datetime64] to ndarray[int] based on time unit.

    - Support units from `'D'` to `'ns'`.
    - Percision will be lost if the original datetime64 unit is smaller than 'unit'.
    """
    # Get unit & values
    try:
        dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(arr[0])
        values: np.ndarray = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    except Exception:
        raise ValueError("Expected `ndarray[datetime64]`, got\n%s" % arr)

    # Convert from [ns] to desired [unit]
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        if unit == "ns":
            return values
        elif unit == "us":
            return values // NS_MICROSECOND
        elif unit == "ms":
            return values // NS_MILLISECOND
        elif unit == "s":
            return values // NS_SECOND
        elif unit == "m":
            return values // NS_MINUTE
        elif unit == "h":
            return values // NS_HOUR
        elif unit == "D":
            return values // NS_DAY

    # Convert from [us] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        if unit == "ns":
            return values * NS_MICROSECOND
        elif unit == "us":
            return values
        elif unit == "ms":
            return values // US_MILLISECOND
        elif unit == "s":
            return values // US_SECOND
        elif unit == "m":
            return values // US_MINUTE
        elif unit == "h":
            return values // US_HOUR
        elif unit == "D":
            return values // US_DAY

    # Convert from [ms] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        if unit == "ns":
            return values * NS_MILLISECOND
        elif unit == "us":
            return values * US_MILLISECOND
        elif unit == "ms":
            return values
        elif unit == "s":
            return values // MS_SECOND
        elif unit == "m":
            return values // MS_MINUTE
        elif unit == "h":
            return values // MS_HOUR
        elif unit == "D":
            return values // MS_DAY

    # Convert from [s] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        if unit == "ns":
            return values * NS_SECOND
        elif unit == "us":
            return values * US_SECOND
        elif unit == "ms":
            return values * MS_SECOND
        elif unit == "s":
            return values
        elif unit == "m":
            return values // SEC_MINUTE
        elif unit == "h":
            return values // SEC_HOUR
        elif unit == "D":
            return values // SEC_DAY

    # Convert from [m] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        if unit == "ns":
            return values * NS_MINUTE
        elif unit == "us":
            return values * US_MINUTE
        elif unit == "ms":
            return values * MS_MINUTE
        elif unit == "s":
            return values * SEC_MINUTE
        elif unit == "m":
            return values
        elif unit == "h":
            return values // MIN_HOUR
        elif unit == "D":
            return values // MIN_DAY

    # Convert from [h] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        if unit == "ns":
            return values * NS_HOUR
        elif unit == "us":
            return values * US_HOUR
        elif unit == "ms":
            return values * MS_HOUR
        elif unit == "s":
            return values * SEC_HOUR
        elif unit == "m":
            return values * MIN_HOUR
        elif unit == "h":
            return values
        elif unit == "D":
            return values // HOUR_DAY

    # Convert from [D] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        if unit == "ns":
            return values * NS_DAY
        elif unit == "us":
            return values * US_DAY
        elif unit == "ms":
            return values * MS_DAY
        elif unit == "s":
            return values * SEC_DAY
        elif unit == "m":
            return values * MIN_DAY
        elif unit == "h":
            return values * HOUR_DAY
        elif unit == "D":
            return values

    raise ValueError("Unsupported unit: %s. Accept units from `'D'` to `'ns'`." % unit)


@cython.cfunc
@cython.inline(True)
def arraydt64_to_arrayint_day(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[datetime64] to ndarray[int] (days).

    - Percision will be lost if the original datetime64 unit is smaller than 'days'.
    """
    # Get unit & values
    try:
        dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(arr[0])
        values: np.ndarray = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    except Exception:
        raise ValueError("Expected `ndarray[datetime64]`, got\n%s" % arr)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return values // NS_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return values // US_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return values // MS_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return values // SEC_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return values // MIN_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return values // HOUR_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return values
    else:
        raise ValueError(
            "Unsupported datetime64 unit: %s. Accept units from `'D'` to `'ns'`."
            % dt_unit
        )


@cython.cfunc
@cython.inline(True)
def arraydt64_to_arrayint_hour(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[datetime64] to ndarray[int] (hours).

    - Percision will be lost if the original datetime64 unit is smaller than 'hours'.
    """
    # Get unit & values
    try:
        dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(arr[0])
        values: np.ndarray = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    except Exception:
        raise ValueError("Expected `ndarray[datetime64]`, got\n%s" % arr)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return values // NS_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return values // US_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return values // MS_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return values // SEC_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return values // MIN_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return values
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return values * HOUR_DAY
    else:
        raise ValueError(
            "Unsupported datetime64 unit: %s. Accept units from `'D'` to `'ns'`."
            % dt_unit
        )


@cython.cfunc
@cython.inline(True)
def arraydt64_to_arrayint_min(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[datetime64] to ndarray[int] (minutes).

    - Percision will be lost if the original datetime64 unit is smaller than 'minutes'.
    """
    # Get unit & values
    try:
        dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(arr[0])
        values: np.ndarray = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    except Exception:
        raise ValueError("Expected `ndarray[datetime64]`, got\n%s" % arr)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return values // NS_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return values // US_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return values // MS_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return values // SEC_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return values
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return values * MIN_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return values * MIN_DAY
    else:
        raise ValueError(
            "Unsupported datetime64 unit: %s. Accept units from `'D'` to `'ns'`."
            % dt_unit
        )


@cython.cfunc
@cython.inline(True)
def arraydt64_to_arrayint_sec(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[datetime64] to ndarray[int] (seconds).

    - Percision will be lost if the original datetime64 unit is smaller than 'seconds'.
    """
    # Get unit & values
    try:
        dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(arr[0])
        values: np.ndarray = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    except Exception:
        raise ValueError("Expected `ndarray[datetime64]`, got\n%s" % arr)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return values // NS_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return values // US_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return values // MS_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return values
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return values * SEC_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return values * SEC_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return values * SEC_DAY
    else:
        raise ValueError(
            "Unsupported datetime64 unit: %s. Accept units from `'D'` to `'ns'`."
            % dt_unit
        )


@cython.cfunc
@cython.inline(True)
def arraydt64_to_arrayint_ms(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[datetime64] to ndarray[int] (miliseconds).

    - Percision will be lost if the original datetime64 unit is smaller than 'miliseconds'.
    """
    # Get unit & values
    try:
        dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(arr[0])
        values: np.ndarray = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    except Exception:
        raise ValueError("Expected `ndarray[datetime64]`, got\n%s" % arr)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return values // NS_MILLISECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return values // US_MILLISECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return values
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return values * MS_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return values * MS_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return values * MS_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return values * MS_DAY
    else:
        raise ValueError(
            "Unsupported datetime64 unit: %s. Accept units from `'D'` to `'ns'`."
            % dt_unit
        )


@cython.cfunc
@cython.inline(True)
def arraydt64_to_arrayint_us(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[datetime64] to ndarray[int] (microseconds).

    - Percision will be lost if the original datetime64 unit is smaller than 'microseconds'.
    """
    # Get unit & values
    try:
        dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(arr[0])
        values: np.ndarray = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    except Exception:
        raise ValueError("Expected `ndarray[datetime64]`, got\n%s" % arr)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return values // NS_MICROSECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return values
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return values * US_MILLISECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return values * US_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return values * US_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return values * US_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return values * US_DAY
    else:
        raise ValueError(
            "Unsupported datetime64 unit: %s. Accept units from `'D'` to `'ns'`."
            % dt_unit
        )


@cython.cfunc
@cython.inline(True)
def arraydt64_to_arrayint_ns(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[datetime64] to ndarray[int] (nanoseconds).

    - Percision will be lost if the original datetime64 unit is smaller than 'nanoseconds'.
    """
    # Get unit & values
    try:
        dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(arr[0])
        values: np.ndarray = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    except Exception:
        raise ValueError("Expected `ndarray[datetime64]`, got\n%s" % arr)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return values
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return values * NS_MICROSECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return values * NS_MILLISECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return values * NS_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return values * NS_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return values * NS_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return values * NS_DAY
    else:
        raise ValueError(
            "Unsupported datetime64 unit: %s. Accept units from `'D'` to `'ns'`."
            % dt_unit
        )


# ndarray[timedelta64] ---------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def arraydelta64_to_arrayint(arr: np.ndarray, unit: str) -> np.ndarray:
    """Convert ndarray[timedelta64] to ndarray[int] based on time unit.

    - Support units from `'D'` to `'ns'`.
    - Percision will be lost if the original timedelta64 unit is smaller than 'unit'.
    """
    # Get unit & values
    try:
        dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(arr[0])
        values: np.ndarray = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    except Exception:
        raise ValueError("Expected `ndarray[timedelta64]`, got\n%s" % arr)

    # Convert from [ns] to desired [unit]
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        if unit == "ns":
            return values
        elif unit == "us":
            return values // NS_MICROSECOND
        elif unit == "ms":
            return values // NS_MILLISECOND
        elif unit == "s":
            return values // NS_SECOND
        elif unit == "m":
            return values // NS_MINUTE
        elif unit == "h":
            return values // NS_HOUR
        elif unit == "D":
            return values // NS_DAY

    # Convert from [us] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        if unit == "ns":
            return values * NS_MICROSECOND
        elif unit == "us":
            return values
        elif unit == "ms":
            return values // US_MILLISECOND
        elif unit == "s":
            return values // US_SECOND
        elif unit == "m":
            return values // US_MINUTE
        elif unit == "h":
            return values // US_HOUR
        elif unit == "D":
            return values // US_DAY

    # Convert from [ms] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        if unit == "ns":
            return values * NS_MILLISECOND
        elif unit == "us":
            return values * US_MILLISECOND
        elif unit == "ms":
            return values
        elif unit == "s":
            return values // MS_SECOND
        elif unit == "m":
            return values // MS_MINUTE
        elif unit == "h":
            return values // MS_HOUR
        elif unit == "D":
            return values // MS_DAY

    # Convert from [s] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        if unit == "ns":
            return values * NS_SECOND
        elif unit == "us":
            return values * US_SECOND
        elif unit == "ms":
            return values * MS_SECOND
        elif unit == "s":
            return values
        elif unit == "m":
            return values // SEC_MINUTE
        elif unit == "h":
            return values // SEC_HOUR
        elif unit == "D":
            return values // SEC_DAY

    # Convert from [m] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        if unit == "ns":
            return values * NS_MINUTE
        elif unit == "us":
            return values * US_MINUTE
        elif unit == "ms":
            return values * MS_MINUTE
        elif unit == "s":
            return values * SEC_MINUTE
        elif unit == "m":
            return values
        elif unit == "h":
            return values // MIN_HOUR
        elif unit == "D":
            return values // MIN_DAY

    # Convert from [h] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        if unit == "ns":
            return values * NS_HOUR
        elif unit == "us":
            return values * US_HOUR
        elif unit == "ms":
            return values * MS_HOUR
        elif unit == "s":
            return values * SEC_HOUR
        elif unit == "m":
            return values * MIN_HOUR
        elif unit == "h":
            return values
        elif unit == "D":
            return values // HOUR_DAY

    # Convert from [D] to desired [unit]
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        if unit == "ns":
            return values * NS_DAY
        elif unit == "us":
            return values * US_DAY
        elif unit == "ms":
            return values * MS_DAY
        elif unit == "s":
            return values * SEC_DAY
        elif unit == "m":
            return values * MIN_DAY
        elif unit == "h":
            return values * HOUR_DAY
        elif unit == "D":
            return values

    raise ValueError("Unsupported unit: %s. Accept units from `'D'` to `'ns'`." % unit)


@cython.cfunc
@cython.inline(True)
def arraydelta64_to_arrayint_day(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[timedelta64] to ndarray[int] (days).

    - Percision will be lost if the original timedelta64 unit is smaller than 'days'.
    """
    # Get unit & values
    try:
        dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(arr[0])
        values: np.ndarray = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    except Exception:
        raise ValueError("Expected `ndarray[timedelta64]`, got\n%s" % arr)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return values // NS_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return values // US_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return values // MS_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return values // SEC_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return values // MIN_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return values // HOUR_DAY
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return values
    else:
        raise ValueError(
            "Unsupported timedelta64 unit: %s. Accept units from `'D'` to `'ns'`."
            % dt_unit
        )


@cython.cfunc
@cython.inline(True)
def arraydelta64_to_arrayint_hour(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[timedelta64] to ndarray[int] (hours).

    - Percision will be lost if the original timedelta64 unit is smaller than 'hours'.
    """
    # Get unit & values
    try:
        dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(arr[0])
        values: np.ndarray = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    except Exception:
        raise ValueError("Expected `ndarray[timedelta64]`, got\n%s" % arr)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return values // NS_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return values // US_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return values // MS_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return values // SEC_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return values // MIN_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return values
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return values * HOUR_DAY
    else:
        raise ValueError(
            "Unsupported timedelta64 unit: %s. Accept units from `'D'` to `'ns'`."
            % dt_unit
        )


@cython.cfunc
@cython.inline(True)
def arraydelta64_to_arrayint_min(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[timedelta64] to ndarray[int] (minutes).

    - Percision will be lost if the original timedelta64 unit is smaller than 'minutes'.
    """
    # Get unit & values
    try:
        dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(arr[0])
        values: np.ndarray = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    except Exception:
        raise ValueError("Expected `ndarray[timedelta64]`, got\n%s" % arr)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return values // NS_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return values // US_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return values // MS_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return values // SEC_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return values
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return values * MIN_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return values * MIN_DAY
    else:
        raise ValueError(
            "Unsupported timedelta64 unit: %s. Accept units from `'D'` to `'ns'`."
            % dt_unit
        )


@cython.cfunc
@cython.inline(True)
def arraydelta64_to_arrayint_sec(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[timedelta64] to ndarray[int] (seconds).

    - Percision will be lost if the original timedelta64 unit is smaller than 'seconds'.
    """
    # Get unit & values
    try:
        dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(arr[0])
        values: np.ndarray = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    except Exception:
        raise ValueError("Expected `ndarray[timedelta64]`, got\n%s" % arr)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return values // NS_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return values // US_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return values // MS_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return values
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return values * SEC_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return values * SEC_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return values * SEC_DAY
    else:
        raise ValueError(
            "Unsupported timedelta64 unit: %s. Accept units from `'D'` to `'ns'`."
            % dt_unit
        )


@cython.cfunc
@cython.inline(True)
def arraydelta64_to_arrayint_ms(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[timedelta64] to ndarray[int] (miliseconds).

    - Percision will be lost if the original timedelta64 unit is smaller than 'miliseconds'.
    """
    # Get unit & values
    try:
        dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(arr[0])
        values: np.ndarray = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    except Exception:
        raise ValueError("Expected `ndarray[timedelta64]`, got\n%s" % arr)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return values // NS_MILLISECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return values // US_MILLISECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return values
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return values * MS_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return values * MS_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return values * MS_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return values * MS_DAY
    else:
        raise ValueError(
            "Unsupported timedelta64 unit: %s. Accept units from `'D'` to `'ns'`."
            % dt_unit
        )


@cython.cfunc
@cython.inline(True)
def arraydelta64_to_arrayint_us(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[timedelta64] to ndarray[int] (microseconds).

    - Percision will be lost if the original timedelta64 unit is smaller than 'microseconds'.
    """
    # Get unit & values
    try:
        dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(arr[0])
        values: np.ndarray = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    except Exception:
        raise ValueError("Expected `ndarray[timedelta64]`, got\n%s" % arr)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return values // NS_MICROSECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return values
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return values * US_MILLISECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return values * US_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return values * US_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return values * US_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return values * US_DAY
    else:
        raise ValueError(
            "Unsupported timedelta64 unit: %s. Accept units from `'D'` to `'ns'`."
            % dt_unit
        )


@cython.cfunc
@cython.inline(True)
def arraydelta64_to_arrayint_ns(arr: np.ndarray) -> np.ndarray:
    """Convert ndarray[timedelta64] to ndarray[int] (nanoseconds).

    - Percision will be lost if the original timedelta64 unit is smaller than 'nanoseconds'.
    """
    # Get unit & values
    try:
        dt_unit: np.NPY_DATETIMEUNIT = _get_datetime64_unit(arr[0])
        values: np.ndarray = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    except Exception:
        raise ValueError("Expected `ndarray[timedelta64]`, got\n%s" % arr)

    # Conversion
    if dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return values
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return values * NS_MICROSECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return values * NS_MILLISECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return values * NS_SECOND
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return values * NS_MINUTE
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return values * NS_HOUR
    elif dt_unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return values * NS_DAY
    else:
        raise ValueError(
            "Unsupported timedelta64 unit: %s. Accept units from `'D'` to `'ns'`."
            % dt_unit
        )


# pandas.Series[datetime64] ----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def seriesdt64_to_arrayint(series: pd.Series, unit: str) -> np.ndarray:
    """Convert pandas.Series[datetime64] to ndarray[int] based on time unit.

    - Support units from `'D'` to `'ns'`.
    - Percision will be lost if the original datetime64 unit is smaller than 'unit'.
    """
    return arraydt64_to_arrayint(series.values, unit)


@cython.cfunc
@cython.inline(True)
def seriesdt64_to_arrayint_day(series: pd.Series) -> np.ndarray:
    """Convert pandas.Series[datetime64] to ndarray[int] (days).

    - Percision will be lost if the original datetime64 unit is smaller than 'days'.
    """
    return arraydt64_to_arrayint_day(series.values)


@cython.cfunc
@cython.inline(True)
def seriesdt64_to_arrayint_hour(series: pd.Series) -> np.ndarray:
    """Convert pandas.Series[datetime64] to ndarray[int] (hours).

    - Percision will be lost if the original datetime64 unit is smaller than 'hours'.
    """
    return arraydt64_to_arrayint_hour(series.values)


@cython.cfunc
@cython.inline(True)
def seriesdt64_to_arrayint_min(series: pd.Series) -> np.ndarray:
    """Convert pandas.Series[datetime64] to ndarray[int] (minutes).

    - Percision will be lost if the original datetime64 unit is smaller than 'minutes'.
    """
    return arraydt64_to_arrayint_min(series.values)


@cython.cfunc
@cython.inline(True)
def seriesdt64_to_arrayint_sec(series: pd.Series) -> np.ndarray:
    """Convert pandas.Series[datetime64] to ndarray[int] (seconds).

    - Percision will be lost if the original datetime64 unit is smaller than 'seconds'.
    """
    return arraydt64_to_arrayint_sec(series.values)


@cython.cfunc
@cython.inline(True)
def seriesdt64_to_arrayint_ms(series: pd.Series) -> np.ndarray:
    """Convert pandas.Series[datetime64] to ndarray[int] (miliseconds).

    - Percision will be lost if the original datetime64 unit is smaller than 'miliseconds'.
    """
    return arraydt64_to_arrayint_ms(series.values)


@cython.cfunc
@cython.inline(True)
def seriesdt64_to_arrayint_us(series: pd.Series) -> np.ndarray:
    """Convert pandas.Series[datetime64] to ndarray[int] (microseconds).

    - Percision will be lost if the original datetime64 unit is smaller than 'microseconds'.
    """
    return arraydt64_to_arrayint_us(series.values)


@cython.cfunc
@cython.inline(True)
def seriesdt64_to_arrayint_ns(series: pd.Series) -> np.ndarray:
    """Convert pandas.Series[datetime64] to ndarray[int] (nanoseconds).

    - Percision will be lost if the original datetime64 unit is smaller than 'nanoseconds'.
    """
    return arraydt64_to_arrayint_ns(series.values)


@cython.cfunc
@cython.inline(True)
def seriesdt64_adjust_to_ns(series: pd.Series) -> object:
    """Adjust pandas.Series[datetime64] to 'datetime64[ns]'
    Support both timezone-naive and timezone-aware series.

    - Percision will be lost if the original datetime64 unit is smaller than 'nanoseconds'.
    """
    dtype: str = series.dtype.str
    if dtype == "<M8[ns]" or dtype == "|M8[ns]":
        return series

    kind: str = series.dtype.kind
    if kind == "M":
        # Timestamp to nanosecond
        values: np.ndarray = arraydt64_to_arrayint_ns(series.values)
        # Reconstruction
        return pd.Series(pd.DatetimeIndex(values, tz=series.dt.tz))
    else:
        raise ValueError(
            "Not a Series of datetime64: %s (dtype: %s)" % (series, series.dtype)
        )


@cython.cfunc
@cython.inline(True)
def seriesdt64_to_ordinal(series: pd.Series) -> object:
    "Convert Series[datetime64] to Series[int] (ordinal)."
    # Convert to days
    values: np.ndarray = seriesdt64_to_arrayint_day(series) + EPOCH_DAY
    # Reconstruction
    return pd.Series(values, index=series.index)


@cython.cfunc
@cython.inline(True)
def seriesdt64_to_seconds(series: pd.Series) -> object:
    "Convert Series[datetime64] to Series[float] (total seconds)."
    # Convert to seconds
    values: np.ndarray = seriesdt64_to_arrayint_us(series) / US_SECOND
    # Reconstruction
    return pd.Series(values, index=series.index)


@cython.cfunc
@cython.inline(True)
def seriesdt64_to_microseconds(series: pd.Series) -> object:
    "Convert Series[datetime64] to Series[float] (total seconds)."
    # Convert to seconds
    values: np.ndarray = seriesdt64_to_arrayint_us(series)
    # Reconstruction
    return pd.Series(values, index=series.index)


# pandas.Series[timedelta64] ---------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def seriesdelta64_to_arrayint(series: pd.Series, unit: str) -> np.ndarray:
    """Convert pandas.Series[timedelta64] to ndarray[int] based on time unit.

    - Support units from `'D'` to `'ns'`.
    - Percision will be lost if the original timedelta64 unit is smaller than 'unit'.
    """
    return arraydt64_to_arrayint(series.values, unit)


@cython.cfunc
@cython.inline(True)
def seriesdelta64_to_arrayint_day(series: pd.Series) -> np.ndarray:
    """Convert pandas.Series[timedelta64] to ndarray[int] (days).

    - Percision will be lost if the original timedelta64 unit is smaller than 'days'.
    """
    return arraydelta64_to_arrayint_day(series.values)


@cython.cfunc
@cython.inline(True)
def seriesdelta64_to_arrayint_hour(series: pd.Series) -> np.ndarray:
    """Convert pandas.Series[timedelta64] to ndarray[int] (hours).

    - Percision will be lost if the original timedelta64 unit is smaller than 'hours'.
    """
    return arraydelta64_to_arrayint_hour(series.values)


@cython.cfunc
@cython.inline(True)
def seriesdelta64_to_arrayint_min(series: pd.Series) -> np.ndarray:
    """Convert pandas.Series[timedelta64] to ndarray[int] (minutes).

    - Percision will be lost if the original timedelta64 unit is smaller than 'minutes'.
    """
    return arraydelta64_to_arrayint_min(series.values)


@cython.cfunc
@cython.inline(True)
def seriesdelta64_to_arrayint_sec(series: pd.Series) -> np.ndarray:
    """Convert pandas.Series[timedelta64] to ndarray[int] (seconds).

    - Percision will be lost if the original timedelta64 unit is smaller than 'seconds'.
    """
    return arraydelta64_to_arrayint_sec(series.values)


@cython.cfunc
@cython.inline(True)
def seriesdelta64_to_arrayint_ms(series: pd.Series) -> np.ndarray:
    """Convert pandas.Series[timedelta64] to ndarray[int] (miliseconds).

    - Percision will be lost if the original timedelta64 unit is smaller than 'miliseconds'.
    """
    return arraydelta64_to_arrayint_ms(series.values)


@cython.cfunc
@cython.inline(True)
def seriesdelta64_to_arrayint_us(series: pd.Series) -> np.ndarray:
    """Convert pandas.Series[timedelta64] to ndarray[int] (microseconds).

    - Percision will be lost if the original timedelta64 unit is smaller than 'microseconds'.
    """
    return arraydelta64_to_arrayint_us(series.values)


@cython.cfunc
@cython.inline(True)
def seriesdelta64_to_arrayint_ns(series: pd.Series) -> np.ndarray:
    """Convert pandas.Series[timedelta64] to ndarray[int] (nanoseconds).

    - Percision will be lost if the original timedelta64 unit is smaller than 'nanoseconds'.
    """
    return arraydelta64_to_arrayint_ns(series.values)


@cython.cfunc
@cython.inline(True)
def seriesdelta64_adjust_to_ns(series: pd.Series) -> object:
    """Adjust pandas.Series[datetime64] to 'datetime64[ns]'
    Support both timezone-naive and timezone-aware series.
    """
    dtype: str = series.dtype.str
    if dtype == "<m8[ns]" or dtype == "|m8[ns]":
        return series

    kind: str = series.dtype.kind
    if kind == "m":
        # Timedelta to nanosecond
        values: np.ndarray = arraydelta64_to_arrayint_ns(series.values)
        # Reconstruction
        return pd.Series(pd.TimedeltaIndex(values, unit="ns"))
    else:
        raise ValueError(
            "Not a Series of timedelta64: %s (dtype: %s)" % (series, series.dtype)
        )
