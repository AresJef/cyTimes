# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from __future__ import annotations

# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.libc import math  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cytimes import typeref, utils  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
import datetime
from dateutil.relativedelta import relativedelta
from cytimes import typeref, utils

__all__ = ["Delta"]


# Contants ------------------------------------------------------------------------------------
# . weekday
WEEKDAY_REPRS: tuple[str, ...] = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")


# Utils ---------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_delta(obj: object) -> cython.bint:
    """(cfunc) Check if an object is an instance of 'Delta' `<'bool'>`.

    Equivalent to:
    >>> isinstance(obj, Delta)
    """
    return isinstance(obj, Delta)


@cython.cfunc
@cython.inline(True)
def _date_add_delta(
    date: object,
    years: cython.int,
    months: cython.int,
    days: cython.int,
    hours: cython.int,
    minutes: cython.int,
    seconds: cython.int,
    microseconds: cython.int,
    year: cython.int,
    month: cython.int,
    day: cython.int,
    weekday: cython.int,
    hour: cython.int,
    minute: cython.int,
    second: cython.int,
    microsecond: cython.int,
) -> object:
    """(internal) Add relative & absolute delta to a date object,
    returns `<'datetime.date'>` or the original subclass type.

    ### Notes:
    - This function is specifically designed for the <'Delta'> class.
      Do `NOT` use this function directly.
    - Argument 'date' must be an instance or subclass of `datetime.date`.
    - Relative delta fields must be normalized.
    - Absolute delta fields must be in valid range to the corresponding
      field, or set to `-1` to retain the original value.
    """
    # Calculate delta
    # . year
    dt_yy: cython.int = datetime.date_year(date)
    yy: cython.int = (year if year != -1 else dt_yy) + years
    ymd_eq: cython.bint = yy == dt_yy
    # . month
    dt_mm: cython.int = datetime.date_month(date)
    mm: cython.int = (month if month != -1 else dt_mm) + months
    if mm != dt_mm:
        if mm > 12:
            yy += 1
            mm -= 12
        elif mm < 1:
            yy -= 1
            mm += 12
        if ymd_eq:
            ymd_eq = mm == dt_mm
    # . day
    dt_dd: cython.int = datetime.date_day(date)
    dd: cython.int = day if day != -1 else dt_dd
    if dd != dt_dd:
        ymd_eq = False
    # ----------------------------------------------------
    us: cython.int = (microsecond if microsecond != -1 else 0) + microseconds
    ss: cython.int = (second if second != -1 else 0) + seconds
    mi: cython.int = (minute if minute != -1 else 0) + minutes
    hh: cython.int = (hour if hour != -1 else 0) + hours
    # ----------------------------------------------------
    # . microseconds
    if us != 0:
        if us > 999_999:
            ss += 1
        elif us < 0:
            ss -= 1
    # . seconds
    if ss != 0:
        if ss > 59:
            mi += 1
        elif ss < 0:
            mi -= 1
    # . minutes
    if mi != 0:
        if mi > 59:
            hh += 1
        elif mi < 0:
            hh -= 1
    # . hours & days
    if hh != 0:
        if hh > 23:
            days += 1
        elif hh < 0:
            days -= 1

    # Add delta
    if days != 0:
        _ymd = utils.ymd_fr_ordinal(utils.ymd_to_ordinal(yy, mm, dd) + days)
        yy, mm, dd = _ymd.year, _ymd.month, _ymd.day
    elif ymd_eq:
        return date  # exit: no change
    elif dd > 28:
        dd = min(dd, utils.days_in_month(yy, mm))

    # Adjust weekday
    if weekday != -1:
        wkd: cython.int = utils.ymd_weekday(yy, mm, dd)
        if wkd != weekday:
            _ymd = utils.ymd_fr_ordinal(
                utils.ymd_to_ordinal(yy, mm, dd) + weekday - wkd
            )
            yy, mm, dd = _ymd.year, _ymd.month, _ymd.day

    # New date
    # . subclass of datetime.date
    if not utils.is_date_exact(date):
        try:
            return date.__class__(yy, mm, dd)
        except Exception:
            pass
    # . native datetime.date / fallback
    return datetime.date_new(yy, mm, dd)


@cython.cfunc
@cython.inline(True)
def _dt_add_delta(
    dt: object,
    years: cython.int,
    months: cython.int,
    days: cython.int,
    hours: cython.int,
    minutes: cython.int,
    seconds: cython.int,
    microseconds: cython.int,
    year: cython.int,
    month: cython.int,
    day: cython.int,
    weekday: cython.int,
    hour: cython.int,
    minute: cython.int,
    second: cython.int,
    microsecond: cython.int,
) -> object:
    """(internal) Add relative & absolute delta to a datetime object,
    returns `<'datetime.datetime'>` or the original subclass type.

    ### Notes:
    - This function is specifically designed for the <'Delta'> class.
      Do `NOT` use this function directly.
    - Argument 'dt' must be an instance or subclass of `datetime.datetime`.
    - Relative delta fields must be normalized.
    - Absolute delta fields must be in valid range to the corresponding
      field, or set to `-1` to retain the original value.
    """
    # Calculate delta
    # . year
    dt_yy: cython.int = datetime.datetime_year(dt)
    yy: cython.int = (year if year != -1 else dt_yy) + years
    ymd_eq: cython.bint = yy == dt_yy
    # . month
    dt_mm: cython.int = datetime.datetime_month(dt)
    mm: cython.int = (month if month != -1 else dt_mm) + months
    if mm != dt_mm:
        if mm > 12:
            yy += 1
            mm -= 12
        elif mm < 1:
            yy -= 1
            mm += 12
        if ymd_eq:
            ymd_eq = mm == dt_mm
    # . day
    dt_dd: cython.int = datetime.datetime_day(dt)
    dd: cython.int = day if day != -1 else dt_dd
    if dd != dt_dd:
        ymd_eq = False
    # ----------------------------------------------------
    dt_us: cython.int = datetime.datetime_microsecond(dt)
    us: cython.int = (microsecond if microsecond != -1 else dt_us) + microseconds
    dt_ss: cython.int = datetime.datetime_second(dt)
    ss: cython.int = (second if second != -1 else dt_ss) + seconds
    dt_mi: cython.int = datetime.datetime_minute(dt)
    mi: cython.int = (minute if minute != -1 else dt_mi) + minutes
    dt_hh: cython.int = datetime.datetime_hour(dt)
    hh: cython.int = (hour if hour != -1 else dt_hh) + hours
    # ----------------------------------------------------
    # . microseconds
    if us != dt_us:
        if us > 999_999:
            ss += 1
            us -= 1_000_000
        elif us < 0:
            ss -= 1
            us += 1_000_000
        hmsf_eq: cython.bint = us == dt_us
    else:
        hmsf_eq: cython.bint = True
    # . seconds
    if ss != dt_ss:
        if ss > 59:
            mi += 1
            ss -= 60
        elif ss < 0:
            mi -= 1
            ss += 60
        if hmsf_eq:
            hmsf_eq = ss == dt_ss
    # . minutes
    if mi != dt_mi:
        if mi > 59:
            hh += 1
            mi -= 60
        elif mi < 0:
            hh -= 1
            mi += 60
        if hmsf_eq:
            hmsf_eq = mi == dt_mi
    # . hours & days
    if hh != dt_hh:
        if hh > 23:
            days += 1
            hh -= 24
        elif hh < 0:
            days -= 1
            hh += 24
        if hmsf_eq:
            hmsf_eq = hh == dt_hh

    # Add delta
    if days != 0:
        _ymd = utils.ymd_fr_ordinal(utils.ymd_to_ordinal(yy, mm, dd) + days)
        yy, mm, dd = _ymd.year, _ymd.month, _ymd.day
    elif ymd_eq:
        if hmsf_eq:
            return dt  # exit: no change
    elif dd > 28:
        dd = min(dd, utils.days_in_month(yy, mm))

    # Adjust weekday
    if weekday != -1:
        wkd: cython.int = utils.ymd_weekday(yy, mm, dd)
        if wkd != weekday:
            _ymd = utils.ymd_fr_ordinal(
                utils.ymd_to_ordinal(yy, mm, dd) + weekday - wkd
            )
            yy, mm, dd = _ymd.year, _ymd.month, _ymd.day

    # New datetime
    tz = datetime.datetime_tzinfo(dt)
    fold: cython.int = datetime.datetime_fold(dt)
    # . subclass of datetime.datetime
    if not utils.is_dt_exact(dt):
        try:
            if fold == 1:
                return dt.__class__(yy, mm, dd, hh, mi, ss, us, tz, fold=1)
            else:
                return dt.__class__(yy, mm, dd, hh, mi, ss, us, tz)
        except Exception:
            pass
    # . native datetime.datetime / fallback
    return datetime.datetime_new(yy, mm, dd, hh, mi, ss, us, tz, fold)


# Delta ---------------------------------------------------------------------------------------
@cython.cclass
class Delta:
    """Represent the difference between two datetime objects at both relative
    and absolute levels. The `<'Delta'>`class supports arithmetic operations
    and is compatible with various datetime and timedelta types.
    """

    _years: cython.int
    _months: cython.int
    _days: cython.int
    _hours: cython.int
    _minutes: cython.int
    _seconds: cython.int
    _microseconds: cython.int
    _year: cython.int
    _month: cython.int
    _day: cython.int
    _weekday: cython.int
    _hour: cython.int
    _minute: cython.int
    _second: cython.int
    _microsecond: cython.int
    _hashcode: cython.longlong

    def __init__(
        self,
        years: cython.int = 0,
        quarters: cython.int = 0,
        months: cython.int = 0,
        weeks: cython.int = 0,
        days: cython.int = 0,
        hours: cython.int = 0,
        minutes: cython.int = 0,
        seconds: cython.int = 0,
        milliseconds: cython.int = 0,
        microseconds: cython.int = 0,
        year: cython.int = -1,
        month: cython.int = -1,
        day: cython.int = -1,
        weekday: cython.int = -1,
        hour: cython.int = -1,
        minute: cython.int = -1,
        second: cython.int = -1,
        millisecond: cython.int = -1,
        microsecond: cython.int = -1,
    ):
        """The difference between two datetime objects at both relative
        and absolute levels. The `<'Delta'>`class supports arithmetic
        operations and is compatible with various datetime and timedelta
        types.

        ## Absolute Deltas (Replace specified fields)

        :param year `<'int'>`: Absolute year, defaults to `-1` (original value).
        :param month `<'int'>`: Absolute month, defaults to `-1` (original value).
        :param day `<'int'>`: Absolute day, defaults to `-1` (original value).
        :param weekday `<'int'>`: Absolute weekday (0=Mon...6=Sun), defaults to `-1` (original value).
        :param hour `<'int'>`: Absolute hour, defaults to `-1` (original value).
        :param minute `<'int'>`: Absolute minute, defaults to `-1` (original value).
        :param second `<'int'>`: Absolute second, defaults to `-1` (original value).
        :param millisecond `<'int'>`: Absolute millisecond, defaults to `-1` (original value).
        :param microsecond `<'int'>`: Absolute microsecond, defaults to `-1` (original value).

        ## Relative Deltas (Add to specified fields)

        :param years `<'int'>`: Relative years, defaults to `0`.
        :param quarters `<'int'>`: Relative quarters (3 months), defaults to `0`.
        :param months `<'int'>`: Relative months, defaults to `0`.
        :param weeks `<'int'>`: Relative weeks (7 days), defaults to `0`.
        :param days `<'int'>`: Relative days, defaults to `0`.
        :param hours `<'int'>`: Relative hours, defaults to `0`.
        :param minutes `<'int'>`: Relative minutes, defaults to `0`.
        :param seconds `<'int'>`: Relative seconds, defaults to `0`.
        :param milliseconds `<'int'>`: Relative milliseconds (`1000 us`), defaults to `0`.
        :param microseconds `<'int'>`: Relative microseconds, defaults to `0`.

        ## Arithmetic Operations
        - Addition (`+`):
            - With datetime instance or subclass (`datetime.datetime`, `cytimes.Pydt`, `pandas.Timestamp`, etc):
                - Applies absolute deltas first (excluding 'weekday'), then adds relative deltas.
                - Adjusts the Y/M/D to the absolute 'weekday' if specified.
                - Returns `<'datetime.datetime'>` or the original subclass type.
            - With date instance or subclass (`datetime.date`, `pendulum.Date`, etc):
                - Similar to datetime addition, but returns `<'datetime.date'>`
                  or the original subclass type instead.
            - With delta object (`datetime.timedelta`, `cytimes.Delta`, `pandas.Timedelta`, etc):
                - Sums corresponding relative delta fields of both objects.
                - For delta objects with absolute delta fields (`cytimes.Delta`, `dateutils.relativedelta`),
                  the right operand's absolute values overwrite the left's.
                - Returns `<'cytimes.Delta'>`.
            - With integer or float (`int`, `float`):
                - Adds the numeric value to all relative delta fields.
                - For float values, each field is normalized (rounded), and the remainder
                  carries over to lower fields.
                - Returns `<'cytimes.Delta'>`.
        - Subtraction (`-`):
            - With datetime instance or subclass (`datetime.datetime`, `cytimes.Pydt`, `pandas.Timestamp`, etc):
                - Only support `RIGHT` operand (i.e., datetime - Delta).
                - Similar to addition, but subtracts the relative deltas instead.
                - Returns `<'datetime.datetime'>` or the original subclass type.
            - With date instance or subclass (`datetime.date`, `pendulum.Date`, etc):
                - Only support `RIGHT` operand (i.e., date - Delta).
                - Similar to datetime subtraction, but returns `<'datetime.date'>`
                  or the original subclass type instead.
            - With delta object (`datetime.timedelta`, `cytimes.Delta`, `pandas.Timedelta`, etc):
                - Subtracts corresponding relative delta fields (left - right).
                - For delta objects with absolute delta fields (`cytimes.Delta`, `dateutils.relativedelta`),
                  the left operand's absolute values are kept.
                - Returns `<'cytimes.Delta'>`.
            - With integer or float (`int`, `float`):
                - Subtracts the numeric value from all relative delta fields.
                - For float values, each field is normalized (rounded), and the remainder
                  carries over to lower fields.
                - Returns `<'cytimes.Delta'>`.
        - Multiplication (`*`) and Division (`/`):
            - Supports multiplication and division with both `<'int'>` and `<'float'>`.
              Division is limited to left operand (i.e., Delta / number).
            - Multiplies or divides all relative delta fields by the numeric value.
            - For float values or division, each field is normalized (rounded), and the
              remainder carries over to lower fields.
            - Returns `<'cytimes.Delta'>`.
        - Negation and Absolute Value:
            - '-Delta' negates all relative delta fields.
            - 'abs(Delta)' converts all relative delta fields to their absolute value.
            - Absolute delta fields remain unchanged.
            - Returns `<'cytimes.Delta'>`.

        ### Compatibility with `relativedelta`
        - Supports direct addition and subtraction with `<'relativedelta'>`,
          returns `<'cytimes.Delta'>`.
        - Arithmetic operations yield equivalent results when `relativedelta`'s
          `weekday` is `None`.

        ### Compatibility with `numpy.datetime64/timedelta64`
        - Supports left operand addition and subtraction with `numpy.timedelta64`
          (i.e., Delta - timedelta64), returns `<'cytimes.Delta'>`.
        - Supports left operand addition with `numpy.datetime64`
          (i.e., Delta + datetime64), returns `<'datetime.datetime'>`.
        - Resolution is limited to microseconds; higher resolutions are truncated.
        """
        # Relative delta
        # . microseconds
        us: cython.longlong = microseconds
        us += milliseconds * 1_000
        if us > 999_999:
            seconds += us // 1_000_000
            self._microseconds = us % 1_000_000
        elif us < -999_999:
            us = -us
            seconds -= us // 1_000_000
            self._microseconds = -(us % 1_000_000)
        else:
            self._microseconds = us
        # . seconds
        if seconds > 59:
            minutes += seconds // 60
            self._seconds = seconds % 60
        elif seconds < -59:
            seconds = -seconds
            minutes -= seconds // 60
            self._seconds = -(seconds % 60)
        else:
            self._seconds = seconds
        # . minutes
        if minutes > 59:
            hours += minutes // 60
            self._minutes = minutes % 60
        elif minutes < -59:
            minutes = -minutes
            hours -= minutes // 60
            self._minutes = -(minutes % 60)
        else:
            self._minutes = minutes
        # . hours
        if hours > 23:
            days += hours // 24
            self._hours = hours % 24
        elif hours < -23:
            hours = -hours
            days -= hours // 24
            self._hours = -(hours % 24)
        else:
            self._hours = hours
        # . days
        self._days = days + weeks * 7
        # . months
        months += quarters * 3
        if months > 11:
            years += months // 12
            self._months = months % 12
        elif months < -11:
            months = -months
            years -= months // 12
            self._months = -(months % 12)
        else:
            self._months = months
        # . years
        self._years = years

        # Absolute delta
        self._year = min(year, 9_999) if year > 0 else -1
        self._month = min(month, 12) if month > 0 else -1
        self._day = min(day, 31) if day > 0 else -1
        self._weekday = min(weekday, 6) if weekday >= 0 else -1
        self._hour = min(hour, 23) if hour >= 0 else -1
        self._minute = min(minute, 59) if minute >= 0 else -1
        self._second = min(second, 59) if second >= 0 else -1
        if millisecond >= 0:
            millisecond = min(millisecond, 999) * 1_000
            if microsecond > 0:
                self._microsecond = millisecond + microsecond % 1_000
            else:
                self._microsecond = millisecond
        elif microsecond >= 0:
            self._microsecond = min(microsecond, 999_999)
        else:
            self._microsecond = -1

        # Initial hashcode
        self._hashcode = -1

    # Property: relative delta -----------------------------------------------
    @property
    def years(self) -> int:
        """The relative years `<'int'>`."""
        return self._years

    @property
    def months(self) -> int:
        """The relative months `<'int'>`."""
        return self._months

    @property
    def days(self) -> int:
        """The relative days `<'int'>`."""
        return self._days

    @property
    def hours(self) -> int:
        """The relative hours `<'int'>`."""
        return self._hours

    @property
    def minutes(self) -> int:
        """The relative minutes `<'int'>`."""
        return self._minutes

    @property
    def seconds(self) -> int:
        """The relative seconds `<'int'>`."""
        return self._seconds

    @property
    def microseconds(self) -> int:
        """The relative microseconds `<'int'>`."""
        return self._microseconds

    # Properties: absolute delta ---------------------------------------------
    @property
    def year(self) -> int | None:
        """The absolute year `<'int/None'>`."""
        return None if self._year == -1 else self._year

    @property
    def month(self) -> int | None:
        """The absolute month `<'int/None'>`."""
        return None if self._month == -1 else self._month

    @property
    def day(self) -> int | None:
        """The absolute day `<'int/None'>`."""
        return None if self._day == -1 else self._day

    @property
    def weekday(self) -> int | None:
        """The absolute weekday (0=Mon...6=Sun) `<'int/None'>`."""
        return None if self._weekday == -1 else self._weekday

    @property
    def hour(self) -> int | None:
        """The absolute hour `<'int/None'>`."""
        return None if self._hour == -1 else self._hour

    @property
    def minute(self) -> int | None:
        """The absolute minute `<'int/None'>`."""
        return None if self._minute == -1 else self._minute

    @property
    def second(self) -> int | None:
        """The absolute second `<'int/None'>`."""
        return None if self._second == -1 else self._second

    @property
    def microsecond(self) -> int | None:
        """The absolute microsecond `<'int/None'>`."""
        return None if self._microsecond == -1 else self._microsecond

    # Arithmetic: addition ---------------------------------------------------
    def __add__(self, o: object) -> object:
        """Left operand addition 'self + o' `<'datetime.datetime/Delta'>`.

        - With datetime instance or subclass (`datetime.datetime`, `cytimes.Pydt`, `pandas.Timestamp`, etc):
            - Applies absolute deltas first (excluding 'weekday'), then adds relative deltas.
            - Adjusts the Y/M/D to the absolute 'weekday' if specified.
            - Returns `<'datetime.datetime'>` or the original subclass type.
        - With date instance or subclass (`datetime.date`, `pendulum.Date`, etc):
            - Similar to datetime addition, but returns `<'datetime.date'>`
              or the original subclass type instead.
        - With delta object (`datetime.timedelta`, `cytimes.Delta`, `pandas.Timedelta`, etc):
            - Sums corresponding relative delta fields of both objects.
            - For delta objects with absolute delta fields (`cytimes.Delta`, `dateutils.relativedelta`),
              the right operand's absolute values overwrite the left's.
            - Returns `<'cytimes.Delta'>`.
        - With integer or float (`int`, `float`):
            - Adds the numeric value to all relative delta fields.
            - For float values, each field is normalized (rounded), and the remainder
              carries over to lower fields.
            - Returns `<'cytimes.Delta'>`.
        """
        # . common
        if utils.is_dt(o):
            return self._add_datetime(o)
        if utils.is_date(o):
            return self._add_date(o)
        if is_delta(o):
            return self._add_delta(o)
        if utils.is_td(o):
            return self._add_timedelta(o)
        if isinstance(o, typeref.RELATIVEDELTA):
            return self._add_relativedelta(o)
        # . uncommon
        if utils.is_dt64(o):
            return self._add_datetime(utils.dt64_to_dt(o, None))
        if utils.is_td64(o):
            return self._add_timedelta(utils.td64_to_td(o))
        # . numeric
        if isinstance(o, int):
            return self._add_int(o)
        if isinstance(o, float):
            return self._add_float(o)
        # . unsupported
        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    def _add_date(self, o: object) -> object:
        """(internal) Addition with datetime.date instance or subclass,
        returns `<'datetime.date'>` or the original subclass type.
        """
        return _date_add_delta(
            o,
            self._years,
            self._months,
            self._days,
            self._hours,
            self._minutes,
            self._seconds,
            self._microseconds,
            self._year,
            self._month,
            self._day,
            self._weekday,
            self._hour,
            self._minute,
            self._second,
            self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _add_datetime(self, o: object) -> object:
        """(internal) Addition with datetime.datetime instance or subclass,
        returns `<'datetime.datetime'>` or the original subclass type.
        """
        return _dt_add_delta(
            o,
            self._years,
            self._months,
            self._days,
            self._hours,
            self._minutes,
            self._seconds,
            self._microseconds,
            self._year,
            self._month,
            self._day,
            self._weekday,
            self._hour,
            self._minute,
            self._second,
            self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _add_delta(self, o: Delta) -> Delta:
        """(internal) Addition with another cytimes.Delta,
        returns `<'cytimes.Delta'>`.
        """
        return Delta(
            o._years + self._years,
            0,
            o._months + self._months,
            0,
            o._days + self._days,
            o._hours + self._hours,
            o._minutes + self._minutes,
            o._seconds + self._seconds,
            0,
            o._microseconds + self._microseconds,
            o._year if o._year != -1 else self._year,
            o._month if o._month != -1 else self._month,
            o._day if o._day != -1 else self._day,
            o._weekday if o._weekday != -1 else self._weekday,
            o._hour if o._hour != -1 else self._hour,
            o._minute if o._minute != -1 else self._minute,
            o._second if o._second != -1 else self._second,
            -1,
            o._microsecond if o._microsecond != -1 else self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _add_timedelta(self, o: object) -> Delta:
        """(internal) Addition with datetime.timedelta instance or subclass,
        returns `<'cytimes.Delta'>`."""
        return Delta(
            self._years,
            0,
            self._months,
            0,
            self._days + datetime.timedelta_days(o),
            self._hours,
            self._minutes,
            self._seconds + datetime.timedelta_seconds(o),
            0,
            self._microseconds + datetime.timedelta_microseconds(o),
            self._year,
            self._month,
            self._day,
            self._weekday,
            self._hour,
            self._minute,
            self._second,
            -1,
            self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _add_relativedelta(self, o: relativedelta) -> Delta:
        """(internal) Left operand addition with dateutil.relativedelta,
        (i.e., Delta + relativedelta), returns `<'cytimes.Delta'>`.
        """
        # Normalize
        o = o.normalized()
        # Relative delta
        years: cython.int = o.years
        months: cython.int = o.months
        days: cython.int = o.days
        hours: cython.int = o.hours
        minutes: cython.int = o.minutes
        seconds: cython.int = o.seconds
        microseconds: cython.int = o.microseconds
        # Absolute delta
        o_year = o.year
        year = self._year if o_year is None else o_year
        o_month = o.month
        month = self._month if o_month is None else o_month
        o_day = o.day
        day = self._day if o_day is None else o_day
        o_weekday = o.weekday
        weekday = self._weekday if o_weekday is None else o_weekday.weekday
        o_hour = o.hour
        hour = self._hour if o_hour is None else o_hour
        o_minute = o.minute
        minute = self._minute if o_minute is None else o_minute
        o_second = o.second
        second = self._second if o_second is None else o_second
        o_microsecond = o.microsecond
        microsecond = self._microsecond if o_microsecond is None else o_microsecond
        # Create delta
        return Delta(
            years + self._years,
            0,
            months + self._months,
            0,
            days + self._days,
            hours + self._hours,
            minutes + self._minutes,
            seconds + self._seconds,
            0,
            microseconds + self._microseconds,
            year,
            month,
            day,
            weekday,
            hour,
            minute,
            second,
            -1,
            microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _add_int(self, o: cython.int) -> Delta:
        """(internal) Addition with int, returns `<'cytimes.Delta'>`."""
        return Delta(
            self._years + o,
            0,
            self._months + o,
            0,
            self._days + o,
            self._hours + o,
            self._minutes + o,
            self._seconds + o,
            0,
            self._microseconds + o,
            self._year,
            self._month,
            self._day,
            self._weekday,
            self._hour,
            self._minute,
            self._second,
            -1,
            self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _add_float(self, o: cython.double) -> Delta:
        """(internal) Addition with float, returns `<'cytimes.Delta'>`."""
        # Normalize
        # . years
        value: cython.double = self._years + o
        years: cython.int = math.llround(value)
        # . months
        value = self._months + o + (value - years) * 12
        months: cython.int = math.llround(value)
        # . days
        value = self._days + o
        days: cython.int = math.llround(value)
        # . hours
        value = self._hours + o + (value - days) * 24
        hours: cython.int = math.llround(value)
        # . minutes
        value = self._minutes + o + (value - hours) * 60
        minutes: cython.int = math.llround(value)
        # . seconds
        value = self._seconds + o + (value - minutes) * 60
        seconds: cython.int = math.llround(value)
        # . microseconds
        value = self._microseconds + o + (value - seconds) * 1_000_000
        microseconds: cython.int = math.llround(value)
        # Create delta
        return Delta(
            years,
            0,
            months,
            0,
            days,
            hours,
            minutes,
            seconds,
            0,
            microseconds,
            self._year,
            self._month,
            self._day,
            self._weekday,
            self._hour,
            self._minute,
            self._second,
            -1,
            self._microsecond,
        )

    # Arithmetic: right addition ---------------------------------------------
    def __radd__(self, o: object) -> object:
        """Right operand addition 'o + self' `<'datetime.datetime/Delta'>`.

        - With datetime instance or subclass (`datetime.datetime`, `cytimes.Pydt`, `pandas.Timestamp`, etc):
            - Applies absolute deltas first (excluding 'weekday'), then adds relative deltas.
            - Adjusts the Y/M/D to the absolute 'weekday' if specified.
            - Returns `<'datetime.datetime'>` or the original subclass type.
        - With date instance or subclass (`datetime.date`, `pendulum.Date`, etc):
            - Similar to datetime addition, but returns `<'datetime.date'>`
              or the original subclass type instead.
        - With delta object (`datetime.timedelta`, `cytimes.Delta`, `pandas.Timedelta`, etc):
            - Sums corresponding relative delta fields of both objects.
            - For delta objects with absolute delta fields (`cytimes.Delta`, `dateutils.relativedelta`),
              the right operand's absolute values overwrite the left's.
            - Returns `<'cytimes.Delta'>`.
        - With integer or float (`int`, `float`):
            - Adds the numeric value to all relative delta fields.
            - For float values, each field is normalized (rounded), and the remainder
              carries over to lower fields.
            - Returns `<'cytimes.Delta'>`.
        """
        # . common
        if utils.is_dt(o):
            return self._add_datetime(o)
        if utils.is_date(o):
            return self._add_date(o)
        if utils.is_td(o):
            return self._add_timedelta(o)
        if isinstance(o, typeref.RELATIVEDELTA):
            return self._radd_relativedelta(o)
        # . uncommon
        # TODO: uncommon block does not work
        if utils.is_dt64(o):
            return self._add_datetime(utils.dt64_to_dt(o, None))
        if utils.is_td64(o):
            return self._add_timedelta(utils.td64_to_td(o))
        # . numeric
        if isinstance(o, int):
            return self._add_int(o)
        if isinstance(o, float):
            return self._add_float(o)
        # . unsupported
        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    def _radd_relativedelta(self, o: relativedelta) -> Delta:
        """(internal) Right operand addition with dateutil.relativedelta,
        (i.e., relativedelta + Delta), returns `<'cytimes.Delta'>`.
        """
        # Normalize
        o = o.normalized()
        # Relative delta
        years: cython.int = o.years
        months: cython.int = o.months
        days: cython.int = o.days
        hours: cython.int = o.hours
        minutes: cython.int = o.minutes
        seconds: cython.int = o.seconds
        microseconds: cython.int = o.microseconds
        # Absolute delta
        if self._year != -1:
            year = self._year
        else:
            o_year: object = o.year
            year = -1 if o_year is None else o_year
        if self._month != -1:
            month = self._month
        else:
            o_month: object = o.month
            month = -1 if o_month is None else o_month
        if self._day != -1:
            day = self._day
        else:
            o_day: object = o.day
            day = -1 if o_day is None else o_day
        if self._weekday != -1:
            weekday = self._weekday
        else:
            o_weekday = o.weekday
            weekday = -1 if o_weekday is None else o_weekday.weekday
        if self._hour != -1:
            hour = self._hour
        else:
            o_hour: object = o.hour
            hour = -1 if o_hour is None else o_hour
        if self._minute != -1:
            minute = self._minute
        else:
            o_minute: object = o.minute
            minute = -1 if o_minute is None else o_minute
        if self._second != -1:
            second = self._second
        else:
            o_second: object = o.second
            second = -1 if o_second is None else o_second
        if self._microsecond != -1:
            microsecond = self._microsecond
        else:
            o_microsecond: object = o.microsecond
            microsecond = -1 if o_microsecond is None else o_microsecond
        # Create delta
        return Delta(
            self._years + years,
            0,
            self._months + months,
            0,
            self._days + days,
            self._hours + hours,
            self._minutes + minutes,
            self._seconds + seconds,
            0,
            self._microseconds + microseconds,
            year,
            month,
            day,
            weekday,
            hour,
            minute,
            second,
            -1,
            microsecond,
        )

    # Arithmetic: subtraction ------------------------------------------------
    def __sub__(self, o: object) -> Delta:
        """Left operand subtraction 'self - o' `<'Delta'>`.

        - With delta object (`datetime.timedelta`, `cytimes.Delta`, `pandas.Timedelta`, etc):
            - Subtracts corresponding relative delta fields (left - right).
            - For delta objects with absolute delta fields (`cytimes.Delta`, `dateutils.relativedelta`),
              the left operand's absolute values are kept.
            - Returns `<'cytimes.Delta'>`.
        - With integer or float (`int`, `float`):
            - Subtracts the numeric value from all relative delta fields.
            - For float values, each field is normalized (rounded), and the remainder
              carries over to lower fields.
            - Returns `<'cytimes.Delta'>`.
        """
        # . common
        if is_delta(o):
            return self._sub_delta(o)
        if utils.is_td(o):
            return self._sub_timedelta(o)
        if isinstance(o, typeref.RELATIVEDELTA):
            return self._sub_relativedelta(o)
        # . uncommon
        if utils.is_td64(o):
            return self._sub_timedelta(utils.td64_to_td(o))
        # . numeric
        if isinstance(o, int):
            return self._sub_int(o)
        if isinstance(o, float):
            return self._sub_float(o)
        # . unsupported
        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    def _sub_delta(self, o: Delta) -> Delta:
        """(internal) Subtraction with another cytimes.Delta,
        returns `<'cytimes.Delta'>`.
        """
        # fmt: off
        return Delta(
            self._years - o._years,
            0,
            self._months - o._months,
            0,
            self._days - o._days,
            self._hours - o._hours,
            self._minutes - o._minutes,
            self._seconds - o._seconds,
            0,
            self._microseconds - o._microseconds,
            self._year if self._year != -1 else o._year,
            self._month if self._month != -1 else o._month,
            self._day if self._day != -1 else o._day,
            self._weekday if self._weekday != -1 else o._weekday,
            self._hour if self._hour != -1 else o._hour,
            self._minute if self._minute != -1 else o._minute,
            self._second if self._second != -1 else o._second,
            -1,
            self._microsecond if self._microsecond != -1 else o._microsecond,
        )
        # fmt: on

    @cython.cfunc
    @cython.inline(True)
    def _sub_timedelta(self, o: object) -> Delta:
        """(internal) Left operand subtraction with datetime.timedelta
        (i.e., Delta - timedelta), returns `<'cytimes.Delta'>`.
        """
        return Delta(
            self._years,
            0,
            self._months,
            0,
            self._days - datetime.timedelta_days(o),
            self._hours,
            self._minutes,
            self._seconds - datetime.timedelta_seconds(o),
            0,
            self._microseconds - datetime.timedelta_microseconds(o),
            self._year,
            self._month,
            self._day,
            self._weekday,
            self._hour,
            self._minute,
            self._second,
            -1,
            self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _sub_relativedelta(self, o: relativedelta) -> Delta:
        """(internal) Left operand subtraction with dateutil.relativedelta,
        (i.e., Delta - relativedelta), returns `<'cytimes.Delta'>`."""
        # Normalize
        o = o.normalized()
        # Relative delta
        years: cython.int = o.years
        months: cython.int = o.months
        days: cython.int = o.days
        hours: cython.int = o.hours
        minutes: cython.int = o.minutes
        seconds: cython.int = o.seconds
        microseconds: cython.int = o.microseconds
        # Absolute delta
        if self._year != -1:
            year = self._year
        else:
            o_year = o.year
            year = -1 if o_year is None else o_year
        if self._month != -1:
            month = self._month
        else:
            o_month = o.month
            month = -1 if o_month is None else o_month
        if self._day != -1:
            day = self._day
        else:
            o_day = o.day
            day = -1 if o_day is None else o_day
        if self._weekday != -1:
            weekday = self._weekday
        else:
            o_weekday = o.weekday
            weekday = -1 if o_weekday is None else o_weekday.weekday
        if self._hour != -1:
            hour = self._hour
        else:
            o_hour = o.hour
            hour = -1 if o_hour is None else o_hour
        if self._minute != -1:
            minute = self._minute
        else:
            o_minute = o.minute
            minute = -1 if o_minute is None else o_minute
        if self._second != -1:
            second = self._second
        else:
            o_second = o.second
            second = -1 if o_second is None else o_second
        if self._microsecond != -1:
            microsecond = self._microsecond
        else:
            o_microsecond = o.microsecond
            microsecond = -1 if o_microsecond is None else o_microsecond
        # Create delta
        return Delta(
            self._years - years,
            0,
            self._months - months,
            0,
            self._days - days,
            self._hours - hours,
            self._minutes - minutes,
            self._seconds - seconds,
            0,
            self._microseconds - microseconds,
            year,
            month,
            day,
            weekday,
            hour,
            minute,
            second,
            -1,
            microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _sub_int(self, o: cython.int) -> Delta:
        """(internal) Left operand subtraction with int
        (i.e., Delta - int), returns `<'cytimes.Delta'>`.
        """
        return Delta(
            self._years - o,
            0,
            self._months - o,
            0,
            self._days - o,
            self._hours - o,
            self._minutes - o,
            self._seconds - o,
            0,
            self._microseconds - o,
            self._year,
            self._month,
            self._day,
            self._weekday,
            self._hour,
            self._minute,
            self._second,
            -1,
            self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _sub_float(self, o: cython.double) -> Delta:
        """(internal) Left operand subtraction with float
        (i.e., Delta - float), returns `<'cytimes.Delta'>`.
        """
        # Normalize
        # . years
        value: cython.double = self._years - o
        years: cython.int = math.llround(value)
        # . months
        value = self._months - o + (value - years) * 12
        months: cython.int = math.llround(value)
        # . days
        value = self._days - o
        days: cython.int = math.llround(value)
        # . hours
        value = self._hours - o + (value - days) * 24
        hours: cython.int = math.llround(value)
        # . minutes
        value = self._minutes - o + (value - hours) * 60
        minutes: cython.int = math.llround(value)
        # . seconds
        value = self._seconds - o + (value - minutes) * 60
        seconds: cython.int = math.llround(value)
        # . microseconds
        value = self._microseconds - o + (value - seconds) * 1_000_000
        microseconds: cython.int = math.llround(value)
        # Create delta
        return Delta(
            years,
            0,
            months,
            0,
            days,
            hours,
            minutes,
            seconds,
            0,
            microseconds,
            self._year,
            self._month,
            self._day,
            self._weekday,
            self._hour,
            self._minute,
            self._second,
            -1,
            self._microsecond,
        )

    # Arithmetic: right subtraction ------------------------------------------
    def __rsub__(self, o: object) -> object:
        """Right operand subtraction 'o - self' `<'Delta'>`.

        - With datetime instance or subclass (`datetime.datetime`, `cytimes.Pydt`, `pandas.Timestamp`, etc):
            - Similar to addition, but subtracts the relative deltas instead.
            - Returns `<'datetime.datetime'>` or the original subclass type.
        - With date instance or subclass (`datetime.date`, `pendulum.Date`, etc):
            - Similar to datetime subtraction, but returns `<'datetime.date'>`
              or the original subclass type instead.
        - With delta object (`datetime.timedelta`, `cytimes.Delta`, `pandas.Timedelta`, etc):
            - Subtracts corresponding relative delta fields (left - right).
            - For delta objects with absolute delta fields (`cytimes.Delta`, `dateutils.relativedelta`),
              the left operand's absolute values are kept.
            - Returns `<'cytimes.Delta'>`.
        - With integer or float (`int`, `float`):
            - Subtracts the numeric value from all relative delta fields.
            - For float values, each field is normalized (rounded), and the remainder
              carries over to lower fields.
            - Returns `<'cytimes.Delta'>`.
        """
        # . common
        if utils.is_dt(o):
            return self._rsub_datetime(o)
        if utils.is_date(o):
            return self._rsub_date(o)
        if utils.is_td(o):
            return self._rsub_timedelta(o)
        if isinstance(o, typeref.RELATIVEDELTA):
            return self._rsub_relativedelta(o)
        # . uncommon
        # TODO: uncommon block does not work
        if utils.is_dt64(o):
            return self._rsub_datetime(utils.dt64_to_dt(o, None))
        if utils.is_td64(o):
            return self._rsub_timedelta(utils.td64_to_td(o))
        # . numeric
        if isinstance(o, int):
            return self._rsub_int(o)
        if isinstance(o, float):
            return self._rsub_float(o)
        # . unsupported
        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    def _rsub_date(self, o: object) -> object:
        """(internal) Right operand subtraction with datetime.date instance
        or subclass (i.e., date - Delta), returns `<'datetime.date'>`
        or the original subclass type.
        """
        return _date_add_delta(
            o,
            -self._years,
            -self._months,
            -self._days,
            -self._hours,
            -self._minutes,
            -self._seconds,
            -self._microseconds,
            self._year,
            self._month,
            self._day,
            self._weekday,
            self._hour,
            self._minute,
            self._second,
            self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _rsub_datetime(self, o: object) -> object:
        """(internal) Right operand subtraction with datetime.datetime instance
        or subclass (i.e., datetime - Delta), returns `<'datetime.datetime'>`
        or the original subclass type.
        """
        return _dt_add_delta(
            o,
            -self._years,
            -self._months,
            -self._days,
            -self._hours,
            -self._minutes,
            -self._seconds,
            -self._microseconds,
            self._year,
            self._month,
            self._day,
            self._weekday,
            self._hour,
            self._minute,
            self._second,
            self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _rsub_timedelta(self, o: object) -> Delta:
        """(internal) Right operand subtraction with datetime.timedelta
        (i.e., timedelta - Delta), returns `<'cytimes.Delta'>`.
        """
        return Delta(
            -self._years,
            0,
            -self._months,
            0,
            datetime.timedelta_days(o) - self._days,
            -self._hours,
            -self._minutes,
            datetime.timedelta_seconds(o) - self._seconds,
            0,
            datetime.timedelta_microseconds(o) - self._microseconds,
            self._year,
            self._month,
            self._day,
            self._weekday,
            self._hour,
            self._minute,
            self._second,
            -1,
            self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _rsub_relativedelta(self, o: relativedelta) -> Delta:
        """(internal) Right operand subtraction with dateutil.relativedelta,
        (i.e., relativedelta - Delta), returns `<'cytimes.Delta'>`.
        """
        # Normalize
        o = o.normalized()
        # Relative delta
        years: cython.int = o.years
        months: cython.int = o.months
        days: cython.int = o.days
        hours: cython.int = o.hours
        minutes: cython.int = o.minutes
        seconds: cython.int = o.seconds
        microseconds: cython.int = o.microseconds
        # Absolute delta
        o_year = o.year
        o_month = o.month
        o_day = o.day
        o_weekday = o.weekday
        o_hour = o.hour
        o_minute = o.minute
        o_second = o.second
        o_microsecond = o.microsecond
        # Create delta
        return Delta(
            years - self._years,
            0,
            months - self._months,
            0,
            days - self._days,
            hours - self._hours,
            minutes - self._minutes,
            seconds - self._seconds,
            0,
            microseconds - self._microseconds,
            self._year if o_year is None else o_year,
            self._month if o_month is None else o_month,
            self._day if o_day is None else o_day,
            self._weekday if o_weekday is None else o_weekday.weekday,
            self._hour if o_hour is None else o_hour,
            self._minute if o_minute is None else o_minute,
            self._second if o_second is None else o_second,
            -1,
            self._microsecond if o_microsecond is None else o_microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _rsub_int(self, o: cython.int) -> Delta:
        """(internal) Right operand subtraction with int
        (i.e., int - Delta), returns `<'cytimes.Delta'>`.
        """
        return Delta(
            o - self._years,
            0,
            o - self._months,
            0,
            o - self._days,
            o - self._hours,
            o - self._minutes,
            o - self._seconds,
            0,
            o - self._microseconds,
            self._year,
            self._month,
            self._day,
            self._weekday,
            self._hour,
            self._minute,
            self._second,
            -1,
            self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _rsub_float(self, o: cython.double) -> Delta:
        """(internal) Right operand subtraction with float
        (i.e., float - Delta), returns `<'cytimes.Delta'>`.
        """
        # Normalize
        # . years
        value: cython.double = o - self._years
        years: cython.int = math.llround(value)
        # . months
        value = o - self._months + (value - years) * 12
        months: cython.int = math.llround(value)
        # . days
        value = o - self._days
        days: cython.int = math.llround(value)
        # . hours
        value = o - self._hours + (value - days) * 24
        hours: cython.int = math.llround(value)
        # . minutes
        value = o - self._minutes + (value - hours) * 60
        minutes: cython.int = math.llround(value)
        # . seconds
        value = o - self._seconds + (value - minutes) * 60
        seconds: cython.int = math.llround(value)
        # . microseconds
        value = o - self._microseconds + (value - seconds) * 1_000_000
        microseconds: cython.int = math.llround(value)
        # Create delta
        return Delta(
            years,
            0,
            months,
            0,
            days,
            hours,
            minutes,
            seconds,
            0,
            microseconds,
            self._year,
            self._month,
            self._day,
            self._weekday,
            self._hour,
            self._minute,
            self._second,
            -1,
            self._microsecond,
        )

    # Arithmetic: multiplication ---------------------------------------------
    def __mul__(self, o: object) -> Delta:
        """Left operand multiplication 'self * o' `<'Delta'>`.

        - Supports multiplication with both `<'int'>` and `<'float'>`.
        - For float values, each field is normalized (rounded), and the
          remainder carries over to lower fields.
        - Returns `<'cytimes.Delta'>`.
        """
        if isinstance(o, int):
            return self._mul_int(o)
        if isinstance(o, float):
            return self._mul_float(o)
        try:
            return self._mul_float(float(o))
        except Exception:
            return NotImplemented

    def __rmul__(self, o: object) -> Delta:
        """Right operand multiplication 'o * self' `<'Delta'>`.

        - Supports multiplication with both `<'int'>` and `<'float'>`.
        - For float values, each field is normalized (rounded), and the
          remainder carries over to lower fields.
        - Returns `<'cytimes.Delta'>`.
        """
        if isinstance(o, int):
            return self._mul_int(o)
        if isinstance(o, float):
            return self._mul_float(o)
        try:
            return self._mul_float(float(o))
        except Exception:
            return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    def _mul_int(self, i: cython.int) -> Delta:
        """(internal) Multiplication with int, returns `<'cytimes.Delta'>`."""
        return Delta(
            self._years * i,
            0,
            self._months * i,
            0,
            self._days * i,
            self._hours * i,
            self._minutes * i,
            self._seconds * i,
            0,
            self._microseconds * i,
            self._year,
            self._month,
            self._day,
            self._weekday,
            self._hour,
            self._minute,
            self._second,
            -1,
            self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _mul_float(self, f: cython.double) -> Delta:
        """(internal) Multiplication with float, returns `<'cytimes.Delta'>`."""
        # Normalize
        # . years
        value: cython.double = self._years * f
        years: cython.int = math.llround(value)
        # . months
        value = self._months * f + (value - years) * 12
        months: cython.int = math.llround(value)
        # . days
        value = self._days * f
        days: cython.int = math.llround(value)
        # . hours
        value = self._hours * f + (value - days) * 24
        hours: cython.int = math.llround(value)
        # . minutes
        value = self._minutes * f + (value - hours) * 60
        minutes: cython.int = math.llround(value)
        # . seconds
        value = self._seconds * f + (value - minutes) * 60
        seconds: cython.int = math.llround(value)
        # . microseconds
        value = self._microseconds * f + (value - seconds) * 1_000_000
        microseconds: cython.int = math.llround(value)
        # Create delta
        return Delta(
            years,
            0,
            months,
            0,
            days,
            hours,
            minutes,
            seconds,
            0,
            microseconds,
            self._year,
            self._month,
            self._day,
            self._weekday,
            self._hour,
            self._minute,
            self._second,
            -1,
            self._microsecond,
        )

    # Arithmetic: division ---------------------------------------------------
    def __truediv__(self, o: object) -> Delta:
        """Left operand division 'self / o' `<'Delta'>`.

        - Supports left division with both `<'int'>` and `<'float'>`.
        - For float values, each field is normalized (rounded), and the
          remainder carries over to lower fields.
        - Returns `<'cytimes.Delta'>`.
        """
        try:
            return self._mul_float(1 / float(o))
        except Exception:
            return NotImplemented

    # Arithmetic: negation ---------------------------------------------------
    def __neg__(self) -> Delta:
        """Negation operator `-self` `<'Delta'>`.

        - Negates all relative delta fields.
        - Absolute delta fields remain unchanged.
        - Returns `<'cytimes.Delta'>`.
        """
        return Delta(
            -self._years,
            0,
            -self._months,
            0,
            -self._days,
            -self._hours,
            -self._minutes,
            -self._seconds,
            0,
            -self._microseconds,
            self._year,
            self._month,
            self._day,
            self._weekday,
            self._hour,
            self._minute,
            self._second,
            -1,
            self._microsecond,
        )

    # Arithmetic: absolute ---------------------------------------------------
    def __abs__(self) -> Delta:
        """Absolute value operator `abs(self)` `<'Delta'>`.

        - Convets all relative delta fields to their absolute values.
        - Absolute delta fields remain unchanged.
        - Returns `<'cytimes.Delta'>`.
        """
        return Delta(
            abs(self._years),
            0,
            abs(self._months),
            0,
            abs(self._days),
            abs(self._hours),
            abs(self._minutes),
            abs(self._seconds),
            0,
            abs(self._microseconds),
            self._year,
            self._month,
            self._day,
            self._weekday,
            self._hour,
            self._minute,
            self._second,
            -1,
            self._microsecond,
        )

    # Comparison -------------------------------------------------------------
    def __eq__(self, o: object) -> bool:
        """Equality comparison 'self == o' `<'bool'>`.

        - Supports comparison with datetime.timedelta instance or subclass,
          cytimes.Delta, and dateutil.relativedelta.
        - Equal means two instance should yeild the equal result when added
          or subtracted to a datetime object.
        """
        # . common
        if is_delta(o):
            return self._eq_delta(o)
        if utils.is_td(o):
            return self._eq_timedelta(o)
        if isinstance(o, typeref.RELATIVEDELTA):
            return self._eq_relativedelta(o)
        # . uncommon
        if utils.is_td64(o):
            return self._eq_timedelta(utils.td64_to_td(o))
        # . unsupported
        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _eq_delta(self, o: Delta) -> cython.bint:
        """(internal) Check if equals to another cytimes.Delta `<'bool'>`."""
        return (
            self._years == o._years
            and self._months == o._months
            and self._days == o._days
            and self._hours == o._hours
            and self._minutes == o._minutes
            and self._seconds == o._seconds
            and self._microseconds == o._microseconds
            and self._year == o._year
            and self._month == o._month
            and self._day == o._day
            and self._weekday == o._weekday
            and self._hour == o._hour
            and self._minute == o._minute
            and self._second == o._second
            and self._microsecond == o._microsecond
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _eq_timedelta(self, o: object) -> cython.bint:
        """(internal) Check if equals to datetimetimedelta
        instance or subclass `<'bool'>`.
        """
        # Assure no extra delta
        if not (
            self._years == 0
            and self._months == 0
            and self._year == -1
            and self._month == -1
            and self._day == -1
            and self._weekday == -1
            and self._hour == -1
            and self._minute == -1
            and self._second == -1
            and self._microsecond == -1
        ):
            return False

        # Total microseconds: self
        dd: cython.longlong = self._days
        ss: cython.longlong = self._seconds
        ss += self._hours * 3_600 + self._minutes * 60
        us: cython.longlong = self._microseconds
        m_us: cython.longlong = (dd * 86_400 + ss) * 1_000_000 + us

        # Total microseconds: object
        dd = datetime.timedelta_days(o)
        ss = datetime.timedelta_seconds(o)
        us = datetime.timedelta_microseconds(o)
        o_us: cython.longlong = (dd * 86_400 + ss) * 1_000_000 + us

        # Comparison
        return m_us == o_us

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _eq_relativedelta(self, o: relativedelta) -> cython.bint:
        """(internal) Check if equals to dateutils.relativedelta `<'bool'>`."""
        # Normalize
        o = o.normalized()
        # Absolute weekday
        o_weekday = o.weekday
        if o_weekday is None:
            weekday: cython.int = -1
        elif o_weekday.n is None:
            weekday: cython.int = o_weekday.weekday
        else:
            return False  # exit: can't compare nth weekday
        if self._weekday != weekday:
            return False
        # Relative delta
        years: cython.int = o.years
        if self._years != years:
            return False
        months: cython.int = o.months
        if self._months != months:
            return False
        days: cython.int = o.days
        if self._days != days:
            return False
        hours: cython.int = o.hours
        if self._hours != hours:
            return False
        minutes: cython.int = o.minutes
        if self._minutes != minutes:
            return False
        seconds: cython.int = o.seconds
        if self._seconds != seconds:
            return False
        microseconds: cython.int = o.microseconds
        if self._microseconds != microseconds:
            return False
        # Absolute delta
        o_year = o.year
        year: cython.int = -1 if o_year is None else o_year
        if self._year != year:
            return False
        o_month = o.month
        month: cython.int = -1 if o_month is None else o_month
        if self._month != month:
            return False
        o_day = o.day
        day: cython.int = -1 if o_day is None else o_day
        if self._day != day:
            return False
        o_hour = o.hour
        hour: cython.int = -1 if o_hour is None else o_hour
        if self._hour != hour:
            return False
        o_minute = o.minute
        minute: cython.int = -1 if o_minute is None else o_minute
        if self._minute != minute:
            return False
        o_second = o.second
        second: cython.int = -1 if o_second is None else o_second
        if self._second != second:
            return False
        o_microsecond = o.microsecond
        microsecond: cython.int = -1 if o_microsecond is None else o_microsecond
        if self._microsecond != microsecond:
            return False
        # Equal
        return True

    def __bool__(self) -> bool:
        """Returns `True` if the Delta has any relative
        or absolute delta values `<'bool'>`.
        """
        return (
            self._years != 0
            or self._months != 0
            or self._days != 0
            or self._hours != 0
            or self._minutes != 0
            or self._seconds != 0
            or self._microseconds != 0
            or self._year != -1
            or self._month != -1
            or self._day != -1
            or self._weekday != -1
            or self._hour != -1
            or self._minute != -1
            or self._second != -1
            or self._microsecond != -1
        )

    # Representation ---------------------------------------------------------
    def __repr__(self) -> str:
        reprs: list = []

        # Relative delta
        if self._years != 0:
            reprs.append("years=%d" % self._years)
        if self._months != 0:
            reprs.append("months=%d" % self._months)
        if self._days != 0:
            reprs.append("days=%d" % self._days)
        if self._hours != 0:
            reprs.append("hours=%d" % self._hours)
        if self._minutes != 0:
            reprs.append("minutes=%d" % self._minutes)
        if self._seconds != 0:
            reprs.append("seconds=%d" % self._seconds)
        if self._microseconds != 0:
            reprs.append("microseconds=%d" % self._microseconds)

        # Absolute delta
        if self._year != -1:
            reprs.append("year=%d" % self._year)
        if self._month != -1:
            reprs.append("month=%d" % self._month)
        if self._day != -1:
            reprs.append("day=%d" % self._day)
        if self._weekday != -1:
            reprs.append("weekday=%s" % WEEKDAY_REPRS[self._weekday])
        if self._hour != -1:
            reprs.append("hour=%d" % self._hour)
        if self._minute != -1:
            reprs.append("minute=%d" % self._minute)
        if self._second != -1:
            reprs.append("second=%d" % self._second)
        if self._microsecond != -1:
            reprs.append("microsecond=%d" % self._microsecond)

        # Create
        return "<%s(%s)>" % (self.__class__.__name__, ", ".join(reprs))

    def __hash__(self) -> int:
        if self._hashcode == -1:
            self._hashcode = hash(
                (
                    self._years,
                    self._months,
                    self._days,
                    self._hours,
                    self._minutes,
                    self._seconds,
                    self._microseconds,
                    self._year,
                    self._month,
                    self._day,
                    self._weekday,
                    self._hour,
                    self._minute,
                    self._second,
                    self._microsecond,
                )
            )
        return self._hashcode
