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


# Delta ---------------------------------------------------------------------------------------
@cython.cclass
class Delta:
    """Represent the cythonized version of `dateutil.relativedelta`.
    The main purpose of `<'Delta'>` is to provide a faster and more
    efficient way to calculate and manipulate relative and absolute
    delta for datetime objects.
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
        months: cython.int = 0,
        days: cython.int = 0,
        weeks: cython.int = 0,
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
        """The cythonized version of `dateutil.relativedelta`. The main
        purpose of `<'Delta'>` is to provide a faster and more efficient
        way to calculate and manipulate relative and absolute delta for
        datetime objects.

        ### Absolute Delta

        :param year `<'int'>`: The absolute year. Defaults to `-1 (None)`.
        :param month `<'int'>`: The absolute month. Defaults to `-1 (None)`.
        :param day `<'int'>`: The absolute day. Defaults to `-1 (None)`.
        :param weekday `<'int'>`: The absolute weekday, where Monday=0...Sunday=6. Defaults to `-1 (None)`.
        :param hour `<'int'>`: The absolute hour. Defaults to `-1 (None)`.
        :param minute `<'int'>`: The absolute minute. Defaults to `-1 (None)`.
        :param second `<'int'>`: The absolute second. Defaults to `-1 (None)`.
        :param millisecond `<'int'>`: The absolute millisecond. Defaults to `-1 (None)`.
        :param microsecond `<'int'>`: The absolute microsecond. Defaults to `-1 (None)`.

        ### Relative delta

        :param years `<'int'>`: The relative delta of years. Defaults to `0`.
        :param months `<'int'>`: The relative delta of months. Defaults to `0`.
        :param days `<'int'>`: The relative delta of days. Defaults to `0`.
        :param weeks `<'int'>`: The relative delta of weeks. Defaults to `0`.
        :param hours `<'int'>`: The relative delta of hours. Defaults to `0`.
        :param minutes `<'int'>`: The relative delta of minutes. Defaults to `0`.
        :param seconds `<'int'>`: The relative delta of seconds. Defaults to `0`.
        :param milliseconds `<'int'>`: The relative delta of milliseconds. Defaults to `0`.
        :param microseconds `<'int'>`: The relative delta of microseconds. Defaults to `0`.

        ### Compatibility with `relativedelta`
        - `<'cytimes.Delta'>` supports direct addition and subtraction with
        `<'relativedelta'>`. Meanwhile, arithmetic operations should yeild
        equivalent result, when relativedelta 'weekday=None'.

        ### Compatibility with `pandas.Timestamp/Timedelta` and `numpy.datetime64/timedelta64`
        - `<'cytimes.Delta'>` supports direct addition and subtraction with the
        above types. However, resolution will be limited to the microsecond level,
        and only returns `<'datetime.datetime'>` or `<'cytimes.Delta'>`.

        ### Arithmetic Operations
        - Addition with datetime objects supports both left and right operand, such
        as `<'datetime.date'>`, `<'datetime.datetime'>` and `<'pandas.Timestamp'>`.
        First, the datetime will be replaced by the absolute delta (exclude weekday).
        Then, the relative delta will be added, and adjust the date to the weekday
        of the week (if weekday is specified). Returns `<'datetime.datetime'>`.

        - Addition with delta objects supports both left and right operand, such as
        `<'cytimes.Delta'>`, `<'dateutil.relativedelta'>`, `<'datetime.timedelta'>`
        and `<'pandas.Timedelta'>`. For objects with absolute delta, the value on the
        right operand will always be kept. For relative delta, values will be added
        together. Returns `<'cytimes.Delta>'`.

        - Subtraction with datetime objects only supports right operand, such as
        `<'datetime.date'>`, `<'datetime.datetime'>` and `<'pandas.Timestamp'>`.
        First, the datetime will be replaced by the absolute delta (exclude weekday).
        Then, the relative delta will be subtracted, and adjust the date to the
        weekday of the week (if weekday is specified). Returns `<'datetime.datetime'>`.

        - Substraction with delta objects supports both left and right operand, such as
        `<'cytimes.Delta'>`, `<'dateutil.relativedelta'>`, `<'datetime.timedelta'>`
        and `<'pandas.Timedelta'>`. For objects with absolute delta, the value on the
        left operand will always be kept. For relative delta, value on the right
        will be subtracted from the left. Returns `<'cytimes.Delta>'`.

        - Supports addition, subtraction, multiplication and division with both
        `<'int'>` and `<'float'>`, which only affects relative delta. Returns
        `<'cytimes.Delta'>`.

        - Supports negation and absolute value, which only affects relative delta.
        Returns `<'cytimes.Delta'>`.

        ### Note: Removed Features from `relativedelta`
        - Does not support taking two date/datetime objects as input and calculate
          the relative delta between them. Affected arguments: `dt1` and `dt2`.
        - Does not support taking the `dateutil.relativedelta.weekday` as input,
          instead only support integer to represent the weekday. Affected arguments:
          `weekday`.
        - Does not support specifying the `yearday` and `nlyearday` as absolute
          delta. Affected arguments: `yearday` and `nlyearday`.
        - Does not support specifying the `leapdays` as relative delta. Affected
          arguments: `leapdays`.
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
        self._microsecond = utils.combine_abs_ms_us(millisecond, microsecond)  # type: ignore

        # Initial hashcode
        self._hashcode = -1

    # Property: relative delta -----------------------------------------------
    @property
    def years(self) -> int:
        """The relative delta for years `<'int'>`."""
        return self._years

    @property
    def months(self) -> int:
        """The relative delta for months `<'int'>`."""
        return self._months

    @property
    def days(self) -> int:
        """The relative delta for days `<'int'>`."""
        return self._days

    @property
    def weeks(self) -> int:
        """The relative delta for weeks `<'int'>`."""
        if self._days >= 0:
            return self._days // 7
        else:
            return -(-self._days // 7)

    @property
    def hours(self) -> int:
        """The relative delta for hours `<'int'>`."""
        return self._hours

    @property
    def minutes(self) -> int:
        """The relative delta for minutes `<'int'>`."""
        return self._minutes

    @property
    def seconds(self) -> int:
        """The relative delta for seconds `<'int'>`."""
        return self._seconds

    @property
    def milliseconds(self) -> int:
        """The relative delta for milliseconds `<'int'>`."""
        if self._microseconds >= 0:
            return self._microseconds // 1_000
        else:
            return -(-self._microseconds // 1_000)

    @property
    def microseconds(self) -> int:
        """The relative delta for microseconds `<'int'>`."""
        return self._microseconds

    # Properties: absolute delta ---------------------------------------------
    @property
    def year(self) -> int | None:
        """The absolute delta for year `<'int/None'>`."""
        return None if self._year == -1 else self._year

    @property
    def month(self) -> int | None:
        """The absolute delta for month `<'int/None'>`."""
        return None if self._month == -1 else self._month

    @property
    def day(self) -> int | None:
        """The absolute delta for day `<'int/None'>`."""
        return None if self._day == -1 else self._day

    @property
    def weekday(self) -> int | None:
        """The absolute delta for weekday `<'int/None'>`."""
        return None if self._weekday == -1 else self._weekday

    @property
    def hour(self) -> int | None:
        """The absolute delta for hour `<'int/None'>`."""
        return None if self._hour == -1 else self._hour

    @property
    def minute(self) -> int | None:
        """The absolute delta for minute `<'int/None'>`."""
        return None if self._minute == -1 else self._minute

    @property
    def second(self) -> int | None:
        """The absolute delta for second `<'int/None'>`."""
        return None if self._second == -1 else self._second

    @property
    def millisecond(self) -> int | None:
        """The absolute delta for millisecond `<'int/None'>`."""
        if self._microsecond == -1:
            return None
        ms: cython.int = self._microsecond // 1_000
        return None if ms == 0 else ms

    @property
    def microsecond(self) -> int | None:
        """The absolute delta for microsecond `<'int/None'>`."""
        return None if self._microsecond == -1 else self._microsecond

    # Arithmetic: addition ---------------------------------------------------
    def __add__(self, o: object) -> Delta | datetime.datetime:
        # . common
        if utils.is_dt(o):
            return self._add_datetime(o)
        if utils.is_date(o):
            return self._add_date(o)
        if isinstance(o, Delta):
            return self._add_delta(o)
        if utils.is_td(o):
            return self._add_timedelta(o)
        if isinstance(o, typeref.RELATIVEDELTA):
            return self._add_relativedelta(o)
        # . numeric
        if isinstance(o, int):
            return self._add_int(o)
        if isinstance(o, float):
            return self._add_float(o)
        # . uncommon
        if utils.is_dt64(o):
            return self._add_datetime(utils.dt64_to_dt(o))
        if utils.is_td64(o):
            return self._add_timedelta(utils.td64_to_td(o))
        # . unsupported
        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    def _add_date(self, o: object) -> datetime.datetime:
        """(cfunc) self + datetime.date, returns `<'datetime.datetime'>`."""
        # Y/M/D
        # . year
        yy: cython.int = datetime.date_year(o) if self._year == -1 else self._year
        yy += self._years
        # . month
        mm: cython.int = datetime.date_month(o) if self._month == -1 else self._month
        if self._months != 0:
            mm += self._months
            if mm > 12:
                yy += 1
                mm -= 12
            elif mm < 1:
                yy -= 1
                mm += 12
        yy = min(max(yy, 1), 9_999)
        # . day
        dd: cython.int = datetime.date_day(o) if self._day == -1 else self._day
        dd = min(dd, utils.days_in_month(yy, mm))

        # Create datetime
        # fmt: off
        dt: datetime.datetime = datetime.datetime_new(
            yy, mm, dd,  # Y/M/D
            0 if self._hour == -1 else self._hour,  # h
            0 if self._minute == -1 else self._minute,  # m 
            0 if self._second == -1 else self._second,  # s
            0 if self._microsecond == -1 else self._microsecond,  # us
            None, 0,  # TZ/Fold
        )

        # Adjust relative delta
        dt = utils.dt_add(
            dt,
            self._days, self._seconds, self._microseconds,  # D/s/us
            0, self._minutes, self._hours, 0,  # ms/m/h/weeks
        )
        # fmt: on

        # Adjust absolute weekday
        if self._weekday != -1:
            dt = utils.dt_chg_weekday(dt, self._weekday)

        # Return datetime
        return dt

    @cython.cfunc
    @cython.inline(True)
    def _add_datetime(self, o: object) -> datetime.datetime:
        """(cfunc) self + datetime.datetime, returns `<'datetime.datetime'>`."""
        # Y/M/D
        # . year
        yy: cython.int = datetime.datetime_year(o) if self._year == -1 else self._year
        yy += self._years
        # . month
        mm: cython.int = (
            datetime.datetime_month(o) if self._month == -1 else self._month
        )
        if self._months != 0:
            mm += self._months
            if mm > 12:
                yy += 1
                mm -= 12
            elif mm < 1:
                yy -= 1
                mm += 12
        yy = min(max(yy, 1), 9_999)
        # . day
        dd: cython.int = datetime.datetime_day(o) if self._day == -1 else self._day
        dd = min(dd, utils.days_in_month(yy, mm))

        # Create datetime
        # fmt: off
        dt: datetime.datetime = datetime.datetime_new(
            yy, mm, dd,  # Y/M/D
            datetime.datetime_hour(o) if self._hour == -1 else self._hour,  # h
            datetime.datetime_minute(o) if self._minute == -1 else self._minute,  # m 
            datetime.datetime_second(o) if self._second == -1 else self._second,  # s
            datetime.datetime_microsecond(o) if self._microsecond == -1 else self._microsecond,  # us
            datetime.datetime_tzinfo(o), 0,  # TZ/Fold
        )

        # Adjust relative delta
        dt = utils.dt_add(
            dt,
            self._days, self._seconds, self._microseconds,  # D/s/us
            0, self._minutes, self._hours, 0,  # ms/m/h/weeks
        )
        # fmt: on

        # Adjust absolute weekday
        if self._weekday != -1:
            dt = utils.dt_chg_weekday(dt, self._weekday)

        # Return datetime
        return dt

    @cython.cfunc
    @cython.inline(True)
    def _add_delta(self, o: Delta) -> Delta:
        """(cfunc) self + cytimes.Delta, returns `<'cytimes.Delta'>`."""
        return Delta(
            o._years + self._years,
            o._months + self._months,
            o._days + self._days,
            0,
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
        """(cfunc) self + datetime.timedelta, returns `<'cytimes.Delta'>`."""
        return Delta(
            self._years,
            self._months,
            self._days + datetime.timedelta_days(o),
            0,
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
        """(cfunc) self + dateutil.relativedelta, returns `<'cytimes.Delta'>`."""
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
            months + self._months,
            days + self._days,
            0,
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
        """(cfunc) self + int, returns `<'cytimes.Delta'>`."""
        return Delta(
            self._years + o,
            self._months + o,
            self._days + o,
            0,
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
        """(cfunc) self + float, returns `<'cytimes.Delta'>`."""
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
            months,
            days,
            0,
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
    def __radd__(self, o: object) -> Delta | datetime.datetime:
        # . common
        if utils.is_dt(o):
            return self._add_datetime(o)
        if utils.is_date(o):
            return self._add_date(o)
        if utils.is_td(o):
            return self._add_timedelta(o)
        if isinstance(o, typeref.RELATIVEDELTA):
            return self._radd_relativedelta(o)
        # . numeric
        if isinstance(o, int):
            return self._add_int(o)
        if isinstance(o, float):
            return self._add_float(o)
        # . uncommon
        # TODO: Below does nothing since numpy does not return NotImplemented
        if utils.is_dt64(o):
            return self._add_datetime(utils.dt64_to_dt(o))
        if utils.is_td64(o):
            return self._add_timedelta(utils.td64_to_td(o))
        # . unsupported
        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    def _radd_relativedelta(self, o: relativedelta) -> Delta:
        """(cfunc) dateutil.relativedelta + self, returns `<'cytimes.Delta'>`."""
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
            self._months + months,
            self._days + days,
            0,
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

    # Arithmetic: substraction -----------------------------------------------
    def __sub__(self, o: object) -> Delta:
        # . common
        if isinstance(o, Delta):
            return self._sub_delta(o)
        if utils.is_td(o):
            return self._sub_timedelta(o)
        if isinstance(o, typeref.RELATIVEDELTA):
            return self._sub_relativedelta(o)
        # . numeric
        if isinstance(o, int):
            return self._sub_int(o)
        if isinstance(o, float):
            return self._sub_float(o)
        # . uncommon
        if utils.is_td64(o):
            return self._sub_timedelta(utils.td64_to_td(o))
        # . unsupported
        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    def _sub_delta(self, o: Delta) -> Delta:
        """(cfunc) self - cytimes.Delta, returns `<'cytimes.Delta'>`."""
        # fmt: off
        return Delta(
            self._years - o._years,
            self._months - o._months,
            self._days - o._days,
            0,
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
        """(cfunc) self - datetime.timedelta, returns `<'cytime.Delta'>`."""
        return Delta(
            self._years,
            self._months,
            self._days - datetime.timedelta_days(o),
            0,
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
        """(cfunc) self - dateutil.relativedelta, returns `<'cytimes.Delta'>`."""
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
            self._months - months,
            self._days - days,
            0,
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
        """(cfunc) self - int, returns `<'cytimes.Delta'>`."""
        return Delta(
            self._years - o,
            self._months - o,
            self._days - o,
            0,
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
        """(cfunc) self - float, returns `<'cytimes.Delta'>`."""
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
            months,
            days,
            0,
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

    # Arithmetic: right substraction -----------------------------------------
    def __rsub__(self, o: object) -> Delta | datetime.datetime:
        # . common
        if utils.is_dt(o):
            return self._rsub_datetime(o)
        if utils.is_date(o):
            return self._rsub_date(o)
        if utils.is_td(o):
            return self._rsub_timedelta(o)
        if isinstance(o, typeref.RELATIVEDELTA):
            return self._rsub_relativedelta(o)
        # . numeric
        if isinstance(o, int):
            return self._rsub_int(o)
        if isinstance(o, float):
            return self._rsub_float(o)
        # . uncommon
        # TODO: Below does nothing since numpy does not return NotImplemented
        if utils.is_dt64(o):
            return self._rsub_datetime(utils.dt64_to_dt(o))
        if utils.is_td64(o):
            return self._rsub_timedelta(utils.td64_to_td(o))
        # . unsupported
        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    def _rsub_date(self, o: object) -> datetime.datetime:
        """(cfunc) datetime.date - self, returns `<'datetime.datetime'>`."""
        # Y/M/D
        # . year
        yy: cython.int = datetime.date_year(o) if self._year == -1 else self._year
        yy -= self._years
        # . month
        mm: cython.int = datetime.date_month(o) if self._month == -1 else self._month
        if self._months != 0:
            mm -= self._months
            if mm < 1:
                yy -= 1
                mm += 12
            elif mm > 12:
                yy += 1
                mm -= 12
        yy = min(max(yy, 1), 9_999)
        # . day
        dd: cython.int = datetime.date_day(o) if self._day == -1 else self._day
        dd = min(dd, utils.days_in_month(yy, mm))

        # Create datetime
        # fmt: off
        dt: datetime.datetime = datetime.datetime_new(
            yy, mm, dd,  # Y/M/D
            0 if self._hour == -1 else self._hour,  # h
            0 if self._minute == -1 else self._minute,  # m 
            0 if self._second == -1 else self._second,  # s
            0 if self._microsecond == -1 else self._microsecond,  # us
            None, 0,  # tz/fold
        )

        # Adjust relative delta
        dt = utils.dt_add(
            dt,
            -self._days, -self._seconds, -self._microseconds,  # D/s/us
            0, -self._minutes, -self._hours, 0,  # ms/m/h/weeks
        )
        # fmt: on

        # Adjust absolute weekday
        if self._weekday != -1:
            dt = utils.dt_chg_weekday(dt, self._weekday)

        # Return datetime
        return dt

    @cython.cfunc
    @cython.inline(True)
    def _rsub_datetime(self, o: object) -> datetime.datetime:
        """(cfunc) datetime.datetime - self, returns `<'datetime.datetime'>`."""
        # Y/M/D
        # . year
        yy: cython.int = datetime.datetime_year(o) if self._year == -1 else self._year
        yy -= self._years
        # . month
        mm: cython.int = (
            datetime.datetime_month(o) if self._month == -1 else self._month
        )
        if self._months != 0:
            mm -= self._months
            if mm < 1:
                yy -= 1
                mm += 12
            elif mm > 12:
                yy += 1
                mm -= 12
        yy = min(max(yy, 1), 9_999)
        # . day
        dd: cython.int = datetime.datetime_day(o) if self._day == -1 else self._day
        dd = min(dd, utils.days_in_month(yy, mm))

        # Create datetime
        # fmt: off
        dt: datetime.datetime = datetime.datetime_new(
            yy, mm, dd,  # Y/M/D
            datetime.datetime_hour(o) if self._hour == -1 else self._hour,  # h
            datetime.datetime_minute(o) if self._minute == -1 else self._minute,  # m 
            datetime.datetime_second(o) if self._second == -1 else self._second,  # s
            datetime.datetime_microsecond(o) if self._microsecond == -1 else self._microsecond,  # us
            datetime.datetime_tzinfo(o), 0,  # tz/fold
        )

        # Adjust relative delta
        dt = utils.dt_add(
            dt,
            -self._days, -self._seconds, -self._microseconds,  # D/s/us
            0, -self._minutes, -self._hours, 0,  # ms/m/h/weeks
        )
        # fmt: on

        # Adjust absolute weekday
        if self._weekday != -1:
            dt = utils.dt_chg_weekday(dt, self._weekday)

        # Return datetime
        return dt

    @cython.cfunc
    @cython.inline(True)
    def _rsub_timedelta(self, o: object) -> Delta:
        """(cfunc) datetime.timedelta - self, returns `<'cytimes.Delta'>`."""
        return Delta(
            -self._years,
            -self._months,
            datetime.timedelta_days(o) - self._days,
            0,
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
        """(cfunc) dateutil.relativedelta - self, returns `<'cytimes.Delta'>`."""
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
            months - self._months,
            days - self._days,
            0,
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
        """(cfunc) int - self, returns `<'cytimes.Delta'>`."""
        return Delta(
            o - self._years,
            o - self._months,
            o - self._days,
            0,
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
        """(cfunc) float - self, returns `<'cytimes.Delta'>`."""
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
            months,
            days,
            0,
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
        if isinstance(o, int):
            return self._mul_int(o)
        if isinstance(o, float):
            return self._mul_float(o)
        try:
            return self._mul_float(float(o))
        except Exception:
            return NotImplemented

    def __rmul__(self, o: object) -> Delta:
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
        """(cfunc) self * int, returns `<'cytimes.Delta'>`."""
        return Delta(
            self._years * i,
            self._months * i,
            self._days * i,
            0,
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
        """(cfunc) self * float, returns `<'cytimes.Delta'>`."""
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
            months,
            days,
            0,
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
        try:
            return self._mul_float(1 / float(o))
        except Exception:
            return NotImplemented

    # Arithmetic: negation ---------------------------------------------------
    def __neg__(self) -> Delta:
        return Delta(
            -self._years,
            -self._months,
            -self._days,
            0,
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
        return Delta(
            abs(self._years),
            abs(self._months),
            abs(self._days),
            0,
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
        # . common
        if isinstance(o, Delta):
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
        """(cfunc) Check if self == cytimes.Delta `<'bool'>`."""
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
        """(cfunc) Check if self == datetime.timedelta `<'bool'>`."""
        # Assure no extra delta
        no_extra: cython.bint = (
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
        )
        if not no_extra:
            return False

        # Total microseconds: self
        dd: cython.longlong = self._days
        ss: cython.longlong = self._seconds
        ss += self._hours * 3_600 + self._minutes * 60
        us: cython.longlong = self._microseconds
        m_us: cython.longlong = dd * utils.US_DAY + ss * 1_000_000 + us

        # Total microseconds: object
        dd, ss, us = (
            datetime.timedelta_days(o),
            datetime.timedelta_seconds(o),
            datetime.timedelta_microseconds(o),
        )
        o_us: cython.longlong = dd * utils.US_DAY + ss * 1_000_000 + us

        # Comparison
        return m_us == o_us

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _eq_relativedelta(self, o: relativedelta) -> cython.bint:
        """(cfunc) Check if self == dateutil.relativedelta `<'bool'>`."""
        # Normalize
        o = o.normalized()
        # Check weekday
        o_weekday = o.weekday
        if o_weekday is None:
            weekday: cython.int = -1
        elif o_weekday.n is None:
            weekday: cython.int = o_weekday.weekday
        else:
            return False  # exit: can't compare nth weekday
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
        year: cython.int = -1 if o_year is None else o_year
        o_month = o.month
        month: cython.int = -1 if o_month is None else o_month
        o_day = o.day
        day: cython.int = -1 if o_day is None else o_day
        o_hour = o.hour
        hour: cython.int = -1 if o_hour is None else o_hour
        o_minute = o.minute
        minute: cython.int = -1 if o_minute is None else o_minute
        o_second = o.second
        second: cython.int = -1 if o_second is None else o_second
        o_microsecond = o.microsecond
        microsecond: cython.int = -1 if o_microsecond is None else o_microsecond
        # Comparison
        return (
            self._years == years
            and self._months == months
            and self._days == days
            and self._hours == hours
            and self._minutes == minutes
            and self._seconds == seconds
            and self._microseconds == microseconds
            and self._year == year
            and self._month == month
            and self._day == day
            and self._weekday == weekday
            and self._hour == hour
            and self._minute == minute
            and self._second == second
            and self._microsecond == microsecond
        )

    def __bool__(self) -> bool:
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
