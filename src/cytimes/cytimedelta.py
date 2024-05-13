# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

########## INTEGER LIMITS ##########
# min_int         -2147483648
# max_int         2147483647
# max_uint        4294967295
# min_long        -9223372036854775808
# max_long        9223372036854775807
# max_ulong       18446744073709551615
# min_llong       -9223372036854775808
# max_llong       9223372036854775807
# max_ullong      18446744073709551615

from __future__ import annotations

# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.libc.math import round as roundf  # type: ignore
from cython.cimports.cytimes import cydatetime as cydt, typeref  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
import datetime
from dateutil.relativedelta import relativedelta
from cytimes import cydatetime as cydt, typeref

__all__ = ["cytimedelta"]

# Contants ------------------------------------------------------------------------------------
# . weekday
WEEKDAY_REPRS: tuple[str, ...] = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")


# cytimedelta ---------------------------------------------------------------------------------
@cython.cclass
class cytimedelta:
    """Represent the cythonized version of `dateutil.relativedelta.relativedelta`
    with some features removed. The main purpose of `<'cytimedelta'>` is to
    provide a faster and more efficient way to calculate relative and absolute
    delta for datetime objects."""

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
    _hashcode: cython.Py_ssize_t

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
        """The cythonized version of `dateutil.relativedelta.relativedelta`
        with some features removed. The main purpose of `<'cytimedelta'>` is
        to provide a faster and more efficient way to calculate relative and
        absolute delta for datetime objects.

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

        ### Arithmetic Operations
        - Addition with datetime objects supports both left and right operand, such
        as `<'datetime.date'>`, `<'datetime.datetime'>` and `<'pandas.Timestamp'>`.
        First, the datetime will be replaced by the absolute delta (exclude weekday).
        Then, the relative delta will be added, and adjust the date to the weekday
        of the week (if weekday is specified). Returns `<'datetime.datetime'>`.

        - Addition with delta objects supports both left and right operand, such as
        `<'cytimedelta'>`, `<'dateutil.relativedelta'>`, `<'datetime.timedelta'>`
        and `<'pandas.Timedelta'>`. For objects with absolute delta, the value on the
        right operand will always be kept. For relative delta, values will be added
        together. Returns `<'cytimedelta>'`.

        - Subtraction with datetime objects only supports right operand, such as
        `<'datetime.date'>`, `<'datetime.datetime'>` and `<'pandas.Timestamp'>`.
        First, the datetime will be replaced by the absolute delta (exclude weekday).
        Then, the relative delta will be subtracted, and adjust the date to the
        weekday of the week (if weekday is specified). Returns `<'datetime.datetime'>`.

        - Substraction with delta objects supports both left and right operand, such as
        `<'cytimedelta'>`, `<'relativedelta.relativedelta'>`, `<'datetime.timedelta'>`
        and `<'pandas.Timedelta'>`. For objects with absolute delta, the value on the
        left operand will always be kept. For relative delta, value on the right
        will be subtracted from the left. Returns `<'cytimedelta>'`.

        - Supports addition, subtraction, multiplication and division with both
        `<'int'>` and `<'float'>`, which only affects relative delta. Returns
        `<'cytimedelta'>`.

        - Supports negation and absolute value, which only affects relative delta.
        Returns `<'cytimedelta'>`.

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

        ### Compatibility with `relativedelta`
        - `<'cytimedelta'>` supports direct addition and subtraction with
        `<'relativedelta'>`. Meanwhile, arithmetic operations should yeild
        equivalent result, if 'weekday' argument for relativedelta is not
        specified.
        """
        # Relative delta
        # . microseconds
        us: cython.Py_ssize_t = microseconds
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
        self._microsecond = combine_ms_us(millisecond, microsecond)  # type: ignore

        # Initial hashcode
        self._hashcode = -1

    # Properties: relative delta ---------------------------------------------
    @property
    def years(self) -> int:
        """Access the relative delta of years `<'int'>`."""
        return self._years

    @property
    def months(self) -> int:
        """Access the relative delta of months `<'int'>`."""
        return self._months

    @property
    def days(self) -> int:
        """Access the relative delta of days `<'int'>`."""
        return self._days

    @property
    def weeks(self) -> int:
        """Access the relative delta of weeks `<'int'>`."""
        return int(self._days / 7)

    @property
    def hours(self) -> int:
        """Access the relative delta of hours `<'int'>`."""
        return self._hours

    @property
    def minutes(self) -> int:
        """Access the relative delta of minutes `<'int'>`."""
        return self._minutes

    @property
    def seconds(self) -> int:
        """Access the relative delta of seconds `<'int'>`."""
        return self._seconds

    @property
    def milliseconds(self) -> int:
        """Access the relative delta of milliseconds `<'int'>`."""
        return int(self._microseconds / 1_000)

    @property
    def microseconds(self) -> int:
        """Access the relative delta of microseconds `<'int'>`."""
        return self._microseconds

    # Properties: absolute delta ---------------------------------------------
    @property
    def year(self) -> int:
        """Access the absolute year `<'int'>`.
        (Value of `-1` means `None`)."""
        return self._year

    @property
    def month(self) -> int:
        """Access the absolute month `<'int'>`.
        (Value of `-1` means `None`)."""
        return self._month

    @property
    def day(self) -> int:
        """Access the absolute day `<'int'>`.
        (Value of `-1` means `None`)."""
        return self._day

    @property
    def weekday(self) -> int:
        """Access the absolute weekday `<'int'>`.
        (Value of `-1` means `None`)."""
        return self._weekday

    @property
    def hour(self) -> int:
        """Access the absolute hour `<'int'>`.
        (Value of `-1` means `None`)."""
        return self._hour

    @property
    def minute(self) -> int:
        """Access the absolute minute `<'int'>`.
        (Value of `-1` means `None`)."""
        return self._minute

    @property
    def second(self) -> int:
        """Access the absolute second `<'int'>`.
        (Value of `-1` means `None`)."""
        return self._second

    @property
    def millisecond(self) -> int:
        """Access the absolute millisecond `<'int'>`.
        (Value of `-1` means `None`)."""
        if self._microsecond == -1:
            return -1
        return int(self._microsecond / 1_000)

    @property
    def microsecond(self) -> int:
        """Access the absolute microsecond `<'int'>`.
        (Value of `-1` means `None`)."""
        return self._microsecond

    # Special methods: addition ----------------------------------------------
    def __add__(self, o: object) -> cytimedelta | datetime.datetime:
        # . common
        if datetime.PyDateTime_Check(o):
            return self._add_datetime(o)
        if datetime.PyDate_Check(o):
            return self._add_date(o)
        if isinstance(o, cytimedelta):
            return self._add_cytimedelta(o)
        if datetime.PyDelta_Check(o):
            return self._add_timedelta(o)
        if isinstance(o, typeref.RELATIVEDELTA):
            return self._add_relativedelta(o)
        # . numeric
        if isinstance(o, int):
            return self._add_int(o)
        if isinstance(o, float):
            return self._add_float(o)
        # . unlikely numpy object
        if np.is_datetime64_object(o):
            return self._add_datetime(cydt.dt64_to_dt(o))
        if np.is_timedelta64_object(o):
            return self._add_timedelta(cydt.td64_to_td(o))
        # . unsupported
        return NotImplemented

    def __radd__(self, o: object) -> cytimedelta | datetime.datetime:
        # . common
        if datetime.PyDateTime_Check(o):
            return self._add_datetime(o)
        if datetime.PyDate_Check(o):
            return self._add_date(o)
        if datetime.PyDelta_Check(o):
            return self._add_timedelta(o)
        if isinstance(o, typeref.RELATIVEDELTA):
            return self._radd_relativedelta(o)
        # . numeric
        if isinstance(o, int):
            return self._add_int(o)
        if isinstance(o, float):
            return self._add_float(o)
        # . unlikely numpy object
        # TODO: Below does nothing since numpy does not return NotImplemented
        if np.is_datetime64_object(o):
            return self._add_datetime(cydt.dt64_to_dt(o))
        if np.is_timedelta64_object(o):
            return self._add_timedelta(cydt.td64_to_td(o))
        # . unsupported
        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    def _add_date(self, o: datetime.date) -> datetime.datetime:
        """(cfunc) Addition with `<'datetime.date'>`, returns `<'datetime.datetime'>`."""
        # Calculate YMD
        # . year
        year: cython.int = o.year if self._year == -1 else self._year
        year += self._years  # add relative years
        # . month
        month: cython.int = o.month if self._month == -1 else self._month
        if self._months != 0:
            month += self._months  # add relative months
            if month > 12:
                year += 1
                month -= 12
            elif month < 1:
                year -= 1
                month += 12
        year = min(max(year, 1), 9_999)  # clip year
        # . day
        day: cython.uint = o.day if self._day == -1 else self._day
        day = min(day, cydt.days_in_month(year, month))

        # Generate datetime
        dt: datetime.datetime = cydt.gen_dt(
            year,
            month,
            day,
            0 if self._hour == -1 else self._hour,
            0 if self._minute == -1 else self._minute,
            0 if self._second == -1 else self._second,
            0 if self._microsecond == -1 else self._microsecond,
            None,
            0,
        )

        # Add relative delta
        dt = cydt.dt_add(
            dt,
            self._days,
            self._hours * 3_600 + self._minutes * 60 + self._seconds,
            self._microseconds,
        )

        # Adjust absolute weekday
        if self._weekday != -1:
            dt = cydt.dt_adj_weekday(dt, self._weekday)

        # Return datetime
        return dt

    @cython.cfunc
    @cython.inline(True)
    def _add_datetime(self, o: datetime.datetime) -> datetime.datetime:
        """(cfunc) Addition with `<'datetime.datetime'>`, returns `<'datetime.datetime'>`."""
        # Calculate YMD
        # . year
        year: cython.int = o.year if self._year == -1 else self._year
        year += self._years  # add relative years
        # . month
        month: cython.int = o.month if self._month == -1 else self._month
        if self._months != 0:
            month += self._months  # add relative months
            if month > 12:
                year += 1
                month -= 12
            elif month < 1:
                year -= 1
                month += 12
        year = min(max(year, 1), 9_999)  # clip year
        # . day
        day: cython.uint = o.day if self._day == -1 else self._day
        day = min(day, cydt.days_in_month(year, month))

        # Generate datetime
        # fmt: off
        dt: datetime.datetime = cydt.gen_dt(
            year,
            month,
            day,
            o.hour if self._hour == -1 else self._hour,
            o.minute if self._minute == -1 else self._minute,
            o.second if self._second == -1 else self._second,
            o.microsecond if self._microsecond == -1 else self._microsecond,
            o.tzinfo,
            o.fold,
        )
        # fmt: on

        # Add relative delta
        dt = cydt.dt_add(
            dt,
            self._days,
            self._hours * 3_600 + self._minutes * 60 + self._seconds,
            self._microseconds,
        )

        # Adjust absolute weekday
        if self._weekday != -1:
            dt = cydt.dt_adj_weekday(dt, self._weekday)

        # Return datetime
        return dt

    @cython.cfunc
    @cython.inline(True)
    def _add_timedelta(self, o: datetime.timedelta) -> cytimedelta:
        """(cfunc) Addition with `<'datetime.timedelta'>`, returns `<'cytimedelta'>`."""
        return cytimedelta(
            years=self._years,
            months=self._months,
            days=self._days + o.day,
            hours=self._hours,
            minutes=self._minutes,
            seconds=self._seconds + o.second,
            microseconds=self._microseconds + o.microsecond,
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _add_cytimedelta(self, o: cytimedelta) -> cytimedelta:
        """(cfunc) Addition with `<'cytimedelta'>`, returns `<'cytimedelta'>`."""
        return cytimedelta(
            years=o._years + self._years,
            months=o._months + self._months,
            days=o._days + self._days,
            hours=o._hours + self._hours,
            minutes=o._minutes + self._minutes,
            seconds=o._seconds + self._seconds,
            microseconds=o._microseconds + self._microseconds,
            year=o._year if o._year != -1 else self._year,
            month=o._month if o._month != -1 else self._month,
            day=o._day if o._day != -1 else self._day,
            weekday=o._weekday if o._weekday != -1 else self._weekday,
            hour=o._hour if o._hour != -1 else self._hour,
            minute=o._minute if o._minute != -1 else self._minute,
            second=o._second if o._second != -1 else self._second,
            microsecond=o._microsecond if o._microsecond != -1 else self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _add_relativedelta(self, o: relativedelta) -> cytimedelta:
        """(cfunc) Addition with `<'dateutil.relativedelta'>`, returns `<'cytimedelta'>`."""
        o = o.normalized()
        wday = o.weekday
        years: cython.int = o.years
        months: cython.int = o.months
        days: cython.int = o.days
        hours: cython.int = o.hours
        minutes: cython.int = o.minutes
        seconds: cython.int = o.seconds
        microseconds: cython.int = o.microseconds
        return cytimedelta(
            # fmt: off
            years=years + self._years,
            months=months + self._months,
            days=days + self._days,
            hours=hours + self._hours,
            minutes=minutes + self._minutes,
            seconds=seconds + self._seconds,
            microseconds=microseconds + self._microseconds,
            year=o.year if o.year is not None else self._year,
            month=o.month if o.month is not None else self._month,
            day=o.day if o.day is not None else self._day,
            weekday=wday.weekday if wday is not None else self._weekday,
            hour=o.hour if o.hour is not None else self._hour,
            minute=o.minute if o.minute is not None else self._minute,
            second=o.second if o.second is not None else self._second,
            microsecond=o.microsecond if o.microsecond is not None else self._microsecond,
            # fmt: on
        )

    @cython.cfunc
    @cython.inline(True)
    def _add_int(self, o: cython.int) -> cytimedelta:
        """(cfunc) Addition with `<'int'>`, returns `<'cytimedelta'>`."""
        return cytimedelta(
            years=self._years + o,
            months=self._months + o,
            days=self._days + o,
            hours=self._hours + o,
            minutes=self._minutes + o,
            seconds=self._seconds + o,
            microseconds=self._microseconds + o,
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _add_float(self, o: cython.double) -> cytimedelta:
        """(cfuc) Addition with `<'float'>`, returns `<'cytimedelta'>`."""
        # Normalize
        # . years
        value: cython.double = self._years + o
        years: cython.int = int(roundf(value))
        # . months
        value = self._months + o + (value - years) * 12
        months: cython.int = int(roundf(value))
        # . days
        value = self._days + o
        days: cython.int = int(roundf(value))
        # . hours
        value = self._hours + o + (value - days) * 24
        hours: cython.int = int(roundf(value))
        # . minutes
        value = self._minutes + o + (value - hours) * 60
        minutes: cython.int = int(roundf(value))
        # . seconds
        value = self._seconds + o + (value - minutes) * 60
        seconds: cython.int = int(roundf(value))
        # . microseconds
        value = self._microseconds + o + (value - seconds) * 1_000_000
        microseconds: cython.int = int(roundf(value))
        # Generate
        return cytimedelta(
            years=years,
            months=months,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            microseconds=microseconds,
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _radd_relativedelta(self, o: relativedelta) -> cytimedelta:
        """(cfunc) Right addition with `<'dateutil.relativedelta'>`, returns `<'cytimedelta'>`."""
        o = o.normalized()
        wday = o.weekday
        years: cython.int = o.years
        months: cython.int = o.months
        days: cython.int = o.days
        hours: cython.int = o.hours
        minutes: cython.int = o.minutes
        seconds: cython.int = o.seconds
        microseconds: cython.int = o.microseconds
        return cytimedelta(
            # fmt: off
            years=self._years + years,
            months=self._months + months,
            days=self._days + days,
            hours=self._hours + hours,
            minutes=self._minutes + minutes,
            seconds=self._seconds + seconds,
            microseconds=self._microseconds + microseconds,
            year=self._year if self._year != -1 else (-1 if o.year is None else o.year),
            month=self._month if self._month != -1 else (-1 if o.month is None else o.month),
            day=self._day if self._day != -1 else (-1 if o.day is None else o.day),
            weekday=self._weekday if self._weekday != -1 else (-1 if wday is None else wday.weekday),
            hour=self._hour if self._hour != -1 else (-1 if o.hour is None else o.hour),
            minute=self._minute if self._minute != -1 else (-1 if o.minute is None else o.minute),
            second=self._second if self._second != -1 else (-1 if o.second is None else o.second),
            microsecond=self._microsecond if self._microsecond != -1 else (-1 if o.microsecond is None else o.microsecond),
            # fmt: on
        )

    # Special methods: substraction ------------------------------------------
    def __sub__(self, o: object) -> cytimedelta:
        # . common
        if isinstance(o, cytimedelta):
            return self._sub_cytimedelta(o)
        if datetime.PyDelta_Check(o):
            return self._sub_timedelta(o)
        if isinstance(o, typeref.RELATIVEDELTA):
            return self._sub_relativedelta(o)
        # . numeric
        if isinstance(o, int):
            return self._sub_int(o)
        if isinstance(o, float):
            return self._sub_float(o)
        # . unlikely numpy object
        if np.is_timedelta64_object(o):
            return self._sub_timedelta(cydt.td64_to_td(o))
        # . unsupported
        return NotImplemented

    def __rsub__(self, o: object) -> cytimedelta | datetime.datetime:
        # . common
        if datetime.PyDateTime_Check(o):
            return self._rsub_datetime(o)
        if datetime.PyDate_Check(o):
            return self._rsub_date(o)
        if datetime.PyDelta_Check(o):
            return self._rsub_timedelta(o)
        if isinstance(o, typeref.RELATIVEDELTA):
            return self._rsub_relativedelta(o)
        # . numeric
        if isinstance(o, int):
            return self._rsub_int(o)
        if isinstance(o, float):
            return self._rsub_float(o)
        # . unlikely numpy object
        # TODO: Below does nothing since numpy does not return NotImplemented
        if np.is_datetime64_object(o):
            return self._rsub_datetime(cydt.dt64_to_dt(o))
        if np.is_timedelta64_object(o):
            return self._rsub_timedelta(cydt.td64_to_td(o))
        # . unsupported
        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    def _sub_timedelta(self, o: datetime.timedelta) -> cytimedelta:
        """(cfunc) Substraction with `<'datetime.timedelta'>`, returns `<'cytimedelta'>`."""
        return cytimedelta(
            years=self._years,
            months=self._months,
            days=self._days - o.day,
            hours=self._hours,
            minutes=self._minutes,
            seconds=self._seconds - o.second,
            microseconds=self._microseconds - o.microsecond,
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _sub_cytimedelta(self, o: cytimedelta) -> cytimedelta:
        """(cfunc) Substraction with `<'cytimedelta'>`, returns `<'cytimedelta'>`."""
        # fmt: off
        return cytimedelta(
            years=self._years - o._years,
            months=self._months - o._months,
            days=self._days - o._days,
            hours=self._hours - o._hours,
            minutes=self._minutes - o._minutes,
            seconds=self._seconds - o._seconds,
            microseconds=self._microseconds - o._microseconds,
            year=self._year if self._year != -1 else o._year,
            month=self._month if self._month != -1 else o._month,
            day=self._day if self._day != -1 else o._day,
            weekday=self._weekday if self._weekday != -1 else o._weekday,
            hour=self._hour if self._hour != -1 else o._hour,
            minute=self._minute if self._minute != -1 else o._minute,
            second=self._second if self._second != -1 else o._second,
            microsecond=self._microsecond if self._microsecond != -1 else o._microsecond,
        )
        # fmt: on

    @cython.cfunc
    @cython.inline(True)
    def _sub_relativedelta(self, o: relativedelta) -> cytimedelta:
        """(cfunc) Substraction with `<'dateutil.relativedelta'>`, returns `<'cytimedelta'>`."""
        o = o.normalized()
        wday = o.weekday
        years: cython.int = o.years
        months: cython.int = o.months
        days: cython.int = o.days
        hours: cython.int = o.hours
        minutes: cython.int = o.minutes
        seconds: cython.int = o.seconds
        microseconds: cython.int = o.microseconds
        return cytimedelta(
            # fmt: off
            years=self._years - years,
            months=self._months - months,
            days=self._days - days,
            hours=self._hours - hours,
            minutes=self._minutes - minutes,
            seconds=self._seconds - seconds,
            microseconds=self._microseconds - microseconds,
            year=self._year if self._year != -1 else (-1 if o.year is None else o.year),
            month=self._month if self._month != -1 else (-1 if o.month is None else o.month),
            day=self._day if self._day != -1 else (-1 if o.day is None else o.day),
            weekday=self._weekday if self._weekday != -1 else (-1 if wday is None else wday.weekday),
            hour=self._hour if self._hour != -1 else (-1 if o.hour is None else o.hour),
            minute=self._minute if self._minute != -1 else (-1 if o.minute is None else o.minute),
            second=self._second if self._second != -1 else (-1 if o.second is None else o.second),
            microsecond=self._microsecond if self._microsecond != -1 else (-1 if o.microsecond is None else o.microsecond),
            # fmt: on
        )

    @cython.cfunc
    @cython.inline(True)
    def _sub_int(self, o: cython.int) -> cytimedelta:
        """(cfunc) Addition with `<'int'>`, returns `<'cytimedelta'>`."""
        return cytimedelta(
            years=self._years - o,
            months=self._months - o,
            days=self._days - o,
            hours=self._hours - o,
            minutes=self._minutes - o,
            seconds=self._seconds - o,
            microseconds=self._microseconds - o,
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _sub_float(self, o: cython.double) -> cytimedelta:
        """(cfuc) Addition with `<'float'>`, returns `<'cytimedelta'>`."""
        # Normalize
        # . years
        value: cython.double = self._years - o
        years: cython.int = int(roundf(value))
        # . months
        value = self._months - o + (value - years) * 12
        months: cython.int = int(roundf(value))
        # . days
        value = self._days - o
        days: cython.int = int(roundf(value))
        # . hours
        value = self._hours - o + (value - days) * 24
        hours: cython.int = int(roundf(value))
        # . minutes
        value = self._minutes - o + (value - hours) * 60
        minutes: cython.int = int(roundf(value))
        # . seconds
        value = self._seconds - o + (value - minutes) * 60
        seconds: cython.int = int(roundf(value))
        # . microseconds
        value = self._microseconds - o + (value - seconds) * 1_000_000
        microseconds: cython.int = int(roundf(value))
        # Generate
        return cytimedelta(
            years=years,
            months=months,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            microseconds=microseconds,
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _rsub_date(self, o: datetime.date) -> datetime.datetime:
        """(cfunc) Right substraction with `<'datetime.date'>`, returns `<'datetime.datetime'>`."""
        # Calculate YMD
        # . year
        year: cython.int = o.year if self._year == -1 else self._year
        year -= self._years  # sub relative years
        # . month
        month: cython.int = o.month if self._month == -1 else self._month
        if self._months != 0:
            month -= self._months  # sub relative months
            if month < 1:
                year -= 1
                month += 12
            elif month > 12:
                year += 1
                month -= 12
        year = min(max(year, 1), 9_999)  # clip year
        # . day
        day: cython.uint = o.day if self._day == -1 else self._day
        day = min(day, cydt.days_in_month(year, month))

        # Generate datetime
        dt: datetime.datetime = cydt.gen_dt(
            year,
            month,
            day,
            0 if self._hour == -1 else self._hour,
            0 if self._minute == -1 else self._minute,
            0 if self._second == -1 else self._second,
            0 if self._microsecond == -1 else self._microsecond,
            None,
            0,
        )

        # Sub relative delta
        dt = cydt.dt_add(
            dt,
            -self._days,
            -self._hours * 3_600 - self._minutes * 60 - self._seconds,
            -self._microseconds,
        )

        # Adjust absolute weekday
        if self._weekday != -1:
            dt = cydt.dt_adj_weekday(dt, self._weekday)

        # Return datetime
        return dt

    @cython.cfunc
    @cython.inline(True)
    def _rsub_datetime(self, o: datetime.datetime) -> datetime.datetime:
        """(cfunc) Right substraction with `<'datetime.datetime'>`, returns `<'datetime.datetime'>`."""
        # Calculate YMD
        # . year
        year: cython.int = o.year if self._year == -1 else self._year
        year -= self._years  # sub relative years
        # . month
        month: cython.int = o.month if self._month == -1 else self._month
        if self._months != 0:
            month -= self._months  # sub relative months
            if month < 1:
                year -= 1
                month += 12
            elif month > 12:
                year += 1
                month -= 12
        year = min(max(year, 1), 9_999)  # clip year
        # . day
        day: cython.uint = o.day if self._day == -1 else self._day
        day = min(day, cydt.days_in_month(year, month))

        # Generate datetime
        # fmt: off
        dt: datetime.datetime = cydt.gen_dt(
            year,
            month,
            day,
            o.hour if self._hour == -1 else self._hour,
            o.minute if self._minute == -1 else self._minute,
            o.second if self._second == -1 else self._second,
            o.microsecond if self._microsecond == -1 else self._microsecond,
            o.tzinfo,
            o.fold,
        )
        # fmt: on

        # Sub relative delta
        dt = cydt.dt_add(
            dt,
            -self._days,
            -self._hours * 3_600 - self._minutes * 60 - self._seconds,
            -self._microseconds,
        )

        # Adjust absolute weekday
        if self._weekday != -1:
            dt = cydt.dt_adj_weekday(dt, self._weekday)

        # Return datetime
        return dt

    @cython.cfunc
    @cython.inline(True)
    def _rsub_timedelta(self, o: datetime.timedelta) -> cytimedelta:
        """(cfunc) Right substraction with `<'datetime.timedelta'>`, returns `<'cytimedelta'>`."""
        return cytimedelta(
            years=-self._years,
            months=-self._months,
            days=o.day - self._days,
            hours=-self._hours,
            minutes=-self._minutes,
            seconds=o.second - self._seconds,
            microseconds=o.microsecond - self._microseconds,
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _rsub_relativedelta(self, o: relativedelta) -> cytimedelta:
        """(cfunc) Right substraction with `<'dateutil.relativedelta'>`, returns `<'cytimedelta'>`."""
        o = o.normalized()
        wday = o.weekday
        years: cython.int = o.years
        months: cython.int = o.months
        days: cython.int = o.days
        hours: cython.int = o.hours
        minutes: cython.int = o.minutes
        seconds: cython.int = o.seconds
        microseconds: cython.int = o.microseconds
        return cytimedelta(
            # fmt: off
            years=years - self._years,
            months=months - self._months,
            days=days - self._days,
            hours=hours - self._hours,
            minutes=minutes - self._minutes,
            seconds=seconds - self._seconds,
            microseconds=microseconds - self._microseconds,
            year=o.year if o.year is not None else self._year,
            month=o.month if o.month is not None else self._month,
            day=o.day if o.day is not None else self._day,
            weekday=wday.weekday if wday is not None else self._weekday,
            hour=o.hour if o.hour is not None else self._hour,
            minute=o.minute if o.minute is not None else self._minute,
            second=o.second if o.second is not None else self._second,
            microsecond=o.microsecond if o.microsecond is not None else self._microsecond,
            # fmt: on
        )

    @cython.cfunc
    @cython.inline(True)
    def _rsub_int(self, o: cython.int) -> cytimedelta:
        """(cfunc) Addition with `<'int'>`, returns `<'cytimedelta'>`."""
        return cytimedelta(
            years=o - self._years,
            months=o - self._months,
            days=o - self._days,
            hours=o - self._hours,
            minutes=o - self._minutes,
            seconds=o - self._seconds,
            microseconds=o - self._microseconds,
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _rsub_float(self, o: cython.double) -> cytimedelta:
        """(cfuc) Addition with `<'float'>`, returns `<'cytimedelta'>`."""
        # Normalize
        # . years
        value: cython.double = o - self._years
        years: cython.int = int(roundf(value))
        # . months
        value = o - self._months + (value - years) * 12
        months: cython.int = int(roundf(value))
        # . days
        value = o - self._days
        days: cython.int = int(roundf(value))
        # . hours
        value = o - self._hours + (value - days) * 24
        hours: cython.int = int(roundf(value))
        # . minutes
        value = o - self._minutes + (value - hours) * 60
        minutes: cython.int = int(roundf(value))
        # . seconds
        value = o - self._seconds + (value - minutes) * 60
        seconds: cython.int = int(roundf(value))
        # . microseconds
        value = o - self._microseconds + (value - seconds) * 1_000_000
        microseconds: cython.int = int(roundf(value))
        # Generate
        return cytimedelta(
            years=years,
            months=months,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            microseconds=microseconds,
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    # Special methods: multiplication ----------------------------------------
    def __mul__(self, o: object) -> cytimedelta:
        if isinstance(o, int):
            return self._mul_int(o)
        if isinstance(o, float):
            return self._mul_float(o)
        try:
            factor: cython.double = float(o)
        except Exception:
            return NotImplemented
        return self._mul_float(factor)

    def __rmul__(self, o: object) -> cytimedelta:
        if isinstance(o, int):
            return self._mul_int(o)
        if isinstance(o, float):
            return self._mul_float(o)
        try:
            factor: cython.double = float(o)
        except Exception:
            return NotImplemented
        return self._mul_float(factor)

    @cython.cfunc
    @cython.inline(True)
    def _mul_int(self, factor: cython.int) -> cytimedelta:
        """(cfunc) Multiplication with an `<'int'>` factor, returns `<'cytimedelta'>`."""
        return cytimedelta(
            years=self._years * factor,
            months=self._months * factor,
            days=self._days * factor,
            hours=self._hours * factor,
            minutes=self._minutes * factor,
            seconds=self._seconds * factor,
            microseconds=self._microseconds * factor,
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _mul_float(self, factor: cython.double) -> cytimedelta:
        """(cfunc) Multiplication with a `<'float'>` factor, returns `<'cytimedelta'>`."""
        # Normalize
        # . years
        value: cython.double = self._years * factor
        years: cython.int = int(roundf(value))
        # . months
        value = self._months * factor + (value - years) * 12
        months: cython.int = int(roundf(value))
        # . days
        value = self._days * factor
        days: cython.int = int(roundf(value))
        # . hours
        value = self._hours * factor + (value - days) * 24
        hours: cython.int = int(roundf(value))
        # . minutes
        value = self._minutes * factor + (value - hours) * 60
        minutes: cython.int = int(roundf(value))
        # . seconds
        value = self._seconds * factor + (value - minutes) * 60
        seconds: cython.int = int(roundf(value))
        # . microseconds
        value = self._microseconds * factor + (value - seconds) * 1_000_000
        microseconds: cython.int = int(roundf(value))
        # Generate
        return cytimedelta(
            years=years,
            months=months,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            microseconds=microseconds,
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    # Special methods: division ----------------------------------------------
    def __truediv__(self, o: object) -> cytimedelta:
        try:
            reciprocal: cython.double = 1 / float(o)
        except Exception:
            return NotImplemented
        return self._mul_float(reciprocal)

    # Special methods: negation ----------------------------------------------
    def __neg__(self) -> cytimedelta:
        return cytimedelta(
            years=-self._years,
            months=-self._months,
            days=-self._days,
            hours=-self._hours,
            minutes=-self._minutes,
            seconds=-self._seconds,
            microseconds=-self._microseconds,
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    # Special methods: absolute ----------------------------------------------
    def __abs__(self) -> cytimedelta:
        return cytimedelta(
            years=abs(self._years),
            months=abs(self._months),
            days=abs(self._days),
            hours=abs(self._hours),
            minutes=abs(self._minutes),
            seconds=abs(self._seconds),
            microseconds=abs(self._microseconds),
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    # Special methods: comparison --------------------------------------------
    def __eq__(self, o: object) -> bool:
        if isinstance(o, cytimedelta):
            return self._eq_cytimedelta(o)
        if datetime.PyDelta_Check(o):
            return self._eq_timedelta(o)
        if isinstance(o, typeref.RELATIVEDELTA):
            return self._eq_relativedelta(o)
        # . unlikely numpy object
        if np.is_timedelta64_object(o):
            return self._eq_timedelta(cydt.td64_to_td(o))
        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _eq_timedelta(self, o: datetime.timedelta) -> cython.bint:
        """(cfunc) Check if equals to a datetime.timedelta `<'bool'>`"""
        return (
            self._years == 0
            and self._months == 0
            and self._days * cydt.US_DAY
            + self._hours * cydt.US_HOUR
            + self._minutes * 60_000_000
            + self._seconds * 1_000_000
            + self._microseconds
            == o.day * cydt.US_DAY + o.second * 1_000_000 + o.microsecond
            and self._year == -1
            and self._month == -1
            and self._day == -1
            and self._weekday == -1
            and self._hour == -1
            and self._minute == -1
            and self._second == -1
            and self._microsecond == -1
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _eq_cytimedelta(self, o: cytimedelta) -> cython.bint:
        """(cfunc) Check if equals to a cytimedelta `<'bool'>`."""
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
    def _eq_relativedelta(self, o: relativedelta) -> cython.bint:
        """(cfunc) Check if equals to a relativedelta `<'bool'>`."""
        o = o.normalized()
        wday = o.weekday
        if wday is None:
            weekday: cython.int = -1
        else:
            if wday.n:
                return False  # exit: can't compare nth weekday
            weekday: cython.int = wday.weekday
        years: cython.int = o.years
        months: cython.int = o.months
        days: cython.int = o.days
        hours: cython.int = o.hours
        minutes: cython.int = o.minutes
        seconds: cython.int = o.seconds
        microseconds: cython.int = o.microseconds
        year: cython.int = -1 if o.year is None else o.year
        month: cython.int = -1 if o.month is None else o.month
        day: cython.int = -1 if o.day is None else o.day
        hour: cython.int = -1 if o.hour is None else o.hour
        minute: cython.int = -1 if o.minute is None else o.minute
        second: cython.int = -1 if o.second is None else o.second
        microsecond: cython.int = -1 if o.microsecond is None else o.microsecond
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

    # Special methods: represent ---------------------------------------------
    def __repr__(self) -> str:
        # Representations
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

        # Construct
        return "<%s(%s)>" % (self.__class__.__name__, ", ".join(reprs))

    # Special methods: hash --------------------------------------------------
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
