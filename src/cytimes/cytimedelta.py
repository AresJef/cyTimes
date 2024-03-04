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
from cython.cimports.libc import stdlib  # type: ignore
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cytimes import cydatetime as cydt  # type: ignore

np.import_array()
datetime.import_datetime()

# Python imports
import datetime
from dateutil.relativedelta import relativedelta
from cytimes import cydatetime as cydt

__all__ = ["cytimedelta"]

# Contants ------------------------------------------------------------------------------------
# . weekday
WEEKDAY_REPRS: tuple[str, ...] = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")
# . type
TP_RELATIVEDELTA: object = relativedelta


# cytimedelta ---------------------------------------------------------------------------------
@cython.cclass
class cytimedelta:
    """Represent the cythonized version of `dateutil.relativedelta.relativedelta`
    with some features removed. The main purpose of `cytimedelta` is to provide
    a faster and more efficient way to calculate relative and absolute time
    delta between two date & time objects."""

    _years: cython.longlong
    _months: cython.longlong
    _days: cython.longlong
    _hours: cython.longlong
    _minutes: cython.longlong
    _seconds: cython.longlong
    _microseconds: cython.longlong
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
        """The cythonized version of `dateutil.relativedelta.relativedelta` with
        some features removed. The main purpose of `cytimedelta` is to provide
        a faster and more efficient way to calculate relative and absolute time
        delta between two date & time objects.

        ### Absolute Delta
        :param year `<int>`: The absolute year value. Defaults to `-1 (no change)`.
        :param month `<int>`: The absolute month value. Defaults to `-1 (no change)`.
        :param day `<int>`: The absolute day value. Defaults to `-1 (no change)`.
        :param weekday `<int>`: The absolute weekday value, where Monday is 0 ... Sunday is 6. Defaults to `-1 (no change)`.
        :param hour `<int>`: The absolute hour value. Defaults to `-1 (no change)`.
        :param minute `<int>`: The absolute minute value. Defaults to `-1 (no change)`.
        :param second `<int>`: The absolute second value. Defaults to `-1 (no change)`.
        :param millisecond `<int>`: The absolute millisecond value. Defaults to `-1 (no change)`.
        :param microsecond `<int>`: The absolute microsecond value. Defaults to `-1 (no change)`.

        ### Relative delta
        :param years `<int>`: The relative delta of years. Defaults to `0`.
        :param months `<int>`: The relative delta of months. Defaults to `0`.
        :param days `<int>`: The relative delta of days. Defaults to `0`.
        :param weeks `<int>`: The relative delta of weeks. Defaults to `0`.
        :param hours `<int>`: The relative delta of hours. Defaults to `0`.
        :param minutes `<int>`: The relative delta of minutes. Defaults to `0`.
        :param seconds `<int>`: The relative delta of seconds. Defaults to `0`.
        :param milliseconds `<int>`: The relative delta of milliseconds. Defaults to `0`.
        :param microseconds `<int>`: The relative delta of microseconds. Defaults to `0`.

        ### Arithmetic Operations
        - Addition with date & time objects supports both left and right operand,
        such as `datetime.date`, `datetime.datetime` and `pandas.Timestamp`. First,
        the date & time will be replaced by the absolute delta (exclude weekday).
        Then, the relative delta will be added, and adjust the date to the weekday
        of the week (if weekday is specified). Finally, a new `datetime.datetime`
        object will be returned.

        - Addition with delta objects supports both left and right operand, such as
        `cytimedelta`, `dateutil.relativedelta.relativedelta`, `datetime.timedelta`
        and `pandas.Timedelta`. For objects with absolute delta, the value on the
        right operand will always be kept. For relative delta, values will be added
        together. Finally, a new `cytimedelta` object will be returned.

        - Subtraction with date & time objects only supports right operand, such as
        `datetime.date`, `datetime.datetime` and `pandas.Timestamp`. First, the
        date & time will be replaced by the absolute delta (exclude weekday). Then,
        the relative delta will be subtracted, and adjust the date to the weekday
        of the week (if weekday is specified). Finally, a new `datetime.datetime`
        object will be returned.

        - Substraction with delta objects supports both left and right operand, such
        as `cytimedelta`, `dateutil.relativedelta.relativedelta`, `datetime.timedelta`
        and `pandas.Timedelta`. For objects with absolute delta, the value on the
        left operand will always be kept. For relative delta, value on the right
        will be subtracted from the left. Finally, a new `cytimedelta` object will
        be returned.

        - Multiplication, division, negation and absolute value are supported, but
        only affects the relative delta.

        ### Removed Features
        - Does not support taking two date/datetime objects as input and calculate
          the relative delta between them. Affected arguments: `dt1` and `dt2`.
        - Does not support taking the `dateutil.relativedelta.weekday` as input,
          instead only support integer to represent the weekday. Affected arguments:
          `weekday`.
        - Does not support specifying the `yearday` and `nlyearday` as absolute
          delta. Affected arguments: `yearday` and `nlyearday`.
        - Does not support specifying the `leapdays` as relative delta. Affected
          arguments: `leapdays`.

        ### Compatibility
        - `cytimedelta` supports direct addition and subtraction with `relativedelta`,
          which should yield the same result (if the `weekday` argument is not used)
          as adding / subtracting two `relativedelta` objects.
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
        self._weekday = weekday % 7 if weekday >= 0 else -1
        self._hour = min(hour, 23) if hour >= 0 else -1
        self._minute = min(minute, 59) if minute >= 0 else -1
        self._second = min(second, 59) if second >= 0 else -1
        if millisecond > 0:
            self._microsecond = min(millisecond, 999) * 1_000
            if microsecond > 0:
                self._microsecond += microsecond % 1_000
        elif microsecond > 0:
            self._microsecond = min(microsecond, 999_999)
        else:
            self._microsecond = -1

        # Initial hashcode
        self._hashcode = -1

    # Properties: relative delta ---------------------------------------------
    @property
    def years(self) -> int:
        """Access the relative delta of years `<int>`."""
        return self._years

    @property
    def months(self) -> int:
        """Access the relative delta of months `<int>`."""
        return self._months

    @property
    def days(self) -> int:
        """Access the relative delta of days `<int>`."""
        return self._days

    @property
    def weeks(self) -> int:
        """Access the relative delta of weeks `<int>`."""
        return int(self._days / 7)

    @property
    def hours(self) -> int:
        """Access the relative delta of hours `<int>`."""
        return self._hours

    @property
    def minutes(self) -> int:
        """Access the relative delta of minutes `<int>`."""
        return self._minutes

    @property
    def seconds(self) -> int:
        """Access the relative delta of seconds `<int>`."""
        return self._seconds

    @property
    def milliseconds(self) -> int:
        """Access the relative delta of milliseconds `<int>`."""
        return int(self._microseconds / 1_000)

    @property
    def microseconds(self) -> int:
        """Access the relative delta of microseconds `<int>`."""
        return self._microseconds

    # Properties: absolute delta ---------------------------------------------
    @property
    def year(self) -> int:
        """Access the absolute year value `<int>`.
        (Value of `-1` means not set)."""
        return self._year

    @property
    def month(self) -> int:
        """Access the absolute month value `<int>`.
        (Value of `-1` means not set)."""
        return self._month

    @property
    def day(self) -> int:
        """Access the absolute day value `<int>`.
        (Value of `-1` means not set)."""
        return self._day

    @property
    def weekday(self) -> int:
        """Access the absolute weekday value `<int>`.
        (Value of `-1` means not set)."""
        return self._weekday

    @property
    def hour(self) -> int:
        """Access the absolute hour value `<int>`.
        (Value of `-1` means not set)."""
        return self._hour

    @property
    def minute(self) -> int:
        """Access the absolute minute value `<int>`.
        (Value of `-1` means not set)."""
        return self._minute

    @property
    def second(self) -> int:
        """Access the absolute second value `<int>`.
        (Value of `-1` means not set)."""
        return self._second

    @property
    def millisecond(self) -> int:
        """Access the absolute millisecond value `<int>`.
        (Value of `-1` means not set)."""
        if self._microsecond == -1:
            return -1

        return int(self._microsecond / 1_000)

    @property
    def microsecond(self) -> int:
        """Access the absolute microsecond value `<int>`.
        (Value of `-1` means not set)."""
        return self._microsecond

    # Special methods: addition ----------------------------------------------
    def __add__(self, o: object) -> cytimedelta | datetime.datetime:
        # . common
        if cydt.is_dt(o):
            return self._add_datetime(o)
        if cydt.is_date(o):
            return self._add_date(o)
        if isinstance(o, cytimedelta):
            return self._add_cytimedeta(o)
        if cydt.is_delta(o):
            return self._add_timedelta(o)
        if isinstance(o, TP_RELATIVEDELTA):
            return self._add_relativedelta(o)
        # . unlikely numpy object
        if cydt.is_dt64(o):
            return self._add_datetime(cydt.dt64_to_dt(o))
        if cydt.is_delta64(o):
            return self._add_timedelta(cydt.delta64_to_delta(o))
        # . unsupported
        return NotImplemented

    def __radd__(self, o: object) -> cytimedelta | datetime.datetime:
        # . common
        if cydt.is_dt(o):
            return self._add_datetime(o)
        if cydt.is_date(o):
            return self._add_date(o)
        if cydt.is_delta(o):
            return self._add_timedelta(o)
        if isinstance(o, TP_RELATIVEDELTA):
            return self._radd_relativedelta(o)
        # . unlikely numpy object
        # TODO: Below does nothing since numpy does not return NotImplemented
        if cydt.is_dt64(o):
            return self._add_datetime(cydt.dt64_to_dt(o))
        if cydt.is_delta64(o):
            return self._add_timedelta(cydt.delta64_to_delta(o))
        # . unsupported
        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    def _add_date(self, o: datetime.date) -> datetime.datetime:
        """(Internal) Addition `__add__` with `datetime.date`
        and returns `<datetime.datetime>`."""
        # Calculate date
        # . year
        year: cython.uint = self._year if self._year > 0 else cydt.access_year(o)
        year = min(max(year + self._years, 1), 9_999)  # add relative years
        # . month
        month: cython.uint = self._month if self._month > 0 else cydt.access_month(o)
        if self._months != 0:
            tmp: cython.int = month + self._months  # add relative months
            if tmp > 12:
                if year < 9_999:
                    year += 1
                month = tmp - 12
            elif tmp < 1:
                if year > 1:
                    year -= 1
                month = tmp + 12
            else:
                month = tmp
        # . day
        day: cython.uint = self._day if self._day > 0 else cydt.access_day(o)
        day = min(day, cydt.days_in_month(year, month))

        # Generate datetime
        dt: datetime.datetime = cydt.gen_dt(
            year,
            month,
            day,
            self._hour if self._hour >= 0 else 0,
            self._minute if self._minute >= 0 else 0,
            self._second if self._second >= 0 else 0,
            self._microsecond if self._microsecond >= 0 else 0,
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
        if self._weekday >= 0:
            dt = cydt.dt_adj_weekday(dt, self._weekday)

        # Return datetime
        return dt

    @cython.cfunc
    @cython.inline(True)
    def _add_datetime(self, o: datetime.datetime) -> datetime.datetime:
        """(Internal) Addition `__add__` with `datetime.datetime`
        and returns `<datetime.datetime>`."""
        # Calculate date & time
        # . year
        year: cython.uint = self._year if self._year > 0 else cydt.access_dt_year(o)
        year = min(max(year + self._years, 1), 9_999)  # add relative years
        # . month
        month: cython.uint = self._month if self._month > 0 else cydt.access_dt_month(o)
        if self._months != 0:
            tmp: cython.int = month + self._months  # add relative months
            if tmp > 12:
                if year < 9_999:
                    year += 1
                month = tmp - 12
            elif tmp < 1:
                if year > 1:
                    year -= 1
                month = tmp + 12
            else:
                month = tmp
        # . day
        day: cython.uint = self._day if self._day > 0 else cydt.access_dt_day(o)
        day = min(day, cydt.days_in_month(year, month))

        # Generate datetime
        # fmt: off
        dt: datetime.datetime = cydt.gen_dt(
            year,
            month,
            day,
            self._hour if self._hour >= 0 else cydt.access_dt_hour(o),
            self._minute if self._minute >= 0 else cydt.access_dt_minute(o),
            self._second if self._second >= 0 else cydt.access_dt_second(o),
            self._microsecond if self._microsecond >= 0 else cydt.access_dt_microsecond(o),
            cydt.access_dt_tzinfo(o),
            cydt.access_dt_fold(o),
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
        if self._weekday >= 0:
            dt = cydt.dt_adj_weekday(dt, self._weekday)

        # Return datetime
        return dt

    @cython.cfunc
    @cython.inline(True)
    def _add_cytimedelta(self, o: cytimedelta) -> cytimedelta:
        """(Internal) Addition `__add__` with `cytimedelta`
        and returns `<cytimedelta>`."""
        return cytimedelta(
            years=o._years + self._years,
            months=o._months + self._months,
            days=o._days + self._days,
            hours=o._hours + self._hours,
            minutes=o._minutes + self._minutes,
            seconds=o._seconds + self._seconds,
            microseconds=o._microseconds + self._microseconds,
            year=o._year if o._year > 0 else self._year,
            month=o._month if o._month > 0 else self._month,
            day=o._day if o._day > 0 else self._day,
            weekday=o._weekday if o._weekday >= 0 else self._weekday,
            hour=o._hour if o._hour >= 0 else self._hour,
            minute=o._minute if o._minute >= 0 else self._minute,
            second=o._second if o._second >= 0 else self._second,
            microsecond=o._microsecond if o._microsecond >= 0 else self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _add_timedelta(self, o: datetime.timedelta) -> cytimedelta:
        """(Internal) Addition `__add__` with `datetime.timedelta`
        and returns `<cytimedelta>`."""
        return cytimedelta(
            years=self._years,
            months=self._months,
            days=cydt.access_delta_days(o) + self._days,
            hours=self._hours,
            minutes=self._minutes,
            seconds=cydt.access_delta_seconds(o) + self._seconds,
            microseconds=cydt.access_delta_microseconds(o) + self._microseconds,
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
    def _add_relativedelta(self, o: relativedelta) -> cytimedelta:
        """(Internal) Addition `__add__` with `relativedelta`
        and returns `<cytimedelta>`."""
        o = o.normalized()
        wday = o.weekday
        return cytimedelta(
            # fmt: off
            years=o.years + self._years,
            months=o.months + self._months,
            days=o.days + self._days,
            hours=o.hours + self._hours,
            minutes=o.minutes + self._minutes,
            seconds=o.seconds + self._seconds,
            microseconds=o.microseconds + self._microseconds,
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
    def _radd_relativedelta(self, o: relativedelta) -> cytimedelta:
        """(Internal) Right addition `__radd__` with `relativedelta`
        and returns `<cytimedelta>`."""
        o = o.normalized()
        wday = o.weekday
        return cytimedelta(
            # fmt: off
            years=self._years + o.years,
            months=self._months + o.months,
            days=self._days + o.days,
            hours=self._hours + o.hours,
            minutes=self._minutes + o.minutes,
            seconds=self._seconds + o.seconds,
            microseconds=self._microseconds + o.microseconds,
            year=self._year if self._year > 0 else (-1 if o.year is None else o.year),
            month=self._month if self._month > 0 else (-1 if o.month is None else o.month),
            day=self._day if self._day > 0 else (-1 if o.day is None else o.day),
            weekday=self._weekday if self._weekday >= 0 else (-1 if wday is None else wday.weekday),
            hour=self._hour if self._hour >= 0 else (-1 if o.hour is None else o.hour),
            minute=self._minute if self._minute >= 0 else (-1 if o.minute is None else o.minute),
            second=self._second if self._second >= 0 else (-1 if o.second is None else o.second),
            microsecond=self._microsecond if self._microsecond >= 0 else (-1 if o.microsecond is None else o.microsecond),
            # fmt: on
        )

    # Special methods: substraction ------------------------------------------
    def __sub__(self, o: object) -> cytimedelta:
        # . common
        if isinstance(o, cytimedelta):
            return self._sub_cytimedelta(o)
        if cydt.is_delta(o):
            return self._sub_timedelta(o)
        if isinstance(o, TP_RELATIVEDELTA):
            return self._sub_relativedelta(o)
        # . unlikely numpy object
        if cydt.is_delta64(o):
            return self._sub_timedelta(cydt.delta64_to_delta(o))
        # . unsupported
        return NotImplemented

    def __rsub__(self, o: object) -> cytimedelta | datetime.datetime:
        # . common
        if cydt.is_dt(o):
            return self._rsub_datetime(o)
        if cydt.is_date(o):
            return self._rsub_date(o)
        if cydt.is_delta(o):
            return self._rsub_timedelta(o)
        if isinstance(o, TP_RELATIVEDELTA):
            return self._rsub_relativedelta(o)
        # . unlikely numpy object
        # TODO: Below does nothing since numpy does not return NotImplemented
        if cydt.is_dt64(o):
            return self._rsub_datetime(cydt.dt64_to_dt(o))
        if cydt.is_delta64(o):
            return self._rsub_timedelta(cydt.delta64_to_delta(o))
        # . unsupported
        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    def _sub_cytimedelta(self, o: cytimedelta) -> cytimedelta:
        """(Internal) Substraction `__sub__` with `cytimedelta`
        and returns `<cytimedelta>`."""
        return cytimedelta(
            years=self._years - o._years,
            months=self._months - o._months,
            days=self._days - o._days,
            hours=self._hours - o._hours,
            minutes=self._minutes - o._minutes,
            seconds=self._seconds - o._seconds,
            microseconds=self._microseconds - o._microseconds,
            year=self._year if self._year > 0 else o._year,
            month=self._month if self._month > 0 else o._month,
            day=self._day if self._day > 0 else o._day,
            weekday=self._weekday if self._weekday >= 0 else o._weekday,
            hour=self._hour if self._hour >= 0 else o._hour,
            minute=self._minute if self._minute >= 0 else o._minute,
            second=self._second if self._second >= 0 else o._second,
            microsecond=self._microsecond if self._microsecond >= 0 else o._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    def _sub_timedelta(self, o: datetime.timedelta) -> cytimedelta:
        """(Internal) Substraction `__sub__` with `datetime.timedelta`
        and returns `<cytimedelta>`."""
        return cytimedelta(
            years=self._years,
            months=self._months,
            days=self._days - cydt.access_delta_days(o),
            hours=self._hours,
            minutes=self._minutes,
            seconds=self._seconds - cydt.access_delta_seconds(o),
            microseconds=self._microseconds - cydt.access_delta_microseconds(o),
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
    def _sub_relativedelta(self, o: relativedelta) -> cytimedelta:
        """(Internal) Substraction `__sub__` with `relativedelta`
        and returns `<cytimedelta>`."""
        o = o.normalized()
        wday = o.weekday
        return cytimedelta(
            # fmt: off
            years=self._years - o.years,
            months=self._months - o.months,
            days=self._days - o.days,
            hours=self._hours - o.hours,
            minutes=self._minutes - o.minutes,
            seconds=self._seconds - o.seconds,
            microseconds=self._microseconds - o.microseconds,
            year=self._year if self._year > 0 else (-1 if o.year is None else o.year),
            month=self._month if self._month > 0 else (-1 if o.month is None else o.month),
            day=self._day if self._day > 0 else (-1 if o.day is None else o.day),
            weekday=self._weekday if self._weekday >= 0 else (-1 if wday is None else wday.weekday),
            hour=self._hour if self._hour >= 0 else (-1 if o.hour is None else o.hour),
            minute=self._minute if self._minute >= 0 else (-1 if o.minute is None else o.minute),
            second=self._second if self._second >= 0 else (-1 if o.second is None else o.second),
            microsecond=self._microsecond if self._microsecond >= 0 else (-1 if o.microsecond is None else o.microsecond),
            # fmt: on
        )

    @cython.cfunc
    @cython.inline(True)
    def _rsub_date(self, o: datetime.date) -> datetime.datetime:
        """(Internal) Right substraction `__rsub__` with `datetime.date`
        and returns `<datetime.datetime>`."""
        # Calculate date
        # . year
        year: cython.uint = self._year if self._year > 0 else cydt.access_year(o)
        year = min(max(year - self._years, 1), 9_999)  # sub relative years
        # . month
        month: cython.uint = self._month if self._month > 0 else cydt.access_month(o)
        if self._months != 0:
            tmp: cython.int = month - self._months  # sub relative months
            if tmp < 1:
                if year > 1:
                    year -= 1
                month = tmp + 12
            elif tmp > 12:
                if year < 9_999:
                    year += 1
                month = tmp - 12
            else:
                month = tmp
        # . day
        day: cython.uint = self._day if self._day > 0 else cydt.access_day(o)
        day = min(day, cydt.days_in_month(year, month))

        # Generate datetime
        dt: datetime.datetime = cydt.gen_dt(
            year,
            month,
            day,
            self._hour if self._hour >= 0 else 0,
            self._minute if self._minute >= 0 else 0,
            self._second if self._second >= 0 else 0,
            self._microsecond if self._microsecond >= 0 else 0,
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
        if self._weekday >= 0:
            dt = cydt.dt_adj_weekday(dt, self._weekday)

        # Return datetime
        return dt

    @cython.cfunc
    @cython.inline(True)
    def _rsub_datetime(self, o: datetime.datetime) -> datetime.datetime:
        """(Internal) Right substraction `__rsub__` with `datetime.datetime`
        and returns `<datetime.datetime>`."""
        # Calculate date
        # . year
        year: cython.uint = self._year if self._year > 0 else cydt.access_dt_year(o)
        year = min(max(year - self._years, 1), 9_999)  # sub relative years
        # . month
        month: cython.uint = self._month if self._month > 0 else cydt.access_dt_month(o)
        if self._months != 0:
            tmp: cython.int = month - self._months  # sub relative months
            if tmp < 1:
                if year > 1:
                    year -= 1
                month = tmp + 12
            elif tmp > 12:
                if year < 9_999:
                    year += 1
                month = tmp - 12
            else:
                month = tmp
        # . day
        day: cython.uint = self._day if self._day > 0 else cydt.access_dt_day(o)
        day = min(day, cydt.days_in_month(year, month))

        # Generate datetime
        # fmt: off
        dt: datetime.datetime = cydt.gen_dt(
            year,
            month,
            day,
            self._hour if self._hour >= 0 else cydt.access_dt_hour(o),
            self._minute if self._minute >= 0 else cydt.access_dt_minute(o),
            self._second if self._second >= 0 else cydt.access_dt_second(o),
            self._microsecond if self._microsecond >= 0 else cydt.access_dt_microsecond(o),
            cydt.access_dt_tzinfo(o),
            cydt.access_dt_fold(o),
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
        if self._weekday >= 0:
            dt = cydt.dt_adj_weekday(dt, self._weekday)

        # Return datetime
        return dt

    @cython.cfunc
    @cython.inline(True)
    def _rsub_timedelta(self, o: datetime.timedelta) -> cytimedelta:
        """(Internal) Right substraction `__rsub__` with `datetime.timedelta`
        and returns `<cytimedelta>`."""
        return cytimedelta(
            years=self._years,
            months=self._months,
            days=cydt.access_delta_days(o) - self._days,
            hours=self._hours,
            minutes=self._minutes,
            seconds=cydt.access_delta_seconds(o) - self._seconds,
            microseconds=cydt.access_delta_microseconds(o) - self._microseconds,
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
        """(Internal) Right substraction `__rsub__` with `relativedelta`
        and returns `<cytimedelta>`."""
        o = o.normalized()
        wday = o.weekday
        return cytimedelta(
            # fmt: off
            years=o.years - self._years,
            months=o.months - self._months,
            days=o.days - self._days,
            hours=o.hours - self._hours,
            minutes=o.minutes - self._minutes,
            seconds=o.seconds - self._seconds,
            microseconds=o.microseconds - self._microseconds,
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

    # Special methods: multiplication ----------------------------------------
    def __mul__(self, o: object) -> cytimedelta:
        if isinstance(o, float):
            return self._mul_float(o)
        if isinstance(o, int):
            return self._mul_int(o)
        try:
            factor = float(o)
        except Exception:
            return NotImplemented
        return self._mul_float(factor)

    def __rmul__(self, o: object) -> cytimedelta:
        if isinstance(o, float):
            return self._mul_float(o)
        if isinstance(o, int):
            return self._mul_int(o)
        try:
            factor = float(o)
        except Exception:
            return NotImplemented
        return self._mul_float(factor)

    @cython.cfunc
    @cython.inline(True)
    def _mul_int(self, factor: cython.int) -> cytimedelta:
        """(Internal) Multiplication `__mul__` with a factor
        and returns `<cytimedelta>`."""
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
        """(Internal) Multiplication `__mul__` with a factor
        and returns `<cytimedelta>`."""
        # fmt: off
        years_f = self._years * factor
        years: cython.longlong = int(years_f)
        months_f = self._months * factor + (years_f - years) * 12
        months: cython.longlong = int(months_f)
        days_f = self._days * factor
        days: cython.longlong = int(days_f)
        hours_f = self._hours * factor + (days_f - days) * 24
        hours: cython.longlong = int(hours_f)
        minutes_f = self._minutes * factor + (hours_f - hours) * 60
        minutes: cython.longlong = int(minutes_f)
        seconds_f = self._seconds * factor + (minutes_f - minutes) * 60
        seconds: cython.longlong = int(seconds_f)
        microseconds_f = self._microseconds * factor + (seconds_f - seconds) * 1_000_000
        microseconds: cython.longlong = int(microseconds_f)
        # fmt: on
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
            reciprocal = 1 / float(o)
        except Exception:
            return NotImplemented
        return self._mul_float(reciprocal)

    # Special methods: negative ----------------------------------------------
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
            years=stdlib.llabs(self._years),
            months=stdlib.llabs(self._months),
            days=stdlib.llabs(self._days),
            hours=stdlib.llabs(self._hours),
            minutes=stdlib.llabs(self._minutes),
            seconds=stdlib.llabs(self._seconds),
            microseconds=stdlib.llabs(self._microseconds),
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
        if isinstance(o, TP_RELATIVEDELTA):
            return self._eq_relativedelta(o)
        return False

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _eq_cytimedelta(self, o: cytimedelta) -> cython.bint:
        """(Internal) Check if equals to a `cytimedelta` `<bool>`."""
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
        """(Internal) Check if equals to a `relativedelta` `<bool>`."""
        o = o.normalized()
        wday = o.weekday
        if wday is None:
            weekday: cython.int = -1
        else:
            if wday.n:
                return False  # exit: can't compare nth weekday
            weekday: cython.int = wday.weekday
        years: cython.longlong = o.years
        months: cython.longlong = o.months
        days: cython.longlong = o.days
        hours: cython.longlong = o.hours
        minutes: cython.longlong = o.minutes
        seconds: cython.longlong = o.seconds
        microseconds: cython.longlong = o.microseconds
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
            self._years
            or self._months
            or self._days
            or self._hours
            or self._minutes
            or self._seconds
            or self._microseconds
            or self._year > 0
            or self._month > 0
            or self._day > 0
            or self._weekday >= 0
            or self._hour >= 0
            or self._minute >= 0
            or self._second >= 0
            or self._microsecond >= 0
        )

    # Special methods: represent ---------------------------------------------
    def __repr__(self) -> str:
        # Representations
        reprs: list = []

        # Relative delta
        if self._years:
            reprs.append("years=%d" % self._years)
        if self._months:
            reprs.append("months=%d" % self._months)
        if self._days:
            reprs.append("days=%d" % self._days)
        if self._hours:
            reprs.append("hours=%d" % self._hours)
        if self._minutes:
            reprs.append("minutes=%d" % self._minutes)
        if self._seconds:
            reprs.append("seconds=%d" % self._seconds)
        if self._microseconds:
            reprs.append("microseconds=%d" % self._microseconds)

        # Absolute delta
        if self._year > 0:
            reprs.append("year=%d" % self._year)
        if self._month > 0:
            reprs.append("month=%d" % self._month)
        if self._day > 0:
            reprs.append("day=%d" % self._day)
        if self._weekday >= 0:
            reprs.append("weekday=%s" % WEEKDAY_REPRS[self._weekday])
        if self._hour >= 0:
            reprs.append("hour=%d" % self._hour)
        if self._minute >= 0:
            reprs.append("minute=%d" % self._minute)
        if self._second >= 0:
            reprs.append("second=%d" % self._second)
        if self._microsecond >= 0:
            reprs.append("microsecond=%d" % self._microsecond)

        # Construct
        return "<%s (%s)>" % (self.__class__.__name__, ", ".join(reprs))

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
