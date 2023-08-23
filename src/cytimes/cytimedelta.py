# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from __future__ import annotations

# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cytimes import cymath  # type: ignore
from cython.cimports.cytimes import cydatetime as cydt  # type: ignore

np.import_array()
datetime.import_datetime()

# Python imports
from typing import Union
import datetime, numpy as np
from dateutil.relativedelta import relativedelta
from dateutil.relativedelta import weekday as relweekday
from cytimes import cymath, cydatetime as cydt

__all__ = ["Weekday", "cytimedelta"]


# weekday -------------------------------------------------------------------------------------
@cython.cclass
class Weekday:
    _weekcode: str
    _weekday: cython.int
    _week_offset: cython.int

    def __init__(self, weekday: cython.int, week_offset: cython.int = 0):
        """The Weekday class represents a day of the week.
        Cythonized version of `dateutil.relativedelta.weekday` with slight modification.

        :param weekday: `<int>` The weekday number, accepts integer from -1 to 6, where 0 is Monday, 6 is Sunday. -1 represents `None`.
        :param week_offset: `<int>` The week offset, defaults to 0. If set to x, it means the final delta will be 7x days offset from the weekday.
            This parameter is slightly different from `dateutil.relativedelta.weekday`, where `n` between -1 to 1 takes no effect
            and equals to `week_offset = 0`. When `n` is greater than 1, (let's say `n` is 2), it means the final delta will be 7x days
            offset and equals to `week_offset = 1`. When `n` is less than -1, (let's say `n` is -2), it means the final delta will be -7x days
            offset and equals to `week_offset = -1`.

        ### Comparison
        >>> from cytimes import Weekday
            from dateutil.relativedelta import weekday
            # The following examples compares the differences between `cytimes.Weekday` (left)
            # and `dateutil.relativedelta.weekday` (right). If yeilds `True`, it means they will
            # have the same impact on delta calculation.

        >>> Weekday(0) == weekday(0)
            # True: MO == MO
        >>> Weekday(0) == weekday(0, 1)
            # True: MO == MO(+1)
        >>> Weekday(0, 1) == weekday(0, 2)
            # True: MO(+1) == MO(+2)
        >>> Weekday(0, 2) == weekday(0, 3)
            # True: MO(+2) == MO(+3)
        >>> Weekday(0, 3) == weekday(0, 3)
            # False: MO(+3) != MO(+3)
        >>> Weekday(0) == weekday(0, -1)
            # True: MO == MO(-1)
        >>> Weekday(0, -1) == weekday(0, -2)
            # True: MO(-1) == MO(-2)
        >>> Weekday(0, -2) == weekday(0, -3)
            # True: MO(-2) == MO(-3)
        >>> Weekday(0, -3) == weekday(0, -3)
            # False: MO(-3) != MO(-3)
        """

        if weekday == -1:
            self._weekcode = "NULL"
        elif weekday == 0:
            self._weekcode = "MO"
        elif weekday == 1:
            self._weekcode = "TU"
        elif weekday == 2:
            self._weekcode = "WE"
        elif weekday == 3:
            self._weekcode = "TH"
        elif weekday == 4:
            self._weekcode = "FR"
        elif weekday == 5:
            self._weekcode = "SA"
        elif weekday == 6:
            self._weekcode = "SU"
        else:
            raise ValueError(
                "<Weekday> Invalid weekday: %d. Only accept integer from -1 to 6, where 0 is Monday, 6 is Sunday and -1 represents `None`."
                % weekday
            )
        self._weekday = weekday
        self._week_offset = week_offset

    # Information
    @property
    def weekcode(self) -> str:
        """The weekday code, e.g: `'MO'`, `'TU'`..."""

        return self._weekcode

    @property
    def weekday(self) -> int:
        """The weekday number, where 0 is Monday, 6 is Sunday. -1 represents `None`."""

        return self._weekday

    @property
    def week_offset(self) -> int:
        """The week offset, 7x days offset from the weekday."""

        return self._week_offset

    # Special methods
    def __call__(self, week_offset: cython.int) -> Weekday:
        if week_offset == self._week_offset:
            return self
        else:
            return Weekday(self._weekday, week_offset)

    def __repr__(self) -> str:
        return "<Weekday (weekcode=%s, weekday=%d, week_offset=%d)>" % (
            self._weekcode,
            self._weekday,
            self._week_offset,
        )

    def __str__(self) -> str:
        if self._week_offset == 0:
            return self._weekcode
        else:
            return "%s(%+d)" % (self._weekcode, self._week_offset)

    def __hash__(self) -> cython.int:
        return hash((self._weekday, self._week_offset))

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _equal_weekday(self, other: Weekday) -> cython.bint:
        return (
            self._weekday == other._weekday and self._week_offset == other._week_offset
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _equal_relweekday(self, other: object) -> cython.bint:
        try:
            if self._weekday != other.weekday:
                return False
            if other.n:
                weekday_n: cython.int = other.n
                if weekday_n > 1:
                    return self._week_offset == weekday_n - 1
                if weekday_n < -1:
                    return self._week_offset == weekday_n + 1
            return self._week_offset == 0

        except Exception:
            return False

    def __eq__(self, other: object) -> cython.bint:
        if isinstance(other, Weekday):
            return self._equal_weekday(other)
        elif isinstance(other, relweekday):
            return self._equal_relweekday(other)
        else:
            return False

    def __bool__(self) -> cython.bint:
        return self._weekday != -1


WEEKDAY_NULL: Weekday = Weekday(-1, 0)


# cytimedelta ---------------------------------------------------------------------------------
@cython.cclass
class cytimedelta:
    _years: cython.int
    _months: cython.int
    _days: cython.int
    _hours: cython.longlong
    _minutes: cython.longlong
    _seconds: cython.longlong
    _microseconds: cython.longlong
    _leapdays: cython.int
    _year: cython.int
    _month: cython.int
    _day: cython.int
    _weekday: Weekday
    _hour: cython.int
    _minute: cython.int
    _second: cython.int
    _microsecond: cython.int

    def __init__(
        self,
        years: cython.int = 0,
        months: cython.int = 0,
        days: cython.int = 0,
        weeks: cython.int = 0,
        hours: cython.int = 0,
        minutes: cython.longlong = 0,
        seconds: cython.longlong = 0,
        microseconds: cython.longlong = 0,
        leapdays: cython.int = 0,
        year: cython.int = -1,
        month: cython.int = -1,
        day: cython.int = -1,
        weekday: Union[cython.int, Weekday, relweekday] = -1,
        yearday: cython.int = -1,
        nlyearday: cython.int = -1,
        hour: cython.int = -1,
        minute: cython.int = -1,
        second: cython.int = -1,
        microsecond: cython.int = -1,
    ):
        """Cythonized version of `dateutil.relativedelta.relativedelta` with slight modification.

        `cytimedelta` removes the input of `dt1` and `dt2` and only accepts relative & absolute
        time information. In most cases, `cytimedelta` and `dateutil.relativedelta.relativedelta`
        are interchangeable.

        #### Absolute information
        :param year: `<int>` The absolute year, -1 means `IGNORE`.
        :param month: `<int>` The absolute month, -1 means `IGNORE`.
        :param day: `<int>` The absolute day, -1 means `IGNORE`.
        :param weekday: `<int>` / `<Weekday>` / `<relativedelta.weekday>` The absolute weekday, -1 means `IGNORE`.
            Accepts: 1. interger (from -1 to 6, where 0 is Monday, 6 is Sunday).
            2. `<cytimes.Weekday>`. 3. `<relativedelta.relativedelta.weekday>`
        :param yearday: `<int>` The absolute year day, -1 means `IGNORE`.
            Month, day and leapdays will be re-calculated based on this information.
        :param nlyearday: `<int>` The absolute non-leap year day, -1 means `IGNORE`.
            Month, day and leapdays will be re-calculated based on this information.
            If `yearday` is also given, this information will be ignored.
        :param hour: `<int>` The absolute hour, -1 means `IGNORE`.
        :param minute: `<int>` The absolute minute, -1 means `IGNORE`.
        :param second: `<int>` The absolute second, -1 means `IGNORE`.
        :param microsecond: `<int>` The absolute microsecond, -1 means `IGNORE`.

        - Addition with `datetime.date`, `datetime.datetime` and `pandas.Timestamp`
        supports both left and right operand. The date / datetime / timestamp's
        absolute information will be replaced by delta's absolute information and
        return a new `datetime.datetime`.

        - Addition with `cytimedelta`, `dateutil.relativedelta.relativedelta`,
        `datetime.timedelta` and `pandas.Timedelta` supports both left and right
        operand. The right operand's absolute information will be kept and return
        a new `cytimedelta`.

        - Subtraction with `datetime.date`, `datetime.datetime` and `pandas.Timestamp`
        only supports right operand. The date / datetime / timestamp's absolute
        information will be replaced by delta's absolute information and return
        a new `datetime.datetime`.

        - Subtraction with `cytimedelta`, `dateutil.relativedelta.relativedelta`,
        `datetime.timedelta` and `pandas.Timedelta` supports both left and right
        operand. The left operand's absolute information will be kept and return
        a new `cytimedelta`.

        - Both multiplication and division take no effect on absolute information.

        #### Relative information
        :param years: `<int>` The relative number of years.
        :param months: `<int>` The relative number of months.
        :param days: `<int>` The relative number of days.
        :param weeks: `<int>` The relative number of weeks.
        :param hours: `<int>` The relative number of hours.
        :param minutes: `<int>` The relative number of minutes.
        :param seconds: `<int>` The relative number of seconds.
        :param microseconds: `<int>` The relative number of microseconds.
        :param leapdays: `<int>` The relative number of leapdays.

        - Addition with `datetime.date`, `datetime.datetime` and `pandas.Timestamp`
        supports both left and right operand. The relative information will be
        added to the date / datetime / timestamp's absolute information and
        return a new `datetime.datetime`.

        - Addition with `cytimedelta`, `dateutil.relativedelta.relativedelta`,
        `datetime.timedelta` and `pandas.Timedelta` supports both left and right
        operand. All the relative information except 'leapdays' will be added together
        and return a new `cytimedelta`.

        - Subtraction with `datetime.date`, `datetime.datetime` and `pandas.Timestamp`
        only supports right operand. The relative information will be subtracted
        from the date / datetime / timestamp's absolute information and return
        a new `datetime.datetime`.

        - Subtraction with `cytimedelta`, `dateutil.relativedelta.relativedelta`,
        `datetime.timedelta` and `pandas.Timedelta` supports both left and right
        operand. All the relative information except 'leapdays' will be subtracted
        from the left operand's relative information and return a new `cytimedelta`.

        - Both multiplication and division will round the factor to the nearest
        integer before calculation. For example, `cytimedelta(days=1) * 1.5` will
        be rounded to `cytimedelta(days=2)`. Except 'leapdays', all the relative
        information will be multiplied / divided by the factor and return a new
        `cytimedelta`.

        #### Mix of relative and absolute information
        When both relative and absolute information are given, the absolute information
        will first be used to replace `datetime.date`, `datetime.datetime` and
        `pandas.Timestamp`'s absolute information, then the relative information will
        be added / subtracted to / from that absolute information. A new `datetime.datetime`
        will be returned.
        """

        # Relative information
        self._years = years
        self._months = months
        self._days = days + weeks * 7
        self._hours = hours
        self._minutes = minutes
        self._seconds = seconds
        self._microseconds = microseconds
        self._leapdays = leapdays

        # Absolute information
        self._year = year
        self._month = month
        self._day = day
        if weekday == -1 or weekday is None:
            self._weekday = WEEKDAY_NULL
        elif isinstance(weekday, int):
            self._weekday = Weekday(weekday)
        elif isinstance(weekday, Weekday):
            self._weekday = weekday
        else:
            self._weekday = self._convert_relweekday(weekday)
        self._hour = hour
        self._minute = minute
        self._second = second
        self._microsecond = microsecond

        # Adjust absolute information
        self._adjust_absolute(yearday, nlyearday)

        # Adjust relative information
        self._adjust_relative()

    # Convert relativedelta.weekday
    @cython.cfunc
    @cython.inline(True)
    def _convert_relweekday(self, weekday: object) -> Weekday:
        """Convert `relativedelta.weekday` to `cytimes.Weekday`."""

        try:
            wd: cython.int = weekday.weekday
            wn: cython.int
            if weekday.n is None:
                wn = 0
            else:
                weekday_n: cython.int = weekday.n
                if weekday_n > 1:
                    wn = weekday_n - 1
                elif weekday_n < -1:
                    wn = weekday_n + 1
                else:
                    wn = 0
            return Weekday(wd, wn)
        except Exception as err:
            raise ValueError(
                "<cytimedelta> Invalid 'weekday', accepts <int>, <Weekday> or <relativedelta.weekday>. Instead got: %s %s"
                % (type(weekday), repr(weekday))
            ) from err

    # Adjustments
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _adjust_absolute(self, yearday: cython.int, nlyearday: cython.int):
        """Adjust absolute information."""

        # Validate year
        if self._year != -1:
            self._year = cymath.clip(self._year, 1, 9999)

        # Validate month
        if self._month != -1:
            self._month = cymath.clip(self._month, 1, 12)

        # Validate day
        if self._day != -1:
            self._day = cymath.clip(self._day, 1, 31)

        # Validate hour
        if self._hour != -1:
            self._hour = cymath.clip(self._hour, 0, 23)

        # Validate minute
        if self._minute != -1:
            self._minute = cymath.clip(self._minute, 0, 59)

        # Validate second
        if self._second != -1:
            self._second = cymath.clip(self._second, 0, 59)

        # Validate microsecond
        if self._microsecond != -1:
            self._microsecond = cymath.clip(self._microsecond, 0, 999999)

        # Adjust yearday
        yday: cython.int = 0

        if nlyearday != -1:
            yday = nlyearday
        elif yearday != -1:
            yday = yearday
            if yearday > 59:
                self._leapdays = -1

        if yday > 0:
            i: cython.int
            for i in range(1, 13):
                if yday <= cydt.DAYS_BR_MONTH[i]:
                    self._month = i
                    self._day = yday - cydt.DAYS_BR_MONTH[i - 1]
                    break
            else:
                self._month = 12
                self._day = 31

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _adjust_relative(self):
        """Adjust relative information."""

        sign: cython.int
        qoutient: cython.longlong
        remainder: cython.longlong

        # Adjust microseconds
        if cymath.abs_ll(self._microseconds) > 999999:
            sign = cymath.signfactor_l(self._microseconds)
            qoutient = sign * self._microseconds // 1000000
            remainder = sign * self._microseconds % 1000000
            self._microseconds = remainder * sign
            self._seconds += qoutient * sign

        # Adjust seconds
        if cymath.abs_ll(self._seconds) > 59:
            sign = cymath.signfactor_l(self._seconds)
            qoutient = sign * self._seconds // 60
            remainder = sign * self._seconds % 60
            self._seconds = remainder * sign
            self._minutes += qoutient * sign

        # Adjust minutes
        if cymath.abs_ll(self._minutes) > 59:
            sign = cymath.signfactor_l(self._minutes)
            qoutient = sign * self._minutes // 60
            remainder = sign * self._minutes % 60
            self._minutes = remainder * sign
            self._hours += qoutient * sign

        # Adjust hours
        if cymath.abs_ll(self._hours) > 23:
            sign = cymath.signfactor_l(self._hours)
            qoutient = sign * self._hours // 24
            remainder = sign * self._hours % 24
            self._hours = remainder * sign
            self._days = cymath.clip(self._days + qoutient * sign, -99999999, 99999999)

        # Adjust months
        if cymath.abs(self._months) > 11:
            sign = cymath.signfactor(self._months)
            qoutient = sign * self._months // 12
            remainder = sign * self._months % 12
            self._months = remainder * sign
            self._years = cymath.clip(self._years + qoutient * sign, -9999, 9999)

    # Relative information
    @property
    def years(self) -> int:
        """The relative number of years."""

        return self._years

    @property
    def months(self) -> int:
        """The relative number of months."""

        return self._months

    @property
    def days(self) -> int:
        """The relative number of days."""

        return self._days

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _cal_weeks(self) -> cython.int:
        return int(self._days / 7)

    @property
    def weeks(self) -> int:
        """The relative number of weeks."""

        return self._cal_weeks()

    @property
    def hours(self) -> int:
        """The relative number of hours."""

        return self._hours

    @property
    def minutes(self) -> int:
        """The relative number of minutes."""

        return self._minutes

    @property
    def seconds(self) -> int:
        """The relative number of seconds."""

        return self._seconds

    @property
    def microseconds(self) -> int:
        """The relative number of microseconds."""

        return self._microseconds

    @property
    def leapdays(self) -> int:
        """The relative number of leapdays."""

        return self._leapdays

    # Absolute information
    @property
    def year(self) -> int:
        """The absolute year."""

        return self._year

    @property
    def month(self) -> int:
        """The absolute month."""

        return self._month

    @property
    def day(self) -> int:
        """The absolute day."""

        return self._day

    @property
    def weekday(self) -> Weekday:
        """The absolute weekday."""

        return self._weekday

    @property
    def hour(self) -> int:
        """The absolute hour."""

        return self._hour

    @property
    def minute(self) -> int:
        """The absolute minute."""

        return self._minute

    @property
    def second(self) -> int:
        """The absolute second."""

        return self._second

    @property
    def microsecond(self) -> int:
        """The absolute microsecond."""

        return self._microsecond

    # Special methods - addition
    @cython.cfunc
    @cython.inline(True)
    def _pref_relativedelta_weekday(self, other: object) -> Weekday:
        """Get weekday from relativedelta if possible."""
        if not other.weekday:
            return self._weekday
        else:
            return self._convert_relweekday(other.weekday)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _add_date_time(self, other: datetime.date) -> datetime.datetime:
        """cytimedelta + datetime.date / datetime.datetime / pandas.Timestamp"""

        # Year
        year: cython.int
        if self._year != -1:
            year = self._year + self._years
        else:
            year = cydt.get_year(other) + self._years

        # Month
        month: cython.int
        if self._month != -1:
            month = self._month
        else:
            month = cydt.get_month(other)
        if self._months:
            month += self._months
            if month > 12:
                year += 1
                month -= 12
            elif month < 1:
                year -= 1
                month += 12
        year = cymath.clip(year, 1, 9999)

        # Day
        day: cython.int = self._day if self._day != -1 else cydt.get_day(other)
        day = min(day, cydt.days_in_month(year, month))

        # Get hour, minute, second & mircoseconds
        hour: cython.int
        minute: cython.int
        second: cython.int
        microsecond: cython.int
        tzinfo: object
        if cydt.is_dt(other):
            hour = self._hour if self._hour != -1 else cydt.get_dt_hour(other)
            minute = self._minute if self._minute != -1 else cydt.get_dt_minute(other)
            second = self._second if self._second != -1 else cydt.get_dt_second(other)
            microsecond = (
                self._microsecond
                if self._microsecond != -1
                else cydt.get_dt_microsecond(other)
            )
            tzinfo = cydt.get_dt_tzinfo(other)
        else:
            hour = self._hour if self._hour != -1 else 0
            minute = self._minute if self._minute != -1 else 0
            second = self._second if self._second != -1 else 0
            microsecond = self._microsecond if self._microsecond != -1 else 0
            tzinfo = None

        # Create datetime
        dt: datetime.datetime = cydt.gen_dt(
            year, month, day, hour, minute, second, microsecond, tzinfo, 0
        )

        # Adjust delta
        dt = cydt.dt_add(
            dt,
            self._days + self._leapdays
            if self._leapdays and month > 2 and cydt.is_leapyear(year)
            else self._days,
            self._hours * 3600 + self._minutes * 60 + self._seconds,
            self._microseconds,
        )

        # Adjust weekday
        if self._weekday._weekday != -1:
            dt_weekday: cython.int = cydt.get_weekday(dt)
            weekday: cython.int = self._weekday._weekday
            week_offset: cython.int = self._weekday._week_offset
            offset_days: cython.int = cymath.abs(week_offset) * 7
            if week_offset >= 0:
                offset_days = offset_days + (7 - dt_weekday + weekday) % 7
            else:
                offset_days = -(offset_days + (dt_weekday - weekday) % 7)
            dt = cydt.dt_add(dt, offset_days, 0, 0)

        # Return
        return dt

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _add_cytimedelta(self, other: cytimedelta) -> cytimedelta:
        """cytimedelta + cytimedelta"""

        return cytimedelta(
            years=other._years + self._years,
            months=other._months + self._months,
            days=other._days + self._days,
            hours=other._hours + self._hours,
            minutes=other._minutes + self._minutes,
            seconds=other._seconds + self._seconds,
            microseconds=other._microseconds + self._microseconds,
            leapdays=other._leapdays or self._leapdays,
            year=other._year if other._year != -1 else self._year,
            month=other._month if other._month != -1 else self._month,
            day=other._day if other._day != -1 else self._day,
            weekday=other._weekday if other._weekday._weekday != -1 else self._weekday,
            hour=other._hour if other._hour != -1 else self._hour,
            minute=other._minute if other._minute != -1 else self._minute,
            second=other._second if other._second != -1 else self._second,
            microsecond=(
                other._microsecond if other._microsecond != -1 else self._microsecond
            ),
        )

    @cython.cfunc
    @cython.inline(True)
    def _add_relativedelta(self, other: object) -> cytimedelta:
        """cytimedelta + relativedelta"""

        other = other.normalized()
        return cytimedelta(
            years=other.years + self._years,
            months=other.months + self._months,
            days=other.days + self._days,
            hours=other.hours + self._hours,
            minutes=other.minutes + self._minutes,
            seconds=other.seconds + self._seconds,
            microseconds=other.microseconds + self._microseconds,
            leapdays=other.leapdays or self._leapdays,
            year=other.year if other.year is not None else self._year,
            month=other.month if other.month is not None else self._month,
            day=other.day if other.day is not None else self._day,
            weekday=self._pref_relativedelta_weekday(other),
            hour=other.hour if other.hour is not None else self._hour,
            minute=other.minute if other.minute is not None else self._minute,
            second=other.second if other.second is not None else self._second,
            microsecond=other.microsecond
            if other.microsecond is not None
            else self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _add_timedelta(self, other: datetime.timedelta) -> cytimedelta:
        """cytimedelta + datetime.timedelta"""

        return cytimedelta(
            years=self._years,
            months=self._months,
            days=cydt.get_delta_days(other) + self._days,
            hours=self._hours,
            minutes=self._minutes,
            seconds=cydt.get_delta_seconds(other) + self._seconds,
            microseconds=cydt.get_delta_microseconds(other) + self._microseconds,
            leapdays=self._leapdays,
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    def __add__(self, other: object) -> Union[cytimedelta, datetime.datetime]:
        if cydt.is_date(other):
            return self._add_date_time(other)
        if isinstance(other, cytimedelta):
            return self._add_cytimedelta(other)
        if cydt.is_delta(other):
            return self._add_timedelta(other)
        if isinstance(other, relativedelta):
            return self._add_relativedelta(other)
        if cydt.is_dt64(other):
            return self._add_date_time(cydt.dt64_to_dt(other))
        if cydt.is_delta64(other):
            return self._add_timedelta(cydt.delta64_to_delta(other))

        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    def _radd_relativedelta(self, other: object) -> cytimedelta:
        """relativedelta + cytimedelta.

        This function is ncessary. Otherwise, `relativedelta + cytimedelta` will
        keep the relativedelta's absolute information instead of the cytimedelta's.
        """

        other = other.normalized()
        return cytimedelta(
            years=self._years + other.years,
            months=self._months + other.months,
            days=self._days + other.days,
            hours=self._hours + other.hours,
            minutes=self._minutes + other.minutes,
            seconds=self._seconds + other.seconds,
            microseconds=self._microseconds + other.microseconds,
            leapdays=self._leapdays or other.leapdays,
            year=self._year if self._year != -1 else (other.year or -1),
            month=self._month if self._month != -1 else (other.month or -1),
            day=self._day if self._day != -1 else (other.day or -1),
            weekday=self._pref_cytimedelta_weekday(other),
            hour=self._hour if self._hour != -1 else (other.hour or -1),
            minute=self._minute if self._minute != -1 else (other.minute or -1),
            second=self._second if self._second != -1 else (other.second or -1),
            microsecond=self._microsecond
            if self._microsecond != -1
            else (other.microsecond or -1),
        )

    def __radd__(self, other: object) -> Union[cytimedelta, datetime.datetime]:
        if cydt.is_date(other):
            return self._add_date_time(other)
        if cydt.is_delta(other):
            return self._add_timedelta(other)
        if isinstance(other, relativedelta):
            return self._radd_relativedelta(other)
        # TODO: This has no effect since numpy does not return NotImplemented
        if cydt.is_dt64(other):
            return self._add_date_time(cydt.dt64_to_dt(other))
        if cydt.is_delta64(other):
            return self._add_timedelta(cydt.delta64_to_delta(other))

        return NotImplemented

    # Special methods - substraction
    @cython.cfunc
    @cython.inline(True)
    def _pref_cytimedelta_weekday(self, other: object) -> Weekday:
        """Get weekday from cytimedelta if possible."""

        if self._weekday._weekday != -1 or other.weekday is None:
            return self._weekday
        else:
            return self._convert_relweekday(other.weekday)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _sub_cytimedelta(self, other: cytimedelta) -> cytimedelta:
        """cytimedelta - cytimedelta"""

        return cytimedelta(
            years=self._years - other._years,
            months=self._months - other._months,
            days=self._days - other._days,
            hours=self._hours - other._hours,
            minutes=self._minutes - other._minutes,
            seconds=self._seconds - other._seconds,
            microseconds=self._microseconds - other._microseconds,
            leapdays=self._leapdays or other._leapdays,
            year=self._year if self._year != -1 else other._year,
            month=self._month if self._month != -1 else other._month,
            day=self._day if self._day != -1 else other._day,
            weekday=self._weekday if self._weekday._weekday != -1 else other._weekday,
            hour=self._hour if self._hour != -1 else other._hour,
            minute=self._minute if self._minute != -1 else other._minute,
            second=self._second if self._second != -1 else other._second,
            microsecond=(
                self._microsecond if self._microsecond != -1 else other._microsecond
            ),
        )

    @cython.cfunc
    @cython.inline(True)
    def _sub_relativedelta(self, other: object) -> cytimedelta:
        """cytimedelta - relativedelta"""

        other = other.normalized()
        return cytimedelta(
            years=self._years - other.years,
            months=self._months - other.months,
            days=self._days - other.days,
            hours=self._hours - other.hours,
            minutes=self._minutes - other.minutes,
            seconds=self._seconds - other.seconds,
            microseconds=self._microseconds - other.microseconds,
            leapdays=self._leapdays or other.leapdays,
            year=self._year if self._year != -1 else (other.year or -1),
            month=self._month if self._month != -1 else (other.month or -1),
            day=self._day if self._day != -1 else (other.day or -1),
            weekday=self._pref_cytimedelta_weekday(other),
            hour=self._hour if self._hour != -1 else (other.hour or -1),
            minute=self._minute if self._minute != -1 else (other.minute or -1),
            second=self._second if self._second != -1 else (other.second or -1),
            microsecond=(
                self._microsecond
                if self._microsecond != -1
                else (other.microsecond or -1)
            ),
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _sub_timedelta(self, other: datetime.timedelta) -> cytimedelta:
        """cytimedelta - datetime.timedelta"""

        return cytimedelta(
            years=self._years,
            months=self._months,
            days=self._days - cydt.get_delta_days(other),
            hours=self._hours,
            minutes=self._minutes,
            seconds=self._seconds - cydt.get_delta_seconds(other),
            microseconds=self._microseconds - cydt.get_delta_microseconds(other),
            leapdays=self._leapdays,
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    def __sub__(self, other: object) -> cytimedelta:
        if isinstance(other, cytimedelta):
            return self._sub_cytimedelta(other)
        if cydt.is_delta(other):
            return self._sub_timedelta(other)
        if isinstance(other, relativedelta):
            return self._sub_relativedelta(other)
        if cydt.is_delta64(other):
            return self._sub_timedelta(cydt.delta64_to_delta(other))

        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _rsub_date_time(self, other: datetime.date) -> datetime.datetime:
        """datetime.date / datetime.datetime / pandas.Timestamp - cytimedelta"""

        # Year
        year: cython.int
        if self._year != -1:
            year = self._year - self._years
        else:
            year = cydt.get_year(other) - self._years

        # Month
        month: cython.int
        if self._month != -1:
            month = self._month
        else:
            month = cydt.get_month(other)
        if self._months:
            month -= self._months
            if month < 1:
                year -= 1
                month += 12
            elif month > 12:
                year += 1
                month -= 12
        year = cymath.clip(year, 1, 9999)

        # Day
        day: cython.int = self._day if self._day != -1 else cydt.get_day(other)
        day = min(day, cydt.days_in_month(year, month))

        # Get hour, minute, second & mircoseconds
        hour: cython.int
        minute: cython.int
        second: cython.int
        microsecond: cython.int
        tzinfo: object
        if cydt.is_dt(other):
            hour = self._hour if self._hour != -1 else cydt.get_dt_hour(other)
            minute = self._minute if self._minute != -1 else cydt.get_dt_minute(other)
            second = self._second if self._second != -1 else cydt.get_dt_second(other)
            microsecond = (
                self._microsecond
                if self._microsecond != -1
                else cydt.get_dt_microsecond(other)
            )
            tzinfo = cydt.get_dt_tzinfo(other)
        else:
            hour = self._hour if self._hour != -1 else 0
            minute = self._minute if self._minute != -1 else 0
            second = self._second if self._second != -1 else 0
            microsecond = self._microsecond if self._microsecond != -1 else 0
            tzinfo = None

        # Create datetime
        dt: datetime.datetime = cydt.gen_dt(
            year, month, day, hour, minute, second, microsecond, tzinfo, 0
        )

        # Adjust delta
        dt = cydt.dt_add(
            dt,
            -self._days + self._leapdays
            if self._leapdays and month > 2 and cydt.is_leapyear(year)
            else -self._days,
            -self._hours * 3600 + -self._minutes * 60 + -self._seconds,
            -self._microseconds,
        )

        # Adjust weekday
        if self._weekday._weekday != -1:
            dt_weekday: cython.int = cydt.get_weekday(dt)
            weekday: cython.int = self._weekday._weekday
            week_offset: cython.int = self._weekday._week_offset
            offset_days: cython.int = cymath.abs(week_offset) * 7
            if week_offset >= 0:
                offset_days = offset_days + (7 - dt_weekday + weekday) % 7
            else:
                offset_days = -(offset_days + (dt_weekday - weekday) % 7)
            dt = cydt.dt_add(dt, offset_days, 0, 0)

        # Return
        return dt

    @cython.cfunc
    @cython.inline(True)
    def _rsub_relativedelta(self, other: object) -> cytimedelta:
        """relativedelta - cytimedelta"""

        other = other.normalized()
        return cytimedelta(
            years=other.years - self._years,
            months=other.months - self._months,
            days=other.days - self._days,
            hours=other.hours - self._hours,
            minutes=other.minutes - self._minutes,
            seconds=other.seconds - self._seconds,
            microseconds=other.microseconds - self._microseconds,
            leapdays=other.leapdays or self._leapdays,
            year=other.year if other.year is not None else self._year,
            month=other.month if other.month is not None else self._month,
            day=other.day if other.day is not None else self._day,
            weekday=self._pref_relativedelta_weekday(other),
            hour=other.hour if other.hour is not None else self._hour,
            minute=other.minute if other.minute is not None else self._minute,
            second=other.second if other.second is not None else self._second,
            microsecond=other.microsecond
            if other.microsecond is not None
            else self._microsecond,
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _rsub_timedelta(self, other: datetime.timedelta) -> cytimedelta:
        """datetime.timedelta - cytimedelta"""

        return cytimedelta(
            years=-self._years,
            months=-self._months,
            days=cydt.get_delta_days(other) - self._days,
            hours=-self._hours,
            minutes=-self._minutes,
            seconds=cydt.get_delta_seconds(other) - self._seconds,
            microseconds=cydt.get_delta_microseconds(other) - self._microseconds,
            leapdays=self._leapdays,
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    def __rsub__(self, other: object) -> Union[cytimedelta, datetime.timedelta]:
        if cydt.is_date(other):
            return self._rsub_date_time(other)
        if cydt.is_delta(other):
            return self._rsub_timedelta(other)
        if isinstance(other, relativedelta):
            return self._rsub_relativedelta(other)
        # TODO: This has no effect since numpy does not return NotImplemented
        if cydt.is_dt64(other):
            return self._rsub_date_time(cydt.dt64_to_dt(other))
        if cydt.is_delta64(other):
            return self._rsub_timedelta(cydt.delta64_to_delta(other))

        return NotImplemented

    # Special methods - multiplication
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _multiply(self, factor: cython.double) -> cytimedelta:
        """Multiply the cytimedelta by a factor."""

        return cytimedelta(
            years=int(self._years * factor),
            months=int(self._months * factor),
            days=int(self._days * factor),
            hours=int(self._hours * factor),
            minutes=int(self._minutes * factor),
            seconds=int(self._seconds * factor),
            microseconds=int(self._microseconds * factor),
            leapdays=self._leapdays,
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    def __mul__(self, other: object) -> cytimedelta:
        try:
            factor = float(other)
        except Exception:
            return NotImplemented
        else:
            return self._multiply(factor)

    def __rmul__(self, other: object) -> cytimedelta:
        try:
            factor = float(other)
        except Exception:
            return NotImplemented
        else:
            return self._multiply(factor)

    # Special methods - division
    def __truediv__(self, other: object) -> cytimedelta:
        try:
            reciprocal = 1 / float(other)
        except Exception:
            return NotImplemented
        else:
            return self._multiply(reciprocal)

    # Special methods - manipulation
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _negate(self) -> cytimedelta:
        """Negate the cytimedelta (only relative information, except leapdays)."""

        return cytimedelta(
            years=-self._years,
            months=-self._months,
            days=-self._days,
            hours=-self._hours,
            minutes=-self._minutes,
            seconds=-self._seconds,
            microseconds=-self._microseconds,
            leapdays=self._leapdays,
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    def __neg__(self) -> cytimedelta:
        return self._negate()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _absolute(self) -> cytimedelta:
        """Absolute the cytimedelta (only relative information, except leapdays)."""

        return cytimedelta(
            years=cymath.abs(self._years),
            months=cymath.abs(self._months),
            days=cymath.abs(self._days),
            hours=cymath.abs_ll(self._hours),
            minutes=cymath.abs_ll(self._minutes),
            seconds=cymath.abs_ll(self._seconds),
            microseconds=cymath.abs_ll(self._microseconds),
            leapdays=self._leapdays,
            year=self._year,
            month=self._month,
            day=self._day,
            weekday=self._weekday,
            hour=self._hour,
            minute=self._minute,
            second=self._second,
            microsecond=self._microsecond,
        )

    def __abs__(self) -> cytimedelta:
        return self._absolute()

    # Special methods - comparison
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _equal_cytimedelta(self, other: cytimedelta) -> cython.bint:
        """Check if cytimedelta == cytimedelta."""

        return (
            self._years == other._years
            and self._months == other._months
            and self._days == other._days
            and self._hours == other._hours
            and self._minutes == other._minutes
            and self._seconds == other._seconds
            and self._microseconds == other._microseconds
            and self._leapdays == other._leapdays
            and self._year == other._year
            and self._month == other._month
            and self._day == other._day
            and self._weekday._weekday == other._weekday._weekday
            and self._weekday._week_offset == other._weekday._week_offset
            and self._hour == other._hour
            and self._minute == other._minute
            and self._second == other._second
            and self._microsecond == other._microsecond
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _equal_relativedelta(self, other: object) -> cython.bint:
        """Check if cytimedelta == relativedelta."""

        try:
            return (
                self._years == other.years
                and self._months == other.months
                and self._days == other.days
                and self._hours == other.hours
                and self._minutes == other.minutes
                and self._seconds == other.seconds
                and self._microseconds == other.microseconds
                and self._leapdays == other.leapdays
                and self._year == (other.year or -1)
                and self._month == (other.month or -1)
                and self._day == (other.day or -1)
                and self._weekday._weekday == -1
                if other.weekday is None
                else self._weekday == other.weekday
                and self._hour == (other.hour or -1)
                and self._minute == (other.minute or -1)
                and self._second == (other.second or -1)
                and self._microsecond == (other.microsecond or -1)
            )

        except Exception:
            return False

    def __eq__(self, other: object) -> cython.bint:
        if isinstance(other, cytimedelta):
            return self._equal_cytimedelta(other)
        elif isinstance(other, relativedelta):
            return self._equal_relativedelta(other)
        else:
            return False

    def __bool__(self) -> cython.bint:
        return (
            self._years
            or self._months
            or self._days
            or self._hours
            or self._minutes
            or self._seconds
            or self._microseconds
            or self._leapdays
            or self._year != -1
            or self._month != -1
            or self._day != -1
            or self._weekday._weekday != -1
            or self._hour != -1
            or self._minute != -1
            or self._second != -1
            or self._microsecond != -1
        )

    # Special methods - represent
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _represent(self) -> str:
        reprs: list[str] = []
        # Relative
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
        if self._leapdays:
            reprs.append("leapdays=%d" % self._leapdays)

        # Absolute
        if self._year != -1:
            reprs.append("year=%d" % self._year)
        if self._month != -1:
            reprs.append("month=%d" % self._month)
        if self._day != -1:
            reprs.append("day=%d" % self._day)
        if self._weekday._weekday != -1:
            reprs.append("weekday=%s" % self._weekday)
        if self._hour != -1:
            reprs.append("hour=%d" % self._hour)
        if self._minute != -1:
            reprs.append("minute=%d" % self._minute)
        if self._second != -1:
            reprs.append("second=%d" % self._second)
        if self._microsecond != -1:
            reprs.append("microsecond=%d" % self._microsecond)

        # Return
        return ", ".join(reprs)

    def __repr__(self) -> str:
        return "<cytimedelta (%s)>" % self._represent()

    def __hash__(self) -> cython.int:
        return hash(
            (
                self._years,
                self._months,
                self._days,
                self._hours,
                self._minutes,
                self._seconds,
                self._microseconds,
                self._leapdays,
                self._year,
                self._month,
                self._weekday,
                self._day,
                self._hour,
                self._minute,
                self._second,
                self._microsecond,
            )
        )
