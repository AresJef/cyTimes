# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

########## Pandas Timstamp ##########
# MAX: 2262-04-11 23:47:16.854775807 | +9223372036854775807
# MIN: 1677-09-21 00:12:43.145225000 | -9223372036854775000
# MIN: 1677-09-21 00:12:43.145224193 | (Documented)

from __future__ import annotations

# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cpython.set import PySet_Contains as set_contains  # type: ignore
from cython.cimports.cpython.dict import PyDict_GetItem as dict_getitem  # type: ignore
from cython.cimports.cytimes.pydt import pydt, cal_absolute_microsecond, TIMEZONE_AVAILABLE  # type: ignore
from cython.cimports.cytimes import cydatetime as cydt  # type: ignore
from cython.cimports.cytimes.cyparser import CONFIG_MONTH, CONFIG_WEEKDAY  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
from typing import Any, Literal, Iterator
import datetime, numpy as np
from pandas import to_datetime, to_timedelta
from pandas.errors import OutOfBoundsDatetime
from pandas._libs.tslibs.offsets import BaseOffset
from pandas import DatetimeIndex, RangeIndex, TimedeltaIndex
from pandas import Series, DataFrame, Timestamp, Timedelta, offsets
from cytimes import errors
from cytimes.pydt import pydt
from cytimes import cydatetime as cydt

__all__ = ["pddt"]

# Constants -----------------------------------------------------------------------------------
# . calendar
DAYS_BR_QUARTER_NDARRAY: np.ndarray = np.array([0, 90, 181, 273, 365])
# . time
TIME_START: datetime.time = datetime.time(0, 0, 0, 0)
TIME_END: datetime.time = datetime.time(23, 59, 59, 999999)
# . unit
UNIT_FREQUENCY: set[str] = {"D", "h", "m", "s", "ms", "us", "ns"}
# . offset
OFST_DATEOFFSET: object = offsets.DateOffset
OFST_MICRO: object = offsets.Micro
OFST_DAY: object = offsets.Day
OFST_MONTHEND: object = offsets.MonthEnd
OFST_MONTHBEGIN: object = offsets.MonthBegin
OFST_QUARTEREND: object = offsets.QuarterEnd
OFST_QUARTERBEGIN: object = offsets.QuarterBegin
OFST_YEAREND: object = offsets.YearEnd
OFST_YEARBEGIN: object = offsets.YearBegin
# . type
TP_SERIES: object = Series
TP_DATETIME64: object = np.datetime64
TP_DATETIMEINDEX: object = DatetimeIndex
TP_TIMEDELTA: object = Timedelta
TP_TIMEDELTAINDEX: object = TimedeltaIndex
TP_BASEOFFSET: object = BaseOffset
# . function
FN_PD_TODATETIME: object = to_datetime
FN_PD_TOTIMEDELTA: object = to_timedelta
FN_NP_ABS: object = np.abs
FN_NP_FULL: object = np.full
FN_NP_WHERE: object = np.where
FN_NP_MINIMUM: object = np.minimum


# pddt (Pandas Datetime) ----------------------------------------------------------------------
@cython.cclass
class pddt:
    """Represents the pddt (Pandas Datetime) that makes
    working with Series[Timestamp] easier.
    """

    # Config
    _default: object
    _day1st: cython.bint
    _year1st: cython.bint
    _utc: cython.bint
    _format: str
    _exact: cython.bint
    # Datetime
    _dts: Series
    _dts_index: DatetimeIndex
    _is_unit_ns: cython.bint
    _dts_len: cython.uint
    _dts_unit: str
    _dts_value: np.ndarray
    _dts_naive: Series
    _series_index: RangeIndex

    # Constructor -----------------------------------------------------------------------------
    def __init__(
        self,
        dtobj: Series | list,
        default: str | datetime.datetime | datetime.date | pydt | None = None,
        day1st: bool = False,
        year1st: bool = False,
        utc: bool = False,
        format: str = None,
        exact: bool = True,
    ) -> None:
        """The pddt (Pandas Datetime) that makes working with Series[Timestamp] easier.

        ### Datetime Object
        :param dtobj `<object>`: The datetime object to convert to pddt.
        - Supported data types:
            1. Sequence of datetime or datetime string.
            2. `<Series>` of timestamp or datetime object.
            3. Another `<pddt>`.

        ### Parser for 'dtobj'. (Only applicable when 'dtobj' must be parsed into Series of Timestamps).
        :param default `<object>`: The default to fill-in missing datetime elements when parsing string 'dtobj'. Defaults to `None`.
            - `None`: If parser failed to extract Y/M/D values from the string,
               the date of '1970-01-01' will be used to fill-in the missing year,
               month & day values.
            - `<date>`: If parser failed to extract Y/M/D values from the string,
               the give `date` will be used to fill-in the missing year, month &
               day values.
            - `<datetime>`: If parser failed to extract datetime elements from
               the string, the given `datetime` will be used to fill-in the
               missing year, month, day, hour, minute, second and microsecond.

        :param day1st `<bool>`: Whether to interpret first ambiguous date values as day. Defaults to `False`.
        :param year1st `<bool>`: Whether to interpret first the ambiguous date value as year. Defaults to `False`.
            - Both the 'day1st' & 'year1st' arguments works together to determine how
              to interpret ambiguous Y/M/D values.
            - In the case when all three values are ambiguous (e.g. `01/05/09`):
                * If 'day1st=True' and 'year1st=True', the date will be interpreted as `'Y/D/M'`.
                * If 'day1st=False' and 'year1st=True', the date will be interpreted as `'Y/M/D'`.
                * If 'day1st=True' and 'year1st=False', the date will be interpreted as `'D/M/Y'`.
                * If 'day1st=False' and 'year1st=False', the date will be interpreted as `'M/D/Y'`.
            - In the case when the year value is clear (e.g. `2010/01/05` or `99/01/05`):
                * If 'day1st=True', the date will be interpreted as `'Y/D/M'`.
                * If 'day1st=False', the date will be interpreted as `'Y/M/D'`.
            - In the case when only one value is ambiguous (e.g. `01/20/2010` or `01/20/99`):
                *There is no need to set 'day1st' or 'year1st', the date should be interpreted correctly.

        :param utc `<bool>`: Control timezone-related parsing, localization and conversion. Defaults to `False`.
            - If `True`, `ALWAYS` parse to tinzome-aware UTC-localized `pandas.Timetamp`. Timezone-naive inputs
              are `LOCALIZED` as UTC, while timezone-aware inputs are `CONVERTED` to UTC.
            - If `False`, Timezone-naive inputs remain naive, while timezone-aware ones will keep their timezone.
            - For more information, please refer to `pandas.to_datetime()` documentation.
              <https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html>

        :param format `<str>`: The strftime to parse the 'dtobj' with. Defaults to `None`.
            - strftime format (e.g. "%d/%m/%Y"): Note that "%f" will parse all the way to nanoseconds.
              For more infomation, please refer to <https://docs.python.org/3/library/datetime.html
              #strftime-and-strptime-behavior>.
            - "ISO8601": Parse any `ISO8601` time string (not necessarily in exactly the same format).
              For more infomation, please refer to <https://en.wikipedia.org/wiki/ISO_8601>.
            - "mixed": Infer the format for each element individually. This is risky, and should probably use
              it along with `dayfirst`.

        :param exact: `<bool>` Whether to parse with the exact provided 'format'. Defaults to `True`.
            - If `True`, perform an exact 'format' match.
            - If `False`, allow the `format` to match anywhere in the string.
            - Can `NOT` be used alongside `format='ISO8601'` or `format='mixed'`.
        """
        # Config
        self._day1st = bool(day1st)
        self._year1st = bool(year1st)
        self._utc = bool(utc)
        self._format = format
        self._exact = bool(exact)
        self._default = None if default is None else self._parse_datetime(default, None)
        # Parse
        self._dts = self._parse_dtobj(dtobj)
        self._dts_index = self._dts.dt
        self._dts_len = 0
        self._dts_unit = None
        self._dts_value = None
        self._dts_naive = None
        self._series_index = None

    # Access ----------------------------------------------------------------------------------
    @property
    def year(self) -> Series[int]:
        """Access the year of the Series `<Series[int]>`."""
        return self._dts_index.year

    @property
    def quarter(self) -> Series[int]:
        """Access the quarter of the Series `<Series[int]>`."""
        return self._dts_index.quarter

    @property
    def month(self) -> Series[int]:
        """Access the month of the Series `<Series[int]>`."""
        return self._dts_index.month

    @property
    def day(self) -> Series[int]:
        """Access the day of the Series `<Series[int]>`."""
        return self._dts_index.day

    @property
    def hour(self) -> Series[int]:
        """Access the hour of the Series `<Series[int]>`."""
        return self._dts_index.hour

    @property
    def minute(self) -> Series[int]:
        """Access the minute of the Series `<Series[int]>`."""
        return self._dts_index.minute

    @property
    def second(self) -> Series[int]:
        """Access the second of the Series `<Series[int]>`."""
        return self._dts_index.second

    @property
    def microsecond(self) -> Series[int]:
        """Access the microsecond of the Series `<Series[int]>`."""
        return self._dts_index.microsecond

    @property
    def tzinfo(self) -> datetime.tzinfo | None:
        """Access the timezone of the Series `<datetime.tzinfo/None>`."""
        return self._dts_index.tz

    @property
    def dts(self) -> Series[Timestamp]:
        """Access as Series of Timestamps `<Series[Timestamp]>`."""
        return self._dts

    @property
    def dts_str(self) -> Series[str]:
        """Access as Series of Timestamps in string format `<Series[str]>`."""
        return self._dts_index.strftime("%Y-%m-%d %H:%M:%S.%f%Z")

    @property
    def dts_iso(self) -> Series[str]:
        """Access as Series of Timestamps in ISO format `<Series[str]>`."""
        return self._dts_index.strftime("%Y-%m-%dT%H:%M:%S.%f")

    @property
    def dts_isotz(self) -> Series[str]:
        """Access as Series of Timestamps in ISO format with timezone `<Series[str]>`."""
        return self._dts_index.strftime("%Y-%m-%dT%H:%M:%S.%f%Z")

    @property
    def dates(self) -> Series[datetime.date]:
        """Access as Series of dates `<Series[datetime.date]>`."""
        return self._dts_index.date

    @property
    def dates_iso(self) -> Series[str]:
        """Access as Series of dates in ISO format `<Series[str]>`."""
        return self._dts_index.strftime("%Y-%m-%d")

    @property
    def times(self) -> Series[datetime.time]:
        """Access as Series of times `<Series[datetime.time]>`."""
        return self._dts_index.time

    @property
    def timestz(self) -> Series[datetime.time]:
        """Access as Series of times with timezone `<Series[datetime.time]>`."""
        return self._dts_index.timetz

    @property
    def times_iso(self) -> Series[str]:
        """Access as Series of times in ISO format `<Series[str]>`."""
        return self._dts_index.strftime("%H:%M:%S.%f")

    @property
    def dts_py(self) -> list[datetime.datetime]:
        """Access as list of datetime objects `<list[datetime.datetime]>`."""
        return [cydt.dt_fr_dt(dt) for dt in self._dts]

    @property
    def dts_64(self) -> np.ndarray[np.datetime64]:
        """Access as numpy array of datetime64 objects `<ndarray[datetime64]>`."""
        return self._get_dts_value()

    @property
    def ordinals(self) -> Series[int]:
        """Access the ordinal of the Series `<Series[int]>`."""
        return cydt.dt64series_to_ordinals(self._get_dts_naive())

    @property
    def seconds(self) -> Series[int]:
        """Access in total seconds since EPOCH, ignoring
        the timezone (if exists) `<Series[int]>`.

        ### Notice
        This should `NOT` be treated as timestamp.
        """
        return cydt.dt64series_to_seconds(self._get_dts_naive())

    @property
    def seconds_utc(self) -> Series[int]:
        """Access in total seconds since EPOCH `<Series[int]>`.
        - If `timezone-aware`, return total seconds in UTC.
        - If `timezone-naive`, requivalent to `pddt.seconds`.

        ### Notice
        This should `NOT` be treated as timestamp.
        """
        return cydt.dt64series_to_seconds(self._dts)

    @property
    def microseconds(self) -> Series[int]:
        """Access in total microseconds since EPOCH, ignoring
        the timezone (if exists) `<Series[int]>`."""
        return cydt.dt64series_to_microseconds(self._get_dts_naive())

    @property
    def microseconds_utc(self) -> Series[int]:
        """Access in total microseconds since EPOCH `<Series[int]>`.
        - If `timezone-aware`, return total microseconds in UTC.
        - If `timezone-naive`, requivalent to `pddt.microseconds`.
        """
        return cydt.dt64series_to_microseconds(self._dts)

    @property
    def timestamps(self) -> Series[float]:
        """Access as Series of timestamps `<Series[float]>`."""
        return cydt.dt64series_to_timestamps(self._dts)

    # Calendar: Year --------------------------------------------------------------------------
    def is_leapyear(self) -> Series[bool]:
        """Whether the dates are leap years `<Series[bool]>`."""
        return self._dts_index.is_leap_year

    def leap_bt_years(self, year: int) -> Series[int]:
        """Calculate the number of leap years between the dates
        and the given 'year' `<Series[int]>`."""
        y1_arr: np.ndarray = self._dts_index.year.values
        y2_arr: np.ndarray = y1_arr.copy()
        y1_arr = FN_NP_WHERE(y1_arr <= year, y1_arr, year) - 1
        y2_arr = FN_NP_WHERE(y2_arr > year, y2_arr, year) - 1
        arr: np.ndarray = (
            (y2_arr // 4 - y1_arr // 4)
            - (y2_arr // 100 - y1_arr // 100)
            + (y2_arr // 400 - y1_arr // 400)
        )
        return self._array_to_series(arr)

    @property
    def days_in_year(self) -> Series[int]:
        """Get the maximum number of days in the year.
        Expect 365 or 366 (leapyear) `<Series[int]>`."""
        return self._array_to_series(self._dts_index.is_leap_year.values + 365)

    @property
    def days_bf_year(self) -> Series[int]:
        """Get the number of days betweem the 1st day of 1AD and
        the 1st day of the year of the dates `<Series[int]>`."""
        return self.ordinals - self.days_of_year

    @property
    def days_of_year(self) -> Series[int]:
        """Get the number of days between the 1st day
        of the year and the dates `<Series[int]>`."""
        return self._dts_index.day_of_year

    # Manipulate: Year ------------------------------------------------------------------------
    def is_year(self, year: int) -> Series[bool]:
        """Whether the dates are in the given 'year' `<Series[bool]>`."""
        return self._dts_index.year == year

    def is_year_1st(self) -> Series[bool]:
        """Whether the dates are the 1st day of the year `<Series[bool]>`."""
        return self._dts_index.is_year_start

    def is_year_lst(self) -> Series[bool]:
        """Whether the dates are the last day of the year `<Series[bool]>`."""
        return self._dts_index.is_year_end

    def to_year_1st(self) -> pddt:
        """Go to the 1st day of the year `<pddt>`."""
        return self._new(self._dts + OFST_YEAREND(0) - OFST_YEARBEGIN(1, month=1))

    def to_year_lst(self) -> pddt:
        """Go to the 1st day of the year `<pddt>`."""
        return self._new(self._dts + OFST_YEAREND(0))

    def to_curr_year(
        self,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> pddt:
        """Go to specific 'month' and 'day' of the current year `<pddt>`."""
        # Parse 'month' value
        month: int = self._parse_month(month)
        if month == 100:
            return self.to_curr_month(day)  # exit: no adjustment to month

        # Invalid 'day' value
        if day < 1:
            dates = self._dts + OFST_YEAREND(0) - OFST_YEARBEGIN(1, month=month)
            delta = self._gen_timedelta(
                FN_NP_MINIMUM(dates.dt.days_in_month.values, self._dts_index.day.values)
                - 1,
                unit="D",
            )
            return self._new(dates + delta)
        # First day of the month
        if day == 1:
            return self._new(
                self._dts + OFST_YEAREND(0) - OFST_YEARBEGIN(1, month=month)
            )
        # Days before 28th
        if day < 29:
            return self._new(
                self._dts
                + OFST_YEAREND(0)
                - OFST_YEARBEGIN(1, month=month)
                + OFST_DAY(day - 1)
            )
        # Days before 31st
        if day < 31:
            dates = self._dts + OFST_YEAREND(0) - OFST_YEARBEGIN(1, month=month)
            delta = self._gen_timedelta(
                FN_NP_MINIMUM(dates.dt.days_in_month.values, day) - 1, unit="D"
            )
            return self._new(dates + delta)
        # Last day of the month
        return self._new(
            self._dts
            + OFST_YEAREND(0)
            - OFST_YEARBEGIN(1, month=month)
            + OFST_MONTHEND(0)
        )

    def to_next_year(
        self,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> pddt:
        """Go to specific 'month' and 'day' of the next year `<pddt>`."""
        return self.to_year(1, month, day)

    def to_prev_year(
        self,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> pddt:
        """Go to specific 'month' and 'day' of the previous year `<pddt>`."""
        return self.to_year(-1, month, day)

    def to_year(
        self,
        offset: cython.int = 0,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> pddt:
        """Go to specific 'month' and 'day' of the current year (+/-) 'offset' `<pddt>`."""
        # No offset adjustment
        if offset == 0:
            return self.to_curr_year(month, day)  # exit

        # Parse 'month' value
        month: int = self._parse_month(month)
        if month == 100:
            # fmt: off
            return self._new(
                self._dts + OFST_DATEOFFSET(years=offset)
            ).to_curr_month(day)  # exit: no adjustment to month
            # fmt: on

        # Invalid 'day' value
        if day < 1:
            dates = (
                self._dts
                + OFST_YEAREND(0)
                + OFST_DATEOFFSET(years=offset, months=month - 12)
            )
            delta = self._gen_timedelta(
                FN_NP_MINIMUM(dates.dt.days_in_month.values, self._dts_index.day.values)
                - dates.dt.day.values,
                unit="D",
            )
            return self._new(dates + delta)
        # First day of the month
        if day == 1:
            return self._new(
                self._dts
                + OFST_YEAREND(0)
                - OFST_YEARBEGIN(1, month=month)
                + OFST_DATEOFFSET(years=offset)
            )
        # Days before 28th
        if day < 29:
            return self._new(
                self._dts
                + OFST_YEAREND(0)
                - OFST_YEARBEGIN(1, month=month)
                + OFST_DATEOFFSET(years=offset, days=day - 1)
            )
        # Days before 31st
        if day < 31:
            dates = (
                self._dts
                + OFST_YEAREND(0)
                + OFST_DATEOFFSET(years=offset, months=month - 12)
            )
            delta = self._gen_timedelta(
                FN_NP_MINIMUM(dates.dt.days_in_month.values, day) - dates.dt.day.values,
                unit="D",
            )
            return self._new(dates + delta)
        # Last day of the month
        return self._new(
            self._dts
            + OFST_YEAREND(0)
            + OFST_DATEOFFSET(years=offset, months=month - 12)
            + OFST_MONTHEND(0)
        )

    # Calendar: Quarter -----------------------------------------------------------------------
    @property
    def days_in_quarter(self) -> Series[int]:
        """Get the maximum number of days in the quarter `<Series[int]>`."""
        quarter: np.ndarray = self._dts_index.quarter.values
        days: np.ndarray = (
            DAYS_BR_QUARTER_NDARRAY[quarter] - DAYS_BR_QUARTER_NDARRAY[quarter - 1]
        )
        return self._array_to_series(days + self._dts_index.is_leap_year.values)

    @property
    def days_bf_quarter(self) -> Series[int]:
        """Get the number of days between the 1st day of the year and
        the 1st day of the quarter `<Series[int]>`."""
        quarter: np.ndarray = self._dts_index.quarter.values
        days: np.ndarray = DAYS_BR_QUARTER_NDARRAY[quarter - 1]
        leap: np.ndarray = self._dts_index.is_leap_year.values * (quarter > 1)
        return self._array_to_series(days + leap)

    @property
    def days_of_quarter(self) -> Series[int]:
        """Get the number of days between the 1st day of the quarter
        and the dates `<Series[int]>`."""
        days_of_year: np.ndarray = self._dts_index.day_of_year.values
        quarter: np.ndarray = self._dts_index.quarter.values
        days: np.ndarray = DAYS_BR_QUARTER_NDARRAY[quarter - 1]
        leap: np.ndarray = self._dts_index.is_leap_year.values * (quarter > 1)
        return self._array_to_series(days_of_year - days - leap)

    @property
    def quarter_1st_month(self) -> Series[int]:
        """Get the first month of the quarter.
        Expect 1, 4, 7, 10 `<Series[int]>`."""
        quarter: np.ndarray = self._dts_index.quarter.values
        return self._array_to_series(quarter * 3 - 2)

    @property
    def quarter_lst_month(self) -> Series[int]:
        """Get the last month of the quarter.
        Expect 3, 6, 9, 12 `<Series[int]>`."""
        quarter: np.ndarray = self._dts_index.quarter.values
        return self._array_to_series(quarter * 3)

    # Manipulate: Quarter ---------------------------------------------------------------------
    def is_quarter(self, quarter: int) -> Series[bool]:
        """Whether the dates are in the given 'quarter' `<Series[bool]>`."""
        return self._dts_index.quarter == quarter

    def is_quarter_1st(self) -> Series[bool]:
        """Whether the dates are the 1st day of the quarter `<Series[bool]>`."""
        return self._dts_index.is_quarter_start

    def is_quarter_lst(self) -> Series[bool]:
        """Whether the dates are the last day of the quarter `<Series[bool]>`."""
        return self._dts_index.is_quarter_end

    def to_quarter_1st(self) -> pddt:
        """Go to the 1st day of the quarter `<pddt>`."""
        return self._new(
            self._dts + OFST_QUARTEREND(0) - OFST_QUARTERBEGIN(1, startingMonth=1)
        )

    def to_quarter_lst(self) -> pddt:
        """Go to the last day of the quarter `<pddt>`."""
        return self._new(self._dts + OFST_QUARTEREND(0))

    def to_curr_quarter(self, month: cython.int = -1, day: cython.int = -1) -> pddt:
        """Go to specific 'month (of the quarter [1..3])'
        and 'day' of the current quarter `<pddt>`."""
        # Invalid 'month' value
        if month < 1:
            return self.to_curr_month(day)  # exit: no adjustment to month
        month = month % 3 or 3

        # Invalid 'day' value
        if day < 1:
            dates: Series = (
                self._dts
                + OFST_QUARTEREND(0)
                - OFST_QUARTERBEGIN(1, startingMonth=month)
            )
            delta = self._gen_timedelta(
                FN_NP_MINIMUM(dates.dt.days_in_month.values, self._dts_index.day.values)
                - 1,
                unit="D",
            )
            return self._new(dates + delta)
        # First day of the month
        if day == 1:
            return self._new(
                self._dts
                + OFST_QUARTEREND(0)
                - OFST_QUARTERBEGIN(1, startingMonth=month)
            )
        # Days before 28th
        if day < 29:
            return self._new(
                self._dts
                + OFST_QUARTEREND(0)
                - OFST_QUARTERBEGIN(1, startingMonth=month)
                + OFST_DAY(day - 1)
            )
        # Days before 31st
        if day < 31:
            dates = (
                self._dts
                + OFST_QUARTEREND(0)
                - OFST_QUARTERBEGIN(1, startingMonth=month)
            )
            delta = self._gen_timedelta(
                FN_NP_MINIMUM(dates.dt.days_in_month.values, day) - 1, unit="D"
            )
            return self._new(dates + delta)
        # Last day of the month
        return self._new(
            self._dts
            + OFST_QUARTEREND(0)
            - OFST_QUARTERBEGIN(1, startingMonth=month)
            + OFST_MONTHEND(0)
        )

    def to_next_quarter(self, month: cython.int = -1, day: cython.int = -1) -> pddt:
        """Go to specific 'month (of the quarter [1..3])' and
        'day' of the next quarter `<pddt>`."""
        return self.to_quarter(1, month, day)

    def to_prev_quarter(self, month: cython.int = -1, day: cython.int = -1) -> pddt:
        """Go to specific 'month (of the quarter [1..3])' and
        'day' of the previous quarter `<pddt>`."""
        return self.to_quarter(-1, month, day)

    def to_quarter(
        self,
        offset: cython.int = 0,
        month: cython.int = -1,
        day: cython.int = -1,
    ) -> pddt:
        """Go to specific 'month (of the quarter [1..3])' and
        'day' of the current quarter (+/-) 'offset' `<pddt>`."""
        # No offset adjustment
        if offset == 0:
            return self.to_curr_quarter(month, day)

        # Invalid 'month' value
        if month < 1:
            return self._new(
                self._dts + OFST_DATEOFFSET(months=offset * 3)
            ).to_curr_month(day)
        month = month % 3 or 3

        # Invalid 'day' value
        if day < 1:
            dates = (
                self._dts
                + OFST_QUARTEREND(0)
                + OFST_DATEOFFSET(months=offset * 3 + month - 3)
            )
            delta = self._gen_timedelta(
                FN_NP_MINIMUM(dates.dt.days_in_month.values, self._dts_index.day.values)
                - dates.dt.day.values,
                unit="D",
            )
            return self._new(dates + delta)
        # First day of the month
        if day == 1:
            return self._new(
                self._dts
                + OFST_QUARTEREND(0)
                - OFST_QUARTERBEGIN(1, startingMonth=month)
                + OFST_DATEOFFSET(months=offset * 3)
            )
        # Days before 28th
        if day < 29:
            return self._new(
                self._dts
                + OFST_QUARTEREND(0)
                - OFST_QUARTERBEGIN(1, startingMonth=month)
                + OFST_DATEOFFSET(months=offset * 3, days=day - 1)
            )
        # Days before 31st
        if day < 31:
            dates = (
                self._dts
                + OFST_QUARTEREND(0)
                + OFST_DATEOFFSET(months=offset * 3 + month - 3)
            )
            delta = self._gen_timedelta(
                FN_NP_MINIMUM(dates.dt.days_in_month.values, day) - dates.dt.day.values,
                unit="D",
            )
            return self._new(dates + delta)
        # Last day of the month
        return self._new(
            self._dts
            + OFST_QUARTEREND(0)
            + OFST_DATEOFFSET(months=offset * 3 + month - 3)
            + OFST_MONTHEND(0)
        )

    # Calendar: Month -------------------------------------------------------------------------
    @property
    def days_in_month(self) -> Series[int]:
        """Get the maximum number of days in the month `<Series[int]>`."""
        return self._dts_index.days_in_month

    @property
    def days_bf_month(self) -> Series[int]:
        """Get the number of days between the 1st day of
        the year and the 1st day of the month `<Series[int]>`."""
        return self.days_of_year - self.day

    # Manipulate: Month -----------------------------------------------------------------------
    def is_month(self, month: int | str) -> Series[bool]:
        """Whether the dates are in the given 'month' `<Series[bool]>`."""
        return self._dts_index.month == self._parse_month(month)

    def is_month_1st(self) -> Series[bool]:
        """Whether the dates are the 1st day of the month `<Series[bool]>`."""
        return self._dts_index.is_month_start

    def is_month_lst(self) -> Series[bool]:
        """Whether the dates are the last day of the month `<Series[bool]>`."""
        return self._dts_index.is_month_end

    def to_month_1st(self) -> pddt:
        """Go to the 1st day of the month `<pddt>`."""
        return self._new(self._dts + OFST_MONTHEND(0) - OFST_MONTHBEGIN(1))

    def to_month_lst(self) -> pddt:
        """Go to the last day of the month `<pddt>`."""
        return self._new(self._dts + OFST_MONTHEND(0))

    def to_curr_month(self, day: cython.int = -1) -> pddt:
        """Go to specific 'day' of the current month `<pddt>`."""
        # Invalid 'day' value
        if day < 1:
            return self  # exit: invalid day
        # First day of the month
        if day == 1:
            return self.to_month_1st()
        # Days before 28th
        if day < 29:
            return self._new(
                self._dts + OFST_MONTHEND(0) - OFST_MONTHBEGIN(1) + OFST_DAY(day - 1)
            )
        # Days before 31st
        if day < 31:
            delta = self._gen_timedelta(
                FN_NP_MINIMUM(self.days_in_month.values, day) - 1, unit="D"
            )
            return self._new(self._dts + OFST_MONTHEND(0) - OFST_MONTHBEGIN(1) + delta)
        # Last day of the month
        return self.to_month_lst()

    def to_next_month(self, day: cython.int = -1) -> pddt:
        """Go to specific 'day' of the next month `<pddt>`."""
        return self.to_month(1, day)

    def to_prev_month(self, day: cython.int = -1) -> pddt:
        """Go to specific 'day' of the previous month `<pddt>`."""
        return self.to_month(-1, day)

    def to_month(self, offset: cython.int = 0, day: cython.int = -1) -> pddt:
        """Go to specific 'day' of the current month (+/-) 'offset' `<pddt>`."""
        # No offset adjustment
        if offset == 0:
            return self.to_curr_month(day)
        # Invalid 'day' value (+/-) 'offset'
        if day < 1:
            return self._new(self._dts + OFST_DATEOFFSET(months=offset))
        # First day of the month (+/-) 'offset'
        if day == 1:
            return self._new(
                self._dts
                + OFST_MONTHEND(0)
                - OFST_MONTHBEGIN(1)
                + OFST_DATEOFFSET(months=offset)
            )
        # Days before 28th (+/-) 'offset'
        if day < 29:
            return self._new(
                self._dts
                + OFST_MONTHEND(0)
                - OFST_MONTHBEGIN(1)
                + OFST_DATEOFFSET(months=offset, days=day - 1)
            )
        # Days before 31st (+/-) 'offset'
        if day < 31:
            dts: Series = (
                self._dts
                + OFST_MONTHEND(0)
                - OFST_MONTHBEGIN(1)
                + OFST_DATEOFFSET(months=offset)
            )
            delta = self._gen_timedelta(
                FN_NP_MINIMUM(dts.dt.days_in_month.values, day) - 1, unit="D"
            )
            return self._new(dts + delta)
        # Last day of the month (+/-) 'offset'
        return self._new(self._dts + OFST_DATEOFFSET(months=offset) + OFST_MONTHEND(0))

    # Calendar: Weekday -----------------------------------------------------------------------
    @property
    def weekday(self) -> Series[int]:
        """The weekday of the dates `<Series[int]>`.
        Values: 0=Monday...6=Sunday."""
        return self._dts_index.weekday

    @property
    def isoweekday(self) -> Series[int]:
        """The ISO weekday of the dates `<Series[int]>`.
        Values: 1=Monday...7=Sunday."""
        return self._array_to_series(self._dts_index.weekday.values + 1)

    @property
    def isoweek(self) -> Series[int]:
        """The ISO calendar week number of the dates `<Series[int]>`."""
        return self._dts_index.isocalendar()["week"]

    @property
    def isoyear(self) -> Series[int]:
        """The ISO calendar year of the dates `<Series[int]>`."""
        return self._dts_index.isocalendar()["year"]

    @property
    def isocalendar(self) -> DataFrame[int]:
        """The ISO calendar year, week, and weekday of the dates `<DataFrame[int]>`."""
        return self._dts_index.isocalendar()

    # Manipulate: Weekday ---------------------------------------------------------------------
    def is_weekday(self, weekday: int | str) -> Series[bool]:
        """Whether the dates are in the given 'weekday' `<Series[bool]>`."""
        return self._dts_index.weekday == self._parse_weekday(weekday)

    def to_monday(self) -> pddt:
        """Go to Monday of the current week `<pddt>`."""
        return self._new(
            self._dts - self._gen_timedelta(self._dts_index.weekday, unit="D")
        )

    def to_tuesday(self) -> pddt:
        """Go to Tuesday of the current week `<pddt>`."""
        return self._new(
            self._dts - self._gen_timedelta(self._dts_index.weekday - 1, unit="D")
        )

    def to_wednesday(self) -> pddt:
        """Go to Wednesday of the current week `<pddt>`."""
        return self._new(
            self._dts - self._gen_timedelta(self._dts_index.weekday - 2, unit="D")
        )

    def to_thursday(self) -> pddt:
        """Go to Thursday of the current week `<pddt>`."""
        return self._new(
            self._dts - self._gen_timedelta(self._dts_index.weekday - 3, unit="D")
        )

    def to_friday(self) -> pddt:
        """Go to Friday of the current week `<pddt>`."""
        return self._new(
            self._dts - self._gen_timedelta(self._dts_index.weekday - 4, unit="D")
        )

    def to_saturday(self) -> pddt:
        """Go to Saturday of the current week `<pddt>`."""
        return self._new(
            self._dts - self._gen_timedelta(self._dts_index.weekday - 5, unit="D")
        )

    def to_sunday(self) -> pddt:
        """Go to Sunday of the current week `<pddt>`."""
        return self._new(
            self._dts - self._gen_timedelta(self._dts_index.weekday - 6, unit="D")
        )

    def to_curr_weekday(self, weekday: int | str | None = None) -> pddt:
        """Go to specific 'weekday' of the current week `<pddt>`."""
        # Parse 'weekday' value
        wkd: int = self._parse_weekday(weekday)
        if wkd == 100:
            return self

        # Go to specific 'weekday'
        return self._new(
            self._dts - self._gen_timedelta(self._dts_index.weekday - wkd, unit="D")
        )

    def to_next_weekday(self, weekday: int | str | None = None) -> pddt:
        """Go to specific 'weekday' of the next week `<pddt>`."""
        return self.to_weekday(1, weekday)

    def to_prev_weekday(self, weekday: int | str | None = None) -> pddt:
        """Go to specific 'weekday' of the previous week `<pddt>`."""
        return self.to_weekday(-1, weekday)

    def to_weekday(
        self,
        offset: cython.int = 0,
        weekday: int | str | None = None,
    ) -> pddt:
        """Go to specific 'weekday' of the current
        week (+/-) 'offset' `<pddt>`."""
        # No offset adjustment
        if offset == 0:
            return self.to_curr_weekday(weekday)

        # Parse 'weekday' value
        wkd: cython.uint = self._parse_weekday(weekday)
        if wkd == 100:
            return self._new(self._dts + OFST_DAY(offset * 7))

        # Go to specific 'weekday' (+/-) 'offset'
        offset = offset * 7 + wkd
        return self._new(
            self._dts - self._gen_timedelta(self._dts_index.weekday - offset, unit="D")
        )

    # Manipulate: Day -------------------------------------------------------------------------
    def is_day(self, day: int) -> Series[bool]:
        """Whether the dates are in the given 'day' `<Series[bool]>`."""
        return self._dts_index.day == day

    def to_tomorrow(self) -> pddt:
        """Go to the next day `<pddt>`."""
        return self._new(self._dts + OFST_DAY(1))

    def to_yesterday(self) -> pddt:
        """Go to the previous day `<pddt>`."""
        return self._new(self._dts + OFST_DAY(-1))

    def to_day(self, offset: cython.int) -> pddt:
        """Go to the current day (+/-) 'offset' `<pddt>`."""
        # No offset adjustment
        if offset == 0:
            return self

        # Go to the current day (+/-) 'offset'
        return self._new(self._dts + OFST_DAY(offset))

    # Manipulate: Time ------------------------------------------------------------------------
    def is_time_start(self) -> Series[bool]:
        """Whether the current time is the
        start of time (00:00:00.000000) `<Series[bool]>`."""
        return self._dts_index.time == TIME_START

    def is_time_end(self) -> Series[bool]:
        """Whether the current time is the
        end of time (23:59:59.999999) `<Series[bool]>`."""
        return self._dts_index.time == TIME_END

    def to_time_start(self) -> pddt:
        """Go to the start of time (00:00:00.000000) `<pddt>`."""
        return self._new(self._dts_index.floor("D", "infer", "shift_backward"))

    def to_time_end(self) -> pddt:
        """Go to the end of time (23:59:59.999999) `<pddt>`."""
        return self._new(
            self._dts_index.ceil("D", "infer", "shift_forward") - OFST_MICRO(1)
        )

    def to_time(
        self,
        hour: cython.int = -1,
        minute: cython.int = -1,
        second: cython.int = -1,
        millisecond: cython.int = -1,
        microsecond: cython.int = -1,
    ) -> pddt:
        """Go to specific 'hour', 'minute', 'second', 'millisecond',
        and 'microsecond' of the current times `<pddt>`."""
        # Calculate microsecond
        microsecond = cal_absolute_microsecond(millisecond, microsecond)
        # fmt: off
        delta = self._gen_timedelta(
            # . hour
            (
                FN_NP_FULL(len(self), min(hour, 23) * cydt.US_HOUR)
                if hour >= 0 else
                self._dts_index.hour.values * cydt.US_HOUR
            )
            # . minute
            + (
                min(minute, 59) * 60_000_000
                if minute >= 0 else
                self._dts_index.minute.values * 60_000_000
            )
            # . second
            + (
                min(second, 59) * 1_000_000
                if second >= 0 else 
                self._dts_index.second.values * 1_000_000
            )
            # . microsecond
            + (
                microsecond 
                if microsecond >= 0
                else self._dts_index.microsecond.values
            ),
            # . unit            
            unit="us"
        )
        # fmt: on
        # Apply delta
        return self._new(self._dts_index.floor("D", "infer", "shift_backward") + delta)

    # Manipulate: Timezone --------------------------------------------------------------------
    @property
    def tz_available(self) -> set[str]:
        """Access all the available timezone names `<set[str]>`."""
        return TIMEZONE_AVAILABLE

    def tz_localize(
        self,
        tz: str | datetime.tzinfo | None = None,
        ambiguous: bool | Series[bool] | Literal["raise", "infer"] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> pddt:
        """Localize to a specific 'tz' timezone `<pddt>`.

        Equivalent to `Series.dt.tz_localize(tz)`.

        :param tz: `<datetime.tzinfo>`/`<str (timezone name)>`/`None (remove timezone)`. Defaults to `None`.
            - `<str>`: The timezone name to localize to.
            - `<datetime.tzinfo>`: The timezone to localize to.
            - `None`: Remove timezone awareness.

        :param ambiguous: `<bool>`/`<Series[bool]>`/`'raise'`/`'infer'`. Defaults to `'raise'`.
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from 03:00
            DST to 02:00 non-DST, 02:30:00 local time occurs both at 00:30:00 UTC
            and at 01:30:00 UTC. In such a situation, the ambiguous parameter
            dictates how ambiguous times should be handled.
            - `<bool>`: Marks all times as DST time (True) and non-DST time (False).
            - `<Series[bool]>`: Marks specific times (matching index) as DST time (True)
               and non-DST time (False).
            - `'raise'`: Raises an `InvalidTimezoneError` if there are ambiguous times.
            - `'infer'`: Attempt to infer fall dst-transition hours based on order.
            - * Notice: `'NaT'` is not supported.

        :param nonexistent: `'shift_forward'`/`'shift_backward'`/`'raise'`. Defaults to `'raise'`.
            A nonexistent time does not exist in a particular timezone where clocks moved
            forward due to DST.
            - `'shift_forward'`: Shifts nonexistent times forward to the closest existing time.
            - `'shift_backward'`: Shifts nonexistent times backward to the closest existing time.
            - `'raise'`: Raises an `InvalidTimezoneError` if there are nonexistent times.
            - * Notice: `'NaT'` is not supported.
        """
        return self._new(self._tz_localize(self._dts, tz, ambiguous, nonexistent))

    def tz_convert(
        self,
        tz: str | datetime.tzinfo | None = None,
        ambiguous: bool | Series[bool] | Literal["raise", "infer"] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> pddt:
        """Convert to a specific 'tz' timezone `<pddt>`.

        Similar to `Series.dt.tz_convert(tz)`, but behave more like
        `datetime.astimezone(tz)`. The main differences occurs when
        handling timezone-naive Series, where `pandas.tz_convert` will
        raise an error, while this method will first localize to the
        system's local timezone and then convert to the target timezone.

        :param tz: `<datetime.tzinfo>`/`<str (timezone name)>`/`None (local timezone)`. Defaults to `None`.
            - `<str>`: The timezone name to convert to.
            - `<datetime.tzinfo>`: The timezone to convert to.
            - `None`: Convert to system's local timezone.

        :param ambiguous, nonexistent: Please refer to `tz_localize` method for details.
        """
        return self._new(self._tz_convert(self._dts, tz, ambiguous, nonexistent))

    def tz_switch(
        self,
        targ_tz: str | datetime.tzinfo | None,
        base_tz: str | datetime.tzinfo | None = None,
        naive: bool = False,
        ambiguous: bool | Series[bool] | Literal["raise", "infer"] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> pddt:
        """Switch to 'targ_tz' timezone from 'base_tz' timezone `<pddt>`.

        :param targ_tz `<str/tzinfo/None>`: The target timezone to convert.
        :param base_tz `<str/tzinfo/None>`: The base timezone to localize. Defaults to `None`.
        :param naive `<bool>`: Whether to return as timezone-naive. Defaults to `False`.
        :param ambiguous, nonexistent: Please refer to `tz_localize` method for details.
        :return `<pddt>`: pddt after switch of timezone.

        ### Explaination
        - If pddt is timezone-aware, 'base_tz' will be ignored, and only performs
          the convertion to the 'targ_tz' (Series.dt.tz_convert(targ_tz)).

        - If pddt is timezone-naive, first will localize to the given 'base_tz',
          then convert to the 'targ_tz'. In this case, the 'base_tz' must be
          specified, else it is ambiguous on how to convert to the target timezone.
        """
        # Pddt is timezone-aware
        curr_tz = self._dts_index.tz
        if curr_tz is not None:
            # . current => target timezone
            dts = self._tz_convert(self._dts, targ_tz, ambiguous, nonexistent)
            if naive:
                dts = dts.dt.tz_localize(None)

        # Pddt is timezone-naive
        elif isinstance(base_tz, str) or cydt.is_tzinfo(base_tz):
            # . localize to base & base => target
            dts = self._tz_localize(self._dts, base_tz, ambiguous, nonexistent)
            dts = self._tz_convert(dts, targ_tz, ambiguous, nonexistent)
            if naive:
                dts = dts.dt.tz_localize(None)

        # Invalid
        else:
            raise errors.InvalidTimezoneError(
                "<{}>\nCannot switch timezone-naive '<pddt>' without "
                "a valid 'base_tz'.".format(self.__class__.__name__)
            )

        # Generate
        return self._new(dts)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _validate_ambiguous_nonexistent(
        self,
        ambiguous: object,
        nonexistent: object,
    ) -> cython.bint:
        """(Internal) Validate 'ambiguous' and 'nonexistent' arguments `<bool>`."""
        if ambiguous == "NaT":
            raise errors.InvalidTimezoneError(
                "<{}>\nArgument `ambiguous='NaT'` is not supported."
            )
        if nonexistent == "NaT":
            raise errors.InvalidTimezoneError(
                "<{}>\nArgument `nonexistent='NaT'` is not supported."
            )
        return True

    def _tz_localize(
        self,
        dts: Series,
        tz: str | datetime.tzinfo | None,
        ambiguous: bool | Series[bool] | Literal["raise", "infer"] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> Series[Timestamp]:
        """(Internal) Localize to a specific 'tz' timezone `<Series>`.

        Equivalent to `Series.dt.tz_localize(tz)`.
        """
        # Validate 'tz'
        if tz is None and tz is dts.dt.tz:
            return dts  # exit: no change

        # Validate 'ambiguous' and 'nonexistent'
        self._validate_ambiguous_nonexistent(ambiguous, nonexistent)

        # Localize to 'tz'
        try:
            return dts.dt.tz_localize(tz, ambiguous, nonexistent)
        except Exception as err:
            if isinstance(tz, str) and tz in str(err):
                msg = "Invalid 'tz' timezone name: '{}'.".format(tz)
            else:
                msg = "Failed to localize to 'tz': {}".format(err)
            raise errors.InvalidTimezoneError(
                "<{}>\n{}".format(self.__class__.__name__, msg)
            ) from err

    def _tz_convert(
        self,
        dts: Series,
        tz: str | datetime.tzinfo | None,
        ambiguous: bool | Series[bool] | Literal["raise", "infer"] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> Series[Timestamp]:
        """(Internal) Convert to a specific 'tz' timezone `<Series>`.

        Similar to `Series.dt.tz_convert(tz)`, but behave more like
        `datetime.astimezone(tz)`. The main differences occurs when
        handling timezone-naive Series, where `pandas.tz_convert` will
        raise an error, while this method will first localize to the
        system's local timezone and then convert to the target timezone.
        """
        # Resolve target timezone
        curr_tz = dts.dt.tz
        if tz is None:
            targ_tz = cydt.gen_tzinfo_local(None)
            if curr_tz is None:
                return self._tz_localize(dts, targ_tz, ambiguous, nonexistent)
        else:
            targ_tz = tz

        # Resolve current timezone
        if curr_tz is None:
            dts = self._tz_localize(
                dts, cydt.gen_tzinfo_local(None), ambiguous, nonexistent
            )

        # Convert timezone
        try:
            return dts.dt.tz_convert(targ_tz)
        except Exception as err:
            if isinstance(tz, str) and tz in str(err):
                msg = "Invalid 'tz' timezone name: '{}'.".format(tz)
            else:
                msg = "Failed to convert to 'tz': {}".format(err)
            raise errors.InvalidTimezoneError(
                "<{}>\n{}".format(self.__class__.__name__, msg)
            ) from err

    # Manipulate: Frequency -------------------------------------------------------------------
    def freq_round(
        self,
        freq: Literal["D", "h", "m", "s", "ms", "us", "ns"],
        ambiguous: bool | Series[bool] | Literal["raise", "infer"] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> pddt:
        """Perform round operation to specified freqency `<pddt>`.

        :param freq: `<str>` frequency to round to.
            `'D'`: Day / `'h'`: Hour / `'m'`: Minute / `'s'`: Second /
            `'ms'`: Millisecond / `'us'`: Microsecond / `'ns'`: Nanosecond

        :param ambiguous, nonexistent: Please refer to `tz_localize` method for details.
        """
        # Validate 'ambiguous' and 'nonexistent'
        self._validate_ambiguous_nonexistent(ambiguous, nonexistent)
        # Round to frequency
        return self._new(
            self._dts_index.round(self._parse_frequency(freq), ambiguous, nonexistent)
        )

    def freq_ceil(
        self,
        freq: Literal["D", "h", "m", "s", "ms", "us", "ns"],
        ambiguous: bool | Series[bool] | Literal["raise", "infer"] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> pddt:
        """Perform ceil operation to specified freqency `<pddt>`.

        :param freq: `<str>` frequency to ceil to.
            `'D'`: Day / `'h'`: Hour / `'m'`: Minute / `'s'`: Second /
            `'ms'`: Millisecond / `'us'`: Microsecond / `'ns'`: Nanosecond

        :param ambiguous, nonexistent: Please refer to `tz_localize` method for details.
        """
        # Validate 'ambiguous' and 'nonexistent'
        self._validate_ambiguous_nonexistent(ambiguous, nonexistent)
        # Round to frequency
        return self._new(
            self._dts_index.ceil(self._parse_frequency(freq), ambiguous, nonexistent)
        )

    def freq_floor(
        self,
        freq: Literal["D", "h", "m", "s", "ms", "us", "ns"],
        ambiguous: bool | Series[bool] | Literal["raise", "infer"] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> pddt:
        """Perform floor operation to specified freqency `<pddt>`.

        :param freq: `<str>` frequency to floor to.
            `'D'`: Day / `'h'`: Hour / `'m'`: Minute / `'s'`: Second /
            `'ms'`: Millisecond / `'us'`: Microsecond / `'ns'`: Nanosecond

        :param ambiguous, nonexistent: Please refer to `tz_localize` method for details.
        """
        # Validate 'ambiguous' and 'nonexistent'
        self._validate_ambiguous_nonexistent(ambiguous, nonexistent)
        # Round to frequency
        return self._new(
            self._dts_index.floor(self._parse_frequency(freq), ambiguous, nonexistent)
        )

    # Manipulate: Delta -----------------------------------------------------------------------
    def add_delta(
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
    ) -> pddt:
        """Add 'timedelta' to the current `<pddt>`.

        Equivalent to `Series + pandas.offsets.DateOffset()`.

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
        """
        # Adjust delta
        try:
            dts = self._dts + OFST_DATEOFFSET(
                # Relative delta
                years=years,
                months=months,
                weeks=weeks,
                days=days,
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                microseconds=milliseconds * 1_000 + microseconds,
            )
        except OutOfBoundsDatetime as err:
            raise errors.DatetimesOutOfBounds(
                "<{}>\n{}".format(self.__class__.__name__, err)
            ) from err

        # Generate
        return self._new(dts)

    def sub_delta(
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
    ) -> pddt:
        """Substract 'timedelta' to the current `<pddt>`.

        Equivalent to `Series - pandas.offsets.DateOffset()`.

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
        """
        # Adjust Relative delta
        try:
            dts = self._dts - OFST_DATEOFFSET(
                # Relative delta
                years=years,
                months=months,
                weeks=weeks,
                days=days,
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                microseconds=milliseconds * 1_000 + microseconds,
            )
        except OutOfBoundsDatetime as err:
            raise errors.DatetimesOutOfBounds(
                "<{}>\n{}".format(self.__class__.__name__, err)
            ) from err

        # Generate
        return self._new(dts)

    def cal_delta(
        self,
        other: pddt | Series | list | pydt | str | datetime.date | datetime.datetime,
        unit: Literal["Y", "M", "W", "D", "h", "m", "s", "ms", "us"] = "D",
        inclusive: bool = False,
    ) -> Series[int]:
        """Calcuate the `ABSOLUTE` delta between the current pddt
        and the given object based on the specified 'unit' `<Series[int]>`.

        :param other `<pddt/Series/list/pydt/str/datetime>`: The target object.
        :param unit `<str>`: The specific time unit to calculate the delta.
        :param inclusive `<bool>`: Whether to include the endpoint (result + 1). Defaults to `False`.
        :return `<Series[int]>`: The delta values.
        """
        # Parse to Series
        if isinstance(other, (datetime.date, str, pydt, TP_DATETIME64)):
            o_dt = self._parse_datetime(other, self._default)
            o_dts = Series(o_dt, index=self._get_series_index())
            o_dts: Series = self._parse_dtobj(o_dts)
        else:
            o_dts: Series = self._parse_dtobj(other)
            if len(self) != len(o_dts):
                raise errors.InvalidDatetimeObjectError(
                    "<{}>\nUnable to calculate the delta between "
                    "'other' due to mismatched data shape: {} vs {}.".format(
                        self.__class__.__name__, len(self), len(o_dts)
                    )
                )

        # Unit: year
        if unit == "Y":
            delta = FN_NP_ABS(self._dts_index.year.values - o_dts.dt.year.values)
            return self._array_to_series(delta + 1 if inclusive else delta)

        # Unit: month
        if unit == "M":
            delta = FN_NP_ABS(
                (self._dts_index.year.values - o_dts.dt.year.values) * 12
                + (self._dts_index.month.values - o_dts.dt.month.values)
            )
            return self._array_to_series(delta + 1 if inclusive else delta)

        # Unit: week
        if unit == "W":
            m_val = self._get_dts_value()
            o_val = o_dts.values
            delta = FN_NP_ABS(
                cydt.dt64array_to_days(m_val) - cydt.dt64array_to_days(o_val)
            )
            adj = TP_DATETIMEINDEX(FN_NP_MINIMUM(m_val, o_val)).weekday.values
            delta = (delta + adj) // 7
            return self._array_to_series(delta + 1 if inclusive else delta)

        # Unit: day
        if unit == "D":
            m_val = cydt.dt64array_to_days(self._get_dts_value())
            o_val = cydt.dt64array_to_days(o_dts.values)
        # Unit: hour
        elif unit == "h":
            m_val = cydt.dt64array_to_hours(self._get_dts_value())
            o_val = cydt.dt64array_to_hours(o_dts.values)
        # Unit: minute
        elif unit == "m":
            m_val = cydt.dt64array_to_minutes(self._get_dts_value())
            o_val = cydt.dt64array_to_minutes(o_dts.values)
        # Unit: second
        elif unit == "s":
            m_val = cydt.dt64array_to_seconds(self._get_dts_value())
            o_val = cydt.dt64array_to_seconds(o_dts.values)
        # Unit: millisecond
        elif unit == "ms":
            m_val = cydt.dt64array_to_milliseconds(self._get_dts_value())
            o_val = cydt.dt64array_to_milliseconds(o_dts.values)
        # Unit: microsecond
        elif unit == "us":
            m_val = cydt.dt64array_to_microseconds(self._get_dts_value())
            o_val = cydt.dt64array_to_microseconds(o_dts.values)
        # Invalid unit
        else:
            raise errors.InvalidDeltaUnitError(
                "<{}>\nInvalid delta 'unit': {}. "
                "Must be one of the following <str>: "
                "['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us'].".format(
                    self.__class__.__name__, repr(unit)
                )
            )

        # Return delta
        delta = FN_NP_ABS(m_val - o_val)
        return self._array_to_series(delta + 1 if inclusive else delta)

    # Manipulate: Replace ---------------------------------------------------------------------
    def replace(
        self,
        year: cython.int = -1,
        month: cython.int = -1,
        day: cython.int = -1,
        hour: cython.int = -1,
        minute: cython.int = -1,
        second: cython.int = -1,
        millisecond: cython.int = -1,
        microsecond: cython.int = -1,
    ) -> pddt:
        """Replacement for the current `<pddt>`.

        Similar to `datetime.replace()`.

        :param year `<int>`: The absolute year value. Defaults to `-1 (no change)`.
        :param month `<int>`: The absolute month value. Defaults to `-1 (no change)`.
        :param day `<int>`: The absolute day value. Defaults to `-1 (no change)`.
        :param hour `<int>`: The absolute hour value. Defaults to `-1 (no change)`.
        :param minute `<int>`: The absolute minute value. Defaults to `-1 (no change)`.
        :param second `<int>`: The absolute second value. Defaults to `-1 (no change)`.
        :param millisecond `<int>`: The absolute millisecond value. Defaults to `-1 (no change)`.
        :param microsecond `<int>`: The absolute microsecond value. Defaults to `-1 (no change)`.
        :return `<pddt>`: pddt after replacement.

        ### Performance Warning
        Replacement for 'year', 'month' and 'day' does not support vectorized
        operation, which could lead to performance issues when used with large datasets.
        """
        # Non-vertorzied
        if year > 0 or month > 0 or day > 0:
            if year > 0:
                year = min(year, 9_999)
            if month > 0:
                month = min(month, 12)
            if day > 0:
                day = min(day, 31)
            if hour >= 0:
                hour = min(hour, 23)
            if minute >= 0:
                minute = min(minute, 59)
            if second >= 0:
                second = min(second, 59)
            microsecond = cal_absolute_microsecond(millisecond, microsecond)
            dts = self._dts.apply(
                lambda dt: cydt.dt_replace(
                    dt, year, month, day, hour, minute, second, microsecond
                )
            )
            return self._new(dts)

        # Vertorzied (h/m/s/ms/us)
        return self.to_time(hour, minute, second, millisecond, microsecond)

    # Core methods ----------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _new(self, s: Series) -> pddt:
        """(Internal) Create a new `<pddt>`."""
        return pddt(
            s,
            default=self._default,
            day1st=self._day1st,
            year1st=self._year1st,
            utc=self._utc,
            format=self._format,
            exact=self._exact,
        )

    @cython.cfunc
    @cython.inline(True)
    def _parse_dtobj(self, dtobj: object) -> object:
        """(Internal) Parser object to `<Series[Timestamp]>`."""
        # Type `<Series>`
        if isinstance(dtobj, TP_SERIES):
            if dtobj.dtype.kind == "M":
                return dtobj.copy(True)

        # Type `<pddt>`
        elif isinstance(dtobj, pddt):
            return dtobj.dts.copy(True)

        # Type `<DatetimeIndex>`
        elif isinstance(dtobj, TP_DATETIMEINDEX):
            return TP_SERIES(dtobj)

        # Parse 'dtobj'.
        return self._parse_dtseries(dtobj)

    @cython.cfunc
    @cython.inline(True)
    def _parse_dtseries(self, dtobj: object) -> object:
        """(Internal) Parse object to `<Series[Timestamp]>`."""
        try:
            # . parse through pandas
            s = FN_PD_TODATETIME(
                dtobj,
                errors="raise",
                dayfirst=self._day1st,
                yearfirst=self._year1st,
                utc=self._utc,
                format=self._format,
                exact=self._exact,
                unit="ns",
                origin="unix",
                cache=True,
            )
        except Exception as err:
            # . fallback: parse each element through cyparser
            try:
                dts = [self._parse_datetime(dt, self._default) for dt in dtobj]
                return Series(dts).astype("<M8[us]")
            except Exception:
                raise errors.InvalidDatetimeObjectError(
                    "<{}>\nUnable to parse 'dtobj' to pandas Series of "
                    "Timestamps: {}".format(self.__class__.__name__, err)
                ) from err

        # Type `<DatetimeIndex>`
        if isinstance(s, TP_DATETIMEINDEX):
            return TP_SERIES(s)

        # Type `<Series[Timestamp]>`
        if isinstance(s, TP_SERIES):
            return s

        # Parse failed
        raise errors.InvalidDatetimeObjectError(
            "<{}>\nFailed to parse 'dtobj' to pandas Series of "
            "datetime64: Unsupported data type {}.".format(
                self.__class__.__name__, type(dtobj)
            )
        )

    @cython.cfunc
    @cython.inline(True)
    def _parse_datetime(self, dtobj: object, default: object) -> datetime.datetime:
        """(Internal) Parse datetime object to `<datetime.datetime>`."""
        try:
            return pydt(
                dtobj,
                default=default,
                day1st=self._day1st,
                year1st=self._year1st,
                ignoretz=True,
                fuzzy=True,
                cfg=None,
            )._capi_dt()
        except Exception as err:
            raise errors.InvalidDatetimeObjectError(
                "<{}>\nFailed to parse [{}] into a valid "
                "datetime.".format(self.__class__.__name__, repr(dtobj))
            ) from err

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_month(self, month: object) -> cython.uint:
        """(Internal) Prase month into numeric value `<int>`.

        Only accepts integer or string. Returns the month
        value if input is valid, and `100` if not the correct
        data type."""
        # Type of '<int>'
        if isinstance(month, int):
            mth: cython.int = month
            if not 1 <= mth <= 12:
                raise errors.InvalidMonthError(
                    "<{}>\nInvalid 'month' value [{}], must between "
                    "1(Jan)..12(Dec).".format(self.__class__.__name__, mth)
                )
            return mth

        # Type of '<str>'
        if isinstance(month, str):
            month_l = month.lower()
            val = dict_getitem(CONFIG_MONTH, month_l)
            if val == cython.NULL:
                raise errors.InvalidMonthError(
                    "<{}>\nUnable to recognize 'month' string: "
                    "{}.".format(self.__class__.__name__, repr(month))
                )
            return cython.cast(object, val)

        # Incorrect data type
        return 100

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_weekday(self, weekday: object) -> cython.uint:
        """(Internal) Prase weekday into numeric value `<int>`.

        Only accepts integer or string. Returns the weekday
        value if input is valid, and `100` if not the correct
        data type."""
        # Type of '<int>'
        if isinstance(weekday, int):
            wkd: cython.int = weekday
            if not 0 <= wkd <= 6:
                raise errors.InvalidWeekdayError(
                    "<{}>\nInvalid 'weekday' value [{}], must between "
                    "0(Monday)..6(Sunday).".format(self.__class__.__name__, wkd)
                )
            return wkd

        # Type of '<str>'
        if isinstance(weekday, str):
            weekday_l = weekday.lower()
            val = dict_getitem(CONFIG_WEEKDAY, weekday_l)
            if val == cython.NULL:
                raise errors.InvalidWeekdayError(
                    "<{}>\nUnable to recognize 'weekday' string: "
                    "{}.".format(self.__class__.__name__, repr(weekday))
                )
            return cython.cast(object, val)

        # Incorrect data type
        return 100

    @cython.cfunc
    @cython.inline(True)
    def _parse_frequency(self, freq: object) -> object:
        """(Internal) Prase frequency into string value `<str>`."""
        if freq == "m":
            return "min"
        if set_contains(UNIT_FREQUENCY, freq):
            return freq
        raise errors.InvalidFrequencyError(
            "<{}>\nInvalid 'freq': {}. "
            "Must be one of the following `<str>`: "
            "['D', 'h', 'm', 's', 'ms', 'us'].".format(
                self.__class__.__name__, repr(freq)
            )
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _get_is_unit_ns(self) -> cython.bint:
        if self._dts is None:
            self._get_dts_unit()
        return self._is_unit_ns

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _get_dts_len(self) -> cython.uint:
        """(Internal) Access the length of the Series `<int>`."""
        if self._dts_len == 0:
            self._dts_len = len(self._dts)
        return self._dts_len

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _get_dts_unit(self) -> str:
        """(Internal) Access the unit of the Series `<str>`."""
        if self._dts_unit is None:
            self._dts_unit = cydt.get_series_unit(self._dts)
            self._is_unit_ns = self._dts_unit == "ns"
        return self._dts_unit

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _get_dts_value(self) -> np.ndarray:
        """(Internal) Access the value of the Series `<ndarray[datetime64]>`."""
        if self._dts_value is None:
            self._dts_value = self._dts.values
        return self._dts_value

    @cython.cfunc
    @cython.inline(True)
    def _get_dts_naive(self) -> object:
        """(Internal) Access the timezone-naive version of the Series `<Series[Timestamp]>`."""
        if self._dts_naive is None:
            if self._dts_index.tz is None:
                return self._dts
            self._dts_naive = self._dts_index.tz_localize(None)
        return self._dts_naive

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _get_series_index(self) -> object:
        if self._series_index is None:
            self._series_index = self._dts.index
        return self._series_index

    @cython.cfunc
    @cython.inline(True)
    def _gen_timedelta(self, arg: object, unit: object) -> object:
        """(Internal) Generate the TimedeltaIndex `<TimedeltaIndex>`."""
        delta = FN_PD_TOTIMEDELTA(arg, unit)
        if not self._get_is_unit_ns() and isinstance(
            delta, (TP_TIMEDELTAINDEX, TP_SERIES)
        ):
            delta = cydt.delta64series_adjust_unit(delta, self._get_dts_unit())
        return delta

    @cython.cfunc
    @cython.inline(True)
    def _array_to_series(self, arr: object) -> object:
        """(Internal) Create the Series from a numpy array `<Series>`."""
        return TP_SERIES(arr, index=self._get_series_index())

    # Special methods: addition ---------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _convert_pddt(self, other: object) -> object:
        """(Internal) Try to convert qualified object to `<pddt>`."""
        if isinstance(other, TP_SERIES) and other.dtype.kind == "M":
            return self._new(other)
        return other  # Not qualified

    def __add__(self, other: object) -> pddt | Any:
        try:
            return self._convert_pddt(self._dts + other)
        except Exception:
            return NotImplemented

    def __radd__(self, other: object) -> pddt | Any:
        try:
            return self._convert_pddt(other + self._dts)
        except Exception:
            return NotImplemented

    # Special methods: substraction -----------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _convert_other(self, other: object) -> object:
        """(Internal) Try to convert 'other' to a valid Series `<Series>`."""
        # . type `<pddt>`
        if isinstance(other, pddt):
            return other._dts
        # . type `<Series>`
        if isinstance(other, TP_SERIES):
            return other
        # . type `<TimedeltaIndex/Timedelta/offsets>`
        if isinstance(other, (TP_TIMEDELTAINDEX, TP_TIMEDELTA, TP_BASEOFFSET)):
            return other
        # . type `<datetime>`
        if isinstance(other, (datetime.date, str, pydt, TP_DATETIME64)):
            o_dt = self._parse_datetime(other, self._default)
            o_dts = Series(o_dt, index=self._get_series_index())
            return self._parse_dtobj(o_dts)
        # . rest
        try:
            other = self._parse_dtobj(other)
        except Exception as err:
            raise errors.InvalidDatetimeObjectError(
                "<{}>\nFailed to parse 'other' to a valid Series: "
                "{}.".format(self.__class__.__name__, repr(err))
            ) from err
        return other

    def __sub__(self, other: object) -> pddt | Series | Any:
        try:
            return self._convert_pddt(self._dts - self._convert_other(other))
        except Exception:
            return NotImplemented

    def __rsub__(self, other: object) -> pddt | Series | Any:
        try:
            return self._convert_pddt(self._convert_other(other) - self._dts)
        except Exception:
            return NotImplemented

    # Special methods - comparison ------------------------------------------------------------
    def __eq__(self, other: object) -> Series[bool]:
        try:
            return self._dts == self._convert_other(other)
        except Exception:
            return NotImplemented

    def __ne__(self, other: object) -> Series[bool]:
        try:
            return self._dts != self._convert_other(other)
        except Exception:
            return NotImplemented

    def __gt__(self, other: object) -> Series[bool]:
        try:
            return self._dts > self._convert_other(other)
        except Exception:
            return NotImplemented

    def __ge__(self, other: object) -> Series[bool]:
        try:
            return self._dts >= self._convert_other(other)
        except Exception:
            return NotImplemented

    def __lt__(self, other: object) -> Series[bool]:
        try:
            return self._dts < self._convert_other(other)
        except Exception:
            return NotImplemented

    def __le__(self, other: object) -> Series[bool]:
        try:
            return self._dts <= self._convert_other(other)
        except Exception:
            return NotImplemented

    def equals(self, other: object) -> bool:
        """Test whether two objects contain the same elements.
        (Equivalent to `pandas.Series.equal()` method).

        Support comparison between `pddt` and `pandas.Series`."""
        try:
            return self._dts.equals(self._convert_other(other))
        except Exception:
            return False

    # Special methods - copy ------------------------------------------------------------------
    def copy(self) -> pddt:
        "Make a (deep)copy of the `<pddt>`."
        return self._new(self._dts)

    def __copy__(self, *args, **kwargs) -> pddt:
        return self._new(self._dts)

    def __deepcopy__(self, *args, **kwargs) -> pddt:
        return self._new(self._dts)

    # Special methods: represent --------------------------------------------------------------
    def __repr__(self) -> str:
        return "<%s (\n%s)>" % (self.__class__.__name__, self._dts.__repr__())

    def __str__(self) -> str:
        return self._dts.__repr__()

    def __len__(self) -> int:
        return self._get_dts_len()

    def __contains__(self, key) -> bool:
        return self._dts.__contains__(key)

    def __getitem__(self, key) -> Timestamp:
        return self._dts.__getitem__(key)

    def __iter__(self) -> Iterator[Timestamp]:
        return self._dts.__iter__()

    def __array__(self) -> np.ndarray:
        return self._dts.__array__()

    def __del__(self):
        self._dts = None
        self._dts_index = None
        self._dts_value = None
        self._dts_naive = None
        self._series_index = None
