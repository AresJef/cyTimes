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
from cython.cimports.cytimes import cydatetime as cydt, typeref  # type: ignore
from cython.cimports.cytimes.cytimedelta import combine_ms_us  # type: ignore
from cython.cimports.cytimes.pydatetime import pydt, access_pydt, TIMEZONE_AVAILABLE  # type: ignore
from cython.cimports.cytimes.cyparser import Config, Parser, CONFIG_MONTH, CONFIG_WEEKDAY  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
from typing import Literal, Iterator
import datetime, numpy as np
from pandas.errors import OutOfBoundsDatetime
from pandas import to_datetime, to_timedelta, date_range
from pandas import Series, DataFrame, Index, Timestamp, TimedeltaIndex
from cytimes.cyparser import Config
from cytimes.pydatetime import pydt
from cytimes import cydatetime as cydt, typeref, errors

__all__ = ["pddt"]

# Constants -----------------------------------------------------------------------------------
# . unit
UNIT_FREQUENCY: set[str] = {"D", "h", "m", "s", "ms", "us", "ns"}
# . function
FN_PD_DATERANGE: object = date_range
FN_PD_TODATETIME: object = to_datetime
FN_PD_TOTIMEDELTA: object = to_timedelta
FN_NP_ABS: object = np.abs
FN_NP_FULL: object = np.full
FN_NP_WHERE: object = np.where
FN_NP_MIN: object = np.minimum


# pddt (Pandas Datetime) ----------------------------------------------------------------------
@cython.cclass
class pddt:
    """Represents the pddt (Pandas Datetime) that makes
    working with Series[Timestamp] easier.
    """

    # Value
    _dt: Series
    # Config
    _cfg: Config
    _day1st: object
    _year1st: object
    _utc: object
    _format: str
    _exact: object
    _default: object

    @classmethod
    def now(
        cls,
        size: int,
        tz: str | datetime.tzinfo | None = None,
        name: str | None = None,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
    ) -> pddt:
        """(Class method) Create pddt with the current time `<'pddt'>`.

        Similar to `datetime.now(tz)`.

        :param size `<'int'>`: The size of the array.
        :param tz `<'str/datetime.tzinfo'>`: The target timezone of the current time. Defaults to `None`.
        :param name `<'str'>`: The name of the array. Defaults to `None`.
        :param unit `<'str'>`: The desired time resolution. Defaults to `None`.
        """
        # Without timezone
        if tz is None:
            dt = cydt.gen_dt_now()
        # With timezone
        else:
            dt = cydt.gen_dt_now_tz(parse_tzinfo(tz))  # type: ignore
        # Generate pddt
        return cls(Series([dt] * size, name=name), unit=unit, copy=False)

    @classmethod
    def from_dtobj(
        cls,
        size: int,
        dtobj: object,
        default: object | None = None,
        day1st: bool | None = None,
        year1st: bool | None = None,
        cfg: Config | None = None,
        name: str | None = None,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
    ) -> pddt:
        """(Class method) Create pddt from datetime objects `<'pddt'>`.

        :param size `<'int'>`: The size of the array.
        :param dtobj `<'object'>`: The datetime object to convert to pddt.
        - Supported data types:
        1. `<'str'>`: The datetime string, e.g. "2021-01-01 12:00:00" or "Jan 12, 2023".
        2. `<'datetime.datetime'>`: Python native datetime or subclass.
        3. `<'datetime.date'>`: Python native date or subclass.
        4. `<'pydt'>`: Another pydt instance.
        6. `<'numpy.datetime64'>`: Numpy datetime64 instance.

        ### Parse 'dtobj' (Only applicable when 'dtobj' is type of `<'str'>`).
        :param default `<'object'>`: The default to fill-in missing datetime elements. Defaults to `None`.
        - `<'date'>`: If parser failed to extract Y/M/D values from the string,
           the give 'date' will be used to fill-in the missing Y/M/D values.
        - `<'datetime'>`: If parser failed to extract datetime elements from
           the string, the given 'datetime' will be used to fill-in the
           missing Y/M/D & H/M/S.f values.
        - `None`: raise error if Y/M/D values are missing.

        :param day1st `<'bool'>`: Whether to interpret first ambiguous date values as day. Defaults to `None`.
        :param year1st `<'bool'>`: Whether to interpret first the ambiguous date value as year. Defaults to `None`.
        - Both the 'day1st' & 'year1st' arguments works together to determine how
          to interpret ambiguous Y/M/D values. If not provided (set to `None`),
          defaults to the 'day1st' & 'year1st' settings of the Parser `<'Config'>`.
        - For more information, please refer to the `cytimes.Parser.parse()` method.

        :param cfg `<'Config'>`: The configurations for the Parser. Defaults to `None`.

        ### Arguments for the array.
        :param name `<'str'>`: The name of the array. Defaults to `None`.
        :param unit `<'str'>`: The desired time resolution. Defaults to `None`.
        """
        # <'str'> datetime string to be parsed
        if isinstance(dtobj, str):
            try:
                dt = Parser(cfg).parse(dtobj, default, day1st, year1st, False, True)
            except Exception as err:
                raise errors.InvalidDatetimeObjectError("<'pddt'> %s." % (err)) from err
        # <'datetime.date'> subclass
        elif datetime.PyDate_Check(dtobj):
            if datetime.PyDateTime_Check(dtobj):
                dt = dtobj
            else:
                dt = cydt.dt_fr_date(dtobj, None, 0)
        # <'pydt'>
        elif isinstance(dtobj, pydt):
            dt = access_pydt(dtobj)
        # <'numpy.datetime64'>
        elif np.is_datetime64_object(dtobj):
            dt = dtobj
        # Invalid
        else:
            raise errors.InvalidDatetimeObjectError(
                "<'pddt'>\nUnsupported 'dtobj' type: %s %s."
                % (type(dtobj), repr(dtobj))
            )
        # Generate pddt
        return cls(Series([dt] * size, name=name), unit=unit, copy=False)

    @classmethod
    def from_datetime(
        cls,
        size: int,
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        microsecond: int = 0,
        nanosecond: int = 0,
        tz: str | datetime.tzinfo | None = None,
        name: str | None = None,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
    ) -> pddt:
        """(Class method) Create pddt from datetime values `<'pddt'>`.

        :param size `<'int'>`: The size of the array.
        :param year `<'int'>`: The year of the datetime.
        :param month `<'int'>`: The month of the datetime.
        :param day `<'int'>`: The day of the datetime.
        :param hour `<'int'>`: The hour of the datetime. Defaults to `0`.
        :param minute `<'int'>`: The minute of the datetime. Defaults to `0`.
        :param second `<'int'>`: The second of the datetime. Defaults to `0`.
        :param microsecond `<'int'>`: The microsecond of the datetime. Defaults to `0`.
        :param nanosecond `<'int'>`: The nanosecond of the datetime. Defaults to `0`.
        :param tz `<'str/datetime.tzinfo/None'>`: The timezone of the datetime. Defaults to `None`.
        :param name `<'str'>`: The name of the array. Defaults to `None`.
        :param unit `<'str'>`: The desired time resolution. Defaults to `None`.
        """
        # fmt: off
        # With nanosecond
        if nanosecond:
            try:
                dt = Timestamp(
                    year=year, month=month, day=day, hour=hour, minute=minute, 
                    second=second, microsecond=microsecond, nanosecond=nanosecond,
                    tz=parse_tzinfo(tz) ) # type: ignore
            except OutOfBoundsDatetime:
                dt = datetime.datetime_new(
                    year, month, day, hour, minute, 
                    second, microsecond, parse_tzinfo(tz), 0 # type: ignore
                ) 
        # Without nanosecond
        else:
            dt = datetime.datetime_new(
                year, month, day, hour, minute, 
                second, microsecond, parse_tzinfo(tz), 0 # type: ignore
            ) 
        # fmt: on
        # Generate pddt
        return cls(Series([dt] * size, name=name), unit=unit, copy=False)

    @classmethod
    def from_range(
        cls,
        start: object = None,
        end: object = None,
        freq: Literal["Y", "M", "W", "D", "B" "h", "m", "s", "ms", "us", "ns"] = "D",
        periods: int = None,
        tz: str | datetime.tzinfo | None = None,
        inclusive: Literal["left", "right", "both", "neither"] = "both",
        normalize: bool = False,
        name: str | None = None,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
    ) -> pddt:
        """(Class method) Create pddt from a fixed frequency `<'pddt'>`.

        Returns the range of equally spaced time points (where the difference between any
        two adjacent points is specified by the given frequency) such that they all
        satisfy `start <[=] x <[=] end`, where the first one and the last one are, resp.,
        the first and last time points in that range that fall on the boundary of ``freq``
        (if given as a frequency string) or that are valid for 'freq'. For more information
        please refer to `pandas.date_range()` documentation.

        :param start `<'object'>`: The left bound of the range. Defaults to `None`.
        :param end `<'object'>`: The right bound of the range. Defaults to `None`.
        :param freq `<'str'>`: The frequency for the range. Defaults to `'D'`.
        :param periods `<'int'>`: The number of periods to generate. Defaults to `None`.
        :param tz `<'str/datetime.tzinfo/None'>`: The timezone of the range. Defaults to `None`.
        :param inclusive `<'str'>`: Include boundaries, whether to set each bound as closed or open. Defaults to `'both'`.
        :param normalize `<'bool'>`: Normalize 'start/end' to midnight before generating date range. Defaults to `False`.
        :param name `<'str'>`: The name of the array. Defaults to `None`.
        :param unit `<'str'>`: The desired time resolution. Defaults to `None`.
        """
        # fmt: off
        return cls(FN_PD_DATERANGE(
            start=start, end=end, periods=periods, freq="min" if freq == "m" else freq, tz=tz, 
            normalize=normalize, name=name, inclusive=inclusive, unit=unit),
            copy=False,
        )
        # fmt: on

    def __init__(
        self,
        dtsobj: object,
        default: object | None = None,
        day1st: bool | None = None,
        year1st: bool | None = None,
        utc: bool = False,
        format: str = None,
        exact: bool = True,
        cfg: Config | None = None,
        name: str | None = None,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        copy: bool = True,
    ) -> None:
        """The pddt (Pandas Datetime) that makes working with Series[Timestamp] easier.

        ### Datetimes Object
        :param dtsobj `<'object'>`: The datetimes object to convert to pddt.
        - Supported data types:
            1. Sequence of <'datetime'> or datetime <'str'>
            2. `<'Series'>` of timestamp or datetime object
            3. `<'DatetimeIndex'>`
            4. `<'ndarray'> of datetime64 or datetime object
            5. `<'pddt'>`

        ### Parser for 'dtsobj'. (Only applicable when 'dtsobj' is sequence of datetime <'str'>).
        :param default `<'object'>`: The default to fill-in missing datetime elements. Defaults to `None`.
        - `<'date'>`: If parser failed to extract Y/M/D values from the string,
           the give 'date' will be used to fill-in the missing Y/M/D values.
        - `<'datetime'>`: If parser failed to extract datetime elements from
           the string, the given 'datetime' will be used to fill-in the
           missing Y/M/D & H/M/S.f values.
        - `None`: raise error if Y/M/D values are missing.

        :param day1st `<'bool'>`: Whether to interpret first ambiguous date values as day. Defaults to `None`.
        :param year1st `<'bool'>`: Whether to interpret first the ambiguous date value as year. Defaults to `None`.
        - Both the 'day1st' & 'year1st' arguments works together to determine how
          to interpret ambiguous Y/M/D values. If not provided (set to `None`),
          defaults to the 'day1st' & 'year1st' settings of the Parser `<'Config'>`.
        - For more information, please refer to the `cytimes.Parser.parse()` method.

        :param utc `<'bool'>`: Control timezone-related parsing, localization and conversion. Defaults to `False`.
        - If `True`, `ALWAYS` parse to tinzome-aware UTC-localized `<'Timestamp'>`. Timezone-naive inputs
          are `LOCALIZED` as UTC, while timezone-aware inputs are `CONVERTED` to UTC.
        - If `False`, Timezone-naive inputs remain naive, while timezone-aware ones will keep their timezone.
        - For more information, please refer to `pandas.to_datetime()` documentation.
          <https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html>

        :param format `<str>`: The strftime to parse the 'dtsobj' with. Defaults to `None`.
        - strftime format (e.g. "%d/%m/%Y"): Note that "%f" will parse all the way to nanoseconds.
          For more infomation, please refer to <https://docs.python.org/3/library/datetime.html
          #strftime-and-strptime-behavior>.
        - "ISO8601": Parse any `ISO8601` time string (not necessarily in exactly the same format).
          For more infomation, please refer to <https://en.wikipedia.org/wiki/ISO_8601>.
        - "mixed": Infer the format for each element individually. This is risky, and should probably
          use it along with `dayfirst`.

        :param exact: `<'bool'>` Whether to parse with the exact provided 'format'. Defaults to `True`.
        - If `True`, perform an exact 'format' match (Only applicable when 'format' is provided).
        - If `False`, allow the 'format' to match anywhere in the string.
        - Can `NOT` be used alongside `format='ISO8601'` or `format='mixed'`.

        :param cfg `<'Config'>`: The configurations for the Parser. Defaults to `None`.

        ### Arguments for the array.
        :param name `<'str'>`: The name of the array. Defaults to `None`.
        :param unit `<'str'>`: The desired time resolution. Defaults to `None`.
        :param copy `<'bool'>`: Whether to copy the data. Defaults to `True`.
        - If `copy=False` and 'name' is given, the original series name will be replaced.
        """
        # Set config
        self._cfg = cfg
        self._day1st = (
            (False if self._cfg is None else self._cfg.day1st)
            if day1st is None
            else bool(day1st)
        )
        self._year1st = (
            (False if self._cfg is None else self._cfg.year1st)
            if year1st is None
            else bool(year1st)
        )
        self._utc = bool(utc)
        self._format = format
        self._exact = bool(exact)
        # Parse default
        if default is not None:
            default = self._parse_dtobj(default, None, True)
        self._default = default
        # Parse pddt value
        self._dt = self._parse_dtsobj(dtsobj, default, name, unit, bool(copy))

    # Access ----------------------------------------------------------------------------------
    @property
    def name(self) -> str | None:
        """Access the 'name' of the Series `<str/None>`."""
        return self._dt.name

    @name.setter
    def name(self, value: str | None) -> None:
        self._dt.name = value

    @property
    def index(self) -> Index:
        """Access the 'index' of the Series `<Index>`."""
        return self._dt.index

    @property
    def unit(self) -> str:
        """Access the time unit of the Series `<'str'>`.
        Such as 'ns', 'us', 'ms', 's'."""
        return self._c_unit()

    @property
    def year(self) -> Series[int]:
        """Access the year `<'Series[int]'>`."""
        return self._dt.dt.year

    @property
    def quarter(self) -> Series[int]:
        """Access the quarter `<'Series[int]'>`."""
        return self._dt.dt.quarter

    @property
    def month(self) -> Series[int]:
        """Access the month `<'Series[int]'>`."""
        return self._dt.dt.month

    @property
    def day(self) -> Series[int]:
        """Access the day `<'Series[int]'>`."""
        return self._dt.dt.day

    @property
    def hour(self) -> Series[int]:
        """Access the hour `<'Series[int]'>`."""
        return self._dt.dt.hour

    @property
    def minute(self) -> Series[int]:
        """Access the minute `<'Series[int]'>`."""
        return self._dt.dt.minute

    @property
    def second(self) -> Series[int]:
        """Access the second `<'Series[int]'>`."""
        return self._dt.dt.second

    @property
    def microsecond(self) -> Series[int]:
        """Access the microsecond `<'Series[int]'>`."""
        return self._dt.dt.microsecond

    @property
    def nanosecond(self) -> Series[int]:
        """Access the nanosecond `<'Series[int]'>`."""
        return self._dt.dt.nanosecond

    @property
    def tzinfo(self) -> datetime.tzinfo | None:
        """Access the timezone `<'datetime.tzinfo/None'>`."""
        return self._dt.dt.tz

    @property
    def tzname(self) -> str | None:
        """Access the timezone name `<'str/None'>`."""
        return cydt.tzinfo_name(self._dt.dt.tz, None)

    @property
    def dt(self) -> Series[Timestamp]:
        """Access as `<'Series[Timestamp]'>`."""
        return self._dt

    @property
    def dt_str(self) -> Series[str]:
        """Access in string format `<'Series[str]'>`.
        e.g: "2021-01-01 12:00:00.000001"""
        return self._dt.dt.strftime("%Y-%m-%d %H:%M:%S.%f")

    @property
    def dt_strtz(self) -> Series[str]:
        """Access in string format with timezone `<'Series[str]'>`.
        e.g: "2021-01-01 12:00:00.000001+00:00"""
        return self._dt.dt.strftime("%Y-%m-%d %H:%M:%S.%f%z")

    @property
    def dt_iso(self) -> str:
        """Access in ISO format `<'Series[str]'>`.
        e.g: "2021-01-01T12:00:00.000001"""
        return self._dt.dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

    @property
    def dt_isotz(self) -> str:
        """Access in ISO format with timezone `<'Series[str]'>`.
        e.g: "2021-01-01T12:00:00.000001+00:00"""
        return self._dt.dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")

    @property
    def dt64(self) -> np.ndarray[np.datetime64]:
        """Access as `<'ndarray[datetime64]'>`."""
        return self._dt.values

    @property
    def dtpy(self) -> np.ndarray[np.datetime64]:
        """Access as `<'ndarray[datetime.datetime]'>`."""
        return self._dt.dt.to_pydatetime()

    @property
    def pydt(self) -> np.ndarray[pydt]:
        """Access as `<'ndarray[pydt]'>`."""
        # Config for pydt
        default = self._default
        day1st = self._day1st
        year1st = self._year1st
        cfg = self._cfg
        # Generate ndarray
        dim: np.npy_intp[1] = [len(self._dt)]
        arr: np.ndarray = np.PyArray_EMPTY(1, dim, np.NPY_TYPES.NPY_OBJECT, 0)
        idx: cython.Py_ssize_t = 0
        for dt in self._dt:
            ndarray_setitem_1d(  # type: ignore
                arr, idx, pydt(dt, default, day1st, year1st, False, True, cfg)
            )
            idx += 1
        return arr

    @property
    def date(self) -> Series[datetime.date]:
        """Access as `<'Series[datetime.date]'>`."""
        return self._dt.dt.date

    @property
    def date_iso(self) -> Series[str]:
        """Access dates in ISO format `<'Series[str]'>`."""
        return self._dt.dt.strftime("%Y-%m-%d")

    @property
    def time(self) -> Series[datetime.time]:
        """Access as `<'Series[datetime.time]'>`."""
        return self._dt.dt.time

    @property
    def timetz(self) -> Series[datetime.time]:
        """Access as `<'Series[datetime.time]'>` with timezone."""
        return self._dt.dt.timetz

    @property
    def time_iso(self) -> Series[str]:
        """Access times in ISO format `<'Series[str]'>`."""
        return self._dt.dt.strftime("%H:%M:%S.%f")

    @property
    def ordinal(self) -> Series[int]:
        """Access in ordinal values `<'Series[int]'>`."""
        if self._dt.dt.tz is not None:
            return cydt.dt64series_to_ordinals(self._dt.dt.tz_localize(None))
        else:
            return cydt.dt64series_to_ordinals(self._dt)

    @property
    def seconds(self) -> Series[int]:
        """Access in total seconds since EPOCH, ignoring
        the timezone (if exists) `<'Series[int]'>`.

        ### Notice
        This should `NOT` be treated as timestamp.
        """
        if self._dt.dt.tz is not None:
            return cydt.dt64series_to_seconds(self._dt.dt.tz_localize(None))
        else:
            return cydt.dt64series_to_seconds(self._dt)

    @property
    def seconds_utc(self) -> Series[int]:
        """Access in total seconds since EPOCH `<'Series[int]'>`.
        - If `timezone-aware`, return total seconds in UTC.
        - If `timezone-naive`, requivalent to `pddt.seconds`.

        ### Notice
        This should `NOT` be treated as timestamp.
        """
        return cydt.dt64series_to_seconds(self._dt)

    @property
    def microseconds(self) -> Series[int]:
        """Access in total microseconds since EPOCH, ignoring
        the timezone (if exists) `<'Series[int]'>`."""
        if self._dt.dt.tz is not None:
            return cydt.dt64series_to_microseconds(self._dt.dt.tz_localize(None))
        else:
            return cydt.dt64series_to_microseconds(self._dt)

    @property
    def microseconds_utc(self) -> Series[int]:
        """Access in total microseconds since EPOCH `<'Series[int]'>`.
        - If `timezone-aware`, return total microseconds in UTC.
        - If `timezone-naive`, requivalent to `pddt.microseconds`.
        """
        return cydt.dt64series_to_microseconds(self._dt)

    @property
    def timestamp(self) -> Series[float]:
        """Access in timestamp values `<'Series[float]'>`."""
        return cydt.dt64series_to_timestamps(self._dt)

    # Year ------------------------------------------------------------------------------------
    def is_leapyear(self) -> Series[bool]:
        """Whether the dates are leap years `<'Series[bool]'>`."""
        return self._dt.dt.is_leap_year

    def leap_bt_years(self, year: int) -> Series[int]:
        """Calculate the number of leap years between the dates
        and the given 'year' `<'Series[int]'>`."""
        y1_arr = self._dt.dt.year.values
        y2_arr = y1_arr.copy()
        y1_arr = FN_NP_WHERE(y1_arr <= year, y1_arr, year) - 1
        y2_arr = FN_NP_WHERE(y2_arr > year, y2_arr, year) - 1
        arr = (
            (y2_arr // 4 - y1_arr // 4)
            - (y2_arr // 100 - y1_arr // 100)
            + (y2_arr // 400 - y1_arr // 400)
        )
        return self._arr_to_series(arr)

    @property
    def days_in_year(self) -> Series[int]:
        """Get the maximum number of days in the year.
        Expect 365 or 366 (leapyear) `<'Series[int]'>`."""
        return self._arr_to_series(self._dt.dt.is_leap_year.values + 365)

    @property
    def days_bf_year(self) -> Series[int]:
        """Get the number of days betweem the 1st day of 1AD and
        the 1st day of the year of the dates `<'Series[int]'>`."""
        return self.ordinal - self._dt.dt.day_of_year

    @property
    def days_of_year(self) -> Series[int]:
        """Get the number of days between the 1st day
        of the year and the dates `<'Series[int]'>`."""
        return self._dt.dt.day_of_year

    def is_year(self, year: int) -> Series[bool]:
        """Whether the dates are in the given 'year' `<'Series[bool]'>`."""
        return self._arr_to_series(self._dt.dt.year.values == year)

    def is_year_1st(self) -> Series[bool]:
        """Whether the dates are the 1st day of the year `<'Series[bool]'>`."""
        return self._dt.dt.is_year_start

    def is_year_lst(self) -> Series[bool]:
        """Whether the dates are the last day of the year `<'Series[bool]'>`."""
        return self._dt.dt.is_year_end

    @cython.ccall
    def to_year_1st(self) -> pddt:
        """Go to the 1st day of the year `<'pddt'>`."""
        try:
            dt: Series = (
                self._dt + typeref.OFST_YEAREND(0) - typeref.OFST_YEARBEGIN(1, month=1)
            )
        except Exception as err:
            self._raise_error(err)
        return self._new(dt, None, None, False)

    @cython.ccall
    def to_year_lst(self) -> pddt:
        """Go to the 1st day of the year `<'pddt'>`."""
        try:
            dt: Series = self._dt + typeref.OFST_YEAREND(0)
        except Exception as err:
            self._raise_error(err)
        return self._new(dt, None, None, False)

    @cython.ccall
    def to_curr_year(
        self,
        month: cython.int | str | None = None,
        day: cython.int = -1,
    ) -> pddt:
        """Go to specific 'month' and 'day' of the current year `<'pddt'>`."""
        # Invalid 'month' value
        mth: cython.int = self._parse_month(month)
        if mth == -1:
            return self.to_curr_month(day)  # exit
        # Invalid 'day' value
        if day < 1:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_YEAREND(0)
                    - typeref.OFST_YEARBEGIN(1, month=mth)
                )
                dt += self._arr_to_tdindex(
                    FN_NP_MIN(dt.dt.days_in_month.values, self._dt.dt.day.values) - 1,
                    "D",
                )
            except Exception as err:
                self._raise_error(err)
        # First day of the month
        elif day == 1:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_YEAREND(0)
                    - typeref.OFST_YEARBEGIN(1, month=mth)
                )
            except Exception as err:
                self._raise_error(err)
        # Days before 28
        elif day < 29:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_YEAREND(0)
                    - typeref.OFST_YEARBEGIN(1, month=mth)
                    + typeref.OFST_DAY(day - 1)
                )
            except Exception as err:
                self._raise_error(err)
        # Days before 31
        elif day < 31:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_YEAREND(0)
                    - typeref.OFST_YEARBEGIN(1, month=mth)
                )
                dt += self._arr_to_tdindex(
                    FN_NP_MIN(dt.dt.days_in_month.values, day) - 1, "D"
                )
            except Exception as err:
                self._raise_error(err)
        # Last day of the month
        else:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_YEAREND(0)
                    - typeref.OFST_YEARBEGIN(1, month=mth)
                    + typeref.OFST_MONTHEND(0)
                )
            except Exception as err:
                self._raise_error(err)
        # Generate new pddt
        return self._new(dt, None, None, False)  # exit

    @cython.ccall
    def to_prev_year(
        self,
        month: cython.int | str | None = None,
        day: cython.int = -1,
    ) -> pddt:
        """Go to specific 'month' and 'day' of the previous year `<'pddt'>`."""
        return self.to_year(-1, month, day)

    @cython.ccall
    def to_next_year(
        self,
        month: cython.int | str | None = None,
        day: cython.int = -1,
    ) -> pddt:
        """Go to specific 'month' and 'day' of the next year `<'pddt'>`."""
        return self.to_year(1, month, day)

    @cython.ccall
    def to_year(
        self,
        offset: cython.int = 0,
        month: cython.int | str | None = None,
        day: cython.int = -1,
    ) -> pddt:
        """Go to specific 'month' and 'day' of the current year (+/-) 'offset' `<'pddt'>`."""
        # No 'offset' adjustment
        if offset == 0:
            return self.to_curr_year(month, day)  # exit
        # Invalid 'month' value
        mth: cython.int = self._parse_month(month)
        if mth == -1:
            try:
                dt: Series = self._dt + typeref.OFST_DATEOFFSET(years=offset)
            except Exception as err:
                self._raise_error(err)
            return self._new(dt, None, None, False).to_curr_month(day)  # exit
        # Invalid 'day' value
        if day < 1:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_YEAREND(0)
                    + typeref.OFST_DATEOFFSET(years=offset, months=mth - 12)
                )
                dt += self._arr_to_tdindex(
                    FN_NP_MIN(dt.dt.days_in_month.values, self._dt.dt.day.values)
                    - dt.dt.day.values,
                    "D",
                )
            except Exception as err:
                self._raise_error(err)
        # First day of the month
        elif day == 1:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_YEAREND(0)
                    - typeref.OFST_YEARBEGIN(1, month=mth)
                    + typeref.OFST_DATEOFFSET(years=offset)
                )
            except Exception as err:
                self._raise_error(err)
        # Days before 28
        elif day < 29:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_YEAREND(0)
                    - typeref.OFST_YEARBEGIN(1, month=mth)
                    + typeref.OFST_DATEOFFSET(years=offset, days=day - 1)
                )
            except Exception as err:
                self._raise_error(err)
        # Days before 31st
        elif day < 31:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_YEAREND(0)
                    + typeref.OFST_DATEOFFSET(years=offset, months=mth - 12)
                )
                dt += self._arr_to_tdindex(
                    FN_NP_MIN(dt.dt.days_in_month.values, day) - dt.dt.day.values, "D"
                )
            except Exception as err:
                self._raise_error(err)
        # Last day of the month
        else:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_YEAREND(0)
                    + typeref.OFST_DATEOFFSET(years=offset, months=mth - 12)
                    + typeref.OFST_MONTHEND(0)
                )
            except Exception as err:
                self._raise_error(err)
        # Generate new pddt
        return self._new(dt, None, None, False)  # exit

    # Quarter ---------------------------------------------------------------------------------
    @property
    def days_in_quarter(self) -> Series[int]:
        """Get the maximum number of days in the quarter `<'Series[int]'>`."""
        quarter = self._dt.dt.quarter.values
        return self._arr_to_series(
            cydt.DAYS_BR_QUARTER_NDARRAY[quarter]
            - cydt.DAYS_BR_QUARTER_NDARRAY[quarter - 1]
            + self._dt.dt.is_leap_year.values
        )

    @property
    def days_bf_quarter(self) -> Series[int]:
        """Get the number of days between the 1st day of the year and
        the 1st day of the quarter `<'Series[int]'>`."""
        quarter = self._dt.dt.quarter.values
        return self._arr_to_series(
            cydt.DAYS_BR_QUARTER_NDARRAY[quarter - 1]
            + self._dt.dt.is_leap_year.values * (quarter > 1)
        )

    @property
    def days_of_quarter(self) -> Series[int]:
        """Get the number of days between the 1st day of the quarter
        and the dates `<'Series[int]'>`."""
        quarter = self._dt.dt.quarter.values
        return self._arr_to_series(
            self._dt.dt.day_of_year.values
            - cydt.DAYS_BR_QUARTER_NDARRAY[quarter - 1]
            - self._dt.dt.is_leap_year.values * (quarter > 1)
        )

    @property
    def quarter_1st_month(self) -> Series[int]:
        """Get the first month of the quarter.
        Expect 1, 4, 7, 10 `<'Series[int]'>`."""
        return self._arr_to_series(self._dt.dt.quarter.values * 3 - 2)

    @property
    def quarter_lst_month(self) -> Series[int]:
        """Get the last month of the quarter.
        Expect 3, 6, 9, 12 `<'Series[int]'>`."""
        return self._arr_to_series(self._dt.dt.quarter.values * 3)

    def is_quarter(self, quarter: int) -> Series[bool]:
        """Whether the dates are in the given 'quarter' `<'Series[bool]'>`."""
        return self._arr_to_series(self._dt.dt.quarter.values == quarter)

    def is_quarter_1st(self) -> Series[bool]:
        """Whether the dates are the 1st day of the quarter `<'Series[bool]'>`."""
        return self._dt.dt.is_quarter_start

    def is_quarter_lst(self) -> Series[bool]:
        """Whether the dates are the last day of the quarter `<'Series[bool]'>`."""
        return self._dt.dt.is_quarter_end

    @cython.ccall
    def to_quarter_1st(self) -> pddt:
        """Go to the 1st day of the quarter `<'pddt'>`."""
        try:
            dt: Series = (
                self._dt
                + typeref.OFST_QUARTEREND(0)
                - typeref.OFST_QUARTERBEGIN(1, startingMonth=1)
            )
        except Exception as err:
            self._raise_error(err)
        return self._new(dt, None, None, False)

    @cython.ccall
    def to_quarter_lst(self) -> pddt:
        """Go to the last day of the quarter `<'pddt'>`."""
        try:
            dt: Series = self._dt + typeref.OFST_QUARTEREND(0)
        except Exception as err:
            self._raise_error(err)
        return self._new(dt, None, None, False)

    @cython.ccall
    def to_curr_quarter(self, month: cython.int = -1, day: cython.int = -1) -> pddt:
        """Go to specific 'month (of the quarter [1..3])'
        and 'day' of the current quarter `<'pddt'>`."""
        # Invalid 'month' value
        if month < 1:
            return self.to_curr_month(day)  # exit
        month = month % 3 or 3
        # Invalid 'day' value
        if day < 1:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_QUARTEREND(0)
                    - typeref.OFST_QUARTERBEGIN(1, startingMonth=month)
                )
                dt += self._arr_to_tdindex(
                    FN_NP_MIN(dt.dt.days_in_month.values, self._dt.dt.day.values) - 1,
                    "D",
                )
            except Exception as err:
                self._raise_error(err)
        # First day of the month
        elif day == 1:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_QUARTEREND(0)
                    - typeref.OFST_QUARTERBEGIN(1, startingMonth=month)
                )
            except Exception as err:
                self._raise_error(err)
        # Days before 28
        elif day < 29:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_QUARTEREND(0)
                    - typeref.OFST_QUARTERBEGIN(1, startingMonth=month)
                    + typeref.OFST_DAY(day - 1)
                )
            except Exception as err:
                self._raise_error(err)
        # Days before 31
        elif day < 31:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_QUARTEREND(0)
                    - typeref.OFST_QUARTERBEGIN(1, startingMonth=month)
                )
                dt += self._arr_to_tdindex(
                    FN_NP_MIN(dt.dt.days_in_month.values, day) - 1, "D"
                )
            except Exception as err:
                self._raise_error(err)
        # Last day of the month
        else:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_QUARTEREND(0)
                    - typeref.OFST_QUARTERBEGIN(1, startingMonth=month)
                    + typeref.OFST_MONTHEND(0)
                )
            except Exception as err:
                self._raise_error(err)
        # Generate new pddt
        return self._new(dt, None, None, False)  # exit

    @cython.ccall
    def to_prev_quarter(self, month: cython.int = -1, day: cython.int = -1) -> pddt:
        """Go to specific 'month (of the quarter [1..3])' and
        'day' of the previous quarter `<'pddt'>`."""
        return self.to_quarter(-1, month, day)

    @cython.ccall
    def to_next_quarter(self, month: cython.int = -1, day: cython.int = -1) -> pddt:
        """Go to specific 'month (of the quarter [1..3])' and
        'day' of the next quarter `<'pddt'>`."""
        return self.to_quarter(1, month, day)

    @cython.ccall
    def to_quarter(
        self,
        offset: cython.int = 0,
        month: cython.int = -1,
        day: cython.int = -1,
    ) -> pddt:
        """Go to specific 'month (of the quarter [1..3])' and
        'day' of the current quarter (+/-) 'offset' `<'pddt'>`."""
        # No 'offset' adjustment
        if offset == 0:
            return self.to_curr_quarter(month, day)  # exit
        # Invalid 'month' value
        if month < 1:
            try:
                dt: Series = self._dt + typeref.OFST_DATEOFFSET(months=offset * 3)
            except Exception as err:
                self._raise_error(err)
            return self._new(dt, None, None, False).to_curr_month(day)  # exit
        month = month % 3 or 3
        # Invalid 'day' value
        if day < 1:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_QUARTEREND(0)
                    + typeref.OFST_DATEOFFSET(months=offset * 3 + month - 3)
                )
                dt += self._arr_to_tdindex(
                    FN_NP_MIN(dt.dt.days_in_month.values, self._dt.dt.day.values)
                    - dt.dt.day.values,
                    "D",
                )
            except Exception as err:
                self._raise_error(err)
        # First day of the month
        elif day == 1:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_QUARTEREND(0)
                    - typeref.OFST_QUARTERBEGIN(1, startingMonth=month)
                    + typeref.OFST_DATEOFFSET(months=offset * 3)
                )
            except Exception as err:
                self._raise_error(err)
        # Days before 28
        elif day < 29:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_QUARTEREND(0)
                    - typeref.OFST_QUARTERBEGIN(1, startingMonth=month)
                    + typeref.OFST_DATEOFFSET(months=offset * 3, days=day - 1)
                )
            except Exception as err:
                self._raise_error(err)
        # Days before 31
        elif day < 31:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_QUARTEREND(0)
                    + typeref.OFST_DATEOFFSET(months=offset * 3 + month - 3)
                )
                dt += self._arr_to_tdindex(
                    FN_NP_MIN(dt.dt.days_in_month.values, day) - dt.dt.day.values, "D"
                )
            except Exception as err:
                self._raise_error(err)
        # Last day of the month
        else:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_QUARTEREND(0)
                    + typeref.OFST_DATEOFFSET(months=offset * 3 + month - 3)
                    + typeref.OFST_MONTHEND(0)
                )
            except Exception as err:
                self._raise_error(err)
        # Generate new pddt
        return self._new(dt, None, None, False)  # exit

    # Month -----------------------------------------------------------------------------------
    @property
    def days_in_month(self) -> Series[int]:
        """Get the maximum number of days in the month `<'Series[int]'>`."""
        return self._dt.dt.days_in_month

    @property
    def days_bf_month(self) -> Series[int]:
        """Get the number of days between the 1st day of
        the year and the 1st day of the month `<'Series[int]'>`."""
        return self._arr_to_series(
            self._dt.dt.day_of_year.values - self._dt.dt.day.values
        )

    def is_month(self, month: int | str) -> Series[bool]:
        """Whether the dates are in the given 'month' `<'Series[bool]'>`."""
        return self._arr_to_series(self._dt.dt.month.values == self._parse_month(month))

    def is_month_1st(self) -> Series[bool]:
        """Whether the dates are the 1st day of the month `<'Series[bool]'>`."""
        return self._dt.dt.is_month_start

    def is_month_lst(self) -> Series[bool]:
        """Whether the dates are the last day of the month `<'Series[bool]'>`."""
        return self._dt.dt.is_month_end

    @cython.ccall
    def to_month_1st(self) -> pddt:
        """Go to the 1st day of the month `<'pddt'>`."""
        try:
            dt: Series = (
                self._dt + typeref.OFST_MONTHEND(0) - typeref.OFST_MONTHBEGIN(1)
            )
        except Exception as err:
            self._raise_error(err)
        return self._new(dt, None, None, False)

    @cython.ccall
    def to_month_lst(self) -> pddt:
        """Go to the last day of the month `<'pddt'>`."""
        try:
            dt: Series = self._dt + typeref.OFST_MONTHEND(0)
        except Exception as err:
            self._raise_error(err)
        return self._new(dt, None, None, False)

    @cython.ccall
    def to_curr_month(self, day: cython.int = -1) -> pddt:
        """Go to specific 'day' of the current month `<'pddt'>`."""
        # Invalid 'day' value
        if day < 1:
            return self._new(self._dt, None, None, True)  # exit
        # Frist day of the month
        if day == 1:
            return self.to_month_1st()  # exit
        # Days before 28
        if day < 29:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_MONTHEND(0)
                    - typeref.OFST_MONTHBEGIN(1)
                    + typeref.OFST_DAY(day - 1)
                )
            except Exception as err:
                self._raise_error(err)
            return self._new(dt, None, None, False)  # exit
        # Days before 31
        if day < 31:
            try:
                dt: Series = (
                    self._dt + typeref.OFST_MONTHEND(0) - typeref.OFST_MONTHBEGIN(1)
                )
                dt += self._arr_to_tdindex(
                    FN_NP_MIN(self._dt.dt.days_in_month.values, day) - 1, "D"
                )
            except Exception as err:
                self._raise_error(err)
            return self._new(dt, None, None, False)  # exit
        # Last day of the month
        return self.to_month_lst()  # exit

    @cython.ccall
    def to_prev_month(self, day: cython.int = -1) -> pddt:
        """Go to specific 'day' of the previous month `<'pddt'>`."""
        return self.to_month(-1, day)

    @cython.ccall
    def to_next_month(self, day: cython.int = -1) -> pddt:
        """Go to specific 'day' of the next month `<'pddt'>`."""
        return self.to_month(1, day)

    @cython.ccall
    def to_month(self, offset: cython.int = 0, day: cython.int = -1) -> pddt:
        """Go to specific 'day' of the current month (+/-) 'offset' `<'pddt'>`."""
        # No 'offset' adjustment
        if offset == 0:
            return self.to_curr_month(day)  # exit
        # Invalid 'day' value (+/-) 'offset'
        if day < 1:
            try:
                dt: Series = self._dt + typeref.OFST_DATEOFFSET(months=offset)
            except Exception as err:
                self._raise_error(err)
        # First day of the month (+/-) 'offset'
        elif day == 1:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_MONTHEND(0)
                    - typeref.OFST_MONTHBEGIN(1)
                    + typeref.OFST_DATEOFFSET(months=offset)
                )
            except Exception as err:
                self._raise_error(err)
        # Days before 28 (+/-) 'offset'
        elif day < 29:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_MONTHEND(0)
                    - typeref.OFST_MONTHBEGIN(1)
                    + typeref.OFST_DATEOFFSET(months=offset, days=day - 1)
                )
            except Exception as err:
                self._raise_error(err)
        # Days before 31 (+/-) 'offset'
        elif day < 31:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_MONTHEND(0)
                    - typeref.OFST_MONTHBEGIN(1)
                    + typeref.OFST_DATEOFFSET(months=offset)
                )
                dt += self._arr_to_tdindex(
                    FN_NP_MIN(dt.dt.days_in_month.values, day) - 1, "D"
                )
            except Exception as err:
                self._raise_error(err)
        # Last day of the month (+/-) 'offset'
        else:
            try:
                dt: Series = (
                    self._dt
                    + typeref.OFST_DATEOFFSET(months=offset)
                    + typeref.OFST_MONTHEND(0)
                )
            except Exception as err:
                self._raise_error(err)
        # Generate new pddt
        return self._new(dt, None, None, False)  # exit

    # Weekday ---------------------------------------------------------------------------------
    @property
    def weekday(self) -> Series[int]:
        """The weekday of the dates `<'Series[int]'>`.
        Values: 0=Monday...6=Sunday."""
        return self._dt.dt.weekday

    @property
    def isoweekday(self) -> Series[int]:
        """The ISO weekday of the dates `<'Series[int]'>`.
        Values: 1=Monday...7=Sunday."""
        return self._arr_to_series(self._dt.dt.weekday.values + 1)

    @property
    def isoweek(self) -> Series[int]:
        """The ISO calendar week number of the dates `<'Series[int]'>`."""
        return self._dt.dt.isocalendar()["week"]

    @property
    def isoyear(self) -> Series[int]:
        """The ISO calendar year of the dates `<'Series[int]'>`."""
        return self._dt.dt.isocalendar()["year"]

    @property
    def isocalendar(self) -> DataFrame[int]:
        """The ISO calendar year, week, and weekday of the dates `<DataFrame[int]>`."""
        return self._dt.dt.isocalendar()

    def is_weekday(self, weekday: int | str) -> Series[bool]:
        """Whether the dates are in the given 'weekday' `<'Series[bool]'>`."""
        return self._arr_to_series(
            self._dt.dt.weekday.values == self._parse_weekday(weekday)
        )

    @cython.ccall
    def to_monday(self) -> pddt:
        """Go to Monday of the current week `<'pddt'>`."""
        return self._c_to_curr_weekday(0)

    @cython.ccall
    def to_tuesday(self) -> pddt:
        """Go to Tuesday of the current week `<'pddt'>`."""
        return self._c_to_curr_weekday(1)

    @cython.ccall
    def to_wednesday(self) -> pddt:
        """Go to Wednesday of the current week `<'pddt'>`."""
        return self._c_to_curr_weekday(2)

    @cython.ccall
    def to_thursday(self) -> pddt:
        """Go to Thursday of the current week `<'pddt'>`."""
        return self._c_to_curr_weekday(3)

    @cython.ccall
    def to_friday(self) -> pddt:
        """Go to Friday of the current week `<'pddt'>`."""
        return self._c_to_curr_weekday(4)

    @cython.ccall
    def to_saturday(self) -> pddt:
        """Go to Saturday of the current week `<'pddt'>`."""
        return self._c_to_curr_weekday(5)

    @cython.ccall
    def to_sunday(self) -> pddt:
        """Go to Sunday of the current week `<'pddt'>`."""
        return self._c_to_curr_weekday(6)

    @cython.ccall
    def to_curr_weekday(self, weekday: int | str | None = None) -> pddt:
        """Go to specific 'weekday' of the current week `<'pddt'>`."""
        # Invalid 'weekday' value
        wkd: cython.int = self._parse_weekday(weekday)
        if wkd == -1:
            return self._new(self._dt, None, None, True)  # exit
        # Go to specific 'weekday'
        try:
            dt: Series = self._dt - self._arr_to_tdindex(
                self._dt.dt.weekday.values - wkd, "D"
            )
        except Exception as err:
            self._raise_error(err)
        return self._new(dt, None, None, False)  # exit

    @cython.ccall
    def to_prev_weekday(self, weekday: int | str | None = None) -> pddt:
        """Go to specific 'weekday' of the previous week `<'pddt'>`."""
        return self.to_weekday(-1, weekday)

    @cython.ccall
    def to_next_weekday(self, weekday: int | str | None = None) -> pddt:
        """Go to specific 'weekday' of the next week `<'pddt'>`."""
        return self.to_weekday(1, weekday)

    @cython.ccall
    def to_weekday(
        self,
        offset: cython.int = 0,
        weekday: cython.int | str | None = None,
    ) -> pddt:
        """Go to specific 'weekday' of the current
        week (+/-) 'offset' `<'pddt'>`."""
        # No 'offset' adjustment
        if offset == 0:
            return self.to_curr_weekday(weekday)  # exit
        # Invalid 'weekday' value
        wkd: cython.int = self._parse_weekday(weekday)
        if wkd == -1:
            try:
                dt: Series = self._dt + typeref.OFST_DAY(offset * 7)
            except Exception as err:
                self._raise_error(err)
        # Go to specific 'weekday' (+/-) 'offset'
        else:
            offset: cython.int = offset * 7 + wkd
            try:
                dt: Series = self._dt - self._arr_to_tdindex(
                    self._dt.dt.weekday.values - offset, "D"
                )
            except Exception as err:
                self._raise_error(err)
        # Generate new pddt
        return self._new(dt, None, None, False)  # exit

    # Day -------------------------------------------------------------------------------------
    def is_day(self, day: int) -> Series[bool]:
        """Whether the dates are in the given 'day' `<'Series[bool]'>`."""
        return self._arr_to_series(self._dt.dt.day.values == day)

    @cython.ccall
    def to_yesterday(self) -> pddt:
        """Go to the previous day `<'pddt'>`."""
        try:
            dt: Series = self._dt + typeref.OFST_DAY(-1)
        except Exception as err:
            self._raise_error(err)
        return self._new(dt, None, None, False)

    @cython.ccall
    def to_tomorrow(self) -> pddt:
        """Go to the next day `<'pddt'>`."""
        try:
            dt: Series = self._dt + typeref.OFST_DAY(1)
        except Exception as err:
            self._raise_error(err)
        return self._new(dt, None, None, False)

    @cython.ccall
    def to_day(self, offset: cython.int) -> pddt:
        """Go to the current day (+/-) 'offset' `<'pddt'>`."""
        # No 'offset' adjustment
        if offset == 0:
            return self._new(self._dt, None, None, True)  # exit
        # Go to the current day (+/-) 'offset'
        try:
            dt: Series = self._dt + typeref.OFST_DAY(offset)
        except Exception as err:
            self._raise_error(err)
        return self._new(dt, None, None, False)

    # Time ------------------------------------------------------------------------------------
    def is_time_start(self) -> Series[bool]:
        """Whether the current time is the
        start of time (00:00:00.000000) `<'Series[bool]'>`."""
        return self._arr_to_series(self._dt.dt.time.values == cydt.TIME_START)

    def is_time_end(self) -> Series[bool]:
        """Whether the current time is the
        end of time (23:59:59.999999) `<'Series[bool]'>`."""
        return self._arr_to_series(self._dt.dt.time.values == cydt.TIME_END)

    @cython.ccall
    def to_time_start(self) -> pddt:
        """Go to the start of time (00:00:00.000000) `<'pddt'>`."""
        try:
            dt: Series = self._dt.dt.floor("D", "infer", "shift_backward")
        except Exception as err:
            self._raise_error(err)
        return self._new(dt, None, None, False)

    @cython.ccall
    def to_time_end(self) -> pddt:
        """Go to the end of time (23:59:59.999999) `<'pddt'>`."""
        try:
            dt: Series = self._dt.dt.ceil(
                "D", "infer", "shift_forward"
            ) - typeref.OFST_MICRO(1)
        except Exception as err:
            self._raise_error(err)
        return self._new(dt, None, None, False)

    @cython.ccall
    def to_time(
        self,
        hour: cython.int = -1,
        minute: cython.int = -1,
        second: cython.int = -1,
        millisecond: cython.int = -1,
        microsecond: cython.int = -1,
    ) -> pddt:
        """Go to specific 'hour', 'minute', 'second', 'millisecond',
        and 'microsecond' of the current times `<'pddt'>`.

        ### Important Warning
        - Calling this method, returns <'pddt'> where nanosecond values
          will be truncated to microsecond resolution (lost / reset to `0`).
        """
        # Calculate microsecond
        microsecond = combine_ms_us(millisecond, microsecond)
        if hour < 0 and minute < 0 and second < 0 and microsecond < 0:
            return self._new(self._dt, None, None, True)  # exit: no adjustment
        # Go to specific time
        try:
            dt: Series = self._dt.dt.floor("D", "infer", "shift_backward")
            # fmt: off
            dt += self._arr_to_tdindex(
                # . hour
                ( min(hour, 23) * cydt.US_HOUR
                    if hour >= 0 else
                    self._dt.dt.hour.values * cydt.US_HOUR )
                # . minute
                + ( min(minute, 59) * 60_000_000
                    if minute >= 0 else
                    self._dt.dt.minute.values * 60_000_000 )
                # . second
                + ( min(second, 59) * 1_000_000
                    if second >= 0 else 
                    self._dt.dt.second.values * 1_000_000 )
                # . microsecond
                + ( FN_NP_FULL(len(self._dt), microsecond)
                    if microsecond >= 0
                    else self._dt.dt.microsecond.values ),
                # . unit            
                "us",
            )
            # fmt: on
        except Exception as err:
            self._raise_error(err)
        return self._new(dt, None, None, False)

    # Timezone --------------------------------------------------------------------------------
    @property
    def tz_available(self) -> set[str]:
        """Access all the available timezone names `<set[str]>`."""
        return TIMEZONE_AVAILABLE

    @cython.ccall
    def tz_localize(
        self,
        tz: str | datetime.tzinfo | None = None,
        ambiguous: bool | Series[bool] | Literal["raise", "infer"] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> pddt:
        """Localize to a specific 'tz' timezone `<'pddt'>`.

        Equivalent to `Series.dt.tz_localize(tz)`.

        :param tz: `<'datetime.tzinfo'>`/`<str (timezone name)>`/`None (remove timezone)`. Defaults to `None`.
            - `<'str'>`: The timezone name to localize to.
            - `<'datetime.tzinfo'>`: The timezone to localize to.
            - `None`: Remove timezone awareness.

        :param ambiguous: `<'bool'>`/`<'Series[bool]'>`/`'raise'`/`'infer'`. Defaults to `'raise'`.
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from 03:00
            DST to 02:00 non-DST, 02:30:00 local time occurs both at 00:30:00 UTC
            and at 01:30:00 UTC. In such a situation, the ambiguous parameter
            dictates how ambiguous times should be handled.
            - `<'bool'>`: Marks all times as DST time (True) and non-DST time (False).
            - `<'Series[bool]'>`: Marks specific times (matching index) as DST time (True)
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
        return self._new(
            self._c_tz_localize(self._dt, tz, ambiguous, nonexistent), None, None, False
        )

    @cython.ccall
    def tz_convert(
        self,
        tz: str | datetime.tzinfo | None = None,
        ambiguous: bool | Series[bool] | Literal["raise", "infer"] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> pddt:
        """Convert to a specific 'tz' timezone `<'pddt'>`.

        Similar to `Series.dt.tz_convert(tz)`, but behave more like
        `datetime.astimezone(tz)`. The main differences occurs when
        handling timezone-naive Series, where `pandas.tz_convert` will
        raise an error, while this method will first localize to the
        system's local timezone and then convert to the target timezone.

        :param tz: `<'datetime.tzinfo'>`/`<str (timezone name)>`/`None (local timezone)`. Defaults to `None`.
            - `<'str'>`: The timezone name to convert to.
            - `<'datetime.tzinfo'>`: The timezone to convert to.
            - `None`: Convert to system's local timezone.

        :param ambiguous, nonexistent: Please refer to `tz_localize` method for details.
        """
        return self._new(
            self._c_tz_convert(self._dt, tz, ambiguous, nonexistent), None, None, False
        )

    @cython.ccall
    def tz_switch(
        self,
        targ_tz: str | datetime.tzinfo | None,
        base_tz: str | datetime.tzinfo | None = None,
        naive: bool = False,
        ambiguous: bool | Series[bool] | Literal["raise", "infer"] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> pddt:
        """Switch to 'targ_tz' timezone from 'base_tz' timezone `<'pddt'>`.

        :param targ_tz `<'str/tzinfo/None'>`: The target timezone to convert.
        :param base_tz `<'str/tzinfo/None'>`: The base timezone to localize. Defaults to `None`.
        :param naive `<'bool'>`: Whether to return as timezone-naive. Defaults to `False`.
        :param ambiguous, nonexistent: Please refer to `tz_localize` method for details.
        :return `<'pddt'>`: pddt after switch of timezone.

        ### Explaination
        - If pddt is timezone-aware, 'base_tz' will be ignored, and only performs
          the convertion to the 'targ_tz' (Series.dt.tz_convert(targ_tz)).

        - If pddt is timezone-naive, first will localize to the given 'base_tz',
          then convert to the 'targ_tz'. In this case, the 'base_tz' must be
          specified, else it is ambiguous on how to convert to the target timezone.
        """
        # Pddt is timezone-aware
        curr_tz = self._dt.dt.tz
        if curr_tz is not None:
            # . current => target timezone
            dt: Series = self._c_tz_convert(self._dt, targ_tz, ambiguous, nonexistent)
            if naive:
                dt = dt.dt.tz_localize(None)
            return self._new(dt, None, None, False)
        # Pddt is timezone-naive
        if base_tz is not None:
            # . localize to base & base => target
            dt: Series = self._c_tz_localize(self._dt, base_tz, ambiguous, nonexistent)
            dt = self._c_tz_convert(dt, targ_tz, ambiguous, nonexistent)
            if naive:
                dt = dt.dt.tz_localize(None)
            return self._new(dt, None, None, False)
        # Invalid
        raise errors.InvalidTimezoneError(
            "<'%s'>\nCannot switch timezone-naive '<'pddt'>' without "
            "a valid 'base_tz': %s." % (self.__class__.__name__, repr(base_tz))
        )

    @cython.ccall
    def astimezone(
        self,
        tz: str | datetime.tzinfo | None = None,
        ambiguous: bool | Series[bool] | Literal["raise", "infer"] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> pddt:
        """Convert to a specific 'tz' timezone `<'pddt'>`.

        Alias of `pddt.tz_convert()`.

        :param tz: `<'datetime.tzinfo'>`/`<str (timezone name)>`/`None (local timezone)`. Defaults to `None`.
            - `<'str'>`: The timezone name to convert to.
            - `<'datetime.tzinfo'>`: The timezone to convert to.
            - `None`: Convert to system's local timezone.

        :param ambiguous, nonexistent: Please refer to `tz_localize` method for details.
        """
        return self._new(
            self._c_tz_convert(self._dt, tz, ambiguous, nonexistent), None, None, False
        )

    # Frequency -------------------------------------------------------------------------------
    @cython.ccall
    def freq_round(
        self,
        freq: Literal["D", "h", "m", "s", "ms", "us", "ns"],
        ambiguous: bool | Series[bool] | Literal["raise", "infer"] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> pddt:
        """Perform round operation to specified freqency `<'pddt'>`.

        :param freq: `<'str'>` frequency to round to:
            - 'D': Day
            - 'h': Hour
            - 'm': Minute
            - 's': Second
            - 'ms': Millisecond
            - 'us': Microsecond
            - 'ns': Nanosecond

        :param ambiguous, nonexistent: Please refer to `tz_localize` method for details.
        """
        self._validate_amb_nonex(ambiguous, nonexistent)
        freq = self._parse_frequency(freq)
        try:
            dt: Series = self._dt.dt.round(freq, ambiguous, nonexistent)
        except Exception as err:
            self._raise_error(err)
        return self._new(dt, None, None, False)

    @cython.ccall
    def freq_ceil(
        self,
        freq: Literal["D", "h", "m", "s", "ms", "us", "ns"],
        ambiguous: bool | Series[bool] | Literal["raise", "infer"] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> pddt:
        """Perform ceil operation to specified freqency `<'pddt'>`.

        :param freq: `<'str'>` frequency to round to:
            - 'D': Day
            - 'h': Hour
            - 'm': Minute
            - 's': Second
            - 'ms': Millisecond
            - 'us': Microsecond
            - 'ns': Nanosecond

        :param ambiguous, nonexistent: Please refer to `tz_localize` method for details.
        """
        self._validate_amb_nonex(ambiguous, nonexistent)
        freq = self._parse_frequency(freq)
        try:
            dt: Series = self._dt.dt.ceil(freq, ambiguous, nonexistent)
        except Exception as err:
            self._raise_error(err)
        return self._new(dt, None, None, False)

    @cython.ccall
    def freq_floor(
        self,
        freq: Literal["D", "h", "m", "s", "ms", "us", "ns"],
        ambiguous: bool | Series[bool] | Literal["raise", "infer"] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> pddt:
        """Perform floor operation to specified freqency `<'pddt'>`.

        :param freq: `<'str'>` frequency to round to:
            - 'D': Day
            - 'h': Hour
            - 'm': Minute
            - 's': Second
            - 'ms': Millisecond
            - 'us': Microsecond
            - 'ns': Nanosecond

        :param ambiguous, nonexistent: Please refer to `tz_localize` method for details.
        """
        self._validate_amb_nonex(ambiguous, nonexistent)
        freq = self._parse_frequency(freq)
        try:
            dt: Series = self._dt.dt.floor(freq, ambiguous, nonexistent)
        except Exception as err:
            self._raise_error(err)
        return self._new(dt, None, None, False)

    # Delta -----------------------------------------------------------------------------------
    @cython.ccall
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
        """Add 'timedelta' to the current `<'pddt'>`.

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
            dt: Series = self._dt + typeref.OFST_DATEOFFSET(
                years=years,
                months=months,
                weeks=weeks,
                days=days,
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                microseconds=milliseconds * 1_000 + microseconds,
            )
        except Exception as err:
            self._raise_error(err)
        # Generate
        return self._new(dt, None, None, False)

    @cython.ccall
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
        """Substract 'timedelta' to the current `<'pddt'>`.

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
        try:
            dt = self._dt - typeref.OFST_DATEOFFSET(
                years=years,
                months=months,
                weeks=weeks,
                days=days,
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                microseconds=milliseconds * 1_000 + microseconds,
            )
        except Exception as err:
            self._raise_error(err)
        return self._new(dt, None, None, False)

    def cal_delta(
        self,
        other: object,
        unit: Literal["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"] = "D",
        inclusive: Literal["one", "both", "neither"] = "both",
    ) -> Series[int]:
        """Calcuate the `ABSOLUTE` delta between the current pddt
        and the given object based on the specified 'unit' `<'Series[int]'>`.

        :param other `<'pddt/Series/list/pydt/str/datetime'>`: The target object.
        :param unit `<'str'>`: The specific time unit to calculate the delta.
        :param inclusive `<'str'>`: Include boundaries, whether to set bounds as closed or open. Defaults to `'both'`.
        :return `<'Series[int]'>`: The `ABSOLUTE` delta values.
        """
        return self._c_cal_delta(other, unit, inclusive)

    # Replace ---------------------------------------------------------------------------------
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
        """Replacement for the current `<'pddt'>`.

        Function similar to `datetime.replace()`.

        :param year `<int>`: The absolute year value. Defaults to `-1 (no change)`.
        :param month `<int>`: The absolute month value. Defaults to `-1 (no change)`.
        :param day `<int>`: The absolute day value. Defaults to `-1 (no change)`.
        :param hour `<int>`: The absolute hour value. Defaults to `-1 (no change)`.
        :param minute `<int>`: The absolute minute value. Defaults to `-1 (no change)`.
        :param second `<int>`: The absolute second value. Defaults to `-1 (no change)`.
        :param millisecond `<int>`: The absolute millisecond value. Defaults to `-1 (no change)`.
        :param microsecond `<int>`: The absolute microsecond value. Defaults to `-1 (no change)`.
        :return `<'pddt'>`: pddt after replacement.

        ### Important Warning
        - Calling this method, returns <'pddt'> where nanosecond values
          will be truncated to microsecond resolution (lost / reset to `0`).
        - Replacement for 'year', 'month' and 'day' does not support
          vectorized operation, which could lead to performance issues
          when working with large dataset.
        """
        # Non-vertorzied (Y/M/D)
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
            microsecond = combine_ms_us(millisecond, microsecond)
            dt = self._dt.apply(
                lambda dt: cydt.dt_replace(
                    dt, year, month, day, hour, minute, second, microsecond
                )
            )
            return self._new(dt, None, None, False)

        # Vertorzied (h/m/s/ms/us)
        return self.to_time(hour, minute, second, millisecond, microsecond)

    # Format ----------------------------------------------------------------------------------
    def strftime(self, format: str) -> Series[str]:
        """Convert to string with the given 'format' `<'Series[str]'>`.

        Equivalent to `Series.dt.strftime(format)`."""
        return self._dt.dt.strftime(format)

    # Unit ------------------------------------------------------------------------------------
    @cython.ccall
    def as_unit(self, unit: Literal["s", "ms", "us", "ns"]) -> pddt:
        """Convert to specific time 'unit' resolution `<'pddt'>`.

        Equivalent to `Series.dt.as_unit(unit)`.

        :param unit `<'str'>`: The specific time unit to convert.
            - 's': Second
            - 'ms': Millisecond
            - 'us': Microsecond
            - 'ns': Nanosecond
        """
        return self._new(self._dt, None, unit, True)

    # Internal methods ------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _new(self, dtsobj: object, name: object, unit: bool, copy: bool) -> pddt:
        """(cfunc) Generate new `<'pddt'>`."""
        # fmt: off
        return pddt(
            dtsobj, self._default, self._day1st, self._year1st, self._utc, 
            self._format, self._exact, self._cfg, name, unit, copy
        )
        # fmt: on

    @cython.cfunc
    @cython.inline(True)
    def _parse_dtobj(
        self,
        dtobj: object,
        default: object,
        strict: cython.bint,
    ) -> datetime.datetime:
        """(cfunc) Parse 'dtobj' into `<'datetime.datetime'>`."""
        # <'str'> datetime string to be parsed
        if isinstance(dtobj, str):
            try:
                return Parser(self._cfg).parse(
                    dtobj, default, self._day1st, self._year1st, False, True
                )
            except Exception as err:
                raise errors.InvalidDatetimeObjectError(
                    "<'%s'> %s" % (self.__class__.__name__, err)
                ) from err
        # <'datetime.date'> subclass
        if datetime.PyDate_Check(dtobj):
            if datetime.PyDateTime_Check(dtobj):
                return cydt.dt_fr_dt(dtobj) if strict else dtobj
            else:
                return cydt.dt_fr_date(dtobj, None, 0)
        # <'pydt'>
        if isinstance(dtobj, pydt):
            return access_pydt(dtobj)
        # <'numpy.datetime64'>
        if np.is_datetime64_object(dtobj):
            return cydt.dt64_to_dt(dtobj) if strict else dtobj
        # Invalid
        raise errors.InvalidDatetimeObjectError(
            "<'%s'>\nUnsupported 'dtobj' type: %s %s."
            % (self.__class__.__name__, type(dtobj), repr(dtobj))
        )

    @cython.cfunc
    @cython.inline(True)
    def _parse_dtsobj(
        self,
        dtsobj: object,
        default: object,
        name: object,
        unit: object,
        copy: cython.bint,
    ) -> object:
        """(cfunc) Parse 'dtsobj' into `<'Series[Timestamp]'>`."""
        is_series: cython.bint = False
        # <'Series'>
        if isinstance(dtsobj, typeref.SERIES):
            if isinstance(dtsobj.dtype, (typeref.DT64_ARRAY, typeref.DT64TZ_ARRAY)):
                dt = self._adjust_unit(dtsobj, unit, copy)
                if name is not None:
                    dt.name = name
                return dt
            is_series = True
        # <'DatetimeIndex'>
        elif isinstance(dtsobj, typeref.DATETIMEINDEX):
            if name is not None:
                dt = typeref.SERIES(dtsobj, name=name)
            else:
                dt = typeref.SERIES(dtsobj)
            return self._adjust_unit(dt, unit, False)
        # <'pddt'>
        elif isinstance(dtsobj, pddt):
            dt = self._adjust_unit(access_pddt(dtsobj), unit, copy)  # type: ignore
            if name is not None:
                dt.name = name
            return dt
        # Parse 'dtsobj'
        try:
            dt = FN_PD_TODATETIME(
                dtsobj,
                errors="raise",
                dayfirst=self._day1st,
                yearfirst=self._year1st,
                utc=self._utc,
                format=self._format,
                exact=self._exact,
                cache=True,
            )
        # Fallback
        except Exception as err:
            return self._parse_dtsobj_fallback(
                dtsobj, default, name, unit, is_series, err
            )
        # <'Series'>
        if isinstance(dt, typeref.SERIES):
            if name is not None:
                dt.name = name
            return self._adjust_unit(dt, unit, False)
        # <'DatetimeIndex'>
        if isinstance(dt, typeref.DATETIMEINDEX):
            if name is not None:
                dt = typeref.SERIES(dt, name=name)
            else:
                dt = typeref.SERIES(dt)
            return self._adjust_unit(dt, unit, False)
        # Invalid
        raise errors.InvalidDatetimeObjectError(
            "<'%s'>\nUnsupported 'dtsobj' type: %s."
            % (self.__class__.__name__, type(dtsobj))
        )

    @cython.cfunc
    @cython.inline(True)
    def _parse_dtsobj_fallback(
        self,
        dtsobj: object,
        default: object,
        name: object,
        unit: object,
        is_series: cython.bint,
        exc: Exception,
    ) -> object:
        """(cfunc) The fallback when pandas failed to parse 'dtsobj' `<'Series[Timestamp]'>.

        ### Notice
        The resulting Series will only have a maximum resolution of 'us' instead of 'ns'.
        """
        # Parse each element into timezone-naive datetime
        dts: list = []
        default: object = self._default
        utc: cython.bint = self._utc
        tz: object = None
        try:
            for obj in dtsobj:
                i: datetime.datetime = self._parse_dtobj(obj, default, True)
                if (tzinfo := i.tzinfo) is not None:
                    # . convert to utc & remove timezone
                    if utc:
                        i = cydt.dt_replace_tzinfo(
                            cydt.dt_astimezone(i, cydt.UTC), None
                        )
                    # . remove timezone
                    elif tz is tzinfo:
                        i = cydt.dt_replace_tzinfo(i, None)
                    # . set & remove timezone
                    elif tz is None:
                        i = cydt.dt_replace_tzinfo(i, None)
                        tz = tzinfo
                    # . multiple timezones
                    else:
                        raise errors.InvalidDatetimeObjectError(
                            "<'%s'>\nCannot parse 'dtsobj' into DatetimeArray when it "
                            "contains different timezones, unless setting 'utc=True'.\n"
                            "Conflict timezones: '%s' and '%s'."
                            % (self.__class__.__name__, tz, tzinfo)
                        )
                dts.append(i)
        except errors.PddtError as err:
            raise err from exc
        except Exception:
            raise errors.InvalidDatetimeObjectError(
                "<'%s'>\nUnsupported 'dtsobj' type: %s."
                % (self.__class__.__name__, type(dtsobj))
            ) from exc
        # Construact Series[Timestamp]
        try:
            if is_series:
                # fmt: off
                dt: Series = typeref.SERIES(
                    dts, index=dtsobj.index, dtype="<M8[us]",
                    name=name if name is not None else dtsobj.name,
                )
                # fmt: on
            elif name is not None:
                dt: Series = typeref.SERIES(dts, dtype="<M8[us]", name=name)
            else:
                dt: Series = typeref.SERIES(dts, dtype="<M8[us]")
        except Exception as err:
            raise errors.InvalidDatetimeObjectError(
                "<'%s'>\nFailed to parse 'dtsobj': %s" % (self.__class__.__name__, err)
            ) from exc
        # Adjust unit
        dt = self._adjust_unit(dt, unit, False)
        # Adjust timezone
        if utc:
            dt = dt.dt.tz_localize(cydt.UTC)
        elif tz is not None:
            dt = dt.dt.tz_localize(tz)
        # Return
        return dt

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-2, check=False)
    def _parse_month(self, month: object) -> cython.int:
        """(cfunc) Prase 'month' into numeric value `<'int'>`.

        Accepts both integer or string. Returns -1 if the 'month'
        is None or -1, else raise `InvalidMonthError`."""
        # <'NoneType'>
        if month is None:
            return -1  # exit

        # <'str'>
        if isinstance(month, str):
            mth: str = month
            val = dict_getitem(
                CONFIG_MONTH if self._cfg is None else self._cfg._month,
                mth.lower(),
            )
            if val == cython.NULL:
                raise errors.InvalidMonthError(
                    "<'%s'>\nFailed to parse 'month' string: '%s'."
                    % (self.__class__.__name__, month)
                )
            return cython.cast(object, val)  # exit

        # <'int'>
        if isinstance(month, int):
            try:
                num: cython.int = month
                if num == -1:
                    return -1  # exit
                if not 1 <= num <= 12:
                    raise ValueError
                return num  # exit
            except Exception as err:
                raise errors.InvalidMonthError(
                    "<'%s'>\nInvalid 'month' value '%d', must between "
                    "1(Jan)..12(Dec)." % (self.__class__.__name__, month)
                ) from err

        # Incorrect data type
        raise errors.InvalidMonthError(
            "<'%s'>\nInvalid 'month' data type: %s %s."
            % (self.__class__.__name__, type(month), repr(month))
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-2, check=False)
    def _parse_weekday(self, weekday: object) -> cython.int:
        """(cfunc) Prase 'weekday' into numeric value `<'int'>`.

        Accepts both integer or string. Returns -1 if the 'weekday'
        is None or -1, else raise `InvalidWeekdayError`."""
        # <'NoneType'>
        if weekday is None:
            return -1  # exit

        # <'str'>
        if isinstance(weekday, str):
            wkd: str = weekday
            val = dict_getitem(
                CONFIG_WEEKDAY if self._cfg is None else self._cfg._weekday,
                wkd.lower(),
            )
            if val == cython.NULL:
                raise errors.InvalidWeekdayError(
                    "<'%s'>\nFailed to parse 'weekday' string: '%s'."
                    % (self.__class__.__name__, weekday)
                )
            return cython.cast(object, val)  # exit

        # <'int'>
        if isinstance(weekday, int):
            try:
                num: cython.int = weekday
                if num == -1:
                    return num  # exit
                if not 0 <= num <= 6:
                    raise ValueError
                return num  # exit
            except Exception as err:
                raise errors.InvalidWeekdayError(
                    "<'%s'>\nInvalid 'weekday' value '%d', must between "
                    "0(Monday)..6(Sunday)." % (self.__class__.__name__, weekday)
                ) from err

        # Incorrect data type
        raise errors.InvalidWeekdayError(
            "<'%s'>\nInvalid 'weekday' data type: %s %s."
            % (self.__class__.__name__, type(weekday), repr(weekday))
        )

    @cython.cfunc
    @cython.inline(True)
    def _parse_frequency(self, freq: object) -> object:
        """(cfunc) Prase 'freq' into coresponding string value `<'str'>`."""
        if freq == "m":
            return "min"
        if set_contains(UNIT_FREQUENCY, freq):
            return freq
        raise errors.InvalidFrequencyError(
            "<'%s'>\nInvalid 'freq' input: %s. Must be one of the following <'str'>: "
            "['D', 'h', 'm', 's', 'ms', 'us']." % (self.__class__.__name__, repr(freq))
        )

    @cython.cfunc
    @cython.inline(True)
    def _adjust_unit(
        self,
        dt: Series,
        unit: object,
        copy: cython.bint,
    ) -> object:
        """(cfunc) Adjust Series to specific time 'unit' resolution `<'Series[Timestamp]'>`."""
        if unit is not None and cydt.get_dt64series_unit(dt) != unit:
            try:
                return dt.dt.as_unit(unit)
            except Exception as err:
                raise errors.InvalidTimeUnitError(
                    "<'%s'>\nUnable to adjust time unit from '[%s]' to '[%s]': %s."
                    % (self.__class__.__name__, cydt.get_dt64series_unit(dt), unit, err)
                ) from err
        return dt.copy(True) if copy else dt

    @cython.cfunc
    @cython.inline(True)
    def _arr_to_series(self, arr: object) -> object:
        """(cfunc) Generate Series from the numpy ndarray `<'Series'>`."""
        return typeref.SERIES(arr, index=self._dt.index, name=self._dt.name)

    @cython.cfunc
    @cython.inline(True)
    def _arr_to_tdindex(self, arr: object, unit: object) -> object:
        """(cfunc) Generate TimedeltaIndex from the numpy ndarray `<'TimedeltaIndex'>`"""
        td: TimedeltaIndex = FN_PD_TOTIMEDELTA(arr, unit)
        dt_unit: str = cydt.get_dt64series_unit(self._dt)
        if dt_unit != cydt.get_td64series_unit(td):
            td = td.as_unit(dt_unit)
        return td

    @cython.cfunc
    @cython.inline(True)
    def _obj_to_pddt(self, obj: object, name: object, unit: object) -> object:
        """(cfunc) Try to convert compatible object to `<'object'>`.

        For Incompatible object, return the object itself.
        Used for internal arithmetic operations."""
        if isinstance(obj, typeref.SERIES):
            if isinstance(obj.dtype, (typeref.DT64_ARRAY, typeref.DT64TZ_ARRAY)):
                return self._new(obj, name, unit, False)
            return obj
        elif isinstance(obj, typeref.DATETIMEINDEX):
            return self._new(obj, name, unit, False)
        else:
            return obj

    @cython.cfunc
    @cython.inline(True)
    def _obj_to_series(self, obj: object) -> object:
        """(cfunc) Try to convert compatible object to `<'Series'>`.

        For Incompatible object, return the object itself.
        Used for internal arithmetic operations."""
        # <'pddt'>
        if isinstance(obj, pddt):
            return access_pddt(obj)  # type: ignore
        # <'Series/TimedeltaIndex/Timedelta/offsets'>
        if isinstance(
            obj,
            (
                typeref.SERIES,
                typeref.TIMEDELTAINDEX,
                typeref.TIMEDELTA,
                typeref.BASEOFFSET,
                np.ndarray,
                datetime.timedelta,
                typeref.TIMEDELTA64,
                typeref.DATETIME64,
            ),
        ):
            return obj
        # <'datetime/str/pydt'>
        if isinstance(obj, (datetime.date, str, pydt)):
            return self._parse_dtobj(obj, self._default, False)
        # Other
        try:
            return self._parse_dtsobj(obj, self._default, None, self._c_unit(), False)
        except Exception:
            return obj

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _validate_amb_nonex(
        self,
        ambiguous: object,
        nonexistent: object,
    ) -> cython.bint:
        """(cfunc) Validate the 'ambiguous' and 'nonexistent' arguments `<'bool'>`."""
        if ambiguous == "NaT":
            raise errors.InvalidTimezoneError(
                "<'%s'>\nArgument `ambiguous='NaT'` is not supported."
                % self.__class__.__name__
            )
        if nonexistent == "NaT":
            raise errors.InvalidTimezoneError(
                "<'%s'>\nArgument `nonexistent='NaT'` is not supported."
                % self.__class__.__name__
            )
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _raise_error(self, err: Exception, msg: str = None) -> cython.bint:
        if isinstance(err, (OverflowError, OutOfBoundsDatetime)):
            if msg is None:
                try:
                    msg = (
                        "Value out of bounds for datetime64[%s]"
                        % cydt.get_dt64series_unit(self._dt)
                    )
                except Exception:
                    msg = "Value out of bounds"
            raise errors.DatetimeOutOfBoundsError(
                "<'%s'>\n%s: %s." % (self.__class__.__name__, msg, err)
            ) from err
        elif msg is not None:
            raise errors.PddtValueError(
                "<'%s'>\n%s: %s" % (self.__class__.__name__, msg, err)
            ) from err
        elif msg is None:
            raise errors.PddtValueError(
                "<'%s'>\n%s" % (self.__class__.__name__, err)
            ) from err
        else:
            return -1

    # Special methods: addition ---------------------------------------------------------------
    def __add__(self, o: object) -> pddt:
        try:
            obj = self._dt + o
        except Exception as err:
            self._raise_error(err, "Addition failed")
        return self._new(obj, self._dt.name, None, False)

    def __radd__(self, o: object) -> pddt:
        try:
            obj = o + self._dt
        except Exception as err:
            self._raise_error(err, "Addition failed")
        return self._new(obj, self._dt.name, None, False)

    # Special methods: substraction -----------------------------------------------------------
    def __sub__(self, o: object) -> pddt | Series:
        o = self._obj_to_series(o)
        try:
            obj = self._dt - o
        except Exception as err:
            self._raise_error(err, "Substraction failed")
        return self._obj_to_pddt(obj, self._dt.name, None)

    def __rsub__(self, o: object) -> pddt | Series:
        o = self._obj_to_series(o)
        try:
            obj = o - self._dt
        except Exception as err:
            self._raise_error(err, "Substraction failed")
        return self._obj_to_pddt(obj, self._dt.name, None)

    # Special methods - comparison ------------------------------------------------------------
    def __eq__(self, o: object) -> Series[bool]:
        o = self._obj_to_series(o)
        try:
            return self._dt == o
        except Exception as err:
            self._raise_error(err, "Comparison failed")

    def __ne__(self, o: object) -> Series[bool]:
        o = self._obj_to_series(o)
        try:
            return self._dt != o
        except Exception as err:
            self._raise_error(err, "Comparison failed")

    def __gt__(self, o: object) -> Series[bool]:
        o = self._obj_to_series(o)
        try:
            return self._dt > o
        except Exception as err:
            self._raise_error(err, "Comparison failed")

    def __ge__(self, o: object) -> Series[bool]:
        o = self._obj_to_series(o)
        try:
            return self._dt >= o
        except Exception as err:
            self._raise_error(err, "Comparison failed")

    def __lt__(self, o: object) -> Series[bool]:
        o = self._obj_to_series(o)
        try:
            return self._dt < o
        except Exception as err:
            self._raise_error(err, "Comparison failed")

    def __le__(self, o: object) -> Series[bool]:
        o = self._obj_to_series(o)
        try:
            return self._dt <= o
        except Exception as err:
            self._raise_error(err, "Comparison failed")

    def equals(self, o: object) -> bool:
        """Test whether two objects contain the same elements `<'bool'>`.
        (Equivalent to `pandas.Series.equal()` method).

        Support comparison between `pddt` and `pandas.Series`."""
        o = self._obj_to_series(o)
        try:
            return self._dt.equals(o)
        except Exception as err:
            self._raise_error(err, "Comparison failed")

    # Special methods - copy ------------------------------------------------------------------
    @cython.ccall
    def copy(self) -> pddt:
        "Make a (deep)copy of the `<'pddt'>`."
        return self._new(self._dt, None, None, True)

    def __copy__(self, *args, **kwargs) -> pddt:
        return self._new(self._dt, None, None, True)

    def __deepcopy__(self, *args, **kwargs) -> pddt:
        return self._new(self._dt, None, None, True)

    # Special methods: represent --------------------------------------------------------------
    def __repr__(self) -> str:
        return "<%s(\n%s)>" % (self.__class__.__name__, self._dt.__repr__())

    def __str__(self) -> str:
        return self._dt.__repr__()

    # Special methods: others -----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._dt)

    def __contains__(self, key) -> bool:
        return self._dt.__contains__(key)

    def __getitem__(self, key) -> Timestamp:
        return self._dt.__getitem__(key)

    def __iter__(self) -> Iterator[Timestamp]:
        return self._dt.__iter__()

    def __array__(self) -> np.ndarray:
        return self._dt.__array__()

    # C-API: Access ---------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _c_unit(self) -> str:
        """(cfunc) Access the time unit of the Series `<'str'>`.
        Such as 'ns', 'us', 'ms', 's'."""
        return cydt.get_dt64series_unit(self._dt)

    # C-API: Weekday --------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _c_to_curr_weekday(self, weekday: cython.int) -> pddt:
        """(cfunc) Go to specific 'weekday (<'int'>)'
        of the current week `<'pddt'>`."""
        try:
            if weekday == 0:
                dt: Series = self._dt - self._arr_to_tdindex(
                    self._dt.dt.weekday.values, "D"
                )
            else:
                dt: Series = self._dt - self._arr_to_tdindex(
                    self._dt.dt.weekday.values - weekday, "D"
                )
        except Exception as err:
            self._raise_error(err)
        return self._new(dt, None, None, False)

    # C-API: Timezone -------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _c_tz_localize(
        self,
        dt: Series,
        tz: object,
        ambiguous: object,
        nonexistent: object,
    ) -> object:
        """(cfunc) Localize 'dt' Series to a specific 'tz' timezone `<'Series[Timestamp]'>`.
        Equivalent to `Series.dt.tz_localize(tz)`."""
        # No need to change timezone
        if tz is None and tz is dt.dt.tz:
            return dt
        # Localize to timezone
        self._validate_amb_nonex(ambiguous, nonexistent)
        try:
            return dt.dt.tz_localize(tz, ambiguous, nonexistent)
        except Exception as err:
            raise errors.InvalidTimezoneError(
                "<'%s'>\nInvalid 'tz' timezone: %s"
                % (self.__class__.__name__, repr(tz))
            ) from err

    @cython.cfunc
    @cython.inline(True)
    def _c_tz_convert(
        self,
        dt: Series,
        tz: object,
        ambiguous: object,
        nonexistent: object,
    ) -> object:
        """(cfunc) Convert 'dt' Series to a specific 'tz' timezone `<'Series[Timestamp]'>`."""
        # Resolve target timezone
        curr_tz = dt.dt.tz
        if tz is None:
            targ_tz = cydt.gen_tzinfo_local(None)
            if curr_tz is None:
                return self._c_tz_localize(dt, targ_tz, ambiguous, nonexistent)
        else:
            targ_tz = tz
        # Resolve current timezone
        if curr_tz is None:
            dt = self._c_tz_localize(
                dt, cydt.gen_tzinfo_local(None), ambiguous, nonexistent
            )
        # Convert timezone
        try:
            return dt.dt.tz_convert(targ_tz)
        except Exception as err:
            raise errors.InvalidTimezoneError(
                "<'%s'>\nInvalid 'tz' timezone: %s"
                % (self.__class__.__name__, repr(tz))
            ) from err

    # C-API: Delta -----------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _c_cal_delta(
        self,
        other: object,
        unit: object,
        inclusive: object,
    ) -> object:
        """(cfunc) Calcuate the `ABSOLUTE` delta between the current pddt
        and the given object based on the specified 'unit' `<'Series[int]'>`.

        :param other `<'pddt/Series/list/pydt/str/datetime'>`: The target object.
        :param unit `<'str'>`: The specific time unit to calculate the delta.
        :param inclusive `<'str'>`: Include boundaries, whether to set bounds as closed or open.
        :return `<'Series[int]'>`: The `ABSOLUTE` delta values.
        """
        # Parse 'other' to Series
        if isinstance(other, (datetime.date, str, pydt, typeref.DATETIME64)):
            other = self._parse_dtobj(other, self._default, False)
            o_dt: Series = self._parse_dtsobj(
                Series(other, index=self._dt.index), None, None, None, False
            )
        else:
            o_dt: Series = self._parse_dtsobj(other, self._default, None, None, False)
            if len(self) != len(o_dt):
                raise errors.InvalidDatetimeObjectError(
                    "<'%s'>\nCannot calculate 'delta' for object with different shape: "
                    "%d vs %d." % (self.__class__.__name__, len(self._dt), len(o_dt))
                )

        # Inclusive bounds
        incl: cython.int
        if inclusive == "both":
            incl = 1
        elif inclusive == "one":
            incl = 0
        elif inclusive == "neither":
            incl = -1
        else:
            raise errors.InvalidDeltaInclusiveError(
                "<'%s'>\nInvalid 'inclusive' input: %s. "
                "Must be one of the following <'str'>: ['one', 'both', 'neither']."
                % (self.__class__.__name__, repr(inclusive))
            )

        # Unit: year
        if unit == "Y":
            delta = FN_NP_ABS(self._dt.dt.year.values - o_dt.dt.year.values)
            return self._arr_to_series(delta + incl if incl else delta)
        # Unit: month
        if unit == "M":
            delta = FN_NP_ABS(
                (self._dt.dt.year.values - o_dt.dt.year.values) * 12
                + (self._dt.dt.month.values - o_dt.dt.month.values)
            )
            return self._arr_to_series(delta + incl if incl else delta)
        # Unit: week
        if unit == "W":
            m_val = self._dt.values
            o_val = o_dt.values
            delta = FN_NP_ABS(
                cydt.dt64array_to_days(m_val) - cydt.dt64array_to_days(o_val)
            )
            adj = typeref.DATETIMEINDEX(FN_NP_MIN(m_val, o_val)).weekday.values
            delta = (delta + adj) // 7
            return self._arr_to_series(delta + incl if incl else delta)
        # Unit: day
        if unit == "D":
            m_val = cydt.dt64array_to_days(self._dt.values)
            o_val = cydt.dt64array_to_days(o_dt.values)
        # Unit: hour
        elif unit == "h":
            m_val = cydt.dt64array_to_hours(self._dt.values)
            o_val = cydt.dt64array_to_hours(o_dt.values)
        # Unit: minute
        elif unit == "m":
            m_val = cydt.dt64array_to_minutes(self._dt.values)
            o_val = cydt.dt64array_to_minutes(o_dt.values)
        # Unit: second
        elif unit == "s":
            m_val = cydt.dt64array_to_seconds(self._dt.values)
            o_val = cydt.dt64array_to_seconds(o_dt.values)
        # Unit: millisecond
        elif unit == "ms":
            m_val = cydt.dt64array_to_milliseconds(self._dt.values)
            o_val = cydt.dt64array_to_milliseconds(o_dt.values)
        # Unit: microsecond
        elif unit == "us":
            m_val = cydt.dt64array_to_microseconds(self._dt.values)
            o_val = cydt.dt64array_to_microseconds(o_dt.values)
        # Unit: nanosecond
        elif unit == "ns":
            m_val = cydt.dt64array_to_nanoseconds(self._dt.values)
            o_val = cydt.dt64array_to_nanoseconds(o_dt.values)
        # Invalid unit
        else:
            raise errors.InvalidDeltaUnitError(
                "<'%s'>\nInvalid delta 'unit' input: %s. "
                "Must be one of the following <'str'>: "
                "['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns']."
                % (self.__class__.__name__, repr(unit))
            )
        # Return delta
        delta = FN_NP_ABS(m_val - o_val)
        return self._arr_to_series(delta + incl if incl else delta)
