from typing import Literal, Hashable
from pandas._libs import lib
import datetime, numpy as np, pandas as pd
from typing_extensions import Self
from cytimes.parser import Configs

# Pddt
class Pddt(pd.DatetimeIndex):
    def __init__(
        self,
        data: object,
        freq: str | datetime.timedelta | pd.offsets.BaseOffset | None = None,
        tz: datetime.tzinfo | str | None = None,
        ambiguous: np.ndarray | Literal["infer", "raise"] = "raise",
        year1st: bool | None = None,
        day1st: bool | None = None,
        cfg: Configs | None = None,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
        copy: bool = False,
    ) -> Self:
        """A drop-in replacement for Pandas `<'DatetimeIndex'>`
        class, providing additional functionalities for more
        convenient datetime operations.

        :param data `<'object'>`: Datetime-like data to construct with.

        :param freq `<'str/timedelta/BaseOffset/None'>`: The frequency of the 'data', defaults to `None`.
            - `<'str'>` A frequency string (e.g. 'D', 'h', 's', 'ms'), or 'infer' for auto-detection.
            - `<'timedelta'>` A datetime.timedelta instance representing the frequency.
            - `<'BaseOffset'>` A pandas Offset instance representing the frequency.
            - `<'None'>` No specified frequency.

        :param tz `<'str/tzinfo/None'>`: Set the timezone of the index, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.

        :param ambiguous `<'str/ndarray'>`: How to handle ambiguous times, defaults to `'raise'`.
            - `<'str'>` Accepts 'infer' or 'raise' for ambiguous times handling.
            - `<'ndarray'>` A boolean array to specify ambiguous times ('True' for DST time).

        :param year1st `<'bool/None'>`: Interpret the first ambiguous Y/M/D value as year, defaults to `None`.
            If 'year1st=None', use `cfg.year1st` if 'cfg' is specified; otherwise, defaults to `False`.

        :param day1st `<'bool/None'>`: Interpret the first ambiguous Y/M/D values as day, defaults to `None`.
            If 'day1st=None', use `cfg.day1st` if 'cfg' is specified; otherwise, defaults to `False`.

        :param cfg `<'Configs/None'>`: Custom parser configurations, defaults to `None`.

        :param unit `<'str/None'>`: Set the datetime unit of the index, defaults to `None`.
            Supported datetime units: 's', 'ms', 'us', 'ns'.

        :param name `<'Hashable/None'>`: The name assigned to the index, defaults to `None`.

        :param copy `<'bool'>`: Whether to make a copy of the 'data', defaults to `False`.
        """

    def __new__(
        cls,
        data: object,
        freq: str | datetime.timedelta | pd.offsets.BaseOffset | None = None,
        tz: datetime.tzinfo | str | None = None,
        ambiguous: np.ndarray | Literal["infer", "raise"] = "raise",
        year1st: bool | None = None,
        day1st: bool | None = None,
        cfg: Configs | None = None,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
        copy: bool = False,
    ) -> Self: ...
    # Constructor ------------------------------------------------------
    @classmethod
    def date_range(
        cls,
        start: str | datetime.datetime | None = None,
        end: str | datetime.datetime | None = None,
        periods: int | None = None,
        freq: str | datetime.timedelta | pd.offsets.BaseOffset | None = "D",
        tz: datetime.tzinfo | str | None = None,
        normalize: bool = False,
        inclusive: Literal["left", "right", "both", "neither"] = "both",
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
    ) -> Self: ...
    @classmethod
    def parse(
        cls,
        data: object,
        freq: str | datetime.timedelta | pd.offsets.BaseOffset | None = None,
        tz: datetime.tzinfo | str | None = None,
        ambiguous: np.ndarray | Literal["infer", "raise"] = "raise",
        year1st: bool | None = None,
        day1st: bool | None = None,
        cfg: Configs | None = None,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
        copy: bool = False,
    ) -> Self: ...
    @classmethod
    def now(
        cls,
        tz: datetime.tzinfo | str | None = None,
        size: int | str | bytes | object = 1,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
    ) -> Self: ...
    @classmethod
    def utcnow(
        cls,
        size: int | str | bytes | object = 1,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
    ) -> Self: ...
    @classmethod
    def today(
        cls,
        size: int | str | bytes | object = 1,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
    ) -> Self: ...
    @classmethod
    def combine(
        cls,
        date: datetime.date | str | None = None,
        time: datetime.time | str | None = None,
        tz: datetime.tzinfo | str | None = None,
        size: int | str | bytes | object = 1,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
    ) -> Self: ...
    @classmethod
    def fromordinal(
        cls,
        ordinal: int | object,
        tz: datetime.tzinfo | str | None = None,
        size: int | str | bytes | object = 1,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
    ) -> Self: ...
    @classmethod
    def fromseconds(
        cls,
        seconds: int | float | object,
        tz: datetime.tzinfo | str | None = None,
        size: int | str | bytes | object = 1,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
    ) -> Self: ...
    @classmethod
    def fromicroseconds(
        cls,
        us: int | object,
        tz: datetime.tzinfo | str | None = None,
        size: int | str | bytes | object = 1,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
    ) -> Self: ...
    @classmethod
    def fromtimestamp(
        cls,
        ts: int | float | object,
        tz: datetime.tzinfo | str | None = None,
        size: int | str | bytes | object = 1,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
    ) -> Self: ...
    @classmethod
    def utcfromtimestamp(
        cls,
        ts: int | float | object,
        size: int | str | bytes | object = 1,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
    ) -> Self: ...
    @classmethod
    def fromisoformat(
        cls,
        dtstr: str | object,
        size: int | str | bytes | object = 1,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
    ) -> Self: ...
    @classmethod
    def fromisocalendar(
        cls,
        iso: dict | list | tuple | pd.DataFrame,
        year: int | None = None,
        week: int | None = None,
        weekday: int | None = None,
        tz: datetime.tzinfo | str | None = None,
        size: int | str | bytes | object = 1,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
    ) -> Self: ...
    @classmethod
    def fromdate(
        cls,
        date: datetime.date | object,
        tz: datetime.tzinfo | str | None = None,
        size: int | str | bytes | object = 1,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
    ) -> Self: ...
    @classmethod
    def fromdatetime(
        cls,
        dt: datetime.datetime | object,
        size: int | str | bytes | object = 1,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
    ) -> Self: ...
    @classmethod
    def fromdatetime64(
        cls,
        dt64: np.datetime64 | object,
        tz: datetime.tzinfo | str | None = None,
        size: int | str | bytes | object = 1,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
    ) -> Self: ...
    @classmethod
    def strptime(
        cls,
        dtstr: str | object,
        fmt: str,
        size: int | str | bytes | object = 1,
        unit: Literal["s", "ms", "us", "ns"] | None = None,
        name: Hashable | None = None,
    ) -> Self: ...
    # Convertor --------------------------------------------------------
    def ctime(self) -> pd.Index[str]: ...
    def strftime(self, fmt: str) -> pd.Index[str]:
        """Convert to index of strings according to the given format `<'Index[str]'>`.

        :param format `<'str'>`: The format of the datetime strings.
        """

    def isoformat(self, sep: str = "T") -> pd.Index[str]: ...
    def timedf(self) -> pd.DataFrame: ...
    def utctimedf(self) -> pd.DataFrame: ...
    def toordinal(self) -> pd.Index[np.int64]: ...
    def seconds(self, utc: bool = False) -> pd.Index[np.float64]: ...
    def microseconds(self, utc: bool = False) -> pd.Index[np.int64]: ...
    def timestamp(self) -> pd.Index[np.float64]: ...
    def datetime(self) -> np.ndarray[datetime.datetime]: ...
    def date(self) -> np.ndarray[datetime.date]: ...
    def time(self) -> np.ndarray[datetime.time]: ...
    def timetz(self) -> np.ndarray[datetime.time]: ...
    def to_period(self, freq: str | pd.offsets.BaseOffset | None) -> pd.PeriodIndex:
        """Cast to PeriodIndex at a particular frequency `<PeriodIndex>`."""

    def to_series(
        self,
        index: pd.Index | None = None,
        name: Hashable | None = None,
    ) -> pd.Series: ...
    def to_frame(
        self,
        index: bool = True,
        name: Hashable = lib.no_default,
    ) -> pd.DataFrame: ...
    # Manipulator ------------------------------------------------------
    def replace(
        self,
        year: int = -1,
        month: int = -1,
        day: int = -1,
        hour: int = -1,
        minute: int = -1,
        second: int = -1,
        microsecond: int = -1,
        nanosecond: int = -1,
        tzinfo: datetime.tzinfo | str | None = -1,
    ) -> Self: ...
    # . year
    def to_curr_year(self, month: int | str | None = None, day: int = -1) -> Self: ...
    def to_prev_year(self, month: int | str | None = None, day: int = -1) -> Self: ...
    def to_next_year(self, month: int | str | None = None, day: int = -1) -> Self: ...
    def to_year(
        self,
        offset: int,
        month: int | str | None = None,
        day: int = -1,
    ) -> Self: ...
    # . quarter
    def to_curr_quarter(self, month: int = -1, day: int = -1) -> Self: ...
    def to_prev_quarter(self, month: int = -1, day: int = -1) -> Self: ...
    def to_next_quarter(self, month: int = -1, day: int = -1) -> Self: ...
    def to_quarter(self, offset: int, month: int = -1, day: int = -1) -> Self: ...
    # . month
    def to_curr_month(self, day: int = -1) -> Self: ...
    def to_prev_month(self, day: int = -1) -> Self: ...
    def to_next_month(self, day: int = -1) -> Self: ...
    def to_month(self, offset: int, day: int = -1) -> Self: ...
    # . weekday
    def to_monday(self) -> Self: ...
    def to_tuesday(self) -> Self: ...
    def to_wednesday(self) -> Self: ...
    def to_thursday(self) -> Self: ...
    def to_friday(self) -> Self: ...
    def to_saturday(self) -> Self: ...
    def to_sunday(self) -> Self: ...
    def to_curr_weekday(self, weekday: int | str | None = None) -> Self: ...
    def to_prev_weekday(self, weekday: int | str | None = None) -> Self: ...
    def to_next_weekday(self, weekday: int | str | None = None) -> Self: ...
    def to_weekday(self, offset: int, weekday: int | str | None = None) -> Self: ...
    # . day
    def to_yesterday(self) -> Self: ...
    def to_tomorrow(self) -> Self: ...
    def to_day(self, offset: int) -> Self: ...
    # . date&time
    def normalize(self) -> Self:
        """Set the time fields to midnight (i.e. 00:00:00) `<'Pddt'>`.

        This method is useful in cases, when the time does not matter.
        Length is unaltered. The timezones are unaffected.
        """

    def snap(self, freq: str | pd.offsets.BaseOffset) -> Self: ...
    def to_datetime(
        self,
        year: int = -1,
        month: int = -1,
        day: int = -1,
        hour: int = -1,
        minute: int = -1,
        second: int = -1,
        microsecond: int = -1,
        nanosecond: int = -1,
    ) -> Self: ...
    def to_date(self, year: int = -1, month: int = -1, day: int = -1) -> Self: ...
    def to_time(
        self,
        hour: int = -1,
        minute: int = -1,
        second: int = -1,
        microsecond: int = -1,
        nanosecond: int = -1,
    ) -> Self: ...
    def to_first_of(self, unit: str | Literal["Y", "Q", "M", "W"]) -> Self: ...
    def to_last_of(self, unit: str | Literal["Y", "Q", "M", "W"]) -> Self: ...
    def to_start_of(
        self,
        unit: str | Literal["Y", "Q", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"],
    ) -> Self: ...
    def to_end_of(
        self,
        unit: str | Literal["Y", "Q", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"],
    ) -> Self: ...
    # . round / ceil / floor
    def round(self, unit: Literal["D", "h", "m", "s", "ms", "us", "ns"]) -> Self: ...
    def ceil(self, unit: Literal["D", "h", "m", "s", "ms", "us", "ns"]) -> Self: ...
    def floor(self, unit: Literal["D", "h", "m", "s", "ms", "us", "ns"]) -> Self: ...
    # Calendar ---------------------------------------------------------
    # . iso
    def isocalendar(self) -> pd.DataFrame: ...
    def isoyear(self) -> pd.Index[np.int64]: ...
    def isoweek(self) -> pd.Index[np.int64]: ...
    def isoweekday(self) -> pd.Index[np.int64]: ...
    # . year
    @property
    def year(self) -> pd.Index[np.int64]: ...
    def is_leap_year(self) -> pd.Index[bool]: ...
    def is_long_year(self) -> pd.Index[bool]: ...
    def leap_bt_year(self, year: int) -> pd.Index[np.int64]: ...
    def days_in_year(self) -> pd.Index[np.int64]: ...
    def days_bf_year(self) -> pd.Index[np.int64]: ...
    def days_of_year(self) -> pd.Index[np.int64]: ...
    def is_year(self, year: int) -> pd.Index[bool]: ...
    # . quarter
    @property
    def quarter(self) -> pd.Index[np.int64]: ...
    def days_in_quarter(self) -> pd.Index[np.int64]: ...
    def days_bf_quarter(self) -> pd.Index[np.int64]: ...
    def days_of_quarter(self) -> pd.Index[np.int64]: ...
    def is_quarter(self, quarter: int) -> pd.Index[bool]: ...
    # . month
    @property
    def month(self) -> pd.Index[np.int64]: ...
    def days_in_month(self) -> pd.Index[np.int64]: ...
    def days_bf_month(self) -> pd.Index[np.int64]: ...
    def days_of_month(self) -> pd.Index[np.int64]: ...
    def is_month(self, month: str | int) -> pd.Index[bool]: ...
    def month_name(self, locale: str | None = None) -> pd.Index[str]:
        """Return the month names with specified locale `<'Index[str]'>`.

        :param locale `<'str/None'>`: The locale to use for month names, defaults to `None`.
            - Locale determining the language in which to return the month
              name. Default (`None`) is English locale ('en_US.utf8').
            - Use the command locale -a on your terminal on Unix systems to
              find your locale language code.
        """
    # . weekday
    @property
    def weekday(self) -> pd.Index[np.int64]: ...
    def is_weekday(self, weekday: int | str) -> pd.Index[bool]: ...
    # . day
    @property
    def day(self) -> pd.Index[np.int64]: ...
    def is_day(self, day: int) -> pd.Index[bool]: ...
    def day_name(self, locale: str | None = None) -> pd.Index[str]:
        """Return the weekday names with specified locale `<'Index[str]'>`.

        :param locale `<'str/None'>`: The locale to use for weekday names, defaults to `None`.
            - Locale determining the language in which to return the weekday
              name. Default (`None`) is English locale ('en_US.utf8').
            - Use the command locale -a on your terminal on Unix systems to
              find your locale language code.
        """
    # . time
    @property
    def hour(self) -> pd.Index[np.int64]: ...
    @property
    def minute(self) -> pd.Index[np.int64]: ...
    @property
    def second(self) -> pd.Index[np.int64]: ...
    @property
    def millisecond(self) -> pd.Index[np.int64]: ...
    @property
    def microsecond(self) -> pd.Index[np.int64]: ...
    @property
    def nanosecond(self) -> pd.Index[np.int64]: ...
    # . date&time
    def is_first_of(
        self,
        unit: str | Literal["Y", "Q", "M", "W"],
    ) -> pd.Index[bool]: ...
    def is_last_of(
        self,
        unit: str | Literal["Y", "Q", "M", "W"],
    ) -> pd.Index[bool]: ...
    def is_start_of(
        self,
        unit: str | Literal["Y", "Q", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"],
    ) -> pd.Index[bool]: ...
    def is_end_of(
        self,
        unit: str | Literal["Y", "Q", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"],
    ) -> pd.Index[bool]: ...
    # Timezone -----------------------------------------------------------------------------
    @property
    def tz_available(self) -> set[str]: ...
    @property
    def tz(self) -> datetime.tzinfo | None:
        """The timezone information `<'tzinfo/None'>`."""

    @property
    def tzinfo(self) -> datetime.tzinfo | None:
        """The timezone information `<'tzinfo/None'>`."""

    def is_local(self) -> bool: ...
    def is_utc(self) -> bool: ...
    def tzname(self) -> str: ...
    def astimezone(
        self,
        tz: datetime.tzinfo | str | None = None,
        ambiguous: np.ndarray | Literal["infer", "raise"] = "raise",
        nonexistent: (
            datetime.timedelta | Literal["shift_forward", "shift_backward", "raise"]
        ) = "raise",
    ) -> Self: ...
    def tz_localize(
        self,
        tz: datetime.tzinfo | str | None,
        ambiguous: np.ndarray | Literal["infer", "raise"] = "raise",
        nonexistent: (
            datetime.timedelta | Literal["shift_forward", "shift_backward", "raise"]
        ) = "raise",
    ) -> Self: ...
    def tz_convert(self, tz: datetime.tzinfo | str | None) -> Self: ...
    def tz_switch(
        self,
        targ_tz: datetime.tzinfo | str | None,
        base_tz: datetime.tzinfo | str | None = None,
        naive: bool = False,
        ambiguous: np.ndarray | Literal["infer", "raise"] = "raise",
        nonexistent: (
            datetime.timedelta | Literal["shift_forward", "shift_backward", "raise"]
        ) = "raise",
    ) -> Self: ...
    # Values -------------------------------------------------------------------------------
    @property
    def freq(self) -> pd.offsets.BaseOffset | None:
        """Return the frequency object if it's set, otherwise None. `<'BaseOffset/None'>`."""

    @property
    def freqstr(self) -> str | None: ...
    @property
    def inferred_freq(self) -> str | None:
        """Tries to return a string representing a frequency.
        Return `None` if it can't autodetect the frequency.
        """

    @property
    def values_naive(self) -> np.ndarray[np.datetime64]: ...
    def as_unit(self, unit: Literal["s", "ms", "us", "ns"]) -> Self: ...
    # Arithmetic ---------------------------------------------------------------------------
    def add(
        self,
        years: int = 0,
        quarters: int = 0,
        months: int = 0,
        weeks: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        milliseconds: int = 0,
        microseconds: int = 0,
        nanoseconds: int = 0,
    ) -> Self: ...
    def sub(
        self,
        years: int = 0,
        quarters: int = 0,
        months: int = 0,
        weeks: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        milliseconds: int = 0,
        microseconds: int = 0,
        nanoseconds: int = 0,
    ) -> Self: ...
    def diff(
        self,
        data: object,
        unit: Literal["Y", "Q", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"],
        absolute: bool = False,
        inclusive: Literal["one", "both", "neither"] = "both",
    ) -> np.ndarray[np.int64]: ...
    def mean(*, skipna: bool = True, axis: int = 0) -> pd.Timestamp:
        """Return the mean of the values `<'Timestamp'>`.

        :param skipna `<'bool'>`: Exclude NA/null values, defaults to `True`.
        :param axis `<'int'>`: The axis to use, defaults to `0`.
        """
    # Comparison ---------------------------------------------------------------------------
    def is_past(self) -> pd.Index[bool]: ...
    def is_future(self) -> pd.Index[bool]: ...
    def closest(self, data: object) -> Self: ...
    def farthest(self, dtobjs: object) -> Self: ...
