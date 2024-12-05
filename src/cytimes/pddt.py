# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from __future__ import annotations

# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_READ_CHAR as str_read  # type: ignore
from cython.cimports.cpython.dict import PyDict_GetItem as dict_getitem  # type: ignore
from cython.cimports.cytimes.parser import parse_dtobj as _parse, Configs, CONFIG_MONTH, CONFIG_WEEKDAY  # type: ignore
from cython.cimports.cytimes import utils  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
import datetime, numpy as np
from typing_extensions import Self
from typing import Literal, Hashable
from pandas._libs import lib
from pandas.core.arrays.datetimes import DatetimeArray
from pandas import Index, DatetimeIndex, Series, DataFrame, Timestamp
from pandas import errors as pd_err
from pandas._libs.tslibs.parsing import DateParseError
from pytz import exceptions as pytz_err
from zoneinfo import available_timezones as _available_timezones
from cytimes.parser import Configs, parse_dtobj as _parse
from cytimes import utils, errors

__all__ = ["Pddt"]


# Utils ---------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def pddt_new(
    data: object,
    freq: object | None = None,
    tz: datetime.tzinfo | str | None = None,
    ambiguous: object = "raise",
    year1st: bool | None = False,
    day1st: bool | None = False,
    cfg: Configs = None,
    unit: str = None,
    name: Hashable | None = None,
    copy: bool = False,
) -> object:
    """(cfunc) Construct a new Pddt instance `<'Pddt'>`.

    :param data `<'object'>`: Datetime-like data to construct with.

    :param freq `<'str/timedelta/BaseOffset/None'>`: The frequency of the 'data', defaults to `None`.
        - `<'str'>` A frequency string (e.g. 'D', 'h', 's', 'ms'), or 'infer' for auto-detection.
        - `<'BaseOffset'>` A pandas Offset instance representing the frequency.
        - `<'timedelta'>` A datetime.timedelta instance representing the frequency.
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
    try:
        # Default: DatetimeIndex
        y1st = cfg._year1st if year1st is None else year1st
        d1st = cfg._day1st if day1st is None else day1st
        pt: Pddt = _pddt_new(data, freq, tz, ambiguous, y1st, d1st, name, copy)
    except (pd_err.OutOfBoundsDatetime, DateParseError):
        # Fallback: cytimes parser
        pt: Pddt = _pddt_new_cytimes(
            data, freq, tz, ambiguous, year1st, day1st, cfg, name
        )
    except errors.InvalidArgumentError:
        raise
    except pytz_err.AmbiguousTimeError as err:
        raise errors.AmbiguousTimeError(err) from err
    except Exception as err:
        raise errors.InvalidArgumentError(err) from err

    # Convert to specified unit
    return pt if unit is None else pt.as_unit(unit)


@cython.cfunc
@cython.inline(True)
def pddt_new_simple(
    data: object,
    tz: datetime.tzinfo | str | None = None,
    unit: str = None,
    name: Hashable | None = None,
) -> object:
    """(cfunc) Construct a new Pddt instance (simplified arguments) `<'Pddt'>`.

    :param data `<'object'>`: Datetime-like data to construct with.

    :param tz `<'str/tzinfo/None'>`: Set the timezone of the index, defaults to `None`.
        - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
        - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
        - `<'None'>` Timezone-naive.

    :param unit `<'str/None'>`: Set the datetime unit of the index, defaults to `None`.
        Supported datetime units: 's', 'ms', 'us', 'ns'.

    :param name `<'Hashable/None'>`: The name assigned to the index, defaults to `None`.
    """
    pt: Pddt = pddt_new(
        data, None, None, "raise", False, False, None, unit, name, False
    )
    if tz is None:
        return pt
    return pt.tz_localize(tz, "infer", "shift_forward")


@cython.cfunc
@cython.inline(True)
def _pddt_new(
    data: object,
    freq: object | None = None,
    tz: datetime.tzinfo | str | None = None,
    ambiguous: object = "raise",
    year1st: bool = False,
    day1st: bool = False,
    name: Hashable | None = None,
    copy: bool = False,
) -> object:
    """(internal) Construct a new Pddt instance `<'Pddt'>`.

    This is an internal function, use the 'DatetimeIndex.__new__'
    method to instantiate a new 'Pddt'. This function does not
    handle `OutOfBoundsDatetime` or `DateParseError` exceptions.
    """
    # Parse frequency
    if freq is None:
        freq = lib.no_default
    elif freq == "m":
        freq = "min"
    # Prase timezone
    tz = utils.tz_parse(tz)
    if tz is None:
        tz = lib.no_default
    elif ambiguous == "NaT":
        raise errors.InvalidArgumentError("ambiguous='NaT' is not supported.")
    # New instance
    try:
        return DatetimeIndex.__new__(
            Pddt,
            data=data,
            freq=freq,
            tz=tz,
            ambiguous=ambiguous,
            dayfirst=day1st,
            yearfirst=year1st,
            dtype=None,
            copy=copy,
            name=name,
        )
    except (TypeError, ValueError) as err:
        msg: str = str(err)
        if "mixed timezones" not in msg and "unless utc=True" not in msg:
            raise err
        return DatetimeIndex.__new__(
            Pddt,
            data=data,
            freq=freq,
            tz=utils.UTC,
            ambiguous=ambiguous,
            dayfirst=day1st,
            yearfirst=year1st,
            dtype=None,
            copy=copy,
            name=name,
        )


@cython.cfunc
@cython.inline(True)
def _pddt_new_cytimes(
    data: object,
    freq: object | None = None,
    tz: datetime.tzinfo | str | None = None,
    ambiguous: np.ndarray[np.bool_] | Literal["infer", "raise"] = "raise",
    year1st: bool | None = None,
    day1st: bool | None = None,
    cfg: Configs = None,
    name: Hashable | None = None,
) -> object:
    """(internal) Construct a new Pddt instance using
    the 'cytimes.parse()' `<'Pddt'>`.

    This is an internal function, use to handle fallback
    exceptions `OutOfBoundsDatetime` or `DateParseError`
    when '_pddt_new' failed.
    """
    # Get array size
    arr_size = _parse_arr_size(data)

    # Parse with cytimes
    try:
        tz = utils.tz_parse(tz)
        tz_repl = utils.UTC if tz is None else tz
        arr: np.ndarray = np.PyArray_EMPTY(1, [arr_size], np.NPY_TYPES.NPY_INT64, 0)
        arr_ptr = cython.cast(cython.pointer(np.npy_int64), np.PyArray_DATA(arr))
        i: cython.Py_ssize_t = 0
        for item in data:
            dt: datetime.datetime = _parse(
                item, None, year1st, day1st, False, True, cfg
            )
            if dt.tzinfo is not None:
                dt = utils.dt_astimezone(dt, tz_repl)
                if tz is None:
                    tz = utils.UTC
            arr_ptr[i] = utils.dt_to_us(dt, False)
            i += 1
        arr = arr.astype("datetime64[us]")
    except errors.ParserError as err:
        raise errors.InvalidArgumentError(err) from err
    except errors.InvalidArgumentError:
        raise
    except Exception as err:
        raise errors.InvalidArgumentError(err) from err

    # New instance
    pt: Pddt = _pddt_new(arr, freq, None, ambiguous, False, False, name, False)
    # . apply timezone
    if tz is None:
        return pt
    return pt.tz_localize(tz, ambiguous, "shift_forward")


@cython.cfunc
@cython.inline(True)
def _pddt_fr_dt(
    dt: datetime.datetime,
    size: cython.Py_ssize_t,
    unit: str = None,
    name: Hashable | None = None,
) -> object:
    """(internal) Construct a new Pddt instance from
    an instance of datetime `<'Pddt'>`.
    """
    us = utils.dt_to_us(dt, False)
    return _pddt_fr_us(us, size, dt.tzinfo, unit, name)


@cython.cfunc
@cython.inline(True)
def _pddt_fr_us(
    val: cython.longlong,
    size: cython.Py_ssize_t,
    tz: datetime.tzinfo | None = None,
    unit: str = None,
    name: Hashable | None = None,
) -> object:
    """(internal) Construct a new Pddt instance from
    total microseconds since Unix Epoch `<'Pddt'>`.
    """
    arr: np.ndarray = utils.dt64arr_fr_int64(val, "us", size)
    return pddt_new_simple(arr, tz, unit, name)


@cython.cfunc
@cython.inline(True)
def _parse_dtobj(
    dtobj: object,
    default: object | None = None,
    year1st: bool | None = None,
    day1st: bool | None = None,
    ignoretz: cython.bint = False,
    isoformat: cython.bint = True,
    cfg: Configs = None,
) -> datetime.datetime:
    """(internal) Parse datetime-like object into
    datetime instance `<'datetime.datetime'>.

    For information about the parameters, please refer to
    the 'cytimes.parser.parse_dtobj()' function.
    """
    try:
        return _parse(dtobj, default, year1st, day1st, ignoretz, isoformat, cfg)
    except Exception as err:
        raise errors.InvalidArgumentError(err) from err


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-2, check=False)
def _parse_month(month: object, raise_error: cython.bint) -> cython.int:
    """(internal) Parse the 'month' value to month number (1=Jan...12=Dec) `<'int'>`.

    :param month `<'int/str'>`: Month value.
        - `<'int'>` Month number (1=Jan...12=Dec).
        - `<'str'>` Month name (case-insensitive), e.g., 'Jan', 'februar', '三月'.

    :param raise_error `<'bool'>`: Whether to raise error if the 'month' value is invalid.
        - If `True`, raises `InvalidMonthError` for invalid input.
        - If `False`, returns `-1` instead.

    ## Notice:
    - Return `-1` directly if 'month' is `None` or `-1`.
    """
    # <'None'>
    if month is None:
        return -1  # exit

    # <'int'>
    if isinstance(month, int):
        num: cython.longlong = month
        if num == -1:
            return -1  # exit
        if not 1 <= num <= 12:
            if raise_error:
                raise errors.InvalidMonthError(
                    "invalid month number '%d', must betweem 1(Jan)...12(Dec)." % num
                )
            return -1  # exit
        return num  # exit

    # <'str'>
    if isinstance(month, str):
        mth: str = month
        val = dict_getitem(CONFIG_MONTH, mth.lower())
        if val == cython.NULL:
            if raise_error:
                raise errors.InvalidMonthError("invalid month name '%s'." % mth)
            return -1  # eixt
        return cython.cast(object, val)  # exit

    # Invalid
    if raise_error:
        raise errors.InvalidMonthError(
            "unsupported month value type %s, expects <'str/int'>." % type(month)
        )
    return -1


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-2, check=False)
def _parse_weekday(weekday: object, raise_error: cython.bint) -> cython.int:
    """(internal) Parse the 'weekday' value to weekday number (0=Mon...6=Sun) `<'int'>`.

    :param weekday `<'int/str/None'>`: Weekday value.
        - `<'int'>` Weekday number (0=Mon...6=Sun).
        - `<'str'>` Weekday name (case-insensitive), e.g., 'Mon', 'dienstag', '星期三'.

    :param raise_error `<'bool'>`: Whether to raise error if the 'weekday' value is invalid.
        - If `True`, raises `InvalidWeekdayError` for invalid input.
        - If `False`, returns `-1` instead.

    ## Notice:
    - Return `-1` directly if 'weekday' is `None` or `-1`.
    """
    # <'None'>
    if weekday is None:
        return -1

    # <'int'>
    if isinstance(weekday, int):
        num: cython.longlong = weekday
        if num == -1:
            return -1  # exit
        if not 0 <= num <= 6:
            if raise_error:
                raise errors.InvalidWeekdayError(
                    "invalid weekday number '%d', must betweem 0(Mon)...6(Sun)." % num
                )
            return -1
        return num  # exit

    # <'str'>
    if isinstance(weekday, str):
        wkd: str = weekday
        val = dict_getitem(CONFIG_WEEKDAY, wkd.lower())
        if val == cython.NULL:
            if raise_error:
                raise errors.InvalidWeekdayError("invalid weekday name '%s'." % wkd)
            return -1
        return cython.cast(object, val)  # exit

    # Invalid
    if raise_error:
        raise errors.InvalidWeekdayError(
            "unsupported weekday value type %s, expects <'str/int'>." % type(weekday)
        )
    return -1


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-2, check=False)
def _parse_arr_size(obj: object) -> cython.Py_ssize_t:
    """(internal) Parse the object to an integer
    representing the size of an array `<'int'>`.

    - For `<'int/str/float/bytes'>`, converts to an integer representing the array size.
    - For other types, trys to get the length of the object to represent the array size.
    """
    # int / str / float / bytes
    if isinstance(obj, (int, float, str, bytes)):
        try:
            size: cython.Py_ssize_t = int(obj)
        except Exception as err:
            raise errors.InvalidTypeError(
                "cannot convert '%s' to an integer to represent the array size." % obj
            ) from err
    # Possible Array-like
    else:
        try:
            size: cython.Py_ssize_t = len(obj)
        except Exception as err:
            raise errors.InvalidTypeError(
                "cannot get the length of %s to represent the array size." % type(obj)
            ) from err
    # Validate size
    if size < 1:
        raise errors.InvalidArgumentError(
            "array size must be and integer > 0, instead got '%d'." % size
        )
    return size


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-2, check=False)
def _raise_incomparable_error(
    pt1: object,
    pt2: object,
    desc: str = "compare",
) -> cython.bint:
    """(internal) Raise an `IncomparableError` for comparison
    between timezone-naive & timezone-aware DatetimeIndex.

    :param pt1 `<'DatetimeIndex'>`: The first instance.
    :param pt2 `<'DatetimeIndex'>`: The second instance.
    :param desc `<'str'>`: The description for the comparision, defaults to `'compare'`.
        Displayed as: 'cannot [desc] between naive & aware datetimes...'
    """
    d1_tz, d2_tz = pt1.tzinfo, pt2.tzinfo
    assert d1_tz is not d2_tz and (d1_tz is None or d2_tz is None)
    if d1_tz is None:
        raise errors.IncomparableError(
            "cannot %s between naive & aware datetimes:\n"
            "Timezone-naive %s\n"
            "Timezone-aware %s" % (desc, pt1.dtype, pt2.dtype)
        )
    else:
        raise errors.IncomparableError(
            "cannot %s between naive & aware datetimes:\n"
            "Timezone-aware %s\n"
            "Timezone-naive %s" % (desc, pt1.dtype, pt2.dtype)
        )


# Pddt (Pandas Datetime) ----------------------------------------------------------------------
class Pddt(DatetimeIndex):
    """A drop-in replacement for Pandas `<'DatetimeIndex'>`
    class, providing additional functionalities for more
    convenient datetime operations.
    """

    def __new__(
        cls,
        data: object,
        freq: object | None = None,
        tz: datetime.tzinfo | str | None = None,
        ambiguous: object = "raise",
        year1st: bool | None = None,
        day1st: bool | None = None,
        cfg: Configs = None,
        unit: str = None,
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
        # fmt: off
        return pddt_new(
            data, freq, tz, ambiguous, year1st, 
            day1st, cfg, unit, name, copy,
        )
        # fmt: on

    # Constructor --------------------------------------------------------------------------
    @classmethod
    def date_range(
        cls,
        start: object | None = None,
        end: object | None = None,
        periods: int | None = None,
        freq: object | None = "D",
        tz: datetime.tzinfo | str | None = None,
        normalize: bool = False,
        inclusive: str = "both",
        unit: str = None,
        name: Hashable | None = None,
    ) -> Self:
        """Construct a fixed-frequency index `<'Pddt'>`.

        Construct the range of equally spaced time points at a specified frequency.
        The datetimes are generated such that each one falls within the interval defined
        by 'start' and 'end', satisfying 'start <[=] x <[=] end' (the inclusivity can
        be adjusted with the 'inclusive' parameter). If exactly one of 'start', 'end',
        or 'freq' is not provided, it can be inferred by specifying periods, the total
        number of datetimes to generate.

        :param start `<'str/datetime/None'>`: Left bound for generating the index.
        :param end `<'str/datetime/None'>`: Right bound for generating the index.
        :param periods `<'int/None'>`: Number of periods to generate.
        :param freq `<'str/timedelta/BaseOffset/None'>`: The frequency of the index, defaults to `'D'`.
            - `<'str'>` A frequency string (e.g. 'D', 'h', 's', 'ms').
            - `<'BaseOffset'>` A pandas Offset instance representing the frequency.
            - `<'timedelta'>` A datetime.timedelta instance representing the frequency.
            - `<'None'>` No specified frequency.

        :param tz `<'str/tzinfo/None'>`: Set the timezone of the index, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.

        :param normalize `<'bool'>`: Normalize the start/end dates to midnight, defaults to `False`.

        :param inclusive `<'str'>`: Include boundaries, defaults to `'both'`.
            - `'left'` Include the left boundary.
            - `'right'` Include the right boundary.
            - `'both'` Include both boundaries.
            - `'neither'` Exclude both boundaries.

        :param unit `<'str/None'>`: Set the datetime unit of the index, defaults to `None`.
            Supported datetime units: 's', 'ms', 'us', 'ns'.

        :param name `<'Hashable/None'>`: The name assigned to the index, defaults to `None`.
        """
        if freq is None and (periods is None or start is None or end is None):
            freq = "D"
        try:
            arr = DatetimeArray._generate_range(
                start=start,
                end=end,
                periods=periods,
                freq=freq,
                tz=tz,
                normalize=normalize,
                inclusive=inclusive,
                unit=unit,
            )
        except pd_err.OutOfBoundsDatetime as err:
            raise errors.OutOfBoundsDatetimeError(err) from err
        except Exception as err:
            raise errors.InvalidArgumentError(err) from err
        return pddt_new_simple(arr, None, unit, name)

    @classmethod
    def parse(
        cls,
        data: object,
        freq: object | None = None,
        tz: datetime.tzinfo | str | None = None,
        ambiguous: object = "raise",
        year1st: bool | None = None,
        day1st: bool | None = None,
        cfg: Configs = None,
        unit: str = None,
        name: Hashable | None = None,
        copy: bool = False,
    ) -> Self:
        """Parse from a datetime-like data `<'Pddt'>`.

        :param data `<'object'>`: Datetime-like data to construct with.

        :param freq `<'str/timedelta/BaseOffset/None'>`: The frequency of the 'data', defaults to `None`.
            - `<'str'>` A frequency string (e.g. 'D', 'h', 's', 'ms'), or 'infer' for auto-detection.
            - `<'BaseOffset'>` A pandas Offset instance representing the frequency.
            - `<'timedelta'>` A datetime.timedelta instance representing the frequency.
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
        # fmt: off
        return pddt_new(
            data, freq, tz, ambiguous, year1st, 
            day1st, cfg, unit, name, copy,
        )
        # fmt: on

    @classmethod
    def now(
        cls,
        tz: datetime.tzinfo | str | None = None,
        size: object = 1,
        unit: str = None,
        name: Hashable | None = None,
    ) -> Self:
        """Construct from the current datetime with optional timezone `<'Pddt'>`.

        :param tz `<'str/tzinfo/None'>`: The timezone of the index, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.

        :param size `<'int/str/bytes/object'>`: The size of the index to generate, defaults to `1`.
            - `<'int/str/bytes'>` Representing an integer for the number of elements in the index.
            - `<'object'>` Any object that implements `__len__()`, where its length represents the size of the index.

        :param unit `<'str/None'>`: Set the datetime unit of the index, defaults to `None`.
            Supported datetime units: 's', 'ms', 'us', 'ns'.

        :param name `<'Hashable/None'>`: The name assigned to the index, defaults to `None`.
        """
        arr_size, tz = _parse_arr_size(size), utils.tz_parse(tz)
        return _pddt_fr_dt(utils.dt_now(tz), arr_size, unit, name)

    @classmethod
    def utcnow(
        cls,
        size: object = 1,
        unit: str = None,
        name: Hashable | None = None,
    ) -> Self:
        """Construct from the current UTC datetime `<'Pddt'>`.

        :param size `<'int/str/bytes/object'>`: The size of the index to generate, defaults to `1`.
            - `<'int/str/bytes'>` Representing an integer for the number of elements in the index.
            - `<'object'>` Any object that implements `__len__()`, where its length represents the size of the index.

        :param unit `<'str/None'>`: Set the datetime unit of the index, defaults to `None`.
            Supported datetime units: 's', 'ms', 'us', 'ns'.

        :param name `<'Hashable/None'>`: The name assigned to the index, defaults to `None`.
        """
        arr_size = _parse_arr_size(size)
        return _pddt_fr_dt(utils.dt_now(utils.UTC), arr_size, unit, name)

    @classmethod
    def today(
        cls,
        size: object = 1,
        unit: str = None,
        name: Hashable | None = None,
    ) -> Self:
        """Construct from the current local datetime (timezone-naive) `<'Pddt'>`.

        :param size `<'int/str/bytes/object'>`: The size of the index to generate, defaults to `1`.
            - `<'int/str/bytes'>` Representing an integer for the number of elements in the index.
            - `<'object'>` Any object that implements `__len__()`, where its length represents the size of the index.

        :param unit `<'str/None'>`: Set the datetime unit of the index, defaults to `None`.
            Supported datetime units: 's', 'ms', 'us', 'ns'.

        :param name `<'Hashable/None'>`: The name assigned to the index, defaults to `None`.
        """
        arr_size = _parse_arr_size(size)
        return _pddt_fr_dt(utils.dt_now(None), arr_size, unit, name)

    @classmethod
    def combine(
        cls,
        date: datetime.date | str | None = None,
        time: datetime.time | str | None = None,
        tz: datetime.tzinfo | str | None = None,
        size: object = 1,
        unit: str = None,
        name: Hashable | None = None,
    ) -> Self:
        """Combine date and time into datetime index `<'Pddt'>`.

        :param date `<'date/str/None'>`: A date-like object, defaults to `None`.
            - `<'datetime.date'>` An instance of `datetime.date`.
            - `<'str'>` A date string.
            - `<'None'>` Use the current local date.

        :param time `<'time/str/None'>`: A time-like object, defaults to `None`.
            - `<'datetime.time'>` An instance of `datetime.time`.
            - `<'str'>` A time string.
            - `<'None'>` Use the current local time.

        :param tz `<'str/tzinfo/None'>`: The timezone of the index, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.

        :param size `<'int/str/bytes/object'>`: The size of the index to generate, defaults to `1`.
            - `<'int/str/bytes'>` Representing an integer for the number of elements in the index.
            - `<'object'>` Any object that implements `__len__()`, where its length represents the size of the index.

        :param unit `<'str/None'>`: Set the datetime unit of the index, defaults to `None`.
            Supported datetime units: 's', 'ms', 'us', 'ns'.

        :param name `<'Hashable/None'>`: The name assigned to the index, defaults to `None`.
        """
        arr_size, tz = _parse_arr_size(size), utils.tz_parse(tz)
        if date is not None and not utils.is_date(date):
            date = _parse_dtobj(date)
        if time is not None and not utils.is_time(time):
            time = _parse_dtobj(time, datetime.date_new(1970, 1, 1)).timetz()
        return _pddt_fr_dt(utils.dt_combine(date, time, tz), arr_size, unit, name)

    @classmethod
    def fromordinal(
        cls,
        ordinal: int | object,
        tz: datetime.tzinfo | str | None = None,
        size: object = 1,
        unit: str = None,
        name: Hashable | None = None,
    ) -> Self:
        """Construct from Gregorian ordinal days with optional timezone `<'Pddt'>`.

        :param ordinal `<'int/Array-like'>`: The ordinal days to construct with.
            - `<'int'>` An integer representing the ordinal days.
            - `<'Array-like'>` An array-like object containing the ordinal days.

        :param tz `<'str/tzinfo/None'>`: The timezone of the index, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.

        :param size `<'int/str/bytes/object'>`: The size of the index to generate, defaults to `1`.
            - This argument is `ignored` if 'ordinal' is an array-like object.
            - `<'int/str/bytes'>` Representing an integer for the number of elements in the index.
            - `<'object'>` Any object that implements `__len__()`, where its length represents the size of the index.

        :param unit `<'str/None'>`: Set the datetime unit of the index, defaults to `None`.
            Supported datetime units: 's', 'ms', 'us', 'ns'.

        :param name `<'Hashable/None'>`: The name assigned to the index, defaults to `None`.
        """
        # Validate
        arr: np.ndarray
        arr_size: cython.Py_ssize_t
        val: cython.longlong

        # Single value
        if isinstance(ordinal, (int, float, str, bytes)):
            # . convert to microseconds
            arr_size = _parse_arr_size(size)
            try:
                val = int(ordinal)
            except Exception as err:
                raise errors.InvalidTypeError(
                    "cannot convert '%s' to an integer "
                    "to represent the ordinal value." % ordinal
                ) from err
            val = min(max(val, 1), utils.ORDINAL_MAX)
            val = (val - utils.EPOCH_DAY) * utils.US_DAY
            # . new instance
            return _pddt_fr_us(val, arr_size, tz, unit, name)

        # <'list'/'tuple'>
        if isinstance(ordinal, (list, tuple)):
            # . convert to ndarray[datetime64]
            arr_size = len(ordinal)
            arr = np.PyArray_EMPTY(1, [arr_size], np.NPY_TYPES.NPY_INT64, 0)
            arr_ptr = cython.cast(cython.pointer(np.npy_int64), np.PyArray_DATA(arr))
            i: cython.Py_ssize_t = 0
            for item in ordinal:
                try:
                    val = int(item)
                except Exception as err:
                    raise errors.InvalidTypeError(
                        "cannot convert '%s' to an integer "
                        "to represent the ordinal value." % item
                    ) from err
                val = min(max(val, 1), utils.ORDINAL_MAX)
                val = (val - utils.EPOCH_DAY) * utils.US_DAY
                arr_ptr[i] = val
                i += 1
            arr = arr.astype("datetime64[us]")
            # . new instance
            return pddt_new_simple(arr, tz, unit, name)

        # <'pandas.Series'>
        if isinstance(ordinal, (Series, Index)):
            ordinal = ordinal.values

        # <'numpy.ndarray'>
        if isinstance(ordinal, np.ndarray):
            # . validate array
            arr = ordinal
            if arr.ndim != 1:
                raise errors.InvalidTypeError(
                    "must be an 1-dimensional ndarray, instead of %d-dim." % arr.ndim
                )
            # . convert to ndarray[datetime64]
            arr = utils.arr_assure_int64_like(arr)
            arr = utils.arr_clip(arr, 1, utils.ORDINAL_MAX, -utils.EPOCH_DAY)
            arr = utils.arr_mul(arr, utils.US_DAY)
            arr = arr.astype("datetime64[us]")
            # . new instance
            return pddt_new_simple(arr, tz, unit, name)

        # Invalid
        raise errors.InvalidTypeError(
            "unsupported type %s for argument 'ordinal'." % type(ordinal)
        )

    @classmethod
    def fromseconds(
        cls,
        seconds: int | float | object,
        tz: datetime.tzinfo | str | None = None,
        size: object = 1,
        unit: str = None,
        name: Hashable | None = None,
    ) -> Self:
        """Construct from seconds since Unix Epoch with optional timezone `<'Pddt'>`.

        :param seconds `<'int/float/Array-like'>`: The seconds to construct with.
            - `<'int/float'>` A numeric value representing the seconds since Unix Epoch.
            - `<'Array-like'>` An array-like object containing the seconds.

        :param tz `<'str/tzinfo/None'>`: The timezone of the index, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.

        :param size `<'int/str/bytes/object'>`: The size of the index to generate, defaults to `1`.
            - This argument is `ignored` if 'seconds' is an array-like object.
            - `<'int/str/bytes'>` Representing an integer for the number of elements in the index.
            - `<'object'>` Any object that implements `__len__()`, where its length represents the size of the index.

        :param unit `<'str/None'>`: Set the datetime unit of the index, defaults to `None`.
            Supported datetime units: 's', 'ms', 'us', 'ns'.

        :param name `<'Hashable/None'>`: The name assigned to the index, defaults to `None`.
        """
        # Validate
        arr: np.ndarray
        arr_size: cython.Py_ssize_t
        sec: cython.double
        val: cython.longlong

        # Single value
        if isinstance(seconds, (int, float, str, bytes)):
            # . convert to microseconds
            arr_size = _parse_arr_size(size)
            try:
                sec = float(seconds)
            except Exception as err:
                raise errors.InvalidTypeError(
                    "cannot covnert '%s' to float "
                    "to represent the seconds value." % seconds
                ) from err
            val = int(sec * utils.US_SECOND)
            # . new instance
            return _pddt_fr_us(val, arr_size, tz, unit, name)

        # <'list'/'tuple'>
        if isinstance(seconds, (list, tuple)):
            # . convert to ndarray[datetime64]
            arr_size = len(seconds)
            arr = np.PyArray_EMPTY(1, [arr_size], np.NPY_TYPES.NPY_INT64, 0)
            arr_ptr = cython.cast(cython.pointer(np.npy_int64), np.PyArray_DATA(arr))
            i: cython.Py_ssize_t = 0
            for item in seconds:
                try:
                    sec = float(item)
                except Exception as err:
                    raise errors.InvalidTypeError(
                        "cannot covnert '%s' to float "
                        "to represent the seconds value." % item
                    ) from err
                val = int(sec * utils.US_SECOND)
                arr_ptr[i] = val
                i += 1
            arr = arr.astype("datetime64[us]")
            # . new instance
            return pddt_new_simple(arr, tz, unit, name)

        # <'pandas.Series'>
        if isinstance(seconds, (Series, Index)):
            seconds = seconds.values

        # <'numpy.ndarray'>
        if isinstance(seconds, np.ndarray):
            # . validate array
            arr = seconds
            if arr.ndim != 1:
                raise errors.InvalidTypeError(
                    "must be an 1-dimensional ndarray, instead of %d-dim." % arr.ndim
                )
            # . convert to ndarray[datetime64]
            kind: cython.Py_UCS4 = arr.descr.kind
            if kind == "f":  # float64
                arr = utils.arr_assure_float64(arr)
                arr = utils.arr_assure_int64(arr * utils.US_SECOND)
            else:
                arr = utils.arr_assure_int64_like(arr)
                arr = utils.arr_mul(arr, utils.US_SECOND)
            arr = arr.astype("datetime64[us]")
            # . new instance
            return pddt_new_simple(arr, tz, unit, name)

        # Invalid
        raise errors.InvalidTypeError(
            "unsupported type %s for argument 'seconds'." % type(seconds)
        )

    @classmethod
    def fromicroseconds(
        cls,
        us: int | object,
        tz: datetime.tzinfo | str | None = None,
        size: object = 1,
        unit: str = None,
        name: Hashable | None = None,
    ) -> Self:
        """Construct from microseconds since Unix Epoch with optional timezone `<'Pddt'>`.

        :param us `<'int/Array-like'>`: The microseconds to construct with.
            - `<'int'>` An integer value representing the microseconds since Unix Epoch.
            - `<'Array-like'>` An array-like object containing the microseconds.

        :param tz `<'str/tzinfo/None'>`: The timezone of the index, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.

        :param size `<'int/str/bytes/object'>`: The size of the index to generate, defaults to `1`.
            - This argument is `ignored` if 'us' is an array-like object.
            - `<'int/str/bytes'>` Representing an integer for the number of elements in the index.
            - `<'object'>` Any object that implements `__len__()`, where its length represents the size of the index.

        :param unit `<'str/None'>`: Set the datetime unit of the index, defaults to `None`.
            Supported datetime units: 's', 'ms', 'us', 'ns'.

        :param name `<'Hashable/None'>`: The name assigned to the index, defaults to `None`.
        """
        # Validate
        arr: np.ndarray
        arr_size: cython.Py_ssize_t
        val: cython.longlong

        # Single value
        if isinstance(us, (int, float, str, bytes)):
            # . convert to microseconds
            arr_size = _parse_arr_size(size)
            try:
                val = int(us)
            except Exception as err:
                raise errors.InvalidTypeError(
                    "cannot covnert '%s' to an integer "
                    "to represent the microseconds value." % us
                ) from err
            # . new instance
            return _pddt_fr_us(val, arr_size, tz, unit, name)

        # <'list'/'tuple'>
        if isinstance(us, (list, tuple)):
            # . convert to ndarray[datetime64]
            arr_size = len(us)
            arr = np.PyArray_EMPTY(1, [arr_size], np.NPY_TYPES.NPY_INT64, 0)
            arr_ptr = cython.cast(cython.pointer(np.npy_int64), np.PyArray_DATA(arr))
            i: cython.Py_ssize_t = 0
            for item in us:
                try:
                    val = int(item)
                except Exception as err:
                    raise errors.InvalidTypeError(
                        "cannot covnert '%s' to an integer "
                        "to represent the microseconds value." % item
                    ) from err
                arr_ptr[i] = val
                i += 1
            arr = arr.astype("datetime64[us]")
            # . new instance
            return pddt_new_simple(arr, tz, unit, name)

        # <'pandas.Series'>
        if isinstance(us, (Series, Index)):
            us = us.values

        # <'numpy.ndarray'>
        if isinstance(us, np.ndarray):
            # . validate array
            arr = us
            if arr.ndim != 1:
                raise errors.InvalidTypeError(
                    "must be an 1-dimensional ndarray, instead of %d-dim." % arr.ndim
                )
            # . convert to ndarray[datetime64]
            arr = utils.arr_assure_int64_like(arr)
            arr = arr.astype("datetime64[us]")
            # . new instance
            return pddt_new_simple(arr, tz, unit, name)

        # Invalid
        raise errors.InvalidTypeError(
            "unsupported type %s for argument 'us'." % type(us)
        )

    @classmethod
    def fromtimestamp(
        cls,
        ts: int | float | object,
        tz: datetime.tzinfo | str | None = None,
        size: object = 1,
        unit: str = None,
        name: Hashable | None = None,
    ) -> Self:
        """Construct from POSIX timestamps with optional timezone `<'Pddt'>`.

        :param ts `<'int/float/Array-like'>`: The POSIX timestamps to construct with.
            - `<'int/float'>` A numeric value representing the timestamp.
            - `<'Array-like'>` An array-like object containing the timestamps.

        :param tz `<'str/tzinfo/None'>`: The timezone of the index, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.

        :param size `<'int/str/bytes/object'>`: The size of the index to generate, defaults to `1`.
            - This argument is `ignored` if 'us' is an array-like object.
            - `<'int/str/bytes'>` Representing an integer for the number of elements in the index.
            - `<'object'>` Any object that implements `__len__()`, where its length represents the size of the index.

        :param unit `<'str/None'>`: Set the datetime unit of the index, defaults to `None`.
            Supported datetime units: 's', 'ms', 'us', 'ns'.

        :param name `<'Hashable/None'>`: The name assigned to the index, defaults to `None`.
        """
        # Validate
        arr: np.ndarray
        arr_size: cython.Py_ssize_t
        sec: cython.double
        tz = utils.tz_parse(tz)

        # Single value
        if isinstance(ts, (int, float, str, bytes)):
            # . convert to datetime
            arr_size = _parse_arr_size(size)
            try:
                sec = float(ts)
            except Exception as err:
                raise errors.InvalidTypeError(
                    "cannot convert '%s' to float "
                    "to represent the timestamp value." % ts
                ) from err
            dt = utils.dt_fr_ts(sec, tz)
            # . new instance
            return _pddt_fr_dt(dt, arr_size, unit, name)

        # Array-like
        if isinstance(ts, (list, tuple, Series, Index, np.ndarray)):
            # . convert to ndarray[datetime64]
            arr_size = len(ts)
            arr = np.PyArray_EMPTY(1, [arr_size], np.NPY_TYPES.NPY_INT64, 0)
            arr_ptr = cython.cast(cython.pointer(np.npy_int64), np.PyArray_DATA(arr))
            i: cython.Py_ssize_t = 0
            for item in ts:
                try:
                    sec = float(item)
                except Exception as err:
                    raise errors.InvalidTypeError(
                        "cannot convert '%s' to float "
                        "to represent the timestamp value." % item
                    ) from err
                dt = utils.dt_fr_ts(sec, tz)
                arr_ptr[i] = utils.dt_to_us(dt, False)
                i += 1
            arr = arr.astype("datetime64[us]")
            # . new instance
            return pddt_new_simple(arr, tz, unit, name)

        # Invalid
        raise errors.InvalidTypeError(
            "unsupported type %s for argument 'ts'." % type(ts)
        )

    @classmethod
    def utcfromtimestamp(
        cls,
        ts: int | float | object,
        size: object = 1,
        unit: str = None,
        name: Hashable | None = None,
    ) -> Self:
        """Construct from POSIX timestamps with UTC timezone `<'Pddt'>`.

        :param ts `<'int/float/Array-like'>`: The POSIX timestamps to construct with.
            - `<'int/float'>` A numeric value representing the timestamp.
            - `<'Array-like'>` An array-like object containing the timestamps.

        :param size `<'int/str/bytes/object'>`: The size of the index to generate, defaults to `1`.
            - This argument is `ignored` if 'us' is an array-like object.
            - `<'int/str/bytes'>` Representing an integer for the number of elements in the index.
            - `<'object'>` Any object that implements `__len__()`, where its length represents the size of the index.

        :param unit `<'str/None'>`: Set the datetime unit of the index, defaults to `None`.
            Supported datetime units: 's', 'ms', 'us', 'ns'.

        :param name `<'Hashable/None'>`: The name assigned to the index, defaults to `None`.
        """
        return cls.fromtimestamp(ts, utils.UTC, size, unit, name)

    @classmethod
    def fromisoformat(
        cls,
        dtstr: str | object,
        size: object = 1,
        unit: str = None,
        name: Hashable | None = None,
    ) -> Self:
        """Construct from ISO format strings `<'Pddt'>`.

        :param dtstr `<'str/Array-like'>`: The ISO format datetime string(s) to construct with.
            - `<'str'>` A ISO format datetime string.
            - `<'Array-like'>` An array-like object containing the datetime strings.

        :param size `<'int/str/bytes/object'>`: The size of the index to generate, defaults to `1`.
            - This argument is `ignored` if 'us' is an array-like object.
            - `<'int/str/bytes'>` Representing an integer for the number of elements in the index.
            - `<'object'>` Any object that implements `__len__()`, where its length represents the size of the index.

        :param unit `<'str/None'>`: Set the datetime unit of the index, defaults to `None`.
            Supported datetime units: 's', 'ms', 'us', 'ns'.

        :param name `<'Hashable/None'>`: The name assigned to the index, defaults to `None`.
        """
        # <'str'>
        if isinstance(dtstr, str):
            arr_size = _parse_arr_size(size)
            return _pddt_fr_dt(_parse_dtobj(dtstr), arr_size, unit, name)

        # Other types
        return pddt_new_simple(dtstr, None, unit, name)

    @classmethod
    def fromisocalendar(
        cls,
        iso: object,
        year: int | None = None,
        week: int | None = None,
        weekday: int | None = None,
        tz: datetime.tzinfo | str | None = None,
        size: object = 1,
        unit: str = None,
        name: Hashable | None = None,
    ) -> Self:
        """Construct from ISO calendar values `<'Pddt'>`.

        :param iso `<'dict/list/tuple/DataFrame'>`: The ISO calendar values to construct with.
        :param year `<'int/None'>`: The ISO year, defaults to `None`.
        :param week `<'int/None'>`: The ISO week number (1-53), defaults to `None`.
        :param weekday `<'int'>`: The ISO weekday (1=Mon...7=Sun), defaults to `None`.
        :param tz `<'str/tzinfo/None'>`: The timezone of the index, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.

        :param size `<'int/str/bytes/object'>`: The size of the index to generate, defaults to `1`.
            - This argument is `ignored` if 'us' is an array-like object.
            - `<'int/str/bytes'>` Representing an integer for the number of elements in the index.
            - `<'object'>` Any object that implements `__len__()`, where its length represents the size of the index.

        :param unit `<'str/None'>`: Set the datetime unit of the index, defaults to `None`.
            Supported datetime units: 's', 'ms', 'us', 'ns'.

        :param name `<'Hashable/None'>`: The name assigned to the index, defaults to `None`.
        """

        def _parse_dict(item: dict) -> object:
            yy = item.get("year", year)
            ww = item.get("week", week)
            dd = item.get("weekday", item.get("day", weekday))
            try:
                _ymd = utils.ymd_fr_isocalendar(yy, ww, dd)
            except Exception as err:
                raise errors.InvalidArgumentError(
                    "cannot unpack '%s' to iso calendar "
                    "values (year, week, day)." % item
                ) from err
            return datetime.datetime_new(
                _ymd.year, _ymd.month, _ymd.day, 0, 0, 0, 0, tz, 0
            )

        def _parse_sequence(item: object) -> object:
            try:
                if len(item) != 3:
                    raise ValueError("sequence must have 3 items.")
                _ymd = utils.ymd_fr_isocalendar(item[0], item[1], item[2])
            except Exception as err:
                raise errors.InvalidArgumentError(
                    "cannot unpack '%s' to iso calendar "
                    "values (year, week, day)." % item
                ) from err
            return datetime.datetime_new(
                _ymd.year, _ymd.month, _ymd.day, 0, 0, 0, 0, tz, 0
            )

        # Validate
        tz = utils.tz_parse(tz)
        dts: list = []

        # <'ditc'>
        if isinstance(iso, dict):
            dts.append(_parse_dict(iso))

        # <'list'/'tuple'>
        elif isinstance(iso, (list, tuple)):
            for item in iso:
                if isinstance(item, dict):
                    dts.append(_parse_dict(item))
                elif isinstance(item, (list, tuple)):
                    dts.append(_parse_sequence(item))
                elif isinstance(item, int):
                    dts.append(_parse_sequence(iso))
                    break
                else:
                    raise errors.InvalidTypeError(
                        "unsupported type %s for argument 'iso'." % type(item)
                    )

        # <'DataFrame'>
        elif isinstance(iso, DataFrame):
            for item in iso.itertuples(index=False):
                dts.append(_parse_sequence(item))

        # Y/M/D values
        elif year is not None and week is not None and weekday is not None:
            dts.append(_parse_sequence([year, week, weekday]))

        # Invalid
        else:
            raise errors.InvalidTypeError(
                "unsupported type %s for argument 'iso'." % type(iso)
            )

        # Create Pddt
        if len(dts) == 1:
            arr_size = _parse_arr_size(size)
            return _pddt_fr_dt(dts[0], arr_size, unit, name)
        else:
            return pddt_new_simple(dts, None, unit, name)

    @classmethod
    def fromdate(
        cls,
        date: datetime.date | object,
        tz: datetime.tzinfo | str | None = None,
        size: object = 1,
        unit: str = None,
        name: Hashable | None = None,
    ) -> Self:
        """Construct from instances of date (all time fields set to 0) `<'Pddt'>`.

        :param date `<'date/Array-like'>`: The date-like object(s) to construct with.
            - `<'datetime.date'>` An instance of `datetime.date`.
            - `<'Array-like'>` An array-like object containing the date-like objects.

        :param tz `<'str/tzinfo/None'>`: The timezone of the index, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.

        :param size `<'int/str/bytes/object'>`: The size of the index to generate, defaults to `1`.
            - This argument is `ignored` if 'us' is an array-like object.
            - `<'int/str/bytes'>` Representing an integer for the number of elements in the index.
            - `<'object'>` Any object that implements `__len__()`, where its length represents the size of the index.

        :param unit `<'str/None'>`: Set the datetime unit of the index, defaults to `None`.
            Supported datetime units: 's', 'ms', 'us', 'ns'.

        :param name `<'Hashable/None'>`: The name assigned to the index, defaults to `None`.
        """
        # Validate

        # <'date'>
        if utils.is_date(date):
            arr_size, tz = _parse_arr_size(size), utils.tz_parse(tz)
            return _pddt_fr_dt(utils.dt_fr_date(date, tz), arr_size, unit, name)

        # Other types
        return pddt_new_simple(date, tz, unit, name)

    @classmethod
    def fromdatetime(
        cls,
        dt: datetime.datetime | object,
        size: object = 1,
        unit: str = None,
        name: Hashable | None = None,
    ) -> Self:
        """Construct from instances of datetime `<'Pddt'>`.

        :param dt `<'datetime/Array-like'>`: The datetime-like object(s) to construct with.
            - `<'datetime.datetime'>` An instance of `datetime.datetime`.
            - `<'Array-like'>` An array-like object containing the datetime-like objects.

        :param size `<'int/str/bytes/object'>`: The size of the index to generate, defaults to `1`.
            - This argument is `ignored` if 'us' is an array-like object.
            - `<'int/str/bytes'>` Representing an integer for the number of elements in the index.
            - `<'object'>` Any object that implements `__len__()`, where its length represents the size of the index.

        :param unit `<'str/None'>`: Set the datetime unit of the index, defaults to `None`.
            Supported datetime units: 's', 'ms', 'us', 'ns'.

        :param name `<'Hashable/None'>`: The name assigned to the index, defaults to `None`.
        """
        # <'str'/'Timestamp'/'datetime64'>
        if isinstance(dt, (str, Timestamp, np.datetime64)):
            arr_size = _parse_arr_size(size)
            dts = [dt for _ in range(arr_size)]
            return pddt_new_simple(dts, None, unit, name)

        # <'datetime'>
        if utils.is_dt(dt):
            arr_size = _parse_arr_size(size)
            return _pddt_fr_dt(dt, arr_size, unit, name)

        # Other types
        return pddt_new_simple(dt, None, unit, name)

    @classmethod
    def fromdatetime64(
        cls,
        dt64: np.datetime64 | object,
        tz: datetime.tzinfo | str | None = None,
        size: object = 1,
        unit: str = None,
        name: Hashable | None = None,
    ) -> Self:
        """Construct from instances of datetime64 `<'Pddt'>`.

        :param dt64 `<'datetime64/Array-like'>`: The datetime-like object(s) to construct with.
            - `<'np..datetime64'>` An instance of `np.datetime64`.
            - `<'Array-like'>` An array-like object containing the datetime-like objects.

        :param tz `<'str/tzinfo/None'>`: The timezone of the index, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.

        :param size `<'int/str/bytes/object'>`: The size of the index to generate, defaults to `1`.
            - This argument is `ignored` if 'us' is an array-like object.
            - `<'int/str/bytes'>` Representing an integer for the number of elements in the index.
            - `<'object'>` Any object that implements `__len__()`, where its length represents the size of the index.

        :param unit `<'str/None'>`: Set the datetime unit of the index, defaults to `None`.
            Supported datetime units: 's', 'ms', 'us', 'ns'.

        :param name `<'Hashable/None'>`: The name assigned to the index, defaults to `None`.
        """
        # <'datetime64'>
        if utils.is_dt64(dt64):
            arr_size = _parse_arr_size(size)
            dts = [dt64 for _ in range(arr_size)]
            return pddt_new_simple(dts, tz, unit, name)

        # Other types
        return pddt_new_simple(dt64, tz, unit, name)

    @classmethod
    def strptime(
        cls,
        dtstr: str | object,
        fmt: str,
        size: object = 1,
        unit: str = None,
        name: Hashable | None = None,
    ) -> Self:
        """Construct from datetime strings according to the given format `<'Pddt'>`.

        :param dtstr `<'str/Array-like'>`: The datetime string(s) to construct with.
            - `<'str'>` A datetime string.
            - `<'Array-like'>` An array-like object containing the datetime strings.

        :param fmt `<'str'>`: The format used to parse the datetime strings.

        :param size `<'int/str/bytes/object'>`: The size of the index to generate, defaults to `1`.
            - This argument is `ignored` if 'us' is an array-like object.
            - `<'int/str/bytes'>` Representing an integer for the number of elements in the index.
            - `<'object'>` Any object that implements `__len__()`, where its length represents the size of the index.

        :param unit `<'str/None'>`: Set the datetime unit of the index, defaults to `None`.
            Supported datetime units: 's', 'ms', 'us', 'ns'.

        :param name `<'Hashable/None'>`: The name assigned to the index, defaults to `None`.
        """
        # <'str'>
        if isinstance(dtstr, str):
            arr_size = _parse_arr_size(size)
            try:
                dt = datetime.datetime.strptime(dtstr, fmt)
            except Exception as err:
                raise errors.InvalidArgumentError(err) from err
            return _pddt_fr_dt(dt, arr_size, unit, name)

        # Array-like
        if isinstance(dtstr, (list, tuple, Series, Index, np.ndarray)):
            dts: list = []
            for item in dtstr:
                try:
                    dts.append(datetime.datetime.strptime(item, fmt))
                except Exception as err:
                    raise errors.InvalidArgumentError(err) from err
            return pddt_new_simple(dts, None, unit, name)

        # Invalid
        raise errors.InvalidTypeError(
            "unsupported type %s for argument 'dtstr'." % type(dtstr)
        )

    # Convertor ----------------------------------------------------------------------------
    def ctime(self) -> Index[str]:
        """Convert to index of strings in C time format `<'Index[str]'>`.

        - ctime format: 'Tue Oct 10 08:19:05 2024'
        """
        return self.strftime("%a %b %d %H:%M:%S %Y")

    def isoformat(self, sep: str = "T") -> Index[str]:
        """Convert to index of strings in ISO format `<'Index[str]'>`.

        :param sep `<'str'>`: The separator between date and time components, defaults to `'T'`.
        """
        return self.strftime(f"%Y-%m-%d{sep}%H:%M:%S.%f%z")

    def timedf(self) -> DataFrame:
        """Convert to DataFrame of time components `<'DataFrame'>`."""
        arr: np.ndarray = self.values_naive
        return DataFrame(
            {
                "tm_year": utils.dt64arr_year(arr),
                "tm_mon": utils.dt64arr_month(arr),
                "tm_mday": utils.dt64arr_day(arr),
                "tm_hour": utils.dt64arr_hour(arr),
                "tm_min": utils.dt64arr_minute(arr),
                "tm_sec": utils.dt64arr_second(arr),
                "tm_wday": utils.dt64arr_weekday(arr),
                "tm_yday": utils.dt64arr_days_of_year(arr),
            },
        )

    def utctimedf(self) -> DataFrame:
        """Convert to DataFrame of time components representing the UTC time `<'DataFrame'>`."""
        my_tz = self.tzinfo
        if my_tz is None or my_tz is utils.UTC:
            return self.timedf()
        return self.tz_convert(utils.UTC).timedf()

    def toordinal(self) -> Index[np.int64]:
        """Convert to index of proleptic Gregorian ordinal days `<'Index[int64]'>`.

        - Day 1 (ordinal=1) is `0001-01-01`.
        """
        return Index(utils.dt64arr_to_ordinal(self.values), name="ordinal")

    def seconds(self, utc: cython.bint = False) -> Index[np.float64]:
        """Convert to index of seconds since Unix Epoch `<'Index[float64]'>`.

        Unlike 'timesamp()', this method does `NOT` take local
        timezone into consideration at conversion.

        :param utc `<'bool'>`: Whether to subtract the UTC offset from the result, defaults to `False`.
            Only applicable when instance is timezone-aware; otherwise ignored.
        """
        my_tz = self.tzinfo
        if my_tz is None:
            arr = self.values
        elif not utc:
            arr = self.values_naive
        else:
            arr = self.tz_convert(utils.UTC).values_naive
        return Index(utils.dt64arr_to_ts(arr), name="seconds")

    def microseconds(self, utc: cython.bint = False) -> Index[np.int64]:
        """Convert to index of microseconds since Unix Epoch `<'Index[int64]'>`.

        Unlike 'timesamp()', this method does `NOT` take local
        timezone into consideration at conversion.

        :param utc `<'bool'>`: Whether to subtract the UTC offset from the result, defaults to `False`.
            Only applicable when instance is timezone-aware; otherwise ignored.
        """
        my_tz = self.tzinfo
        if my_tz is None:
            arr = self.values
        elif not utc:
            arr = self.values_naive
        else:
            arr = self.tz_convert(utils.UTC).values_naive
        return Index(utils.dt64arr_as_int64_us(arr), name="microseconds")

    def timestamp(self) -> Index[np.float64]:
        """Convert to index of POSIX timestamps `<'Index[float64]'>`."""
        # fmt: off
        if self.tzinfo is None:
            arr = self.tz_localize(utils.tz_local(None), "infer", "shift_forward").values
        else:
            arr = self.values
        # fmt: on
        return Index(utils.dt64arr_to_ts(arr), name="timestamp")

    def datetime(self) -> np.ndarray[datetime.datetime]:
        """Convert to array of datetime.datetime `<'ndarray[datetime]'>`.

        #### Alias of `to_pydatetime()`.
        """
        return super(Pddt, self).to_pydatetime()

    def date(self) -> np.ndarray[datetime.date]:
        """Convert to array of datetime.date `<'ndarray[date]'>`."""
        return super(Pddt, self).date

    def time(self) -> np.ndarray[datetime.time]:
        """Convert to array of datetime.time (`WITHOUT` timezone) `<'ndarray[time]'>`."""
        return super(Pddt, self).time

    def timetz(self) -> np.ndarray[datetime.time]:
        """Convert to array of datetime.time (`WITH` timezone) `<'ndarray[time]'>`."""
        return super(Pddt, self).timetz

    # Manipulator --------------------------------------------------------------------------
    def replace(
        self,
        year: cython.int = -1,
        month: cython.int = -1,
        day: cython.int = -1,
        hour: cython.int = -1,
        minute: cython.int = -1,
        second: cython.int = -1,
        microsecond: cython.int = -1,
        nanosecond: cython.int = -1,
        tzinfo: datetime.tzinfo | str | None = -1,
    ) -> Self:
        """Replace the specified datetime fields with new values `<'Pddt'>`.

        #### Fields set to `-1` means retaining the original values.

        :param year `<'int'>`: Year value, defaults to `-1`.
        :param month `<'int'>`: Yonth value (1-12), defaults to `-1`.
        :param day `<'int'>`: Day value (1-31), automacially clamped to the maximum days in the month, defaults to `-1`.
        :param hour `<'int'>`: Hour value (0-23), defaults to `-1`.
        :param minute `<'int'>`: Minute value (0-59), defaults to `-1`.
        :param second `<'int'>`: Second value (0-59), defaults to `-1`.
        :param microsecond `<'int'>`: Microsecond value (0-999999), defaults to `-1`.
        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `-1`.
            - `<'int'>` Retains the original timezone.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.
        """
        # Access my_tzinfo
        my_tz = self.tzinfo
        pt = self if my_tz is None else self.tz_localize(None)  # timezone-naive

        # Replace datetime
        pt = pt.to_datetime(
            year, month, day, hour, minute, second, microsecond, nanosecond
        )

        # Replace timezone
        # . timezone-naive
        if my_tz is None:
            if isinstance(tzinfo, int):
                return pt  # exit: keep naive
            tzinfo = utils.tz_parse(tzinfo)
            if tzinfo is None:
                return pt  # exit: keep naive
        # . timezone-aware
        else:
            if isinstance(tzinfo, int):
                return pt.tz_localize(
                    my_tz, "infer", "shift_forward"
                )  # exit: keep mytz
            tzinfo = utils.tz_parse(tzinfo)
            if tzinfo is None:
                return pt  # exit: keep naive
        # . localize to new timezone
        return pt.tz_localize(tzinfo, "infer", "shift_forward")

    # . year
    def to_curr_year(
        self,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> Self:
        """Adjust the date to the specified month and day in the current year `<'Pddt'>`.

        :param month `<'int/str/None'>`: Month value, defaults to `None`.
            - `<'int'>` Month number (1=Jan...12=Dec).
            - `<'str'>` Month name (case-insensitive), e.g., 'Jan', 'februar', '三月'.
            - `<'None'>` Retains the original months.

        :param day `<'int'>`: Day value (1-31), defaults to `-1`.
            If `-1`, retains the original days. The final day
            values are clamped to the maximum days in the month.

        ### Example:
        >>> pt.to_curr_year("Feb", 31)  # The last day of February in the current year
        >>> pt.to_curr_year(11)         # The same day of November in the current year
        >>> pt.to_curr_year(day=1)      # The first day of the current month
        """
        # No month adjustment
        mm: cython.int = _parse_month(month, True)
        if mm == -1:
            return self.to_curr_month(day)

        # Access datetime array & info
        arr: np.ndarray = self.values_naive
        my_unit: str = self.unit
        delta: np.ndarray

        # Seperate times
        times = utils.dt64arr_times(arr, my_unit)  # int64[my_unit]

        # Adjust dates
        datesY = utils.dt64arr_as_int64_Y(arr, my_unit)
        datesM = utils.dt64arr_as_int64_M(datesY, "Y", mm - 1)
        # . add back original day
        if day < 1:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
            delta = utils.dt64arr_days_in_month(datesM, "M")
            delta = utils.arr_min_arr(delta, utils.dt64arr_day(arr, my_unit))
            datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]
        # . first day
        elif day == 1:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
        # . days before 29
        elif day < 29:
            datesD = utils.dt64arr_as_int64_D(datesM, "M", day - 1)  # int64[D]
        # . days before 31
        elif day < 31:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
            delta = utils.dt64arr_days_in_month(datesM, "M")
            delta = utils.arr_min(delta, day)
            datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]
        # . last day
        else:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
            delta = utils.dt64arr_days_in_month(datesM, "M")
            datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]

        # Combine dates & times
        my_unit_ns: cython.bint = str_read(my_unit, 0) == "n"
        if my_unit_ns and not utils.is_dt64arr_ns_safe(datesD, "D"):
            dates = utils.arr_mul(datesD, utils.US_DAY)  # int64[us]
            times = utils.arr_floor_div(times, utils.NS_MICROSECOND)  # int64[us]
            dtype = "datetime64[us]"
        else:
            dates = utils.dt64arr_as_int64(datesD, my_unit, "D")  # int64[my_unit]
            dtype = "datetime64[%s]" % my_unit
        arr = utils.arr_add_arr(dates, times).astype(dtype)  # dt64[my_unit]
        return pddt_new_simple(arr, self.tzinfo, None, self.name)

    def to_prev_year(
        self,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> Self:
        """Adjust the date to the specified month and day in the previous year `<'Pddt'>`.

        :param month `<'int/str/None'>`: Month value, defaults to `None`.
            - `<'int'>` Month number (1=Jan...12=Dec).
            - `<'str'>` Month name (case-insensitive), e.g., 'Jan', 'februar', '三月'.
            - `<'None'>` Retains the original months.

        :param day `<'int'>`: Day value (1-31), defaults to `-1`.
            If `-1`, retains the original days. The final day
            values are clamped to the maximum days in the month.

        ### Example:
        >>> pt.to_prev_year("Feb", 31)  # The last day of February in the previous year
        >>> pt.to_prev_year(11)         # The same day of November in the previous year
        >>> pt.to_prev_year(day=1)      # The first day of the current month in the previous year
        """
        return self.to_year(-1, month, day)

    def to_next_year(
        self,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> Self:
        """Adjust the date to the specified month and day in the next year `<'Pddt'>`.

        :param month `<'int/str/None'>`: Month value, defaults to `None`.
            - `<'int'>` Month number (1=Jan...12=Dec).
            - `<'str'>` Month name (case-insensitive), e.g., 'Jan', 'februar', '三月'.
            - `<'None'>` Retains the original months.

        :param day `<'int'>`: Day value (1-31), defaults to `-1`.
            If `-1`, retains the original days. The final day
            values are clamped to the maximum days in the month.

        ### Example:
        >>> pt.to_next_year("Feb", 31)  # The last day of February in the next year
        >>> pt.to_next_year(11)         # The same day of November in the next year
        >>> pt.to_next_year(day=1)      # The first day of the current month in the next year
        """
        return self.to_year(1, month, day)

    def to_year(
        self,
        offset: cython.int,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> Self:
        """Adjust the date to the specified month and day in the year (+/-) offset `<'Pddt'>`.

        :param offset `<'int'>`: The year offset (+/-).

        :param month `<'int/str/None'>`: Month value, defaults to `None`.
            - `<'int'>` Month number (1=Jan...12=Dec).
            - `<'str'>` Month name (case-insensitive), e.g., 'Jan', 'februar', '三月'.
            - `<'None'>` Retains the original months.

        :param day `<'int'>`: Day value (1-31), defaults to `-1`.
            If `-1`, retains the original days. The final day
            values are clamped to the maximum days in the month.

        ### Example:
        >>> pt.to_year(-2, "Feb", 31)  # The last day of February, two years ago
        >>> pt.to_year(2, 11)          # The same day of November, two years later
        >>> pt.to_year(2, day=1)       # The first day of the current month, two years later
        """
        # No offset
        if offset == 0:
            return self.to_curr_year(month, day)
        mm: cython.int = _parse_month(month, True)

        # Access datetime array & info
        arr: np.ndarray = self.values_naive
        my_unit: str = self.unit
        delta: np.ndarray

        # Seperate times
        times = utils.dt64arr_times(arr, my_unit)  # int64[my_unit]

        # Adjust dates
        datesY = utils.dt64arr_as_int64_Y(arr, my_unit, offset)  # int64[Y]+off
        datesM = utils.dt64arr_as_int64_M(datesY, "Y")  # int64[M]
        if mm == -1:
            delta = utils.dt64arr_month(arr, my_unit)
            datesM = utils.arr_add_arr(datesM, delta, -1)  # int64[M]
        else:
            datesM = utils.arr_add(datesM, mm - 1)  # int64[M]
        # . add back original day
        if day < 1:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
            delta = utils.dt64arr_days_in_month(datesM, "M")
            delta = utils.arr_min_arr(delta, utils.dt64arr_day(arr, my_unit))
            datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]
        # . first day
        elif day == 1:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
        # . days before 29
        elif day < 29:
            datesD = utils.dt64arr_as_int64_D(datesM, "M", day - 1)  # int64[D]
        # . days before 31
        elif day < 31:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
            delta = utils.dt64arr_days_in_month(datesM, "M")
            delta = utils.arr_min(delta, day)
            datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]
        # . last day
        else:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
            delta = utils.dt64arr_days_in_month(datesM, "M")
            datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]

        # Combine dates & times
        my_unit_ns: cython.bint = str_read(my_unit, 0) == "n"
        if my_unit_ns and not utils.is_dt64arr_ns_safe(datesD, "D"):
            dates = utils.arr_mul(datesD, utils.US_DAY)  # int64[us]
            times = utils.arr_floor_div(times, utils.NS_MICROSECOND)  # int64[us]
            dtype = "datetime64[us]"
        else:
            dates = utils.dt64arr_as_int64(datesD, my_unit, "D")  # int64[my_unit]
            dtype = "datetime64[%s]" % my_unit
        arr = utils.arr_add_arr(dates, times).astype(dtype)  # dt64[my_unit]
        return pddt_new_simple(arr, self.tzinfo, None, self.name)

    # . quarter
    def to_curr_quarter(self, month: cython.int = -1, day: cython.int = -1) -> Self:
        """Adjust the date to the specified month and day in the current quarter. `<'Pddt'>`.

        :param month `<'int'>`: Month (1-3) of the quarter, defaults to `-1`.
            If `-1`, retains the original months of the quarter.

        :param day `<'int'>`: Day value (1-31), defaults to `-1`.
            If `-1`, retains the original days. The final day
            values are clamped to the maximum days in the month.

        ### Example:
        >>> pt.to_curr_quarter(1, 31)  # The last day of the first quarter month in the current quarter
        >>> pt.to_curr_quarter(2)      # The same day of the second quarter month in the current quarter
        >>> pt.to_curr_quarter(day=1)  # The first day of the current quarter month in the current quarter
        """
        # No month adjustment
        if month < 1:
            return self.to_curr_month(day)

        # Access datetime array & info
        arr: np.ndarray = self.values_naive
        my_unit: str = self.unit

        # Seperate times
        times = utils.dt64arr_times(arr, my_unit)  # int64[my_unit]

        # Adjust dates
        datesQ = utils.dt64arr_as_int64_Q(arr, my_unit)  # int64[Q]
        datesM = utils.arr_mul(datesQ, 3, min(month, 3) - 1)  # int64[M]+Moff
        # . add back original day
        if day < 1:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
            delta = utils.dt64arr_days_in_month(datesM, "M")
            delta = utils.arr_min_arr(delta, utils.dt64arr_day(arr, my_unit))
            datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]
        # . first day
        elif day == 1:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
        # . days before 29
        elif day < 29:
            datesD = utils.dt64arr_as_int64_D(datesM, "M", day - 1)  # int64[D]
        # . days before 31
        elif day < 31:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
            delta = utils.dt64arr_days_in_month(datesM, "M")
            delta = utils.arr_min(delta, day)
            datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]
        # . last day
        else:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
            delta = utils.dt64arr_days_in_month(datesM, "M")
            datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]

        # Combine dates & times
        my_unit_ns: cython.bint = str_read(my_unit, 0) == "n"
        if my_unit_ns and not utils.is_dt64arr_ns_safe(datesD, "D"):
            dates = utils.arr_mul(datesD, utils.US_DAY)  # int64[us]
            times = utils.arr_floor_div(times, utils.NS_MICROSECOND)  # int64[us]
            dtype = "datetime64[us]"
        else:
            dates = utils.dt64arr_as_int64(datesD, my_unit, "D")  # int64[my_unit]
            dtype = "datetime64[%s]" % my_unit
        arr = utils.arr_add_arr(dates, times).astype(dtype)  # dt64[my_unit]
        return pddt_new_simple(arr, self.tzinfo, None, self.name)

    def to_prev_quarter(self, month: cython.int = -1, day: cython.int = -1) -> Self:
        """Adjust the date to the specified month and day in the previous quarter. `<'Pddt'>`.

        :param month `<'int'>`: Month (1-3) of the quarter, defaults to `-1`.
            If `-1`, retains the original months of the quarter.

        :param day `<'int'>`: Day value (1-31), defaults to `-1`.
            If `-1`, retains the original days. The final day
            values are clamped to the maximum days in the month.

        ### Example:
        >>> pt.to_prev_quarter(1, 31)  # The last day of the first quarter month in the previous quarter
        >>> pt.to_prev_quarter(2)      # The same day of the second quarter month in the previous quarter
        >>> pt.to_prev_quarter(day=1)  # The first day of the current quarter month in the previous quarter
        """
        return self.to_quarter(-1, month, day)

    def to_next_quarter(self, month: cython.int = -1, day: cython.int = -1) -> Self:
        """Adjust the date to the specified month and day in the next quarter. `<'Pddt'>`.

        :param month `<'int'>`: Month (1-3) of the quarter, defaults to `-1`.
            If `-1`, retains the original months of the quarter.

        :param day `<'int'>`: Day value (1-31), defaults to `-1`.
            If `-1`, retains the original days. The final day
            values are clamped to the maximum days in the month.

        ### Example:
        >>> pt.to_next_quarter(1, 31)  # The last day of the first quarter month in the next quarter
        >>> pt.to_next_quarter(2)      # The same day of the second quarter month in the next quarter
        >>> pt.to_next_quarter(day=1)  # The first day of the current quarter month in the next quarter
        """
        return self.to_quarter(1, month, day)

    def to_quarter(
        self,
        offset: cython.int,
        month: cython.int = -1,
        day: cython.int = -1,
    ) -> Self:
        """Adjust the date to the specified month and day in the quarter (+/-) 'offset'. `<'Pddt'>`.

        :param offset `<'int'>`: The quarter offset (+/-).

        :param month `<'int'>`: Month (1-3) of the quarter, defaults to `-1`.
            If `-1`, retains the original months of the quarter.

        :param day `<'int'>`: Day value (1-31), defaults to `-1`.
            If `-1`, retains the original days. The final day
            values are clamped to the maximum days in the month.

        ### Example:
        >>> pt.to_quarter(-2, 1, 31)  # The last day of the first quarter month, two quarters ago
        >>> pt.to_quarter(2, 2)       # The same day of the second quarter month, two quarters later
        >>> pt.to_quarter(2, day=1)   # The first day of the current quarter month, two quarters later
        """
        # No offset
        if offset == 0:
            return self.to_curr_quarter(month, day)

        # Access datetime array & info
        arr: np.ndarray = self.values_naive
        my_unit: str = self.unit

        # Seperate times
        times = utils.dt64arr_times(arr, my_unit)  # int64[my_unit]

        # Adjust dates
        if month < 1:
            datesM = utils.dt64arr_as_int64_M(arr, my_unit, offset * 3)
        else:
            datesQ = utils.dt64arr_as_int64_Q(arr, my_unit)
            datesM = utils.arr_mul(datesQ, 3, offset * 3 + min(month, 3) - 1)
        # . add back original day
        if day < 1:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
            delta = utils.dt64arr_days_in_month(datesM, "M")
            delta = utils.arr_min_arr(delta, utils.dt64arr_day(arr, my_unit))
            datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]
        # . first day
        elif day == 1:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
        # . days before 29
        elif day < 29:
            datesD = utils.dt64arr_as_int64_D(datesM, "M", day - 1)  # int64[D]
        # . days before 31
        elif day < 31:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
            delta = utils.dt64arr_days_in_month(datesM, "M")
            delta = utils.arr_min(delta, day)
            datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]
        # . last day
        else:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
            delta = utils.dt64arr_days_in_month(datesM, "M")
            datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]

        # Combine dates & times
        my_unit_ns: cython.bint = str_read(my_unit, 0) == "n"
        if my_unit_ns and not utils.is_dt64arr_ns_safe(datesD, "D"):
            dates = utils.arr_mul(datesD, utils.US_DAY)  # int64[us]
            times = utils.arr_floor_div(times, utils.NS_MICROSECOND)  # int64[us]
            dtype = "datetime64[us]"
        else:
            dates = utils.dt64arr_as_int64(datesD, my_unit, "D")  # int64[my_unit]
            dtype = "datetime64[%s]" % my_unit
        arr = utils.arr_add_arr(dates, times).astype(dtype)  # dt64[my_unit]
        return pddt_new_simple(arr, self.tzinfo, None, self.name)

    # . month
    def to_curr_month(self, day: cython.int = -1) -> Self:
        """Adjust the date to the specified day of the current month `<'Pddt'>`.

        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            If `-1`, retains the original days. The final day
            values are clamped to the maximum days in the month.

        ### Example:
        >>> pt.to_curr_month(31)  # The last day of the current month
        >>> pt.to_curr_month(1)   # The first day of the current month
        """
        # No adjustment
        if day < 1:
            return self  # exit

        # Access datetime array & info
        arr: np.ndarray = self.values_naive
        my_unit: str = self.unit
        delta: np.ndarray

        # Seperate times
        times = utils.dt64arr_times(arr, my_unit)  # int64[my_unit]

        # Adjust dates
        datesM = utils.dt64arr_as_int64_M(arr, my_unit)  # int64[M]
        # . first day
        if day == 1:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
        # . days before 29
        elif day < 29:
            datesD = utils.dt64arr_as_int64_D(datesM, "M", day - 1)  # int64[D]
        # . days before 31
        elif day < 31:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
            delta = utils.dt64arr_days_in_month(datesM, "M")
            delta = utils.arr_min(delta, day)
            datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]
        # . last day
        else:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
            delta = utils.dt64arr_days_in_month(datesM, "M")
            datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]

        # Combine dates & times
        my_unit_ns: cython.bint = str_read(my_unit, 0) == "n"
        if my_unit_ns and not utils.is_dt64arr_ns_safe(datesD, "D"):
            dates = utils.arr_mul(datesD, utils.US_DAY)  # int64[us]
            times = utils.arr_floor_div(times, utils.NS_MICROSECOND)  # int64[us]
            dtype = "datetime64[us]"
        else:
            dates = utils.dt64arr_as_int64(datesD, my_unit, "D")  # int64[my_unit]
            dtype = "datetime64[%s]" % my_unit
        arr = utils.arr_add_arr(dates, times).astype(dtype)  # dt64[my_unit]
        return pddt_new_simple(arr, self.tzinfo, None, self.name)

    def to_prev_month(self, day: cython.int = -1) -> Self:
        """Adjust the date to the specified day of the previous month `<'Pddt'>`.

        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            If `-1`, retains the original days. The final day
            values are clamped to the maximum days in the month.

        ### Example:
        >>> pt.to_prev_month(31)  # The last day of the previous month
        >>> pt.to_prev_month(1)   # The first day of the previous month
        """
        return self.to_month(-1, day)

    def to_next_month(self, day: cython.int = -1) -> Self:
        """Adjust the date to the specified day of the next month `<'Pddt'>`.

        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            If `-1`, retains the original days. The final day
            values are clamped to the maximum days in the month.

        ### Example:
        >>> pt.to_next_month(31)  # The last day of the next month
        >>> pt.to_next_month(1)   # The first day of the next month
        """
        return self.to_month(1, day)

    def to_month(self, offset: cython.int, day: cython.int = -1) -> Self:
        """Adjust the date to the specified day of the month (+/-) offest `<'Pddt'>`.

        :param offset `<'int'>`: The month offset (+/-).
        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            If `-1`, retains the original days. The final day
            values are clamped to the maximum days in the month.

        ### Example:
        >>> pt.to_month(-2, 31)  # The last day of the month, two months ago
        >>> pt.to_month(2, 1)    # The first day of the month, two months later
        """
        # No offset
        if offset == 0:
            return self.to_curr_month(day)

        # Access datetime array & info
        arr: np.ndarray = self.values_naive
        my_unit: str = self.unit
        delta: np.ndarray

        # Seperate times
        times = utils.dt64arr_times(arr, my_unit)  # int64[my_unit]

        # Adjust dates
        datesM = utils.dt64arr_as_int64_M(arr, my_unit, offset)  # int64[M]+off
        # . add back original day
        if day < 1:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
            delta = utils.dt64arr_days_in_month(datesM, "M")
            delta = utils.arr_min_arr(delta, utils.dt64arr_day(arr, my_unit))
            datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]
        # . first day
        elif day == 1:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
        # . days before 29
        elif day < 29:
            datesD = utils.dt64arr_as_int64_D(datesM, "M", day - 1)  # int64[D]
        # . days before 31
        elif day < 31:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
            delta = utils.dt64arr_days_in_month(datesM, "M")
            delta = utils.arr_min(delta, day)
            datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]
        # . last day
        else:
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
            delta = utils.dt64arr_days_in_month(datesM, "M")
            datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]

        # Combine dates & times
        my_unit_ns: cython.bint = str_read(my_unit, 0) == "n"
        if my_unit_ns and not utils.is_dt64arr_ns_safe(datesD, "D"):
            dates = utils.arr_mul(datesD, utils.US_DAY)  # int64[us]
            times = utils.arr_floor_div(times, utils.NS_MICROSECOND)  # int64[us]
            dtype = "datetime64[us]"
        else:
            dates = utils.dt64arr_as_int64(datesD, my_unit, "D")  # int64[my_unit]
            dtype = "datetime64[%s]" % my_unit
        arr = utils.arr_add_arr(dates, times).astype(dtype)  # dt64[my_unit]
        return pddt_new_simple(arr, self.tzinfo, None, self.name)

    # . weekday
    def to_monday(self) -> Self:
        """Adjust the date to the Monday of the current week `<'Pddt'>`."""
        return self._to_curr_weekday(0)

    def to_tuesday(self) -> Self:
        """Adjust the date to the Tuesday of the current week `<'Pddt'>`."""
        return self._to_curr_weekday(1)

    def to_wednesday(self) -> Self:
        """Adjust the date to the Wednesday of the current week `<'Pddt'>`."""
        return self._to_curr_weekday(2)

    def to_thursday(self) -> Self:
        """Adjust the date to the Thursday of the current week `<'Pddt'>`."""
        return self._to_curr_weekday(3)

    def to_friday(self) -> Self:
        """Adjust the date to the Friday of the current week `<'Pddt'>`."""
        return self._to_curr_weekday(4)

    def to_saturday(self) -> Self:
        """Adjust the date to the Saturday of the current week `<'Pddt'>`."""
        return self._to_curr_weekday(5)

    def to_sunday(self) -> Self:
        """Adjust the date to the Sunday of the current week `<'Pddt'>`."""
        return self._to_curr_weekday(6)

    def to_curr_weekday(self, weekday: int | str | None = None) -> Self:
        """Adjust the date to the specific weekday of the current week `<'Pddt'>`.

        :param weekday `<'int/str/None'>`: Weekday value, defaults to `None`.
            - `<'int'>` Weekday number (0=Mon...6=Sun).
            - `<'str'>` Weekday name (case-insensitive), e.g., 'Mon', 'dienstag', '星期三'.
            - `<'None'>` Retains the original weekdays.

        ### Example:
        >>> pt.to_curr_weekday(0)      # The Monday of the current week
        >>> pt.to_curr_weekday("Tue")  # The Tuesday of the current week
        """
        return self._to_curr_weekday(_parse_weekday(weekday, True))

    def to_prev_weekday(self, weekday: int | str | None = None) -> Self:
        """Adjust the date to the specific weekday of the previous week `<'Pddt'>`.

        :param weekday `<'int/str/None'>`: Weekday value, defaults to `None`.
            - `<'int'>` Weekday number (0=Mon...6=Sun).
            - `<'str'>` Weekday name (case-insensitive), e.g., 'Mon', 'dienstag', '星期三'.
            - `<'None'>` Retains the original weekdays.

        ### Example:
        >>> pt.to_prev_weekday(0)      # The Monday of the previous week
        >>> pt.to_prev_weekday("Tue")  # The Tuesday of the previous week
        """
        return self.to_weekday(-1, weekday)

    def to_next_weekday(self, weekday: int | str | None = None) -> Self:
        """Adjust the date to the specific weekday of the next week `<'Pddt'>`.

        :param weekday `<'int/str/None'>`: Weekday value, defaults to `None`.
            - `<'int'>` Weekday number (0=Mon...6=Sun).
            - `<'str'>` Weekday name (case-insensitive), e.g., 'Mon', 'dienstag', '星期三'.
            - `<'None'>` Retains the original weekdays.

        ### Example:
        >>> pt.to_next_weekday(0)      # The Monday of the next week
        >>> pt.to_next_weekday("Tue")  # The Tuesday of the next week
        """
        return self.to_weekday(1, weekday)

    def to_weekday(self, offset: cython.int, weekday: int | str | None = None) -> Self:
        """Adjust the date to the specific weekday of the week (+/-) offset `<'Pddt'>`.

        :param offset `<'int'>`: The week offset (+/-).
        :param weekday `<'int/str/None'>`: Weekday value, defaults to `None`.
            - `<'int'>` Weekday number (0=Mon...6=Sun).
            - `<'str'>` Weekday name (case-insensitive), e.g., 'Mon', 'dienstag', '星期三'.
            - `<'None'>` Retains the original weekdays.

        ### Example:
        >>> pt.to_weekday(-2, 0)     # The Monday of the week, two weeks ago
        >>> pt.to_weekday(2, "Tue")  # The Tuesday of the week, two weeks later
        >>> pt.to_weekday(2)         # The same weekday of the week, two weeks later
        """
        # No offset
        wkd: cython.int = _parse_weekday(weekday, True)
        if offset == 0:
            return self._to_curr_weekday(wkd)

        # Access datetime array & info
        arr: np.ndarray = self.values_naive
        my_unit: str = self.unit
        delta: np.ndarray

        # Seperate times
        times = utils.dt64arr_times(arr, my_unit)  # int64[my_unit]

        # Adjust dates
        datesD = utils.dt64arr_as_int64_D(arr, my_unit)  # int64[D]
        if wkd == -1:
            datesD = utils.arr_add(datesD, offset * 7)  # int64[D]
        else:
            delta = utils.dt64arr_weekday(datesD, "D", -(offset * 7 + wkd))
            datesD = utils.arr_sub_arr(datesD, delta)  # int64[D]

        # Combine dates & times
        my_unit_ns: cython.bint = str_read(my_unit, 0) == "n"
        if my_unit_ns and not utils.is_dt64arr_ns_safe(datesD, "D"):
            dates = utils.arr_mul(datesD, utils.US_DAY)  # int64[us]
            times = utils.arr_floor_div(times, utils.NS_MICROSECOND)  # int64[us]
            dtype = "datetime64[us]"
        else:
            dates = utils.dt64arr_as_int64(datesD, my_unit, "D")  # int64[my_unit]
            dtype = "datetime64[%s]" % my_unit
        arr = utils.arr_add_arr(dates, times).astype(dtype)  # dt64[my_unit]
        return pddt_new_simple(arr, self.tzinfo, None, self.name)

    def _to_curr_weekday(self, weekday: cython.int) -> Self:
        """(internal) Adjust the date to the specific weekday of the current week `<'Pddt'>`.

        :param weekday `<'int'>`: Weekday number (0=Mon...6=Sun).
        """
        # No adjustment
        if weekday < 0:
            return self

        # Access datetime array & info
        arr: np.ndarray = self.values_naive
        my_unit: str = self.unit
        delta: np.ndarray

        # Seperate times
        times = utils.dt64arr_times(arr, my_unit)  # int64[my_unit]

        # Adjust dates
        datesD = utils.dt64arr_as_int64_D(arr, my_unit)  # int64[D]
        if weekday == 0:
            delta = utils.dt64arr_weekday(datesD, "D")  # weekday
        else:
            delta = utils.dt64arr_weekday(datesD, "D", -min(weekday, 6))  # weekday-off
        datesD = utils.arr_sub_arr(datesD, delta)  # int64[D]

        # Combine dates & times
        my_unit_ns: cython.bint = str_read(my_unit, 0) == "n"
        if my_unit_ns and not utils.is_dt64arr_ns_safe(datesD, "D"):
            dates = utils.arr_mul(datesD, utils.US_DAY)  # int64[us]
            times = utils.arr_floor_div(times, utils.NS_MICROSECOND)  # int64[us]
            dtype = "datetime64[us]"
        else:
            dates = utils.dt64arr_as_int64(datesD, my_unit, "D")  # int64[my_unit]
            dtype = "datetime64[%s]" % my_unit
        arr = utils.arr_add_arr(dates, times).astype(dtype)  # dt64[my_unit]
        return pddt_new_simple(arr, self.tzinfo, None, self.name)

    # . day
    def to_yesterday(self) -> Self:
        """Adjust the date to Yesterday `<'Pddt'>`."""
        return self.to_day(-1)

    def to_tomorrow(self) -> Self:
        """Adjust the date to Tomorrow `<'Pddt'>`."""
        return self.to_day(1)

    def to_day(self, offset: cython.int) -> Self:
        """Adjust the date to day (+/-) offset `<'Pddt'>`.

        :param offset `<'int'>`: The day offset (+/-).
        """
        # No offset
        if offset == 0:
            return self

        # Access datetime array & info
        arr: np.ndarray = self.values_naive
        my_unit: str = self.unit

        # Seperate times
        times: np.ndarray = utils.dt64arr_times(arr, my_unit)  # int64[my_unit]

        # Adjust dates
        datesD: np.ndarray = utils.dt64arr_as_int64_D(arr, my_unit, offset)  # int64[D]

        # Combine dates & times
        my_unit_ns: cython.bint = str_read(my_unit, 0) == "n"
        if my_unit_ns and not utils.is_dt64arr_ns_safe(datesD, "D"):
            dates = utils.arr_mul(datesD, utils.US_DAY)  # int64[us]
            times = utils.arr_floor_div(times, utils.NS_MICROSECOND)  # int64[us]
            dtype = "datetime64[us]"
        else:
            dates = utils.dt64arr_as_int64(datesD, my_unit, "D")  # int64[my_unit]
            dtype = "datetime64[%s]" % my_unit
        arr = utils.arr_add_arr(dates, times).astype(dtype)  # dt64[my_unit]
        return pddt_new_simple(arr, self.tzinfo, None, self.name)

    # . date&time
    def snap(self, freq: object = "S") -> Self:
        """Snap to nearest occurring frequency `<'Pddt'>`.

        Examples
        --------
        >>> idx = pd.DatetimeIndex(['2023-01-01', '2023-01-02',
        ...                        '2023-02-01', '2023-02-02'])
        >>> idx
        DatetimeIndex(['2023-01-01', '2023-01-02', '2023-02-01', '2023-02-02'],
        dtype='datetime64[ns]', freq=None)
        >>> idx.snap('MS')
        DatetimeIndex(['2023-01-01', '2023-01-01', '2023-02-01', '2023-02-01'],
        dtype='datetime64[ns]', freq=None)
        """
        return pddt_new_simple(super(Pddt, self).snap(freq))

    def to_datetime(
        self,
        year: cython.int = -1,
        month: cython.int = -1,
        day: cython.int = -1,
        hour: cython.int = -1,
        minute: cython.int = -1,
        second: cython.int = -1,
        microsecond: cython.int = -1,
        nanosecond: cython.int = -1,
    ) -> Self:
        """Adjust the date and time fields with new values `<'Pddt'>`.

        #### Fields set to `-1` means retaining the original values.

        :param year `<'int'>`: Year value, defaults to `-1`.
        :param month `<'int'>`: Yonth value (1-12), defaults to `-1`.
        :param day `<'int'>`: Day value (1-31), automacially clamped to the maximum days in the month, defaults to `-1`.
        :param hour `<'int'>`: Hour value (0-23), defaults to `-1`.
        :param minute `<'int'>`: Minute value (0-59), defaults to `-1`.
        :param second `<'int'>`: Second value (0-59), defaults to `-1`.
        :param microsecond `<'int'>`: Microsecond value (0-999999), defaults to `-1`.

        ### Equivalent to:
        >>> pt.replace(
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=second,
                microsecond=microsecond,
            )
        """
        # Fast-path
        # fmt: off
        if year <= 0 and month <= 0 and day <= 0:
            return self.to_time(hour, minute, second, microsecond, nanosecond)
        if hour < 0 and minute < 0 and second < 0 and microsecond < 0 and nanosecond < 0:
            return self.to_date(year, month, day)
        # fmt: on

        # Access datetime array & info
        arr: np.ndarray = self.values_naive
        my_unit: str = self.unit
        my_unit_ch: cython.Py_UCS4 = str_read(my_unit, 0)
        delta: np.ndarray
        hh_fac: cython.longlong
        mi_fac: cython.longlong
        ss_fac: cython.longlong
        us_fac: cython.longlong

        # Construct dates
        # . year
        if year > 0:
            months = (year - utils.EPOCH_YEAR) * 12  # year+off
            datesM = utils.arr_full_int64(months, arr.shape[0])  # int64[M]
        else:
            datesY = utils.dt64arr_as_int64_Y(arr, my_unit)  # int64[Y]
            datesM = utils.dt64arr_as_int64_M(datesY, "Y")  # int64[M]
        # . month
        if month > 0:
            datesM = utils.arr_add(datesM, min(month, 12) - 1)  # int64[M]
        else:
            delta = utils.dt64arr_month(arr, my_unit)
            datesM = utils.arr_add_arr(datesM, delta, -1)  # int64[M]
        # . day
        datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
        delta = utils.dt64arr_days_in_month(datesM, "M")
        if day > 0:
            delta = utils.arr_min(delta, day)
        else:
            delta = utils.arr_min_arr(delta, utils.dt64arr_day(arr, my_unit))
        datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]

        # Construct times
        times: np.ndarray = utils.arr_zero_int64(arr.shape[0])  # int64[my_unit]
        # . get conversion factors
        my_unit_adj: str = my_unit
        if my_unit_ch == "n" and utils.is_dt64arr_ns_safe(datesD, "D"):
            hh_fac = utils.NS_HOUR
            mi_fac = utils.NS_MINUTE
            ss_fac = utils.NS_SECOND
            us_fac = utils.NS_MICROSECOND
        elif my_unit_ch in ("n", "u"):
            hh_fac = utils.US_HOUR
            mi_fac = utils.US_MINUTE
            ss_fac = utils.US_SECOND
            us_fac = 1
            if my_unit_ch == "n":
                my_unit_ch, my_unit_adj = "u", "us"
        elif my_unit_ch == "m":
            hh_fac = utils.MS_HOUR
            mi_fac = utils.MS_MINUTE
            ss_fac = utils.MS_SECOND
            us_fac = 1
            if microsecond != 0:
                microsecond = min(microsecond, 999_999) // 1_000
        else:
            hh_fac = utils.SS_HOUR
            mi_fac = utils.SS_MINUTE
            ss_fac = 1
        # . hour
        if hour >= 0:
            times = utils.arr_add(times, min(hour, 23) * hh_fac)
        else:
            delta = utils.arr_mul(utils.dt64arr_hour(arr, my_unit), hh_fac)
            times = utils.arr_add_arr(times, delta)
        # . minute
        if minute >= 0:
            times = utils.arr_add(times, min(minute, 59) * mi_fac)
        else:
            delta = utils.arr_mul(utils.dt64arr_minute(arr, my_unit), mi_fac)
            times = utils.arr_add_arr(times, delta)
        # . second
        if second >= 0:
            times = utils.arr_add(times, min(second, 59) * ss_fac)
        else:
            delta = utils.arr_mul(utils.dt64arr_second(arr, my_unit), ss_fac)
            times = utils.arr_add_arr(times, delta)
        # . microsecond
        if my_unit_ch != "s":
            if microsecond >= 0:
                times = utils.arr_add(times, min(microsecond, 999_999) * us_fac)
            elif my_unit_ch == "m":
                delta = utils.dt64arr_millisecond(arr, my_unit)
                times = utils.arr_add_arr(times, delta)
            else:
                delta = utils.arr_mul(utils.dt64arr_microsecond(arr, my_unit), us_fac)
                times = utils.arr_add_arr(times, delta)
        # . nanosecond
        if my_unit_ch == "n":
            if nanosecond >= 0:
                times = utils.arr_add(times, min(nanosecond, 999))
            else:
                delta = utils.dt64arr_nanosecond(arr, my_unit)
                times = utils.arr_add_arr(times, delta)

        # Combine dates & times
        dtype = "datetime64[%s]" % my_unit_adj
        dates = utils.dt64arr_as_int64(datesD, my_unit_adj, "D")  # int64[my_unit]
        arr = utils.arr_add_arr(dates, times).astype(dtype)  # dt64[my_unit]
        return pddt_new_simple(arr, self.tzinfo, None, self.name)

    def to_date(
        self,
        year: cython.int = -1,
        month: cython.int = -1,
        day: cython.int = -1,
    ) -> Self:
        """Adjust the date with new values `<'Pddt'>`.

        #### Fields set to `-1` means retaining the original values.

        :param year `<'int'>`: Year value, defaults to `-1`.
        :param month `<'int'>`: Yonth value (1-12), defaults to `-1`.
        :param day `<'int'>`: Day value (1-31), automacially clamped to the maximum days in the month, defaults to `-1`.

        ### Equivalent to:
        >>> pt.replace(year=year, month=month, day=day)
        """
        # Fast-path
        if year <= 0:
            if month <= 0:
                # . D replacements
                return self.to_curr_month(day)
            # . M/D replacements
            return self.to_curr_year(month, day)

        # Access datetime array & info
        arr: np.ndarray = self.values_naive
        my_unit: str = self.unit
        delta: np.ndarray

        # Seperate times
        times = utils.dt64arr_times(arr, my_unit)  # int64[my_unit]

        # Construct dates
        months = (year - utils.EPOCH_YEAR) * 12  # year+off
        datesM = utils.arr_full_int64(months, arr.shape[0])  # int64[M]
        if month > 0:
            datesM = utils.arr_add(datesM, min(month, 12) - 1)  # int64[M]
        else:
            delta = utils.dt64arr_month(arr, my_unit)
            datesM = utils.arr_add_arr(datesM, delta, -1)  # int64[M]
        datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
        delta = utils.dt64arr_days_in_month(datesM, "M")
        if day > 0:
            delta = utils.arr_min(delta, day)
        else:
            delta = utils.arr_min_arr(delta, utils.dt64arr_day(arr, my_unit))
        datesD = utils.arr_add_arr(datesD, delta, -1)  # int64[D]

        # Combine dates & times
        my_unit_ns: cython.bint = str_read(my_unit, 0) == "n"
        if my_unit_ns and not utils.is_dt64arr_ns_safe(datesD, "D"):
            dates = utils.arr_mul(datesD, utils.US_DAY)  # int64[us]
            times = utils.arr_floor_div(times, utils.NS_MICROSECOND)  # int64[us]
            dtype = "datetime64[us]"
        else:
            dates = utils.dt64arr_as_int64(datesD, my_unit, "D")  # int64[my_unit]
            dtype = "datetime64[%s]" % my_unit
        arr = utils.arr_add_arr(dates, times).astype(dtype)  # dt64[my_unit]
        return pddt_new_simple(arr, self.tzinfo, None, self.name)

    def to_time(
        self,
        hour: cython.int = -1,
        minute: cython.int = -1,
        second: cython.int = -1,
        microsecond: cython.int = -1,
        nanosecond: cython.int = -1,
    ) -> Self:
        """Adjust the time fields with new values `<'Pddt'>`.

        #### Fields set to `-1` means retaining the original values.

        :param hour `<'int'>`: Hour value (0-23), defaults to `-1`.
        :param minute `<'int'>`: Minute value (0-59), defaults to `-1`.
        :param second `<'int'>`: Second value (0-59), defaults to `-1`.
        :param microsecond `<'int'>`: Microsecond value (0-999999), defaults to `-1`.

        ### Equivalent to:
        >>> pt.replace(
                hour=hour,
                minute=minute,
                second=second,
                microsecond=microsecond,
            )
        """
        # Fast-path: no changes
        if (
            hour < 0
            and minute < 0
            and second < 0
            and microsecond < 0
            and nanosecond < 0
        ):
            return self

        # Access datetime array & info
        arr: np.ndarray = self.values_naive
        my_unit: str = self.unit
        my_unit_ch: cython.Py_UCS4 = str_read(my_unit, 0)
        delta: np.ndarray
        hh_fac: cython.longlong
        mi_fac: cython.longlong
        ss_fac: cython.longlong
        us_fac: cython.longlong

        # Seperate dates
        datesD: np.ndarray = utils.dt64arr_as_int64_D(arr, my_unit)  # int64[D]

        # Get conversion factors
        my_unit_adj: str = my_unit
        if my_unit_ch == "n" and utils.is_dt64arr_ns_safe(datesD, "D"):
            hh_fac = utils.NS_HOUR
            mi_fac = utils.NS_MINUTE
            ss_fac = utils.NS_SECOND
            us_fac = utils.NS_MICROSECOND
        elif my_unit_ch in ("n", "u"):
            hh_fac = utils.US_HOUR
            mi_fac = utils.US_MINUTE
            ss_fac = utils.US_SECOND
            us_fac = 1
            if my_unit_ch == "n":
                my_unit_ch, my_unit_adj = "u", "us"
        elif my_unit_ch == "m":
            hh_fac = utils.MS_HOUR
            mi_fac = utils.MS_MINUTE
            ss_fac = utils.MS_SECOND
            us_fac = 1
            if microsecond != 0:
                microsecond = min(microsecond, 999_999) // 1_000
        else:
            hh_fac = utils.SS_HOUR
            mi_fac = utils.SS_MINUTE
            ss_fac = 1

        # Construct times
        times: np.ndarray = utils.arr_zero_int64(arr.shape[0])  # int64[my_unit]
        # . hour
        if hour >= 0:
            times = utils.arr_add(times, min(hour, 23) * hh_fac)
        else:
            delta = utils.arr_mul(utils.dt64arr_hour(arr, my_unit), hh_fac)
            times = utils.arr_add_arr(times, delta)
        # . minute
        if minute >= 0:
            times = utils.arr_add(times, min(minute, 59) * mi_fac)
        else:
            delta = utils.arr_mul(utils.dt64arr_minute(arr, my_unit), mi_fac)
            times = utils.arr_add_arr(times, delta)
        # . second
        if second >= 0:
            times = utils.arr_add(times, min(second, 59) * ss_fac)
        else:
            delta = utils.arr_mul(utils.dt64arr_second(arr, my_unit), ss_fac)
            times = utils.arr_add_arr(times, delta)
        # . microsecond
        if my_unit_ch != "s":
            if microsecond >= 0:
                times = utils.arr_add(times, min(microsecond, 999_999) * us_fac)
            elif my_unit_ch == "m":
                delta = utils.dt64arr_millisecond(arr, my_unit)
                times = utils.arr_add_arr(times, delta)
            else:
                delta = utils.arr_mul(utils.dt64arr_microsecond(arr, my_unit), us_fac)
                times = utils.arr_add_arr(times, delta)
        # . nanosecond
        if my_unit_ch == "n":
            if nanosecond >= 0:
                times = utils.arr_add(times, min(nanosecond, 999))
            else:
                delta = utils.dt64arr_nanosecond(arr, my_unit)
                times = utils.arr_add_arr(times, delta)

        # Combine dates & times
        dtype = "datetime64[%s]" % my_unit_adj
        dates = utils.dt64arr_as_int64(datesD, my_unit_adj, "D")  # int64[my_unit]
        arr = utils.arr_add_arr(dates, times).astype(dtype)  # dt64[my_unit]
        return pddt_new_simple(arr, self.tzinfo, None, self.name)

    def to_first_of(self, unit: str) -> Self:
        """Adjust the date to the first day of the specified datetime unit `<'Pddt'>`.

        :param unit `<'str'>`: The datetime unit.
            - `'Y'`: Sets to the first day of the current year.
            - `'Q'`: Sets to the first day of the current quarter.
            - `'M'`: Sets to the first day of the current month.
            - `'W'`: Sets to the first day (Monday) of the current week.
            - Month name (e.g., `'Jan'`, `'February'`, `'三月'`): Sets to the first day of that month.
        """
        # Access datetime array & info
        arr: np.ndarray = self.values_naive
        my_unit: str = self.unit

        # Seperate times
        times: np.ndarray = utils.dt64arr_times(arr, my_unit)  # int64[my_unit]

        # To weekday
        if unit == "W":
            datesD = utils.dt64arr_as_int64_D(arr, my_unit)  # int64[D]
            delta = utils.dt64arr_weekday(datesD, "D")  # weekday
            datesD = utils.arr_sub_arr(datesD, delta)  # int64[D]

        # To month
        elif unit == "M":
            datesM = utils.dt64arr_as_int64_M(arr, my_unit)  # int64[M]
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]

        # To quarter
        elif unit == "Q":
            datesQ = utils.dt64arr_as_int64_Q(arr, my_unit)  # int64[Q]
            datesM = utils.arr_mul(datesQ, 3)  # int64[M]
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]

        # To year
        elif unit == "Y":
            datesY = utils.dt64arr_as_int64_Y(arr, my_unit)  # int64[Y]
            datesD = utils.dt64arr_as_int64_D(datesY, "Y")  # int64[D]

        # Custom
        else:
            # Month name
            val: cython.int
            if (val := _parse_month(unit, False)) != -1:
                datesY = utils.dt64arr_as_int64_Y(arr, my_unit)  # int64[Y]
                datesM = utils.dt64arr_as_int64_M(datesY, "Y", val - 1)  # int64[M]
                datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]

            # Invalid
            else:
                raise errors.InvalidTimeUnitError(
                    "invalid 'first of' datetime unit '%s'.\nSupports: "
                    "['Y', 'Q', 'M', 'W'] or Month name." % unit
                )

        # Combine dates & times
        my_unit_ns: cython.bint = str_read(my_unit, 0) == "n"
        if my_unit_ns and not utils.is_dt64arr_ns_safe(datesD, "D"):
            dates = utils.arr_mul(datesD, utils.US_DAY)  # int64[us]
            times = utils.arr_floor_div(times, utils.NS_MICROSECOND)  # int64[us]
            dtype = "datetime64[us]"
        else:
            dates = utils.dt64arr_as_int64(datesD, my_unit, "D")  # int64[my_unit]
            dtype = "datetime64[%s]" % my_unit
        arr = utils.arr_add_arr(dates, times).astype(dtype)  # dt64[my_unit]
        return pddt_new_simple(arr, self.tzinfo, None, self.name)

    def to_last_of(self, unit: str) -> Self:
        """Adjust the date to the last day of the specified datetime unit `<'Pddt'>`.

        :param unit `<'str'>`: The datetime unit.
            - `'Y'`: Sets to the last day of the current year.
            - `'Q'`: Sets to the last day of the current quarter.
            - `'M'`: Sets to the last day of the current month.
            - `'W'`: Sets to the last day (Sunday) of the current week.
            - Month name (e.g., `'Jan'`, `'February'`, `'三月'`): Sets to the last day of that month.
        """
        # Access datetime array & info
        arr: np.ndarray = self.values_naive
        my_unit: str = self.unit

        # Seperate times
        times: np.ndarray = utils.dt64arr_times(arr, my_unit)  # int64[my_unit]

        # To weekday
        if unit == "W":
            datesD = utils.dt64arr_as_int64_D(arr, my_unit)  # int64[D]
            delta = utils.dt64arr_weekday(datesD, "D", -6)  # weekday-6
            datesD = utils.arr_sub_arr(datesD, delta)  # int64[D]

        # To month
        elif unit == "M":
            datesM = utils.dt64arr_as_int64_M(arr, my_unit, 1)  # int64[M]+1
            datesD = utils.dt64arr_as_int64_D(datesM, "M", -1)  # int64[D]-1

        # To quarter
        elif unit == "Q":
            datesQ = utils.dt64arr_as_int64_Q(arr, my_unit, 1)  # int64[Q]+1
            datesM = utils.arr_mul(datesQ, 3)  # int64[M]
            datesD = utils.dt64arr_as_int64_D(datesM, "M", -1)  # int64[D]-1

        # To year
        elif unit == "Y":
            datesY = utils.dt64arr_as_int64_Y(arr, my_unit, 1)  # int64[Y]+1
            datesD = utils.dt64arr_as_int64_D(datesY, "Y", -1)  # int64[D]-1

        # Custom
        else:
            # Month name
            val: cython.int
            if (val := _parse_month(unit, False)) != -1:
                datesY = utils.dt64arr_as_int64_Y(arr, my_unit)  # int64[Y]
                datesM = utils.dt64arr_as_int64_M(datesY, "Y", val)  # int64[M]+val
                datesD = utils.dt64arr_as_int64_D(datesM, "M", -1)  # int64[D]-1

            # Invalid
            else:
                raise errors.InvalidTimeUnitError(
                    "invalid 'last of' datetime unit '%s'.\nSupports: "
                    "['Y', 'Q', 'M', 'W'] or Month name." % unit
                )

        # Combine dates & times
        my_unit_ns: cython.bint = str_read(my_unit, 0) == "n"
        if my_unit_ns and not utils.is_dt64arr_ns_safe(datesD, "D"):
            dates = utils.arr_mul(datesD, utils.US_DAY)  # int64[us]
            times = utils.arr_floor_div(times, utils.NS_MICROSECOND)  # int64[us]
            dtype = "datetime64[us]"
        else:
            dates = utils.dt64arr_as_int64(datesD, my_unit, "D")  # int64[my_unit]
            dtype = "datetime64[%s]" % my_unit
        arr = utils.arr_add_arr(dates, times).astype(dtype)  # dt64[my_unit]
        return pddt_new_simple(arr, self.tzinfo, None, self.name)

    def to_start_of(self, unit: str) -> Self:
        """Adjust the datetime to the start of the specified datetime unit `<'Pddt'>`.

        :param unit `<'str'>`: The datetime unit.
            - `'Y'`: Sets date to the first day of the current year & time to '00:00:00.000000'.
            - `'Q'`: Sets date to the first day of the current quarter & time to '00:00:00.000000'.
            - `'M'`: Sets date to the first day of the current month & time to '00:00:00.000000'.
            - `'W'`: Sets date to the first day (Monday) of the current week & time to '00:00:00.000000'.
            - `'D'`: Retains the original date & sets time to '00:00:00.000000'.
            - `'h'`: Retains the original date & sets time to 'XX:00:00.000000'.
            - `'m'`: Retains the original date & sets time to 'XX:XX:00.000000'.
            - `'s'`: Retains the original date & sets time to 'XX:XX:XX.000000'.
            - `'ms'`: Retains the original date & sets time to 'XX:XX:XX.XXX000'.
            - `'us'`: Return the instance itself.
            - Month name (e.g., `'Jan'`, `'February'`, `'三月'`): Sets date to first day of that month & time to '00:00:00.000000'.
            - Weekday name (e.g., `'Mon'`, `'Tuesday'`, `'星期三'`): Sets date to that weekday & time to '00:00:00.000000'.
        """
        # Fast-path
        if unit is None:
            return self  # exit: invalid
        if unit == "ns":
            return self  # exit: to nanosecond
        if unit in ("D", "h", "m", "min", "s", "ms", "us"):
            return self.floor(unit)  # exit: floor to unit

        # Access datetime array & info
        arr: np.ndarray = self.values_naive
        my_unit: str = self.unit

        # To weekday
        if unit == "W":
            datesD = utils.dt64arr_as_int64_D(arr, my_unit)  # int64[D]
            delta = utils.dt64arr_weekday(datesD, "D")  # weekday
            datesD = utils.arr_sub_arr(datesD, delta)  # int64[D]
            arr = utils.dt64arr_as_unit(datesD, my_unit, "D")  # dt64[my_unit]

        # To month
        elif unit == "M":
            datesM = utils.dt64arr_as_int64_M(arr, my_unit)  # int64[M]
            arr = utils.dt64arr_as_unit(datesM, my_unit, "M")  # dt64[my_unit]

        # To quarter
        elif unit == "Q":
            datesQ = utils.dt64arr_as_int64_Q(arr, my_unit)  # int64[Q]
            datesM = utils.arr_mul(datesQ, 3)  # int64[M]
            arr = utils.dt64arr_as_unit(datesM, my_unit, "M")  # dt64[my_unit]

        # To year
        elif unit == "Y":
            datesY = utils.dt64arr_as_int64_Y(arr, my_unit)  # int64[Y]
            arr = utils.dt64arr_as_unit(datesY, my_unit, "Y")  # dt64[my_unit]

        # Custom
        else:
            val: cython.int
            # Month name
            if (val := _parse_month(unit, False)) != -1:
                datesY = utils.dt64arr_as_int64_Y(arr, my_unit)  # int64[Y]
                datesM = utils.dt64arr_as_int64_M(datesY, "Y", val - 1)  # int64[M]
                arr = utils.dt64arr_as_unit(datesM, my_unit, "M")  # dt64[my_unit]

            # Weekday name
            elif (val := _parse_weekday(unit, False)) != -1:
                datesD = utils.dt64arr_as_int64_D(arr, my_unit)  # int64[D]
                delta = utils.dt64arr_weekday(datesD, "D", -val)  # weekday-val
                datesD = utils.arr_sub_arr(datesD, delta)  # int64[D]
                arr = utils.dt64arr_as_unit(datesD, my_unit, "D")  # dt64[my_unit]

            # Invalid
            else:
                raise errors.InvalidTimeUnitError(
                    "invalid 'start of' time unit '%s'.\nSupports: "
                    "['Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'] "
                    "or Month/Weekday name." % unit
                )

        # New instance
        return pddt_new_simple(arr, self.tzinfo, None, self.name)

    def to_end_of(self, unit: str) -> Self:
        """Adjust the datetime to the end of the specified datetime unit `<'Pddt'>`.

        :param unit `<'str'>`: The datetime unit.
            - `'Y'`: Sets date to the last day of the current year & time to '23:59:59.999999'.
            - `'Q'`: Sets date to the last day of the current quarter & time to '23:59:59.999999'.
            - `'M'`: Sets date to the last day of the current month & time to '23:59:59.999999'.
            - `'W'`: Sets date to the last day (Sunday) of the current week & time to '23:59:59.999999'.
            - `'D'`: Retains the original date & sets time to '23:59:59.999999'.
            - `'h'`: Retains the original date & sets time to 'XX:59:59.999999'.
            - `'m'`: Retains the original date & sets time to 'XX:XX:59.999999'.
            - `'s'`: Retains the original date & sets time to 'XX:XX:XX.999999'.
            - `'ms'`: Retains the original date & sets time to 'XX:XX:XX.XXX999'.
            - `'us'`: Return the instance itself.
            - `'Month name (e.g., `'Jan'`, `'February'`, `'三月'`): Sets date to last day of that month & time to '23:59:59.999999'.
            - `'Weekday name (e.g., `'Mon'`, `'Tuesday'`, `'星期三'`): Sets date to that weekday & time to '23:59:59.999999'.
        """
        # Fast-path
        if unit is None:
            return self  # exit: invalid
        if unit == "ns":
            return self  # exit: to nanosecond
        my_unit: str = self.unit
        if unit == my_unit:
            return self  # exit: same unit

        # Access datetime array & info
        arr: np.ndarray = self.values_naive
        my_unit_ch: cython.Py_UCS4 = str_read(my_unit, 0)
        fac: cython.longlong  # floor divisor factor
        mul: cython.longlong  # multiple after floor
        off: cython.longlong  # offset to floor value

        # To microsecond
        if unit == "us":
            # my_unit: [ms, s] -> to_unit [us]
            if my_unit_ch != "n":
                return self  # exit: lower unit
            # my_unit [ns] -> to_unit [us]
            fac = utils.NS_MICROSECOND
            if utils.is_dt64arr_ns_safe(arr, "ns"):
                arr = utils.arr_floor_to_mul(arr, fac, fac, fac - 1)
                dtype = "datetime64[ns]"
            else:
                arr = utils.arr_floor_div(arr, fac)
                dtype = "datetime64[us]"
            # back to dt64[my_unit]
            arr = arr.astype(dtype)

        # To millisecond
        elif unit == "ms":
            # my_unit [s] -> to_unit [ms]
            if my_unit_ch == "s":
                return self  # exit: lower unit
            # my_unit [ns] -> to_unit [ms]
            if my_unit_ch == "n":
                fac = utils.NS_MILLISECOND
                if utils.is_dt64arr_ns_safe(arr, "ns"):
                    mul, off = fac, fac - 1
                    dtype = "datetime64[ns]"
                else:
                    mul, off = utils.US_MILLISECOND, utils.US_MILLISECOND - 1
                    dtype = "datetime64[us]"
            # my_unit [us] -> to_unit [ms]
            else:
                fac = utils.US_MILLISECOND
                mul, off = fac, fac - 1
                dtype = "datetime64[us]"
            # back to dt64[my_unit]
            arr = utils.arr_floor_to_mul(arr, fac, mul, off).astype(dtype)

        # To second
        elif unit == "s":
            # my_unit [ns] -> to_unit [s]
            if my_unit_ch == "n":  # ns
                fac = utils.NS_SECOND
                if utils.is_dt64arr_ns_safe(arr, "ns"):
                    mul, off = fac, fac - 1
                    dtype = "datetime64[ns]"
                else:
                    mul, off = utils.US_SECOND, utils.US_SECOND - 1
                    dtype = "datetime64[us]"
            # my_unit [us, ms] -> to_unit [s]
            else:
                if my_unit_ch == "u":
                    fac, dtype = utils.US_SECOND, "datetime64[us]"
                else:
                    fac, dtype = utils.MS_SECOND, "datetime64[ms]"
                mul, off = fac, fac - 1
            # back to dt64[my_unit]
            arr = utils.arr_floor_to_mul(arr, fac, mul, off).astype(dtype)

        # To minute
        elif unit == "m" or unit == "min":
            # my_unit [ns] -> to_unit [s]
            if my_unit_ch == "n":
                fac = utils.NS_MINUTE
                if utils.is_dt64arr_ns_safe(arr, "ns"):
                    mul, off = fac, fac - 1
                    dtype = "datetime64[ns]"
                else:
                    mul, off = utils.US_MINUTE, utils.US_MINUTE - 1
                    dtype = "datetime64[us]"
            # my_unit [us, ms, s] -> to_unit [m]
            else:
                if my_unit_ch == "u":
                    fac, dtype = utils.US_MINUTE, "datetime64[us]"
                elif my_unit_ch == "m":
                    fac, dtype = utils.MS_MINUTE, "datetime64[ms]"
                else:  # s
                    fac, dtype = utils.SS_MINUTE, "datetime64[s]"
                mul, off = fac, fac - 1
            # back to dt64[my_unit]
            arr = utils.arr_floor_to_mul(arr, fac, mul, off).astype(dtype)

        # To hour
        elif unit == "h":
            # my_unit [ns] -> to_unit [h]
            if my_unit_ch == "n":
                fac = utils.NS_HOUR
                if utils.is_dt64arr_ns_safe(arr, "ns"):
                    mul, off = fac, fac - 1
                    dtype = "datetime64[ns]"
                else:
                    mul, off = utils.US_HOUR, utils.US_HOUR - 1
                    dtype = "datetime64[us]"
            # my_unit [us, ms, s] -> to_unit [h]
            else:
                if my_unit_ch == "u":  # us
                    fac, dtype = utils.US_HOUR, "datetime64[us]"
                elif my_unit_ch == "m":  # ms
                    fac, dtype = utils.MS_HOUR, "datetime64[ms]"
                else:  # s
                    fac, dtype = utils.SS_HOUR, "datetime64[s]"
                mul, off = fac, fac - 1
            # back to dt64[my_unit]
            arr = utils.arr_floor_to_mul(arr, fac, mul, off).astype(dtype)

        # To day
        elif unit == "D":
            # my_unit [ns] -> to_unit [D]
            if my_unit_ch == "n":
                fac = utils.NS_DAY
                if utils.is_dt64arr_ns_safe(arr, "ns"):
                    mul, off = fac, fac - 1
                    dtype = "datetime64[ns]"
                else:
                    mul, off = utils.US_DAY, utils.US_DAY - 1
                    dtype = "datetime64[us]"
            # my_unit [us, ms, s] -> to_unit [D]
            else:
                if my_unit_ch == "u":  # us
                    fac, dtype = utils.US_DAY, "datetime64[us]"
                elif my_unit_ch == "m":  # ms
                    fac, dtype = utils.MS_DAY, "datetime64[ms]"
                else:  # s
                    fac, dtype = utils.SS_DAY, "datetime64[s]"
                mul, off = fac, fac - 1
            # back to dt64[my_unit]
            arr = utils.arr_floor_to_mul(arr, fac, mul, off).astype(dtype)

        # To weekday
        elif unit == "W":
            datesD = utils.dt64arr_as_int64_D(arr, my_unit)  # int64[D]
            delta = utils.dt64arr_weekday(datesD, "D", -7)  # weekday-7
            datesD = utils.arr_sub_arr(datesD, delta)  # int64[D]
            arr = utils.dt64arr_as_unit(datesD, my_unit, "D") - 1  # dt64[my_unit]-1

        # To Month
        elif unit == "M":
            datesM = utils.dt64arr_as_int64_M(arr, my_unit, 1)  # int64[M]+1
            arr = utils.dt64arr_as_unit(datesM, my_unit, "M") - 1  # dt64[my_unit]-1

        # To Quarter
        elif unit == "Q":
            datesQ = utils.dt64arr_as_int64_Q(arr, my_unit, 1)  # int64[Q]+1
            datesM = utils.arr_mul(datesQ, 3)  # int64[M]
            arr = utils.dt64arr_as_unit(datesM, my_unit, "M") - 1  # dt64[my_unit]-1

        # To Year
        elif unit == "Y":
            datesY = utils.dt64arr_as_int64_Y(arr, my_unit, 1)  # int64[Y]
            arr = utils.dt64arr_as_unit(datesY, my_unit, "Y") - 1  # dt64[my_unit]-1

        # Custom
        else:
            val: cython.int
            # Month name
            if (val := _parse_month(unit, False)) != -1:
                datesY = utils.dt64arr_as_int64_Y(arr, my_unit)  # int64[Y]
                datesM = utils.dt64arr_as_int64_M(datesY, "Y", val)  # int64[M]+val
                arr = utils.dt64arr_as_unit(datesM, my_unit, "M") - 1  # dt64[my_unit]-1

            # Weekday name
            elif (val := _parse_weekday(unit, False)) != -1:
                datesD = utils.dt64arr_as_int64_D(arr, my_unit)  # int64[D]
                delta = utils.dt64arr_weekday(datesD, "D", -val - 1)  # weekday-val-1
                datesD = utils.arr_sub_arr(datesD, delta)  # int64[D]
                arr = utils.dt64arr_as_unit(datesD, my_unit, "D") - 1  # dt64[my_unit]-1

            # Invalid
            else:
                raise errors.InvalidTimeUnitError(
                    "invalid 'end of' datetime unit '%s'.\nSupports: "
                    "['Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'] "
                    "or Month/Weekday name." % unit
                )

        # New instance
        return pddt_new_simple(arr, self.tzinfo, None, self.name)

    # . round / ceil / floor
    def round(self, unit: str) -> Self:
        """Perform round operation to the specified datetime unit `<'Pddt'>`.

        :param unit `<'str'>`: The datetime unit to round to.
            Supported datetime units: 'D', 'h', 'm', 's', 'ms', 'us', 'ns'.
        """
        # No change / Same unit
        if unit is None:
            return self  # exit
        my_unit: str = self.unit
        if unit == my_unit:
            return self  # exit

        # Access datetime array & info
        arr: np.ndarray = self.values_naive

        # Perform operation
        try:
            res: np.ndarray = utils.dt64arr_round(arr, unit, my_unit)
        except Exception as err:
            raise errors.InvalidTimeUnitError(err) from err
        if arr is res:
            return self
        return pddt_new_simple(res, self.tzinfo, None, self.name)

    def ceil(self, unit: str) -> Self:
        """Perform ceil operation to the specified datetime unit `<'Pddt'>`.

        :param unit `<'str'>`: The datetime unit to round to.
            Supported datetime units: 'D', 'h', 'm', 's', 'ms', 'us', 'ns'.
        """
        # No change / Same unit
        if unit is None:
            return self  # exit
        my_unit: str = self.unit
        if unit == my_unit:
            return self  # exit

        # Access datetime array & info
        arr: np.ndarray = self.values_naive

        # Perform operation
        try:
            res: np.ndarray = utils.dt64arr_ceil(arr, unit, my_unit)
        except Exception as err:
            raise errors.InvalidTimeUnitError(err) from err
        if arr is res:
            return self
        return pddt_new_simple(res, self.tzinfo, None, self.name)

    def floor(self, unit: str) -> Self:
        """Perform floor operation to the specified datetime unit `<'Pddt'>`.

        :param unit `<'str'>`: The datetime unit to round to.
            Supported datetime units: 'D', 'h', 'm', 's', 'ms', 'us', 'ns'.
        """
        # No change / Same unit
        if unit is None:
            return self  # exit
        my_unit: str = self.unit
        if unit == my_unit:
            return self  # exit

        # Access datetime array & info
        arr: np.ndarray = self.values_naive

        # Perform operation
        try:
            res: np.ndarray = utils.dt64arr_floor(arr, unit, my_unit)
        except Exception as err:
            raise errors.InvalidTimeUnitError(err) from err
        if arr is res:
            return self
        return pddt_new_simple(res, self.tzinfo, None, self.name)

    # Calendar -----------------------------------------------------------------------------
    # . iso
    def isocalendar(self) -> DataFrame:
        """Return the ISO calendar `<'DateFrame'>`."""
        _iso = utils.dt64arr_isocalendar(self.values_naive)
        return DataFrame(_iso, columns=["year", "week", "weekday"], index=self)

    def isoyear(self) -> Index[np.int64]:
        """Return the ISO calendar years `<'Index[int64]'>`."""
        _iso = utils.dt64arr_isocalendar(self.values_naive)
        return Index(_iso[:, 0], name="isoyear")

    def isoweek(self) -> Index[np.int64]:
        """Return the ISO calendar week numbers (1-53) `<'Index[int64]'>`."""
        _iso = utils.dt64arr_isocalendar(self.values_naive)
        return Index(_iso[:, 1], name="isoweek")

    def isoweekday(self) -> Index[np.int64]:
        """Return the ISO calendar weekdays (1=Mon...7=Sun) `<'Index[int64]'>`."""
        _iso = utils.dt64arr_isocalendar(self.values_naive)
        return Index(_iso[:, 2], name="isoweekday")

    # . year
    @property
    def year(self) -> Index[np.int64]:
        """The years `<'Index[int64]'>`."""
        return Index(utils.dt64arr_year(self.values_naive), name="year")

    def is_leap_year(self) -> Index[bool]:
        """Determine if the instance's years are in a leap year `<'Index[bool]'>`."""
        return Index(utils.dt64arr_is_leap_year(self.values_naive), name="is_leap_year")

    def is_long_year(self) -> Index[bool]:
        """Determine if the instance's years are in a long year `<'Index[bool]'>`.

        - Long year: maximum ISO week number is 53.
        """
        return Index(utils.dt64arr_is_long_year(self.values_naive), name="is_long_year")

    def leap_bt_year(self, year: cython.int) -> Index[np.int64]:
        """Compute the number of leap years between the instance's
        years and the passed-in 'year' `<'Index[int64]'>`.
        """
        return Index(
            utils.dt64arr_leap_bt_year(self.values_naive, year),
            name="leap_bt_year",
        )

    def days_in_year(self) -> Index[np.int64]:
        """Return the maximum number of days (365, 366)
        in the instance's years `<'Index[int64]'>`.
        """
        return Index(utils.dt64arr_days_in_year(self.values_naive), name="days_in_year")

    def days_bf_year(self) -> Index[np.int64]:
        """Return the number of days from January 1st, 1st AD,
        to the start of the instance's years `<'Index[int64]'>`.
        """
        return Index(utils.dt64arr_days_bf_year(self.values_naive), name="days_bf_year")

    def days_of_year(self) -> Index[np.int64]:
        """Return the number of days since the start
        of the instance's years `<'Index[int64]'>`.
        """
        return Index(utils.dt64arr_days_of_year(self.values_naive), name="days_of_year")

    def is_year(self, year: cython.int) -> Index[bool]:
        """Determine if the instance's years match
        the passed-in 'year' `<'Index[bool]'>`.
        """
        return Index(
            utils.arr_equal_to(utils.dt64arr_year(self.values_naive), year),
            name="is_year",
        )

    # . quarter
    @property
    def quarter(self) -> Index[np.int64]:
        """The quarters (1-4) `<'Index[int64]'>`."""
        return Index(utils.dt64arr_quarter(self.values_naive), name="quarter")

    def days_in_quarter(self) -> Index[np.int64]:
        """Return the maximum number of days (90-92)
        in the instance's quarters `<'Index[int64]'>`.
        """
        return Index(
            utils.dt64arr_days_in_quarter(self.values_naive),
            name="days_in_quarter",
        )

    def days_bf_quarter(self) -> Index[np.int64]:
        """Return the number of days from the start of the instance's years
        to the start of the instance's quarters `<'Index[int64]'>`.
        """
        return Index(
            utils.dt64arr_days_bf_quarter(self.values_naive),
            name="days_bf_quarter",
        )

    def days_of_quarter(self) -> Index[np.int64]:
        """Return the number of days since the start
        of the instance's quarters `<'Index[int64]'>`.
        """
        return Index(
            utils.dt64arr_days_of_quarter(self.values_naive),
            name="days_of_quarter",
        )

    def is_quarter(self, quarter: cython.int) -> Index[bool]:
        """Determine if the instance's quarters match
        the passed-in 'quarter' `<'Index[bool]'>`.
        """
        return Index(
            utils.arr_equal_to(utils.dt64arr_quarter(self.values_naive), quarter),
            name="is_quarter",
        )

    # . month
    @property
    def month(self) -> Index[np.int64]:
        """The months (1-12) `<'Index[int64]'>`."""
        return Index(utils.dt64arr_month(self.values_naive), name="month")

    def days_in_month(self) -> Index[np.int64]:
        """Return the maximum number of days (28-31)
        in the instance's months `<'Index[int64]'>`.
        """
        return Index(
            utils.dt64arr_days_in_month(self.values_naive),
            name="days_in_month",
        )

    def days_bf_month(self) -> Index[np.int64]:
        """Return the number of days from the start of the instance's
        years to the start of the instance's months `<'Index[int64]'>`.
        """
        return Index(
            utils.dt64arr_days_bf_month(self.values_naive),
            name="days_bf_month",
        )

    def days_of_month(self) -> Index[np.int64]:
        """Return the number of days (1-31) since the start
        of the instance's months `<'Index[int64]'>`.

        ### Equivalent to:
        >>> pt.day
        """
        return Index(utils.dt64arr_day(self.values_naive), name="days_of_month")

    def is_month(self, month: str | int) -> Index[bool]:
        """Determine if the instance's months match
        the passed-in 'month' `<'Index[bool]'>`.
        """
        mm: cython.int = _parse_month(month, True)
        return Index(
            utils.arr_equal_to(utils.dt64arr_month(self.values_naive), mm),
            name="is_month",
        )

    # . weekday
    @property
    def weekday(self) -> Index[np.int64]:
        """The weekdays (0=Mon...6=Sun) `<'Index[int64]'>`."""
        return Index(utils.dt64arr_weekday(self.values_naive), name="weekday")

    def is_weekday(self, weekday: int | str) -> Index[bool]:
        """Determine if the instance's weekdays match
        the passed-in 'weekday' `<'Index[bool]'>`.
        """
        wd: cython.int = _parse_weekday(weekday, True)
        return Index(
            utils.arr_equal_to(utils.dt64arr_weekday(self.values_naive), wd),
            name="is_weekday",
        )

    # . day
    @property
    def day(self) -> Index[np.int64]:
        """The days (1-31) `<'Index[int64]'>`."""
        return Index(utils.dt64arr_day(self.values_naive), name="day")

    def is_day(self, day: cython.int) -> Index[bool]:
        """Determine if the instance's days match
        the passed-in 'day' `<'Index[bool]'>`.
        """
        return Index(
            utils.arr_equal_to(utils.dt64arr_day(self.values_naive), day),
            name="is_day",
        )

    # . time
    @property
    def hour(self) -> Index[np.int64]:
        """The hours (0-23) `<'Index[int64]'>`."""
        return Index(utils.dt64arr_hour(self.values_naive), name="hour")

    @property
    def minute(self) -> Index[np.int64]:
        """The minutes (0-59) `<'Index[int64]'>`."""
        return Index(utils.dt64arr_minute(self.values_naive), name="minute")

    @property
    def second(self) -> Index[np.int64]:
        """The seconds (0-59) `<'Index[int64]'>`."""
        return Index(utils.dt64arr_second(self.values_naive), name="second")

    @property
    def millisecond(self) -> Index[np.int64]:
        """The milliseconds (0-999) `<'Index[int64]'>`."""
        return Index(utils.dt64arr_millisecond(self.values_naive), name="millisecond")

    @property
    def microsecond(self) -> Index[np.int64]:
        """The microseconds (0-999999) `<'Index[int64]'>`."""
        return Index(utils.dt64arr_microsecond(self.values_naive), name="microsecond")

    @property
    def nanosecond(self) -> Index[np.int64]:
        """The nanoseconds (0-999) `<'Index[int64]'>`."""
        return Index(utils.dt64arr_nanosecond(self.values_naive), name="nanosecond")

    # . date&time
    def is_first_of(self, unit: str) -> Index[bool]:
        """Determine if the dates are on the first day of
        the specified datetime unit `<'Index[bool]'>`.

        :param unit `<'str'>`: The datetime unit.
            - `'Y'`: Is on the first day of the current year.
            - `'Q'`: Is on the first day of the current quarter.
            - `'M'`: Is on the first day of the current month.
            - `'W'`: Is on the first day (Monday) of the current week.
            - Month name (e.g., `'Jan'`, `'February'`, `'三月'`): Is the first day of that month.
        """
        # To start of
        pt_1st = self.to_first_of(unit)

        # Compare dates
        datesD_1st = utils.dt64arr_as_int64_D(pt_1st.values)  # int64[D]
        datesD = utils.dt64arr_as_int64_D(self.values)  # int64[D]
        arr = utils.arr_equal_to_arr(datesD, datesD_1st)
        return Index(arr, name="is_first_of[%s]" % unit)

    def is_last_of(self, unit: str) -> Index[bool]:
        """Determine if the dates are on the last day of
        the specified datetime unit `<'Index[bool]'>`.

        :param unit `<'str'>`: The datetime unit.
            - `'Y'`: Is on the last day of the current year.
            - `'Q'`: Is on the last day of the current quarter.
            - `'M'`: Is on the last day of the current month.
            - `'W'`: Is on the last day (Sunday) of the current week.
            - Month name (e.g., `'Jan'`, `'February'`, `'三月'`): Is the last day of that month.
        """
        # To last of
        pt_lst = self.to_last_of(unit)

        # Compare dates
        datesD_lst = utils.dt64arr_as_int64_D(pt_lst.values)  # int64[D]
        datesD = utils.dt64arr_as_int64_D(self.values)  # int64[D]
        arr = utils.arr_equal_to_arr(datesD, datesD_lst)
        return Index(arr, name="is_last_of[%s]" % unit)

    def is_start_of(self, unit: str) -> Index[bool]:
        """Determine if the datetimes are at the start of
        the specified datetime unit `<'Index[bool]'>`.

        :param unit `<'str'>`: The datetime unit.
            - `'Y'`: Is on the first day of the current year at time '00:00:00.000000'.
            - `'Q'`: Is on the first day of the current quarter at time '00:00:00.000000'.
            - `'M'`: Is on the first day of the current month at time '00:00:00.000000'.
            - `'W'`: Is on the first day (Monday) of the current week at time '00:00:00.000000'.
            - `'D'`: Is at time '00:00:00.000000'.
            - `'h'`: Is at time 'XX:00:00.000000'.
            - `'m'`: Is at time 'XX:XX:00.000000'.
            - `'s'`: Is at time 'XX:XX:XX.000000'.
            - `'ms'`: Is at time 'XX:XX:XX.XXX000'.
            - `'us'`: Always return `True`.
            - Month name (e.g., `'Jan'`, `'February'`, `'三月'`): Is on the first day of that month at time '00:00:00.000000'.
            - Weekday name (e.g., `'Mon'`, `'Tuesday'`, `'星期三'`): Is on that weekday at time '00:00:00.000000'.
        """
        # To start of
        pt_stt = self.to_start_of(unit)

        # Compare datetimes
        pt_stt_unit: str = pt_stt.unit
        if pt_stt_unit == self.unit:
            arr = utils.arr_equal_to_arr(self.values, pt_stt.values)
        else:
            arr = utils.arr_equal_to_arr(
                utils.dt64arr_as_unit(self.values, pt_stt_unit), pt_stt.values
            )
        return Index(arr, name="is_start_of[%s]" % unit)

    def is_end_of(self, unit: str) -> Index[bool]:
        """Determine if the datetimes are at the end of
        the specified datetime unit `<'Index[bool]'>`.

        :param unit `<'str'>`: The datetime unit.
            - `'Y'`: Is on the last day of the current year at time '23:59:59.999999'.
            - `'Q'`: Is on the last day of the current quarter at time '23:59:59.999999'.
            - `'M'`: Is on the last day of the current month at time '23:59:59.999999'.
            - `'W'`: Is on the last day (Sunday) of the current week at time '23:59:59.999999'.
            - `'D'`: Is at time '23:59:59.999999'.
            - `'h'`: Is at time 'XX:59:59.999999'.
            - `'m'`: Is at time 'XX:XX:59.999999'.
            - `'s'`: Is at time 'XX:XX:XX.999999'.
            - `'ms'`: Is at time 'XX:XX:XX.XXX999'.
            - `'us'`: Always return `True`.
            - Month name (e.g., `'Jan'`, `'February'`, `'三月'`): Is on the last day of that month at time '23:59:59.999999'.
            - Weekday name (e.g., `'Mon'`, `'Tuesday'`, `'星期三'`): Is on that weekday at time '23:59:59.999999'.
        """
        # To end of
        pt_end = self.to_end_of(unit)

        # Compare datetimes
        pt_end_unit: str = pt_end.unit
        if pt_end_unit == self.unit:
            arr = utils.arr_equal_to_arr(self.values, pt_end.values)
        else:
            arr = utils.arr_equal_to_arr(
                utils.dt64arr_as_unit(self.values, pt_end_unit), pt_end.values
            )
        return Index(arr, name="is_end_of[%s]" % unit)

    # Timezone -----------------------------------------------------------------------------
    @property
    def tz_available(self) -> set[str]:
        """The available timezone names `<'set[str]'>`.

        ### Equivalent to:
        >>> zoneinfo.available_timezones()
        """
        return _available_timezones()

    def is_local(self) -> bool:
        """Determine if the instance is in the local timezone `<'bool'>`.

        #### Timezone-naive instance always returns `False`.
        """
        return self.tzinfo is utils.tz_local(None)

    def is_utc(self) -> bool:
        """Determine if the instance is in the UTC timezone `<'bool'>`.

        #### Timezone-naive instance always returns `False`.
        """
        return self.tzinfo is utils.UTC

    def tzname(self) -> str:
        """Return the timezone name of the instance `<'str/None'>`.

        #### Timezone-naive instance always returns `None`.
        """
        my_tz = self.tzinfo
        return None if my_tz is None else my_tz.tzname(self[0])

    def astimezone(
        self,
        tz: datetime.tzinfo | str | None = None,
        ambiguous: object = "raise",
        nonexistent: object = "raise",
    ) -> Self:
        """Convert the instance to the target timezone
        (retaining the same point in UTC time). `<'Pddt'>`.

        #### Similar to `datetime.datetime.astimezone()`.

        - If the instance is timezone-aware, converts to the target timezone directly.
        - If the instance is timezone-naive, first localizes to the local timezone,
          then converts to the targer timezone.

        :param tz `<'str/tzinfo/None'>`: The target timezone to convert to, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Convert to local timezone.

        :param ambiguous `<'str/ndarray'>`: How to handle ambiguous times, defaults to `'raise'`.
            - `<'str'>` Accepts 'infer' or 'raise' for ambiguous times handling.
            - `<'ndarray'>` A boolean array that specifies ambiguous times ('True' for DST time).

        :param nonexistent `<'str/timedelta'>`: How to handle nonexistent times, defaults to `'raise'`.
            - `<'str'>` Accepts 'shift_forward', 'shift_backward' or 'raise' for nonexistent times handling.
            - `<'timedelta'>` An instance of timedelta to shift the nonexistent times.
        """
        # Adjust target timezone
        tz = utils.tz_parse(tz)
        my_tz = self.tzinfo
        if tz is None:
            tz = utils.tz_local(None)
            if my_tz is None:
                # Since instance is timezone-naive, we
                # simply localize to local timezone.
                return self.tz_localize(tz, ambiguous, nonexistent)

        # Adjust my timezone
        if my_tz is None:
            my_tz = utils.tz_local(None)
            pt = self.tz_localize(my_tz, ambiguous, nonexistent)
        else:
            my_tz = utils.tz_parse(my_tz)
            pt = self

        # Same timezone
        if my_tz is tz:
            return pt  # exit: same timezone

        # Convert timezone
        return pt.tz_convert(tz)

    def tz_localize(
        self,
        tz: datetime.tzinfo | str | None,
        ambiguous: object = "raise",
        nonexistent: object = "raise",
    ) -> Self:
        """Localize timezone-naive instance to the specific timezone;
        or timezone-aware instance to timezone naive (without moving
        the time fields) `<'Pddt'>`.

        :param tz `<'tzinfo/str/None'>`: The timezone to localize to, defaults to `None`.
            - `<'datetime.tzinfo'>` Subclass of datetime.tzinfo.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Localize to timezone-naive.

        :param ambiguous `<'str/ndarray'>`: How to handle ambiguous times, defaults to `'raise'`.
            - `<'str'>` Accepts 'infer' or 'raise' for ambiguous times handling.
            - `<'ndarray'>` A boolean array that specifies ambiguous times ('True' for DST time).

        :param nonexistent `<'str/timedelta'>`: How to handle nonexistent times, defaults to `'raise'`.
            - `<'str'>` Accepts 'shift_forward', 'shift_backward' or 'raise' for nonexistent times handling.
            - `<'timedelta'>` An instance of timedelta to shift the nonexistent times.
        """
        # Timezone-aware
        tz = utils.tz_parse(tz)
        my_tz = self.tzinfo
        if my_tz is not None:
            if tz is not None:
                raise errors.InvalidTimezoneError(
                    "instance is already timezone-aware.\n"
                    "Use 'tz_convert()' or 'tz_switch()' method "
                    "to move to another timezone."
                )
            # . localize: aware => naive
            return super(Pddt, self).tz_localize(None)

        # Timezone-naive
        if tz is None:
            return self  # exit: same timezone
        else:
            if ambiguous == "NaT":
                raise errors.InvalidArgumentError("ambiguous='NaT' is not supported.")
            if nonexistent == "NaT":
                raise errors.InvalidArgumentError("nonexistent='NaT' is not supported.")
        # . localize: naive => aware
        #: To prevent 'ns' overflow issue, we manually check if
        #: the values are in safe 'ns' range before localization.
        if self.unit == "ns" and not utils.is_dt64arr_ns_safe(self.values, "ns"):
            pt = self.as_unit("us")
        else:
            pt = self
        try:
            return super(Pddt, pt).tz_localize(tz, ambiguous, nonexistent)
        except pd_err.OutOfBoundsDatetime as err:
            raise errors.OutOfBoundsDatetimeError(err) from err
        except Exception as err:
            raise errors.InvalidArgumentError(err) from err

    def tz_convert(self, tz: datetime.tzinfo | str | None) -> Self:
        """Convert timezone-aware instance to another timezone `<'Pddt'>`.

        :param tz `<'tzinfo/str/None'>`: The timezone to localize to, defaults to `None`.
            - `<'datetime.tzinfo'>` Subclass of datetime.tzinfo.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Convert to UTC timezone and localize to timezone-naive.
        """
        # Validate
        my_tz = self.tzinfo
        if my_tz is None:
            raise errors.InvalidTimezoneError(
                "instance is timezone-naive.\n"
                "Use 'tz_localize()' method instead to localize to a timezone.\n"
                "Use 'tz_switch()' method to convert to the timezone by "
                "providing a base timezone for the instance."
            )

        # Same timezone
        my_tz = utils.tz_parse(my_tz)
        tz = utils.tz_parse(tz)
        if my_tz is tz:
            return self

        # Convert: aware => aware
        #: To prevent 'ns' overflow issue, we manually check if
        #: the values are in safe 'ns' range before conversion.
        if self.unit == "ns" and not utils.is_dt64arr_ns_safe(self.values_naive, "ns"):
            pt = self.as_unit("us")  # cast to 'us' to prevent overflow
        else:
            pt = self
        try:
            return super(Pddt, pt).tz_convert(tz)
        except pd_err.OutOfBoundsDatetime as err:
            raise errors.OutOfBoundsDatetimeError(err) from err
        except Exception as err:
            raise errors.InvalidArgumentError(err) from err

    def tz_switch(
        self,
        targ_tz: datetime.tzinfo | str | None,
        base_tz: datetime.tzinfo | str | None = None,
        naive: cython.bint = False,
        ambiguous: object = "raise",
        nonexistent: object = "raise",
    ) -> Self:
        """Switch (convert) the instance from base timezone to the target timezone `<'Pddt'>`.

        This method extends the functionality of `astimezone()` by allowing
        user to specify a base timezone for timezone-naive instances before
        converting to the target timezone.

        - If the instance is timezone-aware, the 'base_tz' argument is `ignored`,
          and this method behaves identically to `astimezone()`, converting the
          instance to the target timezone.
        - If the instance is timezone-naive, it first localizes the instance
          to the `base_tz`, and then converts to the target timezone.

        :param targ_tz `<'tzinfo/str/None'>`: The target timezone to convert to.
        :param base_tz `<'tzinfo/str/None'>`: The base timezone for timezone-naive instance, defaults to `None`.
        :param naive `<'bool'>`: If 'True', returns timezone-naive instance after conversion, defaults to `False`.
        :param ambiguous `<'str/ndarray'>`: How to handle ambiguous times, defaults to `'raise'`.
            - `<'str'>` Accepts 'infer' or 'raise' for ambiguous times handling.
            - `<'ndarray'>` A boolean array that specifies ambiguous times ('True' for DST time).

        :param nonexistent `<'str/timedelta'>`: How to handle nonexistent times, defaults to `'raise'`.
            - `<'str'>` Accepts 'shift_forward', 'shift_backward' or 'raise' for nonexistent times handling.
            - `<'timedelta'>` An instance of timedelta to shift the nonexistent times.
        """
        # Timezone-aware
        targ_tz = utils.tz_parse(targ_tz)
        my_tz = self.tzinfo
        if my_tz is not None:
            # . target timezone is None
            if targ_tz is None:
                return self.tz_localize(None)
            # . target timezone is mytz
            elif targ_tz is utils.tz_parse(my_tz):
                return self.tz_localize(None) if naive else self
            # . mytz => target timezone
            else:
                pt = self.tz_convert(targ_tz)
                return pt.tz_localize(None) if naive else pt

        # Timezone-naive
        # . target timezone is None
        if targ_tz is None:
            return self  # exit
        # . base timezone is target timezone
        base_tz = utils.tz_parse(base_tz)
        if base_tz is None:
            raise errors.InvalidTimezoneError(
                "instance is timezone-naive.\n"
                "Cannot switch timezone-naive instance to the "
                "target timezone without providing a 'base_tz'."
            )
        if base_tz is targ_tz:
            return self if naive else self.tz_localize(targ_tz, ambiguous, nonexistent)
        # . base timezone => target timezone
        else:
            pt = self.tz_localize(base_tz, ambiguous, nonexistent)
            pt = pt.tz_convert(targ_tz)
            return pt.tz_localize(None) if naive else pt

    # Values -------------------------------------------------------------------------------
    @property
    def values_naive(self) -> np.ndarray[np.datetime64]:
        """Return an array of the timezone-naive datetime values `<'ndarray[datetime64]'>`.

        - If the instance is timezone-aware, equivalent to 'pt.tz_localize(None).values'.
        - If the instance is timezone-naive, equivalent to 'pt.values'.
        """
        my_tz = self.tzinfo
        if my_tz is None or my_tz is utils.UTC:
            return self.values
        return super(Pddt, self).tz_localize(None).values

    def as_unit(self, unit: str) -> Self:
        """Set the datetime unit resolution of the instance `<'Pddt'>`.

        #### Supports timezone-aware instance.

        :param unit `<'str/None'>`: Set the datetime unit.
            Supported datetime units: 's', 'ms', 'us', 'ns'.
        """
        # No change / Same unit
        if unit is None:
            return self  # exit
        my_unit: str = self.unit
        if unit == my_unit:
            return self  # exit

        # Access datetime array & info
        arr: np.ndarray = self.values_naive

        # Perform operation
        try:
            res: np.ndarray = utils.dt64arr_as_unit(arr, unit, my_unit, True)
        except Exception as err:
            raise errors.InvalidTimeUnitError(err) from err
        if arr is res:
            return self
        return pddt_new_simple(res, self.tzinfo, None, self.name)

    # Arithmetic ---------------------------------------------------------------------------
    def add(
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
        nanoseconds: cython.int = 0,
    ) -> Self:
        """Add relative delta to the instance `<'Pddt'>`.

        :param years `<'int'>`: Relative delta of years, defaults to `0`.
        :param quarters `<'int'>`: Relative delta of quarters (3 months), defaults to `0`.
        :param months `<'int'>`: Relative delta of months, defaults to `0`.
        :param weeks `<'int'>`: Relative delta of weeks (7 days), defaults to `0`.
        :param days `<'int'>`: Relative delta of days, defaults to `0`.
        :param hours `<'int'>`: Relative delta of hours, defaults to `0`.
        :param minutes `<'int'>`: Relative delta of minutes, defaults to `0`.
        :param seconds `<'int'>`: Relative delta of seconds, defaults to `0`.
        :param milliseconds `<'int'>`: Relative delta of milliseconds, defaults to `0`.
        :param microseconds `<'int'>`: Relative delta of microseconds, defaults to `0`.
        :param nanoseconds `<'int'>`: Relative delta of nanoseconds, defaults to `0`.
        """
        # No change
        if (
            years == 0
            and quarters == 0
            and months == 0
            and weeks == 0
            and days == 0
            and hours == 0
            and minutes == 0
            and seconds == 0
            and milliseconds == 0
            and microseconds == 0
            and nanoseconds == 0
        ):
            return self  # exit

        # Access datetime array & info
        arr: np.ndarray = self.values_naive
        my_unit: str = self.unit
        my_unit_ch: cython.Py_UCS4 = str_read(my_unit, 0)
        delta: np.ndarray

        # Adjust dates
        if years != 0 or quarters != 0 or months != 0:
            datesM = utils.dt64arr_as_int64_M(arr, my_unit)  # int64[M]
            datesM = utils.arr_add(datesM, years * 12 + quarters * 3 + months)
            datesD = utils.dt64arr_as_int64_D(datesM, "M")  # int64[D]
            delta = utils.dt64arr_day(arr, my_unit)  # original day
            datesD = utils.arr_add_arr(datesD, delta, weeks * 7 + days - 1)  # int64[D]
        else:
            datesD = utils.dt64arr_as_int64_D(arr, my_unit)  # int64[D]
            datesD = utils.arr_add(datesD, weeks * 7 + days)  # int64[D]

        # Adjust times
        times: np.ndarray = utils.dt64arr_times(arr, my_unit)  # int64[my_unit]
        # . nanosecond
        if my_unit_ch == "n":
            if (
                hours != 0
                or minutes != 0
                or seconds != 0
                or milliseconds != 0
                or microseconds != 0
                or nanoseconds != 0
            ):
                times = utils.arr_add(  # int64[ns]
                    times,
                    (hours * 3600 + minutes * 60 + seconds) * utils.NS_SECOND
                    + milliseconds * utils.NS_MILLISECOND
                    + microseconds * utils.NS_MICROSECOND
                    + nanoseconds,
                )
                delta = utils.arr_floor_div(times, utils.NS_DAY)  # int64[D]
                delta = utils.arr_add_arr(datesD, delta)  # int64[D]
            else:
                delta = datesD  # int64[D]
            # Prevent 'ns' overflow: cast to 'us'
            if not utils.is_dt64arr_ns_safe(delta, "D"):
                times = utils.arr_floor_div(times, utils.NS_MICROSECOND)  # int64[us]
                my_unit = "us"  # change to 'us' from 'ns'
        # . microsecond
        elif my_unit_ch == "u":
            times = utils.arr_add(  # int64[us]
                times,
                (hours * 3600 + minutes * 60 + seconds) * utils.US_SECOND
                + milliseconds * utils.US_MILLISECOND
                + microseconds
                + utils.math_round_div(nanoseconds, utils.NS_MICROSECOND),
            )
        # . millisecond
        elif my_unit_ch == "m":
            times = utils.arr_add(  # int64[ms]
                times,
                (hours * 3600 + minutes * 60 + seconds) * utils.MS_SECOND
                + milliseconds
                + utils.math_round_div(microseconds, utils.US_MILLISECOND)
                + utils.math_round_div(nanoseconds, utils.NS_MILLISECOND),
            )
        # . second
        elif my_unit_ch == "s":
            times = utils.arr_add(  # int64[s]
                times,
                (hours * 3600 + minutes * 60 + seconds)
                + utils.math_round_div(milliseconds, utils.MS_SECOND)
                + utils.math_round_div(microseconds, utils.US_SECOND)
                + utils.math_round_div(nanoseconds, utils.NS_SECOND),
            )

        # Combine dates & times
        dtype = "datetime64[%s]" % my_unit
        dates = utils.dt64arr_as_int64(datesD, my_unit, "D")  # int64[my_unit]
        arr = utils.arr_add_arr(dates, times).astype(dtype)  # dt64[my_unit]
        return pddt_new_simple(arr, self.tzinfo, None, self.name)

    def sub(
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
        nanoseconds: cython.int = 0,
    ) -> Self:
        """Substract relative delta from the instance `<'Pddt'>`.

        :param years `<'int'>`: Relative delta of years, defaults to `0`.
        :param quarters `<'int'>`: Relative delta of quarters (3 months), defaults to `0`.
        :param months `<'int'>`: Relative delta of months, defaults to `0`.
        :param weeks `<'int'>`: Relative delta of weeks (7 days), defaults to `0`.
        :param days `<'int'>`: Relative delta of days, defaults to `0`.
        :param hours `<'int'>`: Relative delta of hours, defaults to `0`.
        :param minutes `<'int'>`: Relative delta of minutes, defaults to `0`.
        :param seconds `<'int'>`: Relative delta of seconds, defaults to `0`.
        :param milliseconds `<'int'>`: Relative delta of milliseconds, defaults to `0`.
        :param microseconds `<'int'>`: Relative delta of microseconds, defaults to `0`.
        :param nanoseconds `<'int'>`: Relative delta of nanoseconds, defaults to `0`.
        """
        # fmt: off
        return self.add(
            -years, -quarters, -months, -weeks, -days,
            -hours, -minutes, -seconds, -milliseconds,
            -microseconds, -nanoseconds
        )
        # fmt: on

    def diff(
        self,
        data: object,
        unit: str,
        absolute: cython.bint = False,
        inclusive: str = "both",
    ) -> np.ndarray[np.int64]:
        """Calculate the difference between the instance and
        another datetime-like data `<'np.ndarray[int64]'>`.

        The differences are computed in the specified datetime 'unit'
        and adjusted based on the 'inclusive' argument to determine
        the inclusivity of the start and end times.

        :param data `<'object'>`: Datetime-like data.
            - `<'Array-Like'>` An array-like object containing datetime information.
            - `<'str'>` A datetime string containing datetime information.
            - `<'datetime.datetime'>` An instance of `datetime.datetime`.
            - `<'datetime.date'>` An instance of `datetime.date` (time fields set to 0).
            - `<'np.datetime64'>` An instance of `np.datetime64`.
            - `<'None'>` Current datetime with the same timezone of the instance.

        :param unit `<'str'>`: The datetime unit for calculating the difference.
            Supports: 'Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'

        :param absolute `<'bool'>`: If 'True', compute the absolute difference, defaults to `False`.

        :param inclusive `<'str'>`: Specifies the inclusivity of the start and end times, defaults to `'both'`.
            - `'one'`: Include either the start or end time.
            - `'both'`: Include both the start and end times.
            - `'neither'`: Exclude both the start and end times.
        """
        # Access 'my' datetime array & info
        my_arr: np.ndarray = self.values
        my_unit: str = self.unit
        my_tz = self.tzinfo
        arr_size: cython.Py_ssize_t = my_arr.shape[0]

        # Parse dtobjs to Pddt
        if data is None:
            pt = self.now(my_tz, arr_size)
        elif isinstance(data, (str, datetime.date, np.datetime64)):
            pt = Pddt([data])
        elif isinstance(data, Pddt):
            pt = data
        else:
            pt = Pddt(data)

        # Access 'pt' datetime array & info
        pt_arr: np.ndarray = pt.values
        pt_unit: str = pt.unit
        pt_tz = pt.tzinfo

        # Validate two arrays
        # . shape
        if arr_size != pt_arr.shape[0]:
            raise errors.IncomparableError(
                "cannot compare between arrays with different shapes ['%d' vs '%d']."
                % (arr_size, pt_arr.shape[0])
            )
        # . tzinfo
        if my_tz is not pt_tz and (my_tz is None or pt_tz is None):
            _raise_incomparable_error(self, pt, "calcuate average")
        # . unit
        my_unit_int = utils.map_nptime_unit_str2int(my_unit)
        pt_unit_int = utils.map_nptime_unit_str2int(pt_unit)
        if my_unit_int != pt_unit_int:
            # Cast to lower resolution
            if my_unit_int < pt_unit_int:
                pt_arr = utils.dt64arr_as_int64(pt_arr, my_unit, pt_unit)  # int64
            else:
                my_arr = utils.dt64arr_as_int64(my_arr, pt_unit, my_unit)  # int64
                my_unit = pt_unit  # change my_unit to pt_unit

        # Handle inclusive
        if inclusive == "both":
            incl_off: cython.int = 1
        elif inclusive == "one":
            incl_off: cython.int = 0
        elif inclusive == "neither":
            incl_off: cython.int = -1
        else:
            raise errors.InvalidArgumentError(
                "invalid input '%s' for inclusive.\n"
                "Supports: ['one', 'both', 'neither']." % inclusive
            )

        # Calculate difference
        if unit == "W":
            my_arr = utils.dt64arr_as_iso_W(my_arr, 1, my_unit)  # int64[W]
            pt_arr = utils.dt64arr_as_iso_W(pt_arr, 1, my_unit)  # int64[W]
        else:
            my_arr = utils.dt64arr_as_int64(my_arr, unit, my_unit)  # int64[unit]
            pt_arr = utils.dt64arr_as_int64(pt_arr, unit, my_unit)  # int64[unit]
        arr = utils.arr_sub_arr(my_arr, pt_arr)
        # . absolute = True
        if absolute:
            arr = utils.arr_abs(arr, incl_off)
        # . absolute = False | adj offset
        elif incl_off != 0:
            i: np.npy_int64
            arr_ptr = cython.cast(cython.pointer(np.npy_int64), np.PyArray_DATA(arr))
            for i in range(arr_size):
                val: np.npy_int64 = arr_ptr[i]
                arr_ptr[i] = val - incl_off if val < 0 else val + incl_off
        return Index(arr, name="diff")

    # Comparison ---------------------------------------------------------------------------
    def is_past(self) -> Index[bool]:
        """Determine if the datetimes are in the past `<'Index[bool]'>`."""
        # Access datetime array & info
        arr: np.ndarray = self.values
        my_tz = self.tzinfo
        my_unit: str = self.unit
        my_unit_ch: cython.Py_UCS4 = str_read(my_unit, 0)

        # Generate current datetime
        dt = utils.dt_now(my_tz)
        dt_val: cython.longlong = utils.dt_to_us(dt, True)
        if my_unit_ch == "n":
            dt_val *= utils.NS_MICROSECOND
        elif my_unit_ch == "m":
            dt_val = utils.math_floor_div(dt_val, utils.US_MILLISECOND)
        elif my_unit_ch == "s":
            dt_val = utils.math_floor_div(dt_val, utils.US_SECOND)

        # Compare
        arr = utils.arr_less_than(arr, dt_val)
        return Index(arr, name="is_past")

    def is_future(self) -> Index[bool]:
        """Determine if the datetimes are in the future `<'Index[bool]'>`."""
        # Access datetime array & info
        arr: np.ndarray = self.values
        my_tz = self.tzinfo
        my_unit: str = self.unit
        my_unit_ch: cython.Py_UCS4 = str_read(my_unit, 0)

        # Generate current datetime
        dt = utils.dt_now(my_tz)
        dt_val: cython.longlong = utils.dt_to_us(dt, True)
        if my_unit_ch == "n":
            dt_val *= utils.NS_MICROSECOND
        elif my_unit_ch == "m":
            dt_val = utils.math_floor_div(dt_val, utils.US_MILLISECOND)
        elif my_unit_ch == "s":
            dt_val = utils.math_floor_div(dt_val, utils.US_SECOND)

        # Compare
        arr = utils.arr_greater_than(arr, dt_val)
        return Index(arr, name="is_future")

    def closest(self, data: object) -> Self:
        """Find the closest datetime to each of the instance values `<'Pddt'>`.

        This method compares each elements of the instance with the
        provided datetime-like data and returns the closest datetime
        (from 'data') to each of the instance values.

        :param data `<'object'>`: Datetime-like data.
            - `<'Array-Like'>` An array-like object containing datetime information.
            - `<'str'>` A datetime string containing datetime information.
            - `<'datetime.datetime'>` An instance of `datetime.datetime`.
            - `<'datetime.date'>` An instance of `datetime.date` (time fields set to 0).
            - `<'np.datetime64'>` An instance of `np.datetime64`.
            - `<'None'>` Current datetime with the same timezone of the instance.
        """
        # Access 'my' datetime array & info
        my_arr: np.ndarray = self.values_naive
        my_unit: str = self.unit
        my_tz = self.tzinfo
        arr_size: cython.Py_ssize_t = my_arr.shape[0]

        # Parse dtobjs to Pddt
        if data is None:
            pt = self.now(my_tz, arr_size)
        elif isinstance(data, (str, datetime.date, np.datetime64)):
            pt = Pddt([data])
        elif isinstance(data, Pddt):
            pt = data
        else:
            pt = Pddt(data)

        # Access 'pt' datetime array & info
        pt_arr: np.ndarray = pt.values_naive
        pt_unit: str = pt.unit
        pt_tz = pt.tzinfo

        # Validate arrays
        # . tzinfo
        if my_tz is not pt_tz and (my_tz is None or pt_tz is None):
            _raise_incomparable_error(self, pt, "find closest")
        # . unit
        my_unit_int = utils.map_nptime_unit_str2int(my_unit)
        pt_unit_int = utils.map_nptime_unit_str2int(pt_unit)
        if my_unit_int != pt_unit_int:
            # Cast to lower resolution
            if my_unit_int < pt_unit_int:
                pt_arr = utils.dt64arr_as_int64(pt_arr, my_unit, pt_unit)  # int64
            else:
                my_arr = utils.dt64arr_as_int64(my_arr, pt_unit, my_unit)  # int64
                my_unit = pt_unit  # change my_unit to pt_unit

        # Find closest
        arr = utils.dt64arr_find_closest(my_arr, pt_arr)
        arr = arr.astype("datetime64[%s]" % my_unit)
        return pddt_new_simple(arr, pt_tz, None, "closest")

    def farthest(self, dtobjs: object) -> Self:
        """Find the farthest datetime to each of the instance values `<'Pddt'>`.

        This method compares each elements of the instance with the
        provided datetime-like data and returns the farthest datetime
        (from 'data') to each of the instance values.

        :param data `<'object'>`: Datetime-like data.
            - `<'Array-Like'>` An array-like object containing datetime information.
            - `<'str'>` A datetime string containing datetime information.
            - `<'datetime.datetime'>` An instance of `datetime.datetime`.
            - `<'datetime.date'>` An instance of `datetime.date` (time fields set to 0).
            - `<'np.datetime64'>` An instance of `np.datetime64`.
            - `<'None'>` Current datetime with the same timezone of the instance.
        """
        # Access 'my' datetime array & info
        my_arr: np.ndarray = self.values_naive
        my_unit: str = self.unit
        my_tz = self.tzinfo
        arr_size: cython.Py_ssize_t = my_arr.shape[0]

        # Parse dtobjs to Pddt
        if dtobjs is None:
            pt = self.now(my_tz, arr_size)
        elif isinstance(dtobjs, (str, datetime.date, np.datetime64)):
            pt = Pddt([dtobjs])
        elif isinstance(dtobjs, Pddt):
            pt = dtobjs
        else:
            pt = Pddt(dtobjs)

        # Access 'pt' datetime array & info
        pt_arr: np.ndarray = pt.values_naive
        pt_unit: str = pt.unit
        pt_tz = pt.tzinfo

        # Validate arrays
        # . tzinfo
        if my_tz is not pt_tz and (my_tz is None or pt_tz is None):
            _raise_incomparable_error(self, pt, "find farthest")
        # . unit
        my_unit_int = utils.map_nptime_unit_str2int(my_unit)
        pt_unit_int = utils.map_nptime_unit_str2int(pt_unit)
        if my_unit_int != pt_unit_int:
            # Cast to lower resolution
            if my_unit_int < pt_unit_int:
                pt_arr = utils.dt64arr_as_int64(pt_arr, my_unit, pt_unit)  # int64
            else:
                my_arr = utils.dt64arr_as_int64(my_arr, pt_unit, my_unit)  # int64
                my_unit = pt_unit  # change my_unit to pt_unit

        # Find farthest
        arr = utils.dt64arr_find_farthest(my_arr, pt_arr)
        arr = arr.astype("datetime64[%s]" % my_unit)
        return pddt_new_simple(arr, pt_tz, None, "farthest")
