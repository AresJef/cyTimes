from pandas import errors as pd_err
from pytz import exceptions as pytz_err


# Base Exceptions ---------------------------------------------------------------------------------
class cyTimesError(Exception):
    """The base error for the cyTimes package."""


class cyTimesTypeError(cyTimesError, TypeError):
    """The base TypeError for the cyTimes package."""


class cyTimesValueError(cyTimesError, ValueError):
    """The base ValueError for the cyTimes package."""


# Parser Exceptions -------------------------------------------------------------------------------
class ParserError(cyTimesError):
    """The base error for the Parser module."""


class ParserFailedError(ParserError, cyTimesValueError):
    """Error for failed parsing"""


class ParserBuildError(ParserFailedError):
    """Error for failed building from result"""


# . configs
class ParserConfigsError(ParserError):
    """Error for the 'parser.Configs' when the settings are invalid."""


class InvalidConfigsValue(ParserConfigsError, cyTimesValueError):
    """Error for the 'parser.Configs' when the value is invalid."""


class InvalidParserInfo(InvalidConfigsValue, cyTimesTypeError):
    """Error for Configs importing invalid 'dateutil.parser.parserinfo'."""


# Pydt/Pddt Exceptions ----------------------------------------------------------------------------
class DatetimeError(cyTimesError):
    """The base error for the datetime module."""


class PydtError(DatetimeError):
    """The base error for the pydt module."""


class PddtError(DatetimeError):
    """The base error for the pddt module."""


class InvalidArgumentError(PydtError, PddtError, cyTimesValueError):
    """Error for invalid arguments."""


class InvalidTypeError(InvalidArgumentError, cyTimesTypeError):
    """Error for invalid type."""


class InvalidTimezoneError(InvalidArgumentError):
    """Error for invalid timezone value."""


class AmbiguousTimeError(InvalidArgumentError, pytz_err.AmbiguousTimeError):
    """Error for ambiguous time."""


class IncomparableError(InvalidTimezoneError, InvalidTypeError):
    """Error for incomparable datetime objects."""


class InvalidTimeUnitError(InvalidArgumentError):
    """Error for invalid time unit value."""


class InvalidMonthError(InvalidArgumentError):
    """Error for invalid month value."""


class InvalidWeekdayError(InvalidArgumentError):
    """Error for invalid weekday value."""


class OutOfBoundsDatetimeError(InvalidArgumentError, pd_err.OutOfBoundsDatetime):
    """Error for 'dtsobj' that has datetimes out of bounds."""
