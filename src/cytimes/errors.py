from pandas.errors import OutOfBoundsDatetime


# Base Exceptions ---------------------------------------------------------------------------------
class cyTimesError(Exception):
    """The base error for the cyTimes package."""


class cyTimesValueError(cyTimesError, ValueError):
    """The base ValueError for the cyTimes package."""


# CyParser Exceptions -----------------------------------------------------------------------------
class cyParserError(cyTimesError):
    """The base error for the cyParser module."""


class cyParserFailedError(cyParserError):
    """Error for failed parsing"""


class cyParserValueError(cyParserError, cyTimesValueError):
    """The base ValueError for cyParser module."""


class InvalidDatetimeStrError(cyParserValueError):
    """Error for invalid 'timestr' to parse."""


class InvalidTokenError(cyParserValueError):
    """Error for invalid token"""


class InvalidNumericToken(InvalidTokenError):
    """Error for token that cannot be converted to numeric value."""


class InvalidMonthToken(InvalidTokenError):
    """Error for token that cannot be converted to month value."""


class InvalidParserInfo(cyParserValueError):
    """Error for Configs importing invalid 'dateutil.parser.parserinfo'."""


class InvalidConfigKeyword(cyParserValueError):
    """Error for the 'cyparser.Configs' when conflicting
    (duplicated) keyword exsit in the settings"""


class InvalidConfigValue(cyParserValueError):
    """Error for the 'cyparser.Configs' when the value
    for a keyword is invalid"""


# Pydt/Pddt Exceptions ----------------------------------------------------------------------------
class DatetimeError(cyTimesError):
    """The base error for the datetime module."""


class PydtError(DatetimeError):
    """The base error for the pydt module."""


class PddtError(DatetimeError):
    """The base error for the pddt module."""


class PydtValueError(PydtError, cyTimesValueError):
    """The base ValueError for pydt module."""


class PddtValueError(PddtError, cyTimesValueError):
    """The base ValueError for pddt module."""


class InvalidDatetimeObjectError(PydtValueError, PddtValueError):
    """Error for invalid 'dtobj' to create a pydt object."""


class DatetimesOutOfBounds(InvalidDatetimeObjectError, OutOfBoundsDatetime):
    """Error for 'dtsobj' that has datetimes out of bounds."""


class InvalidMonthError(PydtValueError, PddtValueError):
    """Error for invalid month value."""


class InvalidWeekdayError(PydtValueError, PddtValueError):
    """Error for invalid weekday value."""


class InvalidTimezoneError(PydtValueError, PddtValueError):
    """Error for invalid timezone value."""


class InvalidFrequencyError(PydtValueError, PddtValueError):
    """Error for invalid frequency value."""


class InvalidDeltaUnitError(PydtValueError, PddtValueError):
    """Error for invalid delta unit value."""
