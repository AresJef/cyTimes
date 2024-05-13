from pandas.errors import OutOfBoundsDatetime


# Base Exceptions ---------------------------------------------------------------------------------
class cyTimesError(Exception):
    """The base error for the cyTimes package."""


class cyTimesTypeError(cyTimesError, TypeError):
    """The base TypeError for the cyTimes package."""


class cyTimesValueError(cyTimesError, ValueError):
    """The base ValueError for the cyTimes package."""


# CyParser Exceptions -----------------------------------------------------------------------------
class cyParserError(cyTimesError):
    """The base error for the cyParser module."""


class cyParserValueError(cyParserError, cyTimesValueError):
    """The base ValueError for cyParser module."""


class cyParserFailedError(cyParserValueError):
    """Error for failed parsing"""


class cyParserBuildError(cyParserFailedError):
    """Error for failed parsing"""


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


class InvalidConfigWord(cyParserValueError):
    """Error for the 'cyparser.Configs' when conflicting
    (duplicated) word exsit in the settings"""


class InvalidConfigValue(cyParserValueError):
    """Error for the 'cyparser.Configs' when the value
    for a word is invalid"""


# Pydt/Pddt Exceptions ----------------------------------------------------------------------------
class DatetimeError(cyTimesError):
    """The base error for the datetime module."""


class PydtError(DatetimeError):
    """The base error for the pydt module."""


class PydtTypeError(PydtError, cyTimesTypeError):
    """The base TypeError for pydt module."""


class PydtValueError(PydtError, cyTimesValueError):
    """The base ValueError for pydt module."""


class PddtError(DatetimeError):
    """The base error for the pddt module."""


class PddtTypeError(PddtError, cyTimesTypeError):
    """The base TypeError for pddt module."""


class PddtValueError(PddtError, cyTimesValueError):
    """The base ValueError for pddt module."""


class InvalidDatetimeObjectError(PydtValueError, PddtValueError):
    """Error for invalid 'dtobj' to create a pydt object."""


class DatetimeOutOfBoundsError(InvalidDatetimeObjectError, OutOfBoundsDatetime):
    """Error for 'dtsobj' that has datetimes out of bounds."""


class InvalidTimeUnitError(DatetimeOutOfBoundsError):
    """Error for invalid time unit value."""


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


class InvalidDeltaInclusiveError(PydtValueError, PddtValueError):
    """Error for invalid delta inclusive value."""


class InvalidIsoCalendarError(PydtValueError, PddtValueError):
    """Error for invalid ISO calendar value."""


class InvalidTimzoneNameError(PydtTypeError, PddtTypeError):
    """Error for invalid timezone name."""
