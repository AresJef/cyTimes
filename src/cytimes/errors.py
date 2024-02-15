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


class InvalidTimestrError(cyParserValueError):
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
