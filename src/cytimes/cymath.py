# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

import cython
from cython.cimports.libc.stdlib import abs as _abs, labs as _labs, llabs as _llabs  # type: ignore
from cython.cimports.libc.math import fabs as _fabas, fabsl as _fbasl  # type: ignore
from cython.cimports.libc.math import ceil as _ceil, ceill as _ceill  # type: ignore
from cython.cimports.libc.math import floor as _floor, floorl as _floorl  # type: ignore
from cython.cimports.libc.math import round as _round, roundl as _roundl  # type: ignore
from cython.cimports.libc.math import fmax as _fmax, fmaxl as _fmaxl  # type: ignore
from cython.cimports.libc.math import fmin as _fmin, fminl as _fminl  # type: ignore
from cython.cimports.libc.math import copysign as _copysign, copysignl as _copysignl  # type: ignore
from cython.cimports.libc.math import isinf as _isinf, isfinite as _isfinite  # type: ignore
from cython.cimports.libc.math import isnan as _isnan, isnormal as _isnormal  # type: ignore


# Absolute ------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def abs(num: cython.int) -> cython.int:
    """Absolute value of an integer.

    :param num `<int>`: An integer.
    :return `<int>`: The absolute value of the integer.
    """
    return _abs(num)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def abs_l(num: cython.long) -> cython.long:
    """Absolute value of a (long) integer.

    :param num `<long>`: A (long) integer.
    :return `<long>`: The absolute value of the (long) integer.
    """
    return _labs(num)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def abs_ll(num: cython.longlong) -> cython.longlong:
    """Absolute value of a (long long) integer.

    :param num `<long long>`: A (long long) integer.
    :return `<long long>`: The absolute value of the (long long) integer.
    """
    return _llabs(num)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def abs_f(num: cython.double) -> cython.double:
    """Absolute value of a (float/double) number.

    :param num `<float/double>`: A (float/double) number.
    :return `<float/double>`: The absolute value of the (float/double) number.
    """
    return _fabas(num)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def abs_lf(num: cython.longdouble) -> cython.longdouble:
    """Absolute value of a (long double) number.

    :param num `<long double>`: A (long double) number.
    :return `<long double>`: The absolute value of the (long double) number.
    """
    return _fbasl(num)


# Ceil -------------------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def ceil(num: cython.double) -> cython.long:
    """Ceil value of a (float/double) number to the nearest integer.

    :param num `<float/double>`: A (float/double) number.
    :return `<long>`: The ceil value of the (float/double) number.
    """
    return int(_ceil(num))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def ceil_l(num: cython.longdouble) -> cython.longlong:
    """Ceil value of a (long double) number to the nearest integer.

    :param num `<long double>`: A (long double) number.
    :return `<long long>`: The ceil value of the (long double) number.
    """
    return int(_ceill(num))


# Floor ------------------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def floor(num: cython.double) -> cython.long:
    """Floor value of a (float/double) number to the nearest integer.

    :param num `<float/double>`: A (float/double) number.
    :return `<long>`: The floor value of the (float/double) number.
    """
    return int(_floor(num))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def floor_l(num: cython.longdouble) -> cython.longlong:
    """Floor value of a (long double) number to the nearest integer.

    :param num `<long double>`: A (long double) number.
    :return `<long long>`: The floor value of the (long double) number.
    """
    return int(_floorl(num))


# Round ------------------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def round(num: cython.double) -> cython.long:
    """Round value of a (float/double) number to the nearest integer.

    :param num `<float/double>`: A (float/double) number.
    :return `<long>`: The round value of the (float/double) number.
    """
    return int(_round(num))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def round_l(num: cython.longdouble) -> cython.longlong:
    """Round value of a (long double) number to the nearest integer.

    :param num `<long double>`: A (long double) number.
    :return `<long long>`: The round value of the (long double) number.
    """
    return int(_roundl(num))


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
@cython.exceptval(check=False)
def round_half_away(
    num: cython.double,
    nth_digits: cython.int = 0,
) -> cython.double:
    """Round a (double) number half away from zero.

    :param num `<double>`: A (long double) number.
    :param nth_digits `<int>`: number of digits after the decimal point to round to. Defaults to `0`.
    :return `<double>`: The round value of the (long double) number.
    """
    factor: cython.longlong = int(10**nth_digits)
    return round_half_away_factor(num, factor)


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
@cython.exceptval(check=False)
def round_half_away_factor(
    num: cython.double,
    factor: cython.longlong = 10,
) -> cython.double:
    """Round a (double) number half away from zero by a factor.

    :param num `<double>`: A (long double) number.
    :param factor `<long long>`: Equivalent to `10**nth_digits`. Defaults to `10`.
        `nth_digit`: number of digits after the decimal point to round to.
    :return `<double>`: The round value of the (long double) number.
    """
    adj: cython.double = 0.5 if num >= 0 else -0.5
    base: cython.longlong = int(num * factor + adj)
    return base / factor


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
@cython.exceptval(check=False)
def round_half_away_l(
    num: cython.longdouble,
    nth_digits: cython.int = 0,
) -> cython.longdouble:
    """Round a (long double) number half away from zero.

    :param num `<long double>`: A (long double) number.
    :param nth_digits `<int>`: number of digits after the decimal point to round to. Defaults to `0`.
    :return `<long double>`: The round value of the (long double) number.
    """
    factor: cython.longlong = int(10**nth_digits)
    return round_half_away_factor_l(num, factor)


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
@cython.exceptval(check=False)
def round_half_away_factor_l(
    num: cython.longdouble,
    factor: cython.longlong = 10,
) -> cython.longdouble:
    """Round a (long double) number half away from zero by a factor.

    :param num `<long double>`: A (long double) number.
    :param factor `<long long>`: Equivalent to `10**nth_digits`. Defaults to `10`.
        `nth_digit`: number of digits after the decimal point to round to.
    :return `<long double>`: The round value of the (long double) number.
    """
    adj: cython.longdouble = 0.5 if num >= 0 else -0.5
    base: cython.longlong = int(num * factor + adj)
    return base / factor


# Maximum -------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def max_f(num1: cython.double, num2: cython.double) -> cython.double:
    """Get the maximum value between two (float/double) numbers.

    :param num1 `<float/double>`: (float/double) number.
    :param num2 `<float/double>`: (float/double) number.
    :return `<float/double>`: The maximum value between the two (float/double) numbers.
    """
    return _fmax(num1, num2)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def max_lf(num1: cython.longdouble, num2: cython.longdouble) -> cython.longdouble:
    """Get the maximum value between two (long double) numbers.

    :param num1 `<long double>`: (long double) number.
    :param num2 `<long double>`: (long double) number.
    :return `<long double>`: The maximum value between the two (long double) numbers.
    """
    return _fmaxl(num1, num2)


# Minimum -------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def min_f(num1: cython.double, num2: cython.double) -> cython.double:
    """Get the minimum value between two (float/double) numbers.

    :param num1 `<float/double>`: (float/double) number.
    :param num2 `<float/double>`: (float/double) number.
    :return `<float/double>`: The minimum value between the two (float/double) numbers.
    """
    return _fmin(num1, num2)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def min_lf(num1: cython.longdouble, num2: cython.longdouble) -> cython.longdouble:
    """Get the minimum value between two (long double) numbers.

    :param num1 `<long double>`: (long double) number.
    :param num2 `<long double>`: (long double) number.
    :return `<long double>`: The minimum value between the two (long double) numbers.
    """
    return _fminl(num1, num2)


# Clipping ------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def clip(
    num: cython.longlong,
    min_val: cython.longlong,
    max_val: cython.longlong,
) -> cython.longlong:
    """Clip the min & max value of an (long long) integer.

    :param num `<long long>`: An (long long) integer.
    :param min_val `<long long>`: The minimum value.
    :param max_val `<long long>`: The maximum value.
    :return `<long long>`: The clipped value.
    """
    return max(min(num, max_val), min_val)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def clip_f(
    num: cython.double,
    min_val: cython.double,
    max_val: cython.double,
) -> cython.double:
    """Clip the min & max value of a (float/double) number.

    :param num `<float/double>`: A (float/double) number.
    :param min_val `<float/double>`: The minimum value.
    :param max_val `<float/double>`: The maximum value.
    :return `<float/double>`: The clipped value.
    """
    return _fmax(_fmin(num, max_val), min_val)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def clip_lf(
    num: cython.longdouble,
    min_val: cython.longdouble,
    max_val: cython.longdouble,
) -> cython.longdouble:
    """Clip the min & max value of a (long double) number.

    :param num `<long double>`: A (long double) number.
    :param min_val `<long double>`: The minimum value.
    :param max_val `<long double>`: The maximum value.
    :return `<long double>`: The clipped value.
    """
    return _fmaxl(_fminl(num, max_val), min_val)


# Sign ----------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def copysign(num: cython.double, sign: cython.double) -> cython.double:
    """Copy the sign of a (float/double) number.

    :param num `<float/double>`: A (float/double) number.
    :param sign `<float/double>`: The sign to copy from.
    :return `<float/double>`: The number with the sign copied.
    """
    return _copysign(num, sign)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def copysign_l(num: cython.longdouble, sign: cython.longdouble) -> cython.longdouble:
    """Copy the sign of a (long double) number.

    :param num `<long double>`: A (long double) number.
    :param sign `<long double>`: The sign to copy from.
    :return `<long double>`: The number with the sign copied.
    """
    return _copysignl(num, sign)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def signfactor(num: cython.double) -> cython.int:
    """Get the sign factor (1 or -1) from a (float/double) number.

    :param num `<float/double>`: A (float/double) number.
    :return `<int>`: The sign factor (1 or -1) of the number.
    """
    return int(_copysign(1, num))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def signfactor_l(num: cython.longdouble) -> cython.int:
    """Get the sign factor (1 or -1) from a (long double) number.

    :param num `<long double>`: A (long double) number.
    :return `<int>`: The sign factor (1 or -1) of the number.
    """
    return int(_copysignl(1, num))


# Validation ----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_inf(num: cython.longdouble) -> cython.bint:
    """Chech if a number is infinite.

    :param `<long double>`: The number to check.
    :return `<bool>`: Whether the number is infinite.
    """
    return _isinf(num)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_finite(num: cython.longdouble) -> cython.bint:
    """Check if a number is finite.

    :param `<long double>`: The number to check.
    :return `<bool>`: Whether the number is finite.
    """
    return _isfinite(num)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_nan(num: cython.longdouble) -> cython.bint:
    """Check if a number is `nan`.

    :param `<long double>`: The number to check.
    :return `<bool>`: Whether the number is `nan`.
    """
    return _isnan(num)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_normal(num: cython.longdouble) -> cython.bint:
    """Check if a number is normal.

    :param `<long double>`: The number to check.
    :return `<bool>`: Whether the number is normal.
    """
    return _isnormal(num)
