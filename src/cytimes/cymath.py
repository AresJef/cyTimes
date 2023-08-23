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
    "Absolute value of an (int) integer."
    return _abs(num)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def abs_l(num: cython.long) -> cython.long:
    "Absolute value of a (long) integer."
    return _labs(num)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def abs_ll(num: cython.longlong) -> cython.longlong:
    "Absolute value of a (long long) integer."
    return _llabs(num)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def abs_f(num: cython.double) -> cython.double:
    "Absolute value of a (float/double) number."
    return _fabas(num)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def abs_lf(num: cython.longdouble) -> cython.longdouble:
    "Absolute value of a (long double) number."
    return _fbasl(num)


# Ceil -------------------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def ceil(num: cython.double) -> cython.long:
    "Ceil value of a (float/double) number to the nearest integer."
    return int(_ceil(num))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def ceil_l(num: cython.longdouble) -> cython.longlong:
    "Ceil value of a (long double) number to the nearest integer."
    return int(_ceill(num))


# Floor ------------------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def floor(num: cython.double) -> cython.long:
    "Floor value of a (float/double) number to the nearest integer."
    return int(_floor(num))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def floor_l(num: cython.longdouble) -> cython.longlong:
    "Floor value of a (long double) number to the nearest integer."
    return int(_floorl(num))


# Round ------------------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def round(num: cython.double) -> cython.long:
    "Round value of a (float/double) number to the nearest integer."
    return int(_round(num))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def round_l(num: cython.longdouble) -> cython.longlong:
    "Round value of a (long double) number to the nearest integer."
    return int(_roundl(num))


# Maximum -------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def max_f(num1: cython.double, num2: cython.double) -> cython.double:
    "Maximum value of two (float/double) numbers."
    return _fmax(num1, num2)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def max_lf(num1: cython.longdouble, num2: cython.longdouble) -> cython.longdouble:
    "Maximum value of two (long double) numbers."
    return _fmaxl(num1, num2)


# Minimum -------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def min_f(num1: cython.double, num2: cython.double) -> cython.double:
    "Minimum value of two (float/double) numbers."
    return _fmin(num1, num2)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def min_lf(num1: cython.longdouble, num2: cython.longdouble) -> cython.longdouble:
    "Minimum value of two (long double) numbers."
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
    "Clip the min & max value of a number. Optimized for integer."
    return max(min(num, max_val), min_val)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def clip_f(
    num: cython.double,
    min_val: cython.double,
    max_val: cython.double,
) -> cython.double:
    "Clip the min & max value of a number. Optimized for float/double."
    return _fmax(_fmin(num, max_val), min_val)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def clip_lf(
    num: cython.longdouble,
    min_val: cython.longdouble,
    max_val: cython.longdouble,
) -> cython.longdouble:
    "Clip the min & max value of a number. Optimized for long double."
    return _fmaxl(_fminl(num, max_val), min_val)


# Sign ----------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def copysign(num: cython.double, sign: cython.double) -> cython.double:
    "Copy the sign of a number. Optimized for float/double/int/long."
    return _copysign(num, sign)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def copysign_l(num: cython.longdouble, sign: cython.longdouble) -> cython.longdouble:
    "Copy the sign of a number. Optimized for long double/long long."
    return _copysignl(num, sign)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def signfactor(num: cython.double) -> cython.int:
    "Get the sign factor (1 or -1) of a number. Optimized for float/double/int/long."
    return int(_copysign(1, num))


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def signfactor_l(num: cython.longdouble) -> cython.int:
    "Get the sign factor (1 or -1) of a number. Optimized for long double/long long."
    return int(_copysignl(1, num))


# Validation ----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_inf(num: cython.longdouble) -> cython.bint:
    "Check if a number is `inf`."
    return _isinf(num)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_finite(num: cython.longdouble) -> cython.bint:
    "Check if a number is `finite`."
    return _isfinite(num)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_nan(num: cython.longdouble) -> cython.bint:
    "Check if a number is `nan`."
    return _isnan(num)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_normal(num: cython.longdouble) -> cython.bint:
    "Check if a numer is `normal`."
    return _isnormal(num)
