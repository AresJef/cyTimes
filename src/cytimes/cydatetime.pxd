# cython: language_level=3

cimport numpy as np
from cpython cimport datetime
from cpython.exc cimport PyErr_SetFromErrno
from cpython.unicode cimport PyUnicode_DecodeUTF8, PyUnicode_AsUTF8
from libc.time cimport time_t, localtime as libc_localtime, strftime

# Constants
cdef:
    # . calendar
    unsigned int[13] DAYS_BR_MONTH
    unsigned int[13] DAYS_IN_MONTH
    unsigned int[5] DAYS_BR_QUARTER
    np.ndarray DAYS_BR_QUARTER_NDARRAY
    unsigned int[5] DAYS_IN_QUARTER
    unsigned int[13] MONTH_TO_QUARTER
    # . datetime
    datetime.tzinfo UTC
    datetime.datetime EPOCH_UTC
    long long EPOCH_US, EPOCH_SEC
    int EPOCH_DAY
    long long DT_MAX_US, DT_MIN_US
    # . time
    datetime.time TIME_START, TIME_END 
    # . microsecond
    long long US_DAY, US_HOUR
    # . nanosecond
    long long NS_DAY, NS_HOUR, NS_MINUTE
    # . timedelta unit adjustment
    set TD_UNIT_ADJUSTMENT
    # . function
    object FN_TIME_LOCALTIME

# Struct
ctypedef struct ymd:
    unsigned int year
    unsigned int month
    unsigned int day

ctypedef struct hms:
    unsigned int hour
    unsigned int minute
    unsigned int second
    unsigned int microsecond

ctypedef struct iso:
    unsigned int year
    unsigned int week
    unsigned int weekday

# Time
cdef extern from "<time.h>" nogil:
    cdef struct tm:
        int  tm_sec
        int  tm_min
        int  tm_hour
        int  tm_mday
        int  tm_mon
        int  tm_year
        int  tm_wday
        int  tm_yday
        int  tm_isdst

# Equivalent to `time.localtime(timestamp)`.
cdef inline tm localize_time(double timestamp) except * nogil:
    cdef:
        time_t tic = <time_t>timestamp
        tm* tms

    tms = libc_localtime(&tic)
    if tms is NULL:
        raise_from_errno()
    # Fix 0-based date values (and the 1900-based year).
    # See tmtotuple() in https://github.com/python/cpython/blob/master/Modules/timemodule.c
    tms.tm_year += 1900
    tms.tm_mon += 1
    tms.tm_wday = (tms.tm_wday + 6) % 7
    tms.tm_yday += 1
    return tms[0]

# Error handling for localize time
cdef inline int raise_from_errno() except -1 with gil:
    PyErr_SetFromErrno(RuntimeError)
    return <int> -1  # type: ignore

# Convert timestamp to a local timestamp
cdef inline long long localize_ts(double timestamp) except *:
    # Localize timestamp
    cdef tm t = localize_time(timestamp)
    # Calculate total seconds
    cdef long long ordinal = ymd_to_ordinal(t.tm_year, t.tm_mon, t.tm_mday)
    cdef long long seconds = ordinal * 86_400 + t.tm_hour * 3_600 + t.tm_min * 60 + t.tm_sec
    # Return seconds since epoch
    return seconds - EPOCH_SEC

# Convert struct:tm to Python string based on the 'format'
cdef inline str stime_to_str(tm tms, str format):
    cdef:
        char buffer[256]
        size_t size
        str result
        const char* fmt = PyUnicode_AsUTF8(format)

    # Revert back to 0-based date values.
    tms.tm_year -= 1_900
    tms.tm_mon -= 1
    tms.tm_wday = (tms.tm_wday + 1) % 7
    tms.tm_yday -= 1
    # Convert to Python string.
    size = strftime(buffer, sizeof(buffer), fmt, &tms)
    if size == 0:
        raise OverflowError("The format string for datetime is too long:\n'%s'" % format)
    return PyUnicode_DecodeUTF8(buffer, size, NULL)

# Calendar: year
cdef bint is_leapyear(unsigned int year) except -1
cdef unsigned int leap_bt_years(unsigned int year1, unsigned int year2) except -1
cdef unsigned int days_in_year(unsigned int year) except -1
cdef unsigned int days_bf_year(unsigned int year) except -1
cdef unsigned int days_of_year(unsigned int year, unsigned int month, unsigned int day) except -1
# Calendar: quarter
cdef unsigned int quarter_of_month(unsigned int month) except -1
cdef unsigned int days_in_quarter(unsigned int year, unsigned int month) except -1
cdef unsigned int days_bf_quarter(unsigned int year, unsigned int month) except -1
cdef unsigned int days_of_quarter(unsigned int year, unsigned int month, unsigned int day) except -1
cdef unsigned int quarter_1st_month(unsigned int month) except -1
cdef unsigned int quarter_lst_month(unsigned int month) except -1
# Calendar: month
cdef unsigned int days_in_month(unsigned int year, unsigned int month) except -1
cdef unsigned int days_bf_month(unsigned int year, unsigned int month) except -1
# Calendar: week
cdef unsigned int ymd_weekday(unsigned int year, unsigned int month, unsigned int day) except -1
cdef unsigned int ymd_isoweekday(unsigned int year, unsigned int month, unsigned int day) except -1
cdef unsigned int ymd_isoweek(unsigned int year, unsigned int month, unsigned int day) except -1
cdef unsigned int ymd_isoyear(unsigned int year, unsigned int month, unsigned int day) except -1
cdef iso ymd_isocalendar(unsigned int year, unsigned int month, unsigned int day) noexcept
cdef unsigned int iso1st_ordinal(unsigned int year) except -1
# Calendar: conversion
cdef unsigned int ymd_to_ordinal(unsigned int year, unsigned int month, unsigned int day) except -1
cdef ymd ordinal_to_ymd(int ordinal) noexcept
cdef ymd isocalendar_to_ymd(unsigned int year, unsigned int week, unsigned int weekday) noexcept
cdef ymd days_of_year_to_ymd(unsigned int year, unsigned int days) noexcept
cdef hms microseconds_to_hms(long long microseconds) noexcept

# Datetime.date: generate
cdef datetime.date gen_date(unsigned int year=?, unsigned int month=?, unsigned int day=?)
cdef datetime.date gen_date_now()
cdef datetime.date gen_date_now_utc()
cdef datetime.date gen_date_now_tz(datetime.tzinfo tzinfo)
# Datetime.date: check types
cdef bint is_date(object obj) except -1
cdef bint is_date_exact(object obj) except -1
# Datetime.date: calendar - year
cdef bint date_is_leapyear(datetime.date date) except -1
cdef unsigned int date_leap_bt_years(datetime.date date1, datetime.date date2) except -1
cdef unsigned int date_days_in_year(datetime.date date) except -1
cdef unsigned int date_days_bf_year(datetime.date date) except -1
cdef unsigned int date_days_of_year(datetime.date date) except -1
# Datetime.date: calendar - quarter
cdef unsigned int date_quarter(datetime.date date) except -1
cdef unsigned int date_days_in_quarter(datetime.date date) except -1
cdef unsigned int date_days_bf_quarter(datetime.date date) except -1
cdef unsigned int date_days_of_quarter(datetime.date date) except -1
cdef unsigned int date_quarter_1st_month(datetime.date date) except -1
cdef unsigned int date_quarter_lst_month(datetime.date date) except -1
# Datetime.date: calendar - month
cdef unsigned int date_days_in_month(datetime.date date) except -1
cdef unsigned int date_days_bf_month(datetime.date date) except -1
# Datetime.date: calendar - week
cdef unsigned int date_weekday(datetime.date date) except -1
cdef unsigned int date_isoweekday(datetime.date date) except -1
cdef unsigned int date_isoweek(datetime.date date) except -1
cdef unsigned int date_isoyear(datetime.date date) except -1
cdef iso date_isocalendar(datetime.date date) noexcept
# Datetime.date: conversion
cdef str date_to_isoformat(datetime.date date)
cdef str date_to_strformat(datetime.date date, str format)
cdef unsigned int date_to_ordinal(datetime.date date) except -1
cdef long long date_to_seconds(datetime.date date) noexcept
cdef long long date_to_microseconds(datetime.date date) noexcept
cdef long long date_to_timestamp(datetime.date) except *
cdef datetime.date date_fr_dt(datetime.datetime dt)
cdef datetime.date date_fr_date(datetime.date date)
cdef datetime.date date_fr_ordinal(int ordinal)
cdef datetime.date date_fr_seconds(double seconds)
cdef datetime.date date_fr_microseconds(long long microseconds)
cdef datetime.date date_fr_timestamp(double timestamp)
# Datetime.date: arithmetic
cdef datetime.date date_add(datetime.date date, int days=?, long long seconds=?, long long microseconds=?)
cdef datetime.date date_add_td(datetime.date date, datetime.timedelta td)
cdef datetime.date date_sub_td(datetime.date date, datetime.timedelta td)
cdef datetime.timedelta date_sub_date(datetime.date date_l, datetime.date date_r)
cdef int date_sub_date_days(datetime.date date_l, datetime.date date_r)
# Datetime.date: manipulation
cdef datetime.date date_replace(datetime.date date, int year=?, int month=?, int day=?)
cdef datetime.date date_adj_weekday(datetime.date date, unsigned int weekday)

# Datetime.datetime: generate
cdef datetime.datetime gen_dt(
    unsigned int year=?, unsigned int month=?, unsigned int day=?, 
    unsigned int hour=?, unsigned int minute=?, unsigned int second=?, 
    unsigned int microsecond=?, datetime.tzinfo tzinfo=?, unsigned int fold=?)
cdef datetime.datetime gen_dt_now()
cdef datetime.datetime gen_dt_now_utc()
cdef datetime.datetime gen_dt_now_tz(datetime.tzinfo tzinfo)
# Datetime.datetime: check types
cdef bint is_dt(object obj) except -1
cdef bint is_dt_exact(object obj) except -1
# Datetime.datetime: access
cdef str dt_tzname(datetime.datetime dt)
cdef datetime.timedelta dt_dst(datetime.datetime dt)
cdef datetime.timedelta dt_utcoffset(datetime.datetime dt)
# Datetime.date: calendar - year
cdef bint dt_is_leapyear(datetime.datetime dt) except -1
cdef unsigned int dt_leap_bt_years(datetime.datetime dt1, datetime.datetime dt2) except -1
cdef unsigned int dt_days_in_year(datetime.datetime dt) except -1
cdef unsigned int dt_days_bf_year(datetime.datetime dt) except -1
cdef unsigned int dt_days_of_year(datetime.datetime dt) except -1
# Datetime.date: calendar - quarter
cdef unsigned int dt_quarter(datetime.datetime dt) except -1
cdef unsigned int dt_days_in_quarter(datetime.datetime dt) except -1
cdef unsigned int dt_days_bf_quarter(datetime.datetime dt) except -1
cdef unsigned int dt_days_of_quarter(datetime.datetime dt) except -1
cdef unsigned int dt_quarter_1st_month(datetime.datetime dt) except -1
cdef unsigned int dt_quarter_lst_month(datetime.datetime dt) except -1
# Datetime.date: calendar - month
cdef unsigned int dt_days_in_month(datetime.datetime dt) except -1
cdef unsigned int dt_days_bf_month(datetime.datetime dt) except -1
# Datetime.date: calendar - week
cdef unsigned int dt_weekday(datetime.datetime dt) except -1
cdef unsigned int dt_isoweekday(datetime.datetime dt) except -1
cdef unsigned int dt_isoweek(datetime.datetime dt) except -1
cdef unsigned int dt_isoyear(datetime.datetime dt) except -1
cdef iso dt_isocalendar(datetime.datetime dt) noexcept
# Datetime.datetime: conversion
cdef str dt_to_isoformat(datetime.datetime dt)
cdef str dt_to_isoformat_tz(datetime.datetime dt)
cdef str dt_to_isospaceformat(datetime.datetime dt)
cdef str dt_to_isospaceformat_tz(datetime.datetime dt)
cdef str dt_to_strformat(datetime.datetime dt, str format)
cdef unsigned int dt_to_ordinal(datetime.datetime dt) except -1
cdef double dt_to_seconds(datetime.datetime dt) noexcept
cdef double dt_to_seconds_utc(datetime.datetime dt) noexcept
cdef long long dt_to_microseconds(datetime.datetime dt) noexcept
cdef long long dt_to_microseconds_utc(datetime.datetime dt) noexcept
cdef long long dt_to_posixts(datetime.datetime dt) except *
cdef double dt_to_timestamp(datetime.datetime dt) except *
cdef tm dt_to_stime(datetime.datetime dt) except *
cdef tm dt_to_stime_utc(datetime.datetime dt) except *
cdef datetime.datetime dt_fr_dt(datetime.datetime dt)
cdef datetime.datetime dt_fr_date(datetime.date date, datetime.tzinfo tzinfo=?, unsigned int fold=?)
cdef datetime.datetime dt_fr_time(datetime.time time)
cdef datetime.datetime dt_fr_date_n_time(datetime.date date, datetime.time time)
cdef datetime.datetime dt_fr_ordinal(int ordinal, datetime.tzinfo tzinfo=?, unsigned int fold=?)
cdef datetime.datetime dt_fr_seconds(double seconds, datetime.tzinfo tzinfo=?, unsigned int fold=?)
cdef datetime.datetime dt_fr_microseconds(long long microseconds, datetime.tzinfo tzinfo=?, unsigned int fold=?)
cdef datetime.datetime dt_fr_timestamp(double timestamp, datetime.tzinfo tzinfo=?)
# Datetime.datetime: arithmetic
cdef datetime.datetime dt_add(datetime.datetime dt, int days=?, long long seconds=?, long long microseconds=?)
cdef datetime.datetime dt_add_td(datetime.datetime dt, datetime.timedelta td)
cdef datetime.datetime dt_sub_td(datetime.datetime dt, datetime.timedelta td)
cdef datetime.timedelta dt_sub_dt(datetime.datetime dt_l, datetime.datetime dt_r)
cdef long long dt_sub_dt_us(datetime.datetime dt_l, datetime.datetime dt_r)
# Datetime.datetime: manipulation 
cdef datetime.datetime dt_replace(
    datetime.datetime dt, int year=?, int month=?, int day=?, int hour=?, 
    int minute=?, int second=?, int microsecond=?, object tzinfo=?, int fold=?)
cdef datetime.datetime dt_replace_tzinfo(datetime.datetime dt, datetime.tzinfo tzinfo)
cdef datetime.datetime dt_replace_fold(datetime.datetime dt, unsigned int fold)
cdef datetime.datetime dt_adj_weekday(datetime.datetime dt, unsigned int weekday)
cdef datetime.datetime dt_astimezone(datetime.datetime dt, datetime.tzinfo tzinfo)

# Datetime.time: generate
cdef datetime.time gen_time(
    unsigned int hour=?, unsigned int minute=?, unsigned int second=?, 
    unsigned int microsecond=?, datetime.tzinfo tzinfo=?, unsigned int fold=?)
cdef datetime.time gen_time_now()
cdef datetime.time gen_time_now_utc()
cdef datetime.time gen_time_now_tz(datetime.tzinfo tzinfo)
# Datetime.time: check types
cdef bint is_time(object obj) except -1
cdef bint is_time_exact(object obj) except -1
# Datetime.time: tzinfo
cdef str time_tzname(datetime.time time)
cdef datetime.timedelta time_dst(datetime.time time)
cdef datetime.timedelta time_utcoffset(datetime.time time)
# Datetime.time: conversion
cdef str time_to_isoformat(datetime.time time)
cdef str time_to_strformat(datetime.time time, str format)
cdef double time_to_seconds(datetime.time time) noexcept
cdef long long time_to_microseconds(datetime.time time) noexcept
cdef datetime.time time_fr_dt(datetime.datetime dt)
cdef datetime.time time_fr_seconds(double seconds, datetime.tzinfo tzinfo=?, unsigned int fold=?)
cdef datetime.time time_fr_microseconds(long long microseconds, datetime.tzinfo tzinfo=?, unsigned int fold=?)
# Datetime.time: manipulation
cdef datetime.time time_replace(
    datetime.time time, int hour=?, int minute=?, int second=?, 
    int microsecond=?, object tzinfo=?, int fold=?)
cdef datetime.time time_replace_tzinfo(datetime.time time, datetime.tzinfo tzinfo)
cdef datetime.time time_replace_fold(datetime.time time, unsigned int fold)

# Datetime.timedelta: generate
cdef datetime.timedelta gen_td(int days=?, int seconds=?, int microseconds=?)
# Datetime.timedelta: check types
cdef bint is_td(object obj) except -1
cdef bint is_td_exact(object obj) except -1
# Datetime.timedelta: conversion
cdef str td_to_isoformat(datetime.timedelta td)
cdef str td_to_utcformat(datetime.timedelta td)
cdef double td_to_seconds(datetime.timedelta td) noexcept
cdef long long td_to_microseconds(datetime.timedelta td) noexcept
cdef datetime.timedelta td_fr_td(datetime.timedelta td)
cdef datetime.timedelta td_fr_seconds(double seconds)
cdef datetime.timedelta td_fr_microseconds(long long microseconds)
# Datetime.timedelta: arithmetic
cdef datetime.timedelta td_add(datetime.timedelta td, int days=?, int seconds=?, int microseconds=?)
cdef datetime.timedelta td_add_td(datetime.timedelta td_l, datetime.timedelta td_r)
cdef datetime.timedelta td_sub_td(datetime.timedelta td_l, datetime.timedelta td_r)

# Datetime.tzinfo: generate
cdef datetime.tzinfo gen_tzinfo(int offset)
cdef datetime.tzinfo gen_tzinfo_local(datetime.datetime dt=?)
# Datetime.tzinfo: check types
cdef bint is_tzinfo(object obj) except -1
cdef bint is_tzinfo_exact(object obj) except -1
# Datetime.tzinfo: access
cdef str tzinfo_name(object tzinfo, datetime.datetime dt)
cdef datetime.timedelta tzinfo_dst(object tzinfo, datetime.datetime dt)
cdef datetime.timedelta tzinfo_utcoffset(object tzinfo, datetime.datetime dt)

# numpy.datetime64: check types
cdef bint is_dt64(object obj) except -1
cdef validate_dt64(object obj)
# numpy.datetime64: conversion
cdef str dt64_to_isoformat(object dt64)
cdef str dt64_to_isospaceformat(object dt64)
cdef long long dt64_to_int(object dt64, object unit)
cdef long long dt64_to_days(object dt64)
cdef long long dt64_to_hours(object dt64)
cdef long long dt64_to_minutes(object dt64)
cdef long long dt64_to_seconds(object dt64)
cdef long long dt64_to_milliseconds(object dt64)
cdef long long dt64_to_microseconds(object dt64)
cdef long long dt64_to_nanoseconds(object dt64)
cdef datetime.date dt64_to_date(object dt64)
cdef datetime.datetime dt64_to_dt(object dt64)
cdef datetime.time dt64_to_time(object dt64)

# numpy.timedelta64: check types
cdef bint is_td64(object obj) except -1
cdef validate_td64(object obj)
# numpy.timedelta64: conversion
cdef str td64_to_isoformat(object td64)
cdef long long td64_to_int(object td64, object unit)
cdef long long td64_to_days(object td64)
cdef long long td64_to_hours(object td64)
cdef long long td64_to_minutes(object td64)
cdef long long td64_to_seconds(object td64)
cdef long long td64_to_milliseconds(object td64)
cdef long long td64_to_microseconds(object td64)
cdef long long td64_to_nanoseconds(object td64)
cdef datetime.timedelta td64_to_td(object td64)

# numpy.ndarray[datetime64]: check types
cdef bint is_dt64array(np.ndarray arr) except -1
cdef validate_dt64array(np.ndarray arr)
cdef str get_dt64array_unit(np.ndarray arr)
# numpy.ndarray[datetime64]: conversion
cdef np.ndarray dt64array_to_int(np.ndarray arr, object unit)
cdef np.ndarray dt64array_to_days(np.ndarray arr)
cdef np.ndarray dt64array_to_hours(np.ndarray arr)
cdef np.ndarray dt64array_to_minutes(np.ndarray arr)
cdef np.ndarray dt64array_to_seconds(np.ndarray arr)
cdef np.ndarray dt64array_to_milliseconds(np.ndarray arr)
cdef np.ndarray dt64array_to_microseconds(np.ndarray arr)
cdef np.ndarray dt64array_to_nanoseconds(np.ndarray arr)
cdef np.ndarray dt64array_to_ordinals(np.ndarray arr)
cdef np.ndarray dt64array_to_timestamps(np.ndarray arr)

# numpy.ndarray[timedelta64]: check types
cdef bint is_td64array(np.ndarray arr) except -1
cdef validate_td64array(np.ndarray arr)
cdef str get_td64array_unit(np.ndarray arr)
# numpy.ndarray[timedelta64]: conversion
cdef np.ndarray td64array_to_int(np.ndarray arr, object unit)
cdef np.ndarray td64array_to_days(np.ndarray arr)
cdef np.ndarray td64array_to_hours(np.ndarray arr)
cdef np.ndarray td64array_to_minutes(np.ndarray arr)
cdef np.ndarray td64array_to_seconds(np.ndarray arr)
cdef np.ndarray td64array_to_milliseconds(np.ndarray arr)
cdef np.ndarray td64array_to_microseconds(np.ndarray arr)
cdef np.ndarray td64array_to_nanoseconds(np.ndarray arr)

# pandas.Series[datetime64]: check types
cdef bint is_dt64series(object obj) except -1
cdef validate_dt64series(object obj)
cdef np.ndarray get_dt64series_arr(object s)
cdef str get_dt64series_unit(object s)
# pandas.Series[datetime64]: conversion
cdef object dt64series_to_int(object s, object unit)
cdef object dt64series_to_days(object s)
cdef object dt64series_to_hours(object s)
cdef object dt64series_to_minutes(object s)
cdef object dt64series_to_seconds(object s)
cdef object dt64series_to_milliseconds(object s)
cdef object dt64series_to_microseconds(object s)
cdef object dt64series_to_nanoseconds(object s)
cdef object dt64series_to_ordinals(object s)
cdef object dt64series_to_timestamps(object s)

# pandas.Series[timedelta64]: check types
cdef bint is_td64series(object obj) except -1
cdef validate_td64series(object obj)
cdef np.ndarray get_td64series_arr(object s)
cdef str get_td64series_unit(object s)
# pandas.Series[timedelta64]: conversion
cdef object td64series_to_int(object s, object unit)
cdef object td64series_to_days(object s)
cdef object td64series_to_hours(object s)
cdef object td64series_to_minutes(object s)
cdef object td64series_to_seconds(object s)
cdef object td64series_to_milliseconds(object s)
cdef object td64series_to_microseconds(object s)
cdef object td64series_to_nanoseconds(object s)
# pandas.Series[timedelta64]: adjustment
cdef object td64series_adjust_unit(object s, object unit)
