from timeit import timeit
from random import randint
import datetime, numpy as np, pandas as pd
from dateutil.parser._parser import parser
from dateutil.relativedelta import relativedelta, weekday as relweekday
from cytimes import Weekday, cytimedelta, Parser, pydt, pddt


def gen_date(year: int = 1, month: int = 1, day: int = 1) -> datetime.date:
    return datetime.date(year, month, day)


def gen_dt(
    year: int = 1,
    month: int = 1,
    day: int = 1,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
    tzinfo: datetime.tzinfo = None,
    fold: int = 0,
) -> datetime.datetime:
    return datetime.datetime(
        year, month, day, hour, minute, second, microsecond, tzinfo, fold=fold
    )


def gen_delta(
    days: int = 0,
    seconds: int = 0,
    microseconds: int = 0,
) -> datetime.timedelta:
    return datetime.timedelta(days, seconds, microseconds)


def gen_time(
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
    tzinfo: datetime.tzinfo = None,
    fold: int = 0,
) -> datetime.time:
    return datetime.time(hour, minute, second, microsecond, tzinfo, fold=fold)


def validate_cytimedelta_relative() -> None:
    def yeild_relative_info_kwargs() -> dict:
        for Ys in range(-9000, 9000):
            for Ms in range(-12, 12):
                yield {
                    "years": Ys,
                    "months": Ms,
                    "days": randint(-9999, 9999),
                    "weeks": randint(-52, 52),
                    "hours": randint(-99999, 99999),
                    "minutes": randint(-99999, 99999),
                    "seconds": randint(-99999, 99999),
                    "microseconds": randint(-99999, 99999),
                    "leapdays": randint(-1, 1),
                }

    def yeild_relative_add_kwargs() -> dict:
        for Ys in range(0, 9000):
            for Ms in range(0, 12):
                yield {
                    "years": Ys,
                    "months": Ms,
                    "days": randint(0, 99999),
                    "weeks": randint(0, 52),
                    "hours": randint(0, 99999),
                    "minutes": randint(0, 99999),
                    "seconds": randint(0, 99999),
                    "microseconds": randint(0, 99999),
                    "leapdays": randint(-1, 1),
                }

    def yeild_relative_sub_kwargs() -> dict:
        for Ys in range(-9000, 0):
            for Ms in range(-12, 0):
                yield {
                    "years": Ys,
                    "months": Ms,
                    "days": randint(-99999, 0),
                    "weeks": randint(-52, 0),
                    "hours": randint(-99999, 0),
                    "minutes": randint(-99999, 0),
                    "seconds": randint(-99999, 0),
                    "microseconds": randint(-99999, 0),
                    "leapdays": randint(-1, 1),
                }

    PADDING = 260
    print(" Compare cytimedelta & relativedelta REL Info".center(80, "-"))
    for kwargs in yeild_relative_info_kwargs():
        cydl = cytimedelta(**kwargs)
        redl = relativedelta(**kwargs)
        msg = "%s | %s | %s" % (cydl == redl, cydl, redl)
        print(msg.ljust(PADDING), end="\r")
        if cydl != redl:
            print("REL Info Not Equal".ljust(PADDING))
            print(cydl == redl, cydl, redl, sep=" | ")
            break
    else:
        print("REL Info All Equal".ljust(PADDING))
        print(cydl == redl, cydl, redl, sep=" | ")
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta ADD REL Info".center(80, "-"))
    base_args = {
        "years": 1,
        "months": 1,
        "days": 1,
        "weeks": 1,
        "hours": 1,
        "minutes": 1,
        "seconds": 1,
        "microseconds": 1,
        "leapdays": 1,
    }
    base_cydl = cytimedelta(**base_args)
    base_redl = relativedelta(**base_args)
    for kwargs in yeild_relative_info_kwargs():
        cydl = base_cydl + cytimedelta(**kwargs)
        cyre = base_cydl + relativedelta(**kwargs)
        redl = base_redl + relativedelta(**kwargs)
        recy = base_redl + cytimedelta(**kwargs)
        eq = cydl == redl == cyre == recy
        msg = "%s | %s | %s" % (eq, cydl, redl)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("ADD REL Info Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl, redl))
            break
    else:
        print("ADD REL Info All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl, redl))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta SUB REL Info".center(80, "-"))
    base_cydl = cytimedelta(**base_args)
    base_redl = relativedelta(**base_args)
    for kwargs in yeild_relative_info_kwargs():
        cydl = base_cydl - cytimedelta(**kwargs)
        cyre = base_cydl - relativedelta(**kwargs)
        redl = base_redl - relativedelta(**kwargs)
        recy = base_redl - cytimedelta(**kwargs)
        eq = cydl == redl == cyre == recy
        msg = "%s | %s | %s" % (eq, cydl, redl)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("SUB REL Info Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl, redl))
            break
    else:
        print("SUB REL Info All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl, redl))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta REL Date ADD".center(80, "-"))
    date = gen_date(1, 1, 1)
    for kwargs in yeild_relative_add_kwargs():
        cydl_l = date + cytimedelta(**kwargs)
        redl_l = date + relativedelta(**kwargs)
        cydl_r = cytimedelta(**kwargs) + date
        redl_r = relativedelta(**kwargs) + date
        eq = cydl_l == redl_l == cydl_r == redl_r
        msg = "%s | %s | %s" % (eq, cydl_l, redl_l)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("Date ADD Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl_l, redl_l))
            break
    else:
        print("Date ADD All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl_l, redl_l))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta REL Datetime ADD".center(80, "-"))
    dt = gen_dt(1, 1, 1)
    for kwargs in yeild_relative_add_kwargs():
        cydl_l = dt + cytimedelta(**kwargs)
        redl_l = dt + relativedelta(**kwargs)
        cydl_r = cytimedelta(**kwargs) + dt
        redl_r = relativedelta(**kwargs) + dt
        eq = cydl_l == redl_l == cydl_r == redl_r
        msg = "%s | %s | %s" % (eq, cydl_l, redl_l)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("Datetime ADD Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl_l, redl_l))
            break
    else:
        print("Datetime ADD All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl_l, redl_l))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta REL Datetime ADD (Neg)".center(80, "-"))
    dt = gen_dt(9999, 12, 31, 23, 59, 59, 999999)
    for kwargs in yeild_relative_sub_kwargs():
        cydl_l = dt + cytimedelta(**kwargs)
        redl_l = dt + relativedelta(**kwargs)
        cydl_r = cytimedelta(**kwargs) + dt
        redl_r = relativedelta(**kwargs) + dt
        eq = cydl_l == redl_l == cydl_r == redl_r
        msg = "%s | %s | %s" % (eq, cydl_l, redl_l)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("Datetime ADD (Neg) Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl_l, redl_l))
            break
    else:
        print("Datetime ADD (Neg) All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl_l, redl_l))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta REL Timedelta ADD".center(80, "-"))
    dl = gen_delta(1, 1, 1)
    for kwargs in yeild_relative_add_kwargs():
        cydl_l = dl + cytimedelta(**kwargs)
        redl_l = dl + relativedelta(**kwargs)
        cydl_r = cytimedelta(**kwargs) + dl
        redl_r = relativedelta(**kwargs) + dl
        eq = cydl_l == redl_l == cydl_r == redl_r
        msg = "%s | %s | %s" % (eq, cydl_l, redl_l)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("Timedelta ADD Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl_l, redl_l))
            break
    else:
        print("Timedelta ADD All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl_l, redl_l))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta REL Date SUB".center(80, "-"))
    date = gen_date(9999, 12, 31)
    for kwargs in yeild_relative_add_kwargs():
        cydl_l = date - cytimedelta(**kwargs)
        redl_l = date - relativedelta(**kwargs)
        eq = cydl_l == redl_l
        msg = "%s | %s | %s" % (eq, cydl_l, redl_l)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("Date SUB Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl_l, redl_l))
            break
    else:
        print("Date SUB All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl_l, redl_l))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta REL Datetime SUB".center(80, "-"))
    dt = gen_dt(9999, 12, 31, 23, 59, 59, 999999)
    for kwargs in yeild_relative_add_kwargs():
        cydl_l = dt - cytimedelta(**kwargs)
        redl_l = dt - relativedelta(**kwargs)
        eq = cydl_l == redl_l
        msg = "%s | %s | %s" % (eq, cydl_l, redl_l)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("Datetime SUB Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl_l, redl_l))
            break
    else:
        print("Datetime SUB All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl_l, redl_l))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta REL Datetime SUB (POS)".center(80, "-"))
    dt = gen_dt(1, 1, 1)
    for kwargs in yeild_relative_sub_kwargs():
        cydl_l = dt - cytimedelta(**kwargs)
        redl_l = dt - relativedelta(**kwargs)
        eq = cydl_l == redl_l
        msg = "%s | %s | %s" % (eq, cydl_l, redl_l)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("Datetime SUB (POS) Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl_l, redl_l))
            break
    else:
        print("Datetime SUB (POS) All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl_l, redl_l))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta REL Timedelta SUB".center(80, "-"))
    dl = gen_delta(1, 1, 1)
    for kwargs in yeild_relative_add_kwargs():
        cydl_l = dl - cytimedelta(**kwargs)
        redl_l = dl - relativedelta(**kwargs)
        eq = cydl_l == redl_l
        msg = "%s | %s | %s" % (eq, cydl_l, redl_l)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("Timedelta SUB Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl_l, redl_l))
            break
    else:
        print("Timedelta SUB All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl_l, redl_l))
    print("-" * 80)
    print()


def validate_cytimedelta_absolute() -> None:
    def yeild_absolute_info_kwargs() -> dict:
        for Ys in range(2, 9000):
            for Ms in range(1, 13):
                yield {
                    "year": Ys,
                    "month": Ms,
                    "day": randint(1, 31),
                    "hour": randint(1, 23),
                    "minute": randint(1, 59),
                    "second": randint(1, 59),
                    "microsecond": randint(1, 999999),
                }

    PADDING = 260
    print(" Compare Weekday & Rel weekday ".center(80, "-"))
    for wd in range(7):
        for wn in range(-30, 30):
            cyw = Weekday(wd, wn)
            rlw = relweekday(wd, wn - 1 if wn < 0 else wn + 1 if wn > 0 else wn)
            cydl1 = cytimedelta(weekday=cyw)
            cydl2 = cytimedelta(weekday=rlw)
            redl = relativedelta(weekday=rlw)
            eq = cyw == rlw and cydl1 == cydl2 == redl
            msg = "%s | %s | %s | %s | %s | %s" % (eq, cyw, rlw, cydl1, cydl2, redl)
            print(msg.ljust(PADDING), end="\r")
            if not eq:
                print("Weekday Not Equal".ljust(PADDING))
                print(
                    "%s | %s | %s | %s | %s | %s" % (eq, cyw, rlw, cydl1, cydl2, redl)
                )
                break
    else:
        print("Weekday All Equal".ljust(PADDING))
        print("%s | %s | %s | %s | %s | %s" % (eq, cyw, rlw, cydl1, cydl2, redl))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta ABS Info".center(80, "-"))
    for kwargs in yeild_absolute_info_kwargs():
        cydl = cytimedelta(**kwargs)
        redl = relativedelta(**kwargs)
        msg = "%s | %s | %s" % (cydl == redl, cydl, redl)
        print(msg.ljust(PADDING), end="\r")
        if cydl != redl:
            print("ABS Info Not Equal".ljust(PADDING))
            print(cydl == redl, cydl, redl, sep=" | ")
            break
    else:
        print("ABS Info All Equal".ljust(PADDING))
        print(cydl == redl, cydl, redl, sep=" | ")
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta ADD ABS Info".center(80, "-"))
    base_cydl = cytimedelta(
        year=1, month=1, day=1, hour=1, minute=1, second=1, microsecond=1
    )
    base_redl = relativedelta(
        year=1, month=1, day=1, hour=1, minute=1, second=1, microsecond=1
    )
    for kwargs in yeild_absolute_info_kwargs():
        cydl = base_cydl + cytimedelta(**kwargs)
        cyre = base_cydl + relativedelta(**kwargs)
        redl = base_redl + relativedelta(**kwargs)
        recy = base_redl + cytimedelta(**kwargs)
        eq = cydl == redl == cyre == recy
        msg = "%s | %s | %s" % (eq, cydl, redl)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("ADD ABS Info Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl, redl))
            break
    else:
        print("ADD ABS Info All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl, redl))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta SUB ABS Info".center(80, "-"))
    base_cydl = cytimedelta(
        year=9999, month=12, day=31, hour=23, minute=59, second=59, microsecond=999999
    )
    base_redl = relativedelta(
        year=9999, month=12, day=31, hour=23, minute=59, second=59, microsecond=999999
    )
    for kwargs in yeild_absolute_info_kwargs():
        cydl = base_cydl - cytimedelta(**kwargs)
        cyre = base_cydl - relativedelta(**kwargs)
        redl = base_redl - relativedelta(**kwargs)
        recy = base_redl - cytimedelta(**kwargs)
        eq = cydl == redl == cyre == recy
        msg = "%s | %s | %s" % (eq, cydl, redl)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("SUB ABS Info Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl, redl))
            print(cydl)
            print(cyre)
            print(redl)
            print(recy)
            break
    else:
        print("SUB ABS Info All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl, redl))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta ABS Date ADD".center(80, "-"))
    date = gen_date(1, 1, 1)
    for kwargs in yeild_absolute_info_kwargs():
        cydl_l = date + cytimedelta(**kwargs)
        redl_l = date + relativedelta(**kwargs)
        cydl_r = cytimedelta(**kwargs) + date
        redl_r = relativedelta(**kwargs) + date
        eq = cydl_l == redl_l == cydl_r == redl_r
        msg = "%s | %s | %s" % (eq, cydl_l, redl_l)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("Date ADD Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl_l, redl_l))
            break
    else:
        print("Date ADD All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl_l, redl_l))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta ABS Datetime ADD".center(80, "-"))
    dt = gen_dt(1, 1, 1)
    for kwargs in yeild_absolute_info_kwargs():
        cydl_l = dt + cytimedelta(**kwargs)
        redl_l = dt + relativedelta(**kwargs)
        cydl_r = cytimedelta(**kwargs) + dt
        redl_r = relativedelta(**kwargs) + dt
        eq = cydl_l == redl_l == cydl_r == redl_r
        msg = "%s | %s | %s" % (eq, cydl_l, redl_l)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("Datetime ADD Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl_l, redl_l))
            break
    else:
        print("Datetime ADD All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl_l, redl_l))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta ABS Date SUB".center(80, "-"))
    date = gen_date(1, 1, 1)
    for kwargs in yeild_absolute_info_kwargs():
        cydl_l = date - cytimedelta(**kwargs)
        redl_l = date - relativedelta(**kwargs)
        eq = cydl_l == redl_l
        msg = "%s | %s | %s" % (eq, cydl_l, redl_l)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("Date SUB Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl_l, redl_l))
            break
    else:
        print("Date SUB All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl_l, redl_l))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta ABS Datetime SUB".center(80, "-"))
    dt = gen_dt(1, 1, 1)
    for kwargs in yeild_absolute_info_kwargs():
        cydl_l = dt - cytimedelta(**kwargs)
        redl_l = dt - relativedelta(**kwargs)
        eq = cydl_l == redl_l
        msg = "%s | %s | %s" % (eq, cydl_l, redl_l)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("Datetime SUB Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl_l, redl_l))
            break
    else:
        print("Datetime SUB All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl_l, redl_l))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta ABS ADD with Weekday".center(80, "-"))
    dt = gen_dt(2000, 1, 1)
    base_cydl = cytimedelta(weekday=Weekday(0, 1))
    base_redl = relativedelta(weekday=relweekday(0, 2))
    for kwargs in yeild_absolute_info_kwargs():
        wd = randint(0, 6)
        wn = randint(-20, 20)
        cyw = Weekday(wd, wn)
        rlw = relweekday(wd, wn - 1 if wn < 0 else wn + 1 if wn > 0 else wn)
        cydl_l = dt + cytimedelta(**kwargs, weekday=cyw)
        redl_l = dt + relativedelta(**kwargs, weekday=rlw)
        cydl_r = cytimedelta(**kwargs, weekday=rlw) + dt
        redl_r = relativedelta(**kwargs, weekday=rlw) + dt
        cadd = base_cydl + cytimedelta(**kwargs, weekday=cyw)
        radd = base_redl + relativedelta(**kwargs, weekday=rlw)
        xadd = base_cydl + cytimedelta(**kwargs, weekday=rlw)
        eq = cydl_l == redl_l == cydl_r == redl_r and cadd == radd == xadd
        msg = "%s | %s | %s" % (eq, cydl_l, redl_l)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("ADD with Weekday Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl_l, redl_l))
            break
    else:
        print("ADD with Weekday All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl_l, redl_l))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta ABS SUB with Weekday".center(80, "-"))
    dt = gen_dt(2000, 1, 1)
    base_cydl = cytimedelta(weekday=Weekday(0, 1))
    base_redl = relativedelta(weekday=relweekday(0, 2))
    for kwargs in yeild_absolute_info_kwargs():
        wd = randint(0, 6)
        wn = randint(-20, 20)
        cyw = Weekday(wd, wn)
        rlw = relweekday(wd, wn - 1 if wn < 0 else wn + 1 if wn > 0 else wn)
        cydl_l = dt - cytimedelta(**kwargs, weekday=cyw)
        redl_l = dt - relativedelta(**kwargs, weekday=rlw)
        cydl_x = dt - cytimedelta(**kwargs, weekday=rlw)
        csub = base_cydl - cytimedelta(**kwargs, weekday=cyw)
        rsub = base_redl - relativedelta(**kwargs, weekday=rlw)
        xsub = base_cydl - cytimedelta(**kwargs, weekday=rlw)
        eq = cydl_l == redl_l == cydl_x and csub == rsub == xsub
        msg = "%s | %s | %s" % (eq, cydl_l, redl_l)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("SUB with Weekday Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl_l, redl_l))
            break
    else:
        print("SUB with Weekday All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl_l, redl_l))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta ABS ADD with YearDay".center(80, "-"))
    dt = gen_dt(2000, 1, 1)
    base_cydl = cytimedelta(yearday=1)
    base_redl = relativedelta(yearday=1)
    for kwargs in yeild_absolute_info_kwargs():
        kwargs = kwargs | {"yearday": randint(1, 364)}
        cydl_l = dt + cytimedelta(**kwargs)
        redl_l = dt + relativedelta(**kwargs)
        cydl_r = cytimedelta(**kwargs) + dt
        redl_r = relativedelta(**kwargs) + dt
        cadd = base_cydl + cytimedelta(**kwargs)
        radd = base_redl + relativedelta(**kwargs)
        xadd = base_cydl + cytimedelta(**kwargs)
        eq = cydl_l == redl_l == cydl_r == redl_r and cadd == radd == xadd
        msg = "%s | %s | %s" % (eq, cydl_l, redl_l)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("ADD with YearDay Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl_l, redl_l))
            break
    else:
        print("ADD with YearDay All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl_l, redl_l))
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta ABS SUB with YearDay".center(80, "-"))
    dt = gen_dt(2000, 1, 1)
    base_cydl = cytimedelta(yearday=1)
    base_redl = relativedelta(yearday=1)
    for kwargs in yeild_absolute_info_kwargs():
        kwargs = kwargs | {"yearday": randint(1, 364)}
        cydl_l = dt - cytimedelta(**kwargs)
        redl_l = dt - relativedelta(**kwargs)
        cadd = base_cydl - cytimedelta(**kwargs)
        radd = base_redl - relativedelta(**kwargs)
        xadd = base_cydl - cytimedelta(**kwargs)
        eq = cydl_l == redl_l and cadd == radd == xadd
        msg = "%s | %s | %s" % (eq, cydl_l, redl_l)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("SUB with YearDay Not Equal".ljust(PADDING))
            print("%s | %s | %s" % (eq, cydl_l, redl_l))
            break
    else:
        print("SUB with YearDay All Equal".ljust(PADDING))
        print("%s | %s | %s" % (eq, cydl_l, redl_l))
    print("-" * 80)
    print()


def cytimedelta_performance() -> None:
    full_args = {
        "years": 4,
        "months": 4,
        "days": 4,
        "weeks": 4,
        "hours": 4,
        "minutes": 4,
        "seconds": 4,
        "microseconds": 4,
        "leapdays": 4,
        "year": 4,
        "month": 4,
        "day": 4,
        "hour": 4,
        "minute": 4,
        "second": 4,
        "microsecond": 4,
        "yearday": 360,
    }
    rel_args = {
        "years": 4,
        "months": 4,
        "days": 4,
        "weeks": 4,
        "hours": 4,
        "minutes": 4,
        "seconds": 4,
        "microseconds": 4,
        "leapdays": 4,
    }

    print(" Compare cytimedelta & relativedelta Initialize".center(80, "-"))
    rounds = 1_000_000
    print("cydl:", t1 := timeit(lambda: cytimedelta(**full_args), number=rounds))
    print("redl:", t2 := timeit(lambda: relativedelta(**full_args), number=rounds))
    print("impx:", t2 / t1)
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta ADD date".center(80, "-"))
    rounds = 1_000_000
    da = gen_date(2000)
    cydl = cytimedelta(**full_args)
    redl = relativedelta(**full_args)
    print("cydl:", t1 := timeit(lambda: da + cydl, number=rounds))
    print("redl:", t2 := timeit(lambda: da + redl, number=rounds))
    print("impx:", t2 / t1)
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta ADD datetime".center(80, "-"))
    rounds = 1_000_000
    dt = gen_dt(2000)
    cydl = cytimedelta(**full_args)
    redl = relativedelta(**full_args)
    print("cydl:", t1 := timeit(lambda: dt + cydl, number=rounds))
    print("redl:", t2 := timeit(lambda: dt + redl, number=rounds))
    print("impx:", t2 / t1)
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta SUB date".center(80, "-"))
    rounds = 1_000_000
    da = gen_date(2000)
    cydl = cytimedelta(**rel_args)
    redl = relativedelta(**rel_args)
    print("cydl:", t1 := timeit(lambda: da - cydl, number=rounds))
    print("redl:", t2 := timeit(lambda: da - redl, number=rounds))
    print("impx:", t2 / t1)
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta SUB datetime".center(80, "-"))
    rounds = 1_000_000
    dt = gen_dt(2000)
    cydl = cytimedelta(**rel_args)
    redl = relativedelta(**rel_args)
    print("cydl:", t1 := timeit(lambda: dt - cydl, number=rounds))
    print("redl:", t2 := timeit(lambda: dt - redl, number=rounds))
    print("impx:", t2 / t1)
    print("-" * 80)
    print()

    print(" Compare cytimedelta & relativedelta ADD self".center(80, "-"))
    rounds = 1_000_000
    base_cydl = cytimedelta(**full_args)
    sec_cydl = base_cydl / 2
    base_redl = relativedelta(**full_args)
    sec_redl = base_redl / 2
    print("cydl:", t1 := timeit(lambda: base_cydl + sec_cydl, number=rounds))
    print("redl:", t2 := timeit(lambda: base_redl + sec_redl, number=rounds))
    print("impx:", t2 / t1)
    print("-" * 80)
    print()


def cytimedelta_demo() -> None:
    print(" Cytimedelta Weekday ".center(80, "-"))
    print("Weekday:".ljust(28), wd := Weekday(0, 1))
    print("Weekday.weekday:".ljust(28), wd.weekday)
    print("weekday.weekcode:".ljust(28), wd.weekcode)
    print("Weekday.week_offset:".ljust(28), wd.week_offset)
    print("Weekday repr".ljust(28), repr(wd))
    print("Weekday hash".ljust(28), hash(wd))
    print("Weekday bool".ljust(28), bool(wd))
    wd2 = Weekday(0, 1)
    print("Weekday compare:".ljust(28), wd == wd2)
    print()

    # fmt: off
    cydl_rel_args = {
        "years": 1,
        "months": 1,
        "days": 1,
        "weeks": 1,
        "hours": 1,
        "minutes": 1,
        "seconds": 1,
        "microseconds": 1,
        "leapdays": 1
    }
    print(" Cytimedelta ADD ".center(80, "-"))
    print("cydl:".ljust(28), cydl := cytimedelta(**cydl_rel_args))
    print("date:".ljust(28), da := gen_date(2000))
    print("cydl + date:".ljust(28), cydl + da)
    print("date + cydl:".ljust(28), da + cydl)
    print("dt:".ljust(28), dt := gen_dt(2000))
    print("cydl + dt:".ljust(28), cydl + dt)
    print("dt + cydl:".ljust(28), dt + cydl)
    print("ts:".ljust(28), ts := pd.Timestamp("2000-01-01"))
    print("cydl + ts:".ljust(28), cydl + ts)
    print("ts + cydl:".ljust(28), ts + cydl)
    print("dt64:".ljust(28), dt64 := np.datetime64("2000-01-01"))
    print("cydl + dt64:".ljust(28), cydl + dt64)
    print("cydl2:".ljust(28), cydl2 := cytimedelta(years=1, seconds=999999, year=2008, weekday=Weekday(6, 1)))
    print("cydl + cydl2:".ljust(28), cydl + cydl2)
    print("cydl2 + cydl:".ljust(28), cydl2 + cydl)
    print("redl:".ljust(28), redl := relativedelta(years=1, seconds=999999, year=2008, weekday=relweekday(6, 2)))
    print("cydl + redl:".ljust(28), cydl + redl)
    print("redl + cydl:".ljust(28), redl + cydl)
    print("dlta:".ljust(28), dlta := gen_delta(1, 1, 1))
    print("cydl + dlta:".ljust(28), cydl + dlta)
    print("dlta + cydl:".ljust(28), dlta + cydl)
    print("dl64:".ljust(28), dl64 := np.timedelta64(1, "ms"))
    print("cydl + dl64:".ljust(28), cydl + dl64)
    print()

    print(" Cytimedelta Substruction ".center(80, "-"))
    print("cydl:".ljust(28), cydl := cytimedelta(**cydl_rel_args))
    print("date:".ljust(28), da := gen_date(2000, 1, 1))
    print("date - cydl:".ljust(28), da - cydl)
    print("dt:".ljust(28), dt := gen_dt(2000, 1, 1))
    print("dt - cydl:".ljust(28), dt - cydl)
    print("ts:".ljust(28), ts := pd.Timestamp("2000-01-01"))
    print("ts - cydl:".ljust(28), ts - cydl)
    print("cydl2:".ljust(28), cydl2 := cytimedelta(years=1, seconds=999999, year=2008, weekday=Weekday(6, 1)))
    print("cydl - cydl2:".ljust(28), cydl - cydl2)
    print("cydl2 - cydl:".ljust(28), cydl2 - cydl)
    print("redl:".ljust(28), redl := relativedelta(years=1, seconds=999999, year=2008, weekday=relweekday(6, 2)))
    print("cydl - redl:".ljust(28), cydl - redl)
    print("redl - cydl:".ljust(28), redl - cydl)
    print("dlta:".ljust(28), dlta := gen_delta(1, 1, 1))
    print("cydl - dlta:".ljust(28), cydl - dlta)
    print("dlta - cydl:".ljust(28), dlta - cydl)
    print("dl64:".ljust(28), dl64 := np.timedelta64(1, "ms"))
    print("cydl - dl64:".ljust(28), cydl - dl64)
    print()

    # fmt: on
    print(" Cytimedelta Multiplication ".center(80, "-"))
    print("cydl:".ljust(28), cydl := cytimedelta(**cydl_rel_args))
    print("cydl * 2:".ljust(28), cydl * 2)
    print("2 * cydl:".ljust(28), 2 * cydl)
    print()

    print(" Cytimedelta Division ".center(80, "-"))
    print("cydl:".ljust(28), cydl := cytimedelta(**cydl_rel_args))
    print("cydl / 2:".ljust(28), cydl / 2)
    print("cydl / 2:".ljust(28), cydl / 3)
    print("cydl / 2:".ljust(28), cydl / 4)
    print("cydl / 2:".ljust(28), cydl / 5)
    print("cydl / 2:".ljust(28), cydl / 6)
    print()

    print(" Cytimedelta Manipulation ".center(80, "-"))
    print("cydl:".ljust(28), cydl := cytimedelta(**cydl_rel_args))
    print("cydl neg:".ljust(28), neg := -cydl)
    print("cydl neg abs:".ljust(28), cydl_abs := abs(neg))
    print("cydl equal neg:".ljust(28), cydl == neg)
    print("cydl qeual abs:".ljust(28), cydl == cydl_abs)
    print("cydl hash:".ljust(28), hash(cydl))
    print()


def gen_dtstr_set() -> list[str]:
    return [
        "2023-08-01 12:00:00",
        "2023/08/01 12:00:00",
        "08-01-2023 12:00:00",
        "01/08/2023 12:00:00",
        "2023-08-01T12:00:00",
        "2023-08-01 12:00:00Z",
        "2023-08-01T12:00:00Z",
        "2023-08-01 12:00:00.000Z",
        "2023-08-01T12:00:00.000Z",
        "08-01-2023 12:00",
        "01/08/2023 12:00",
        "2023-08-01 12:00",
        "2023/08/01 12:00",
        "August 1, 2023 12:00:00",
        "1 August, 2023 12:00:00",
        "Aug 1, 2023 12:00:00",
        "1 Aug, 2023 12:00:00",
        "2023-08-01T12:00:00.000000",
        "2023-08-01 12:00:00.000000",
        "2023-08-01T12:00:00+00:00",
        "2023-08-01 12:00:00+00:00",
        "2023-08-01T12:00:00-00:00",
        "2023-08-01 12:00:00-00:00",
        "2023-08-01T12:00:00.000000Z",
        "2023-08-01 12:00:00.000000Z",
        "2023-08-01T12:00:00.000000+00:00",
        "2023-08-01 12:00:00.000000+00:00",
        "2023-08-01T12:00:00.000000-00:00",
        "2023-08-01 12:00:00.000000-00:00",
        "2023-08-01T00:00:00",
        "2023-08-01 00:00:00",
        "08-01-2023 00:00:00",
        "01/08/2023 00:00:00",
        "August 1, 2023 00:00:00",
        "1 August, 2023 00:00:00",
        "Aug 1, 2023 00:00:00",
        "1 Aug, 2023 00:00:00",
        "2023-08-01T00:00:00.000Z",
        "2023-08-01 00:00:00.000Z",
        "2023-08-01T00:00:00.000000",
        "2023-08-01 00:00:00.000000",
        "2023-08-01T00:00:00+00:00",
        "2023-08-01 00:00:00+00:00",
        "2023-08-01T00:00:00-00:00",
        "2023-08-01 00:00:00-00:00",
        "2023-08-01T00:00:00.000000Z",
        "2023-08-01 00:00:00.000000Z",
        "2023-08-01T00:00:00.000000+00:00",
        "2023-08-01 00:00:00.000000+00:00",
        "2023-08-01T00:00:00.000000-00:00",
        "2023-08-01 00:00:00.000000-00:00",
        "2023-08-01",
        "08-01-2023",
        "01/08/2023",
        "August 1, 2023",
        "1 August, 2023",
        "Aug 1, 2023",
        "1 Aug, 2023",
        "20230801",
        "2023.08.01",
        "01.08.2023",
        "2023/08/01 12:00:00 PM",
        "2023/08/01 12:00:00 AM",
        "2023/08/01 00:00:00 AM",
        "2023/08/01 00:00:00 PM",
        "08-01-2023 12:00 PM",
        "08-01-2023 12:00 AM",
        "08-01-2023 00:00 AM",
        "08-01-2023 00:00 PM",
        "August 1, 2023 12:00 PM",
        "August 1, 2023 12:00 AM",
        "1 August, 2023 12:00 PM",
        "1 August, 2023 12:00 AM",
        "Aug 1, 2023 12:00 PM",
        "Aug 1, 2023 12:00 AM",
        "1 Aug, 2023 12:00 PM",
        "1 Aug, 2023 12:00 AM",
        "2023-08-01 12:00:00 PM",
        "2023-08-01 12:00:00 AM",
        "2023-08-01T12:00:00 PM",
        "2023-08-01T12:00:00 AM",
        "2023-08-01 00:00:00 PM",
        "2023-08-01 00:00:00 AM",
        "2023-08-01T00:00:00 PM",
        "2023-08-01T00:00:00 AM",
        "2023-08-01 12:00 PM",
        "2023-08-01 12:00 AM",
        "2023/08/01 12:00 PM",
        "2023/08/01 12:00 AM",
        "2023-08-01T12:00 PM",
        "2023-08-01T12:00 AM",
        "2023-08-01 00:00 PM",
        "2023-08-01 00:00 AM",
        "2023/08/01 00:00 PM",
        "2023/08/01 00:00 AM",
        "2023-08-01T00:00 PM",
        "2023-08-01T00:00 AM",
        "2023-08-01 12:00:00.000Z PM",
        "2023-08-01 12:00:00.000Z AM",
        "2023-08-01 00:00:00.000Z PM",
        "2023-08-01 00:00:00.000Z AM",
        "2023-08-01 12:00:00+00:00 PM",
        "2023-08-01 12:00:00+00:00 AM",
        "2023-08-01 00:00:00+00:00 PM",
        "2023-08-01 00:00:00+00:00 AM",
        "2023-08-01 12:00:00-00:00 PM",
        "2023-08-01 12:00:00-00:00 AM",
        "2023-08-01 00:00:00-00:00 PM",
        "2023-08-01 00:00:00-00:00 AM",
        "20230801T12",
        "2023T12",
        "2023 T12",
        "20230801120000",
        "20230801 120000",
        "2023-August-01",
        "2023-Aug-01",
        "2023-August-1",
        "2023-Aug-1",
        "2023-August-01 12:00",
        "2023-Aug-01 12:00",
        "2023-August-1 12:00",
        "2023-Aug-1 12:00",
        "2023-August-01 12:00:00",
        "2023-Aug-01 12:00:00",
        "2023-August-1 12:00:00",
        "2023-Aug-1 12:00:00",
        "1 8 2023",
        "1 8 2023 12:00",
        "1 8 2023 12:00:00",
        "01 08 2023",
        "01 08 2023 12:00",
        "01 08 2023 12:00:00",
        "1 08 2023",
        "1 08 2023 12:00",
        "1 08 2023 12:00:00",
        "01 8 2023",
        "01 8 2023 12:00",
        "01 8 2023 12:00:00",
        "08 01 2023",
        "08 01 2023 12:00",
        "08 01 2023 12:00:00",
        "8 01 2023",
        "8 01 2023 12:00",
        "8 01 2023 12:00:00",
        "08 1 2023",
        "08 1 2023 12:00",
        "08 1 2023 12:00:00",
        "8 1 2023",
        "8 1 2023 12:00",
        "8 1 2023 12:00:00",
        "2023.08.01.12",
        "2023.08.01.120000",
        "2023.08.01.1200",
        "2023.08.01 12:00:00",
        "2023.08.01 12:00",
        "2023.08.01 12",
        "2023.08.01 120000",
        "2023.08.01 1200",
        "2023 08 01 12:00:00",
        "2023 08 01 12:00",
        "2023 08 01 12",
        "2023 08 01 120000",
        "2023 08 01 1200",
        "2023-Aug-01",
        "2023 Aug-01",
        "2023 Aug 01",
        "2023-Aug 01",
        "2023-August-01",
        "2023 August-01",
        "2023 August 01",
        "2023-August 01",
        "2023-08-01T12Z",
        "2023-08-01T12+00:00",
        "2023-08-01T12-00:00",
        "2023-08-01 12Z",
        "2023-08-01 12+00:00",
        "2023-08-01 12-00:00",
        "2023-08-01T120000Z",
        "2023-08-01T120000+00:00",
        "2023-08-01T120000-00:00",
        "2023-08-01 120000Z",
        "2023-08-01 120000+00:00",
        "2023-08-01 120000-00:00",
        "2023-08-01T1200Z",
        "2023-08-01T1200+00:00",
        "2023-08-01T1200-00:00",
        "2023-08-01 1200Z",
        "2023-08-01 1200+00:00",
        "2023-08-01 1200-00:00",
        "19990101T2359",
        "19990101T235959",
        "19990101T235959.123",
        "125959.123",
        "53.123123s",
        "20230801",
        "12h32m53",
        "12:32",
        "12:32:53",
        "12:32:53.122",
        "2023-08-01",
        "2023/08/01",
        "2023.08.01",
        "11 PM",
        "3 AM",
        "11pm",
        "3am",
        # "Sat 2023-08-01",
        # "2023-08-01 sun.",
        "Feb-01",
        "Feb-01-99",
        "Feb of 99",
        "11 AM PM",
        "2023-08-01 12:00:00 GMT",
        "2023-08-01 12:00:00 GMT-2",
        "2023-08-01 12:00:00 UTC-0312",
        "2023-08-01 12:00:00 UTC+3:12",
        "2023-08-01 12:00:00 -0300 (BRST)",
        "2023-08-01 12:00:00 CST",
        "Thu Sep 25 10:36:28 2003",
        "Thu Sep 25 2003",
        "2003-09-25T10:49:41",
        "2003-09-25T10:49",
        "2003-09-25T10",
        "2003-09-25",
        "20030925T104941",
        "20030925T1049",
        "20030925T10",
        "20030925",
        "2003-09-25 10:49:41,502",
        "199709020908",
        "19970902090807",
        "09-25-2003",
        "25-09-2003",
        "10-09-2003",
        "10-09-03",
        "2003.09.25",
        "09.25.2003",
        "25.09.2003",
        "10.09.2003",
        "10.09.03",
        "2003/09/25",
        "09/25/2003",
        "25/09/2003",
        "10/09/2003",
        "10/09/03",
        "2003 09 25",
        "09 25 2003",
        "25 09 2003",
        "10 09 2003",
        "10 09 03",
        "25 09 03",
        "03 25 Sep",
        "25 03 Sep",
        "  July   4 ,  1976   12:01:02   am  ",
        "Wed, July 10, '96",
        "1996.July.10 AD 12:08 PM",
        "July 4, 1976",
        "7 4 1976",
        "4 jul 1976",
        "4 Jul 1976",
        "7-4-76",
        "19760704",
        "0:01:02 on July 4, 1976",
        "July 4, 1976 12:01:02 am",
        "Mon Jan  2 04:24:27 1995",
        "04.04.95 00:22",
        "Jan 1 1999 11:23:34.578",
        "950404 122212",
        "3rd of May 2001",
        "5th of March 2001",
        "1st of May 2003",
        "0099-01-01T00:00:00",
        "0031-01-01T00:00:00",
        "20080227T21:26:01.123456789",
        "13NOV2017",
        "0003-03-04",
        "December.0031.30",
        "2016-12-21 04.2h",
        # "Thu Sep 25 10:36:28",
        # "Thu Sep 10:36:28",
        # "Thu 10:36:28",
        "Sep 10:36:28",
        "10:36:28",
        "10:36",
        "Sep 2003",
        "Sep",
        "2003",
        "10h36m28.5s",
        "10h36m28s",
        "10h36m",
        "10h",
        "10 h 36",
        "10 h 36.5",
        "36 m 5",
        "36 m 5 s",
        "36 m 05",
        "36 m 05 s",
        "10h am",
        "10h pm",
        "10am",
        "10pm",
        "10:00 am",
        "10:00 pm",
        "10:00am",
        "10:00pm",
        "10:00a.m",
        "10:00p.m",
        "10:00a.m.",
        "10:00p.m.",
        # "Wed",
        # "Wednesday",
        "October",
        "31-Dec-00",
        "0:01:02",
        "12h 01m02s am",
        "12:08 PM",
        "01h02m03",
        "01h02",
        "01h02s",
        "01m02",
        "01m02h",
        "2004 10 Apr 11h30m",
        "Thu Sep 25 10:36:28 BRST 2003",
        "1996.07.10 AD at 15:08:56 PDT",
        # "Tuesday, April 12, 1952 AD 3:30:42pm PST",
        "April 12, 1952 AD 3:30:42pm PST",
        "November 5, 1994, 8:15:30 am EST",
        "1994-11-05T08:15:30-05:00",
        "1994-11-05T08:15:30Z",
        "1976-07-04T00:01:02Z",
        "1986-07-05T08:15:30z",
        "Tue Apr 4 00:22:12 PDT 1995",
        "20030925T104941-0300",
        "Thu, 25 Sep 2003 10:49:41 -0300",
        "2003-09-25T10:49:41.5-03:00",
        "2003-09-25T10:49:41-03:00",
        "20030925T104941.5-0300",
        "Thu Sep 25 2003",
        "Sep 25 2003",
        "2003-09-25",
        "20030925",
        "2003-Sep-25",
        "25-Sep-2003",
        "Sep-25-2003",
        "09-25-2003",
        "25-09-2003",
        "2003.09.25",
        "2003.Sep.25",
        "25.Sep.2003",
        "Sep.25.2003",
        "09.25.2003",
        "25.09.2003",
        "2003/09/25",
        "2003/Sep/25",
        "25/Sep/2003",
        "Sep/25/2003",
        "09/25/2003",
        "25/09/2003",
        "2003 09 25",
        "2003 Sep 25",
        "25 Sep 2003",
        "09 25 2003",
        "25 09 2003",
        "03 25 Sep",
        "2014 January 19",
        "2014-05-01 08:00:00",
        "2017-02-03 12:40 BRST",
        "2014 January 19 09:00 UTC",
        "Thu Sep 25 10:36:28 BRST 2003",
        "0:00 PM, PST",
        "5:50 A.M. on June 13, 1990",
        "April 2009",
        "00:11:25.01",
        "00:12:10.01",
        "090107",
        "2015 09 25",
        "2015-15-May",
        "02:17NOV2017",
    ]


def gen_fuzzy_dtstr_set() -> list[str]:
    return [
        "Today is 25 of September of 2003, exactly at 10:49:41 with timezone -03:00.",
        "I have a meeting on March 1, 1974.",
        "On June 8th, 2020, I am going to be the first man on Mars",
        "Meet me at the AM/PM on Sunset at 3:00 AM on December 3rd, 2003",
        "Meet me at 3:00AM on December 3rd, 2003 at the AM/PM on Sunset",
        "Jan 29, 1945 14:45 AM I going to see you there?",
        "2017-07-17 06:15:",
    ]


def cyparser_validate() -> None:
    import warnings

    warnings.filterwarnings("ignore")

    cp = Parser()
    rp = parser()

    PADDING = 150
    print("Validate Parser (Default)".center(80, "-"))
    for s in gen_dtstr_set():
        dtn = rp.parse(s)
        dtc = cp.parse(s)
        eq = dtc == dtn
        msg = "%s | %s | %s | %s" % (eq, dtc, dtn, s)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("Parser (Default) Result Not Equal".ljust(PADDING))
            print("%s | %s | %s | %s" % (eq, dtc, dtn, s))
            break
    else:
        print("Parser (Default) Result All Equal".ljust(PADDING))
        print("%s | %s | %s | %s" % (eq, dtc, dtn, s))
    print("-" * 80)
    print()

    print("Validate Parser (Default Datetime)".center(80, "-"))
    dt = gen_dt(1995, 2, 1, 1, 1, 1, 1)
    for s in gen_dtstr_set():
        dtn = rp.parse(s, default=dt)
        dtc = cp.parse(s, default=dt)
        eq = dtc == dtn
        msg = "%s | %s | %s | %s" % (eq, dtc, dtn, s)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("Parser (Default Date) Result Not Equal".ljust(PADDING))
            print("%s | %s | %s | %s" % (eq, dtc, dtn, s))
            break
    else:
        print("Parser (Default Datetime) Result All Equal".ljust(PADDING))
        print("%s | %s | %s | %s" % (eq, dtc, dtn, s))
    print("-" * 80)
    print()

    print("Validate Parser (Fuzzy)".center(80, "-"))
    for s in gen_fuzzy_dtstr_set():
        dtn = rp.parse(s, fuzzy=True)
        dtc = cp.parse(s, fuzzy=True)
        eq = dtc == dtn
        msg = "%s | %s | %s | %s" % (eq, dtc, dtn, s)
        print(msg.ljust(PADDING), end="\r")
        if not eq:
            print("Parser (Fuzzy) Result Not Equal".ljust(PADDING))
            print("%s | %s | %s | %s" % (eq, dtc, dtn, s))
            break
    else:
        print("Parser (Fuzzy) Result All Equal".ljust(PADDING))
        print("%s | %s | %s | %s" % (eq, dtc, dtn, s))
    print("-" * 80)
    print()


def cyparser_performance() -> None:
    import warnings

    warnings.filterwarnings("ignore")

    cp = Parser()
    rp = parser()

    rounds = 100000
    timestr = "2023-08-01T12:00:00"
    print("Parser Performance Compare 1".center(80, "-"))
    print("Cython:", t1 := timeit(lambda: cp.parse(timestr), number=rounds))
    print("Native:", t2 := timeit(lambda: rp.parse(timestr), number=rounds))
    print("P_diff:", t2 / t1)
    print("-" * 80)
    print()

    rounds = 100000
    timestr = "2023-08-01 12:00:00"
    from_str, fmt = datetime.datetime.strptime, "%Y-%m-%d %H:%M:%S"
    print("Parser Performance Compare 2".center(80, "-"))
    print("Cython:", t1 := timeit(lambda: cp.parse(timestr), number=rounds))
    print("Format:", t2 := timeit(lambda: from_str(timestr, fmt), number=rounds))
    print("P_diff:", t2 / t1)
    print("-" * 80)
    print()

    rounds = 500
    print("Parser Performance Compare 3".center(80, "-"))

    def cython_test() -> None:
        for s in gen_dtstr_set():
            cp.parse(s)

    def native_test() -> None:
        for s in gen_dtstr_set():
            rp.parse(s)

    print("Cython:", t1 := timeit(cython_test, number=rounds))
    print("Native:", t2 := timeit(native_test, number=rounds))
    print("P_diff:", t2 / t1)
    print()


def pydt_demo() -> None:
    dt = gen_dt(2023, 3, 10, 12, 0, 1, 1)
    tzinfo = datetime.timezone(datetime.timedelta(hours=3))
    dt = gen_dt(2023, 3, 10, 12, 0, 1, 1, tzinfo=tzinfo)
    pt = pydt(dt)

    # Access
    print(" Access ".center(80, "="))
    print("pydt:".ljust(15), pt)
    print("dt:".ljust(15), i := pt.dt, type(i))
    print("dtiso:".ljust(15), i := pt.dtiso, type(i))
    print("dtisotz:".ljust(15), i := pt.dtisotz, type(i))
    print("date:".ljust(15), i := pt.date, type(i))
    print("dateiso:".ljust(15), i := pt.dateiso, type(i))
    print("time:".ljust(15), i := pt.time, type(i))
    print("timeiso:".ljust(15), i := pt.timeiso, type(i))
    print("timetz:".ljust(15), i := pt.timetz, type(i))
    print("timeisotz:".ljust(15), i := pt.timeisotz, type(i))
    print("ts:".ljust(15), i := pt.ts, type(i))
    print("dt64:".ljust(15), i := pt.dt64, type(i))
    print("ordinal:".ljust(15), i := pt.ordinal, type(i))
    print("seconds:".ljust(15), i := pt.seconds, type(i))
    print("seconds_utc:".ljust(15), i := pt.seconds_utc, type(i))
    print("microseconds:".ljust(15), i := pt.microseconds, type(i))
    print("microseconds_utc:".ljust(15), i := pt.microseconds_utc, type(i))
    print("timestamp:".ljust(15), i := pt.timestamp, type(i))
    print()

    # Absolute
    print(" Absolute ".center(80, "="))
    print("pydt:".ljust(15), pt)
    print("year:".ljust(15), pt.year)
    print("month:".ljust(15), pt.month)
    print("day:".ljust(15), pt.day)
    print("hour:".ljust(15), pt.hour)
    print("minute:".ljust(15), pt.minute)
    print("second:".ljust(15), pt.second)
    print("microsecond:".ljust(15), pt.microsecond)
    print("tzinfo:".ljust(15), pt.tzinfo)
    print()

    # Calendar
    print(" Calendar ".center(80, "="))
    print("pydt:".ljust(15), pt)
    print("is_leapyear:".ljust(15), i := pt.is_leapyear, type(i))
    print("days_in_year:".ljust(15), i := pt.days_in_year, type(i))
    print("days_bf_year:".ljust(15), i := pt.days_bf_year, type(i))
    print("days_of_year:".ljust(15), i := pt.days_of_year, type(i))
    print("quarter:".ljust(15), i := pt.quarter, type(i))
    print("days_in_quarter:".ljust(15), i := pt.days_in_quarter, type(i))
    print("days_bf_quarter:".ljust(15), i := pt.days_bf_quarter, type(i))
    print("days_of_quarter:".ljust(15), i := pt.days_of_quarter, type(i))
    print("days_in_month:".ljust(15), i := pt.days_in_month, type(i))
    print("days_bf_month:".ljust(15), i := pt.days_bf_month, type(i))
    print("weekday:".ljust(15), i := pt.weekday, type(i))
    print("isoweekday:".ljust(15), i := pt.isoweekday, type(i))
    print("isoweek:".ljust(15), i := pt.isoweek, type(i))
    print("isoyear:".ljust(15), i := pt.isoyear, type(i))
    print("isocalendar:".ljust(15), i := pt.isocalendar, type(i))
    print()

    # Time manipulation
    print(" Time manipulation ".center(80, "="))
    print("pydt:".ljust(15), pt)
    print("start_time:".ljust(15), i := pt.start_time, type(i))
    print("end_time:".ljust(15), i := pt.end_time, type(i))
    print()

    # Day manipulation
    print(" Day manipulation ".center(80, "="))
    print("pydt:".ljust(15), pt)
    print("tomorrow:".ljust(15), i := pt.tomorrow)
    print("yesterday:".ljust(15), i := pt.yesterday)
    print()

    # Week manipulation
    print(" Week manipulation ".center(80, "="))
    print("pydt:".ljust(15), pt)
    print("monday:".ljust(15), i := pt.monday)
    print("tuesday:".ljust(15), i := pt.tuesday)
    print("wednesday:".ljust(15), i := pt.wednesday)
    print("thursday:".ljust(15), i := pt.thursday)
    print("friday:".ljust(15), i := pt.friday)
    print("saturday:".ljust(15), i := pt.saturday)
    print("sunday:".ljust(15), i := pt.sunday)
    print("curr_week:".ljust(15), i := pt.curr_week(None))
    print("curr_week:".ljust(15), i := pt.curr_week("Thu"))
    print("next_week:".ljust(15), i := pt.next_week("Thu"))
    print("next_week:".ljust(15), i := pt.next_week())
    print("last_week:".ljust(15), i := pt.last_week("mon"))
    print("last_week:".ljust(15), i := pt.last_week())
    print("to_week:".ljust(15), i := pt.to_week(2, "Wed"))
    print("to_week:".ljust(15), i := pt.to_week(2))
    print("to_week:".ljust(15), i := pt.to_week(-2, "Wed"))
    print("to_week:".ljust(15), i := pt.to_week(-2))
    print("is_weekday:".ljust(15), i := pt.is_weekday("fri"))
    print()

    # Month manipulation
    print(" Month manipulation ".center(80, "="))
    print("pydt:".ljust(15), pt)
    print("month_1st:".ljust(15), i := pt.month_1st)
    print("is_month_1st:".ljust(15), i := pt.month_1st.is_month_1st())
    print("month_lst:".ljust(15), i := pt.month_lst)
    print("is_month_lst:".ljust(15), i := pt.month_lst.is_month_lst())
    print("curr_month:".ljust(15), i := pt.curr_month(0))
    print("curr_month:".ljust(15), i := pt.curr_month(31))
    print("curr_month:".ljust(15), i := pt.curr_month(15))
    print("next_month:".ljust(15), i := pt.next_month())
    print("next_month:".ljust(15), i := pt.next_month(31))
    print("next_month:".ljust(15), i := pt.next_month(15))
    print("last_month:".ljust(15), i := pt.last_month())
    print("last_month:".ljust(15), i := pt.last_month(31))
    print("last_month:".ljust(15), i := pt.last_month(15))
    print("to_month:".ljust(15), i := pt.to_month(-2))
    print("to_month:".ljust(15), i := pt.to_month(-2, 32))
    print("to_month:".ljust(15), i := pt.to_month(2))
    print("to_month:".ljust(15), i := pt.to_month(2, 32))
    print("to_month:".ljust(15), i := pt.to_month(0))
    print("is_month".ljust(15), i := pt.is_month("mar"))
    print()

    # Quarter manipulation
    print(" Quarter manipulation ".center(80, "="))
    print("pydt:".ljust(15), pt)
    print("quarter_1st:".ljust(15), i := pt.quarter_1st)
    print("is_quarter_1st:".ljust(15), i := pt.quarter_1st.is_quarter_1st())
    print("quarter_lst:".ljust(15), i := pt.quarter_lst)
    print("is_quarter_lst:".ljust(15), i := pt.quarter_lst.is_quarter_lst())
    print("curr_quarter:".ljust(15), i := pt.curr_quarter(2))
    print("curr_quarter:".ljust(15), i := pt.curr_quarter(2, 32))
    print("next_quarter:".ljust(15), i := pt.next_quarter(2))
    print("next_quarter:".ljust(15), i := pt.next_quarter(2, 32))
    print("last_quarter:".ljust(15), i := pt.last_quarter(2))
    print("last_quarter:".ljust(15), i := pt.last_quarter(2, 32))
    print("to_quarter:".ljust(15), i := pt.to_quarter(-2, 2))
    print("to_quarter:".ljust(15), i := pt.to_quarter(-2, 2, 32))
    print("to_quarter:".ljust(15), i := pt.to_quarter(2, 2))
    print("to_quarter:".ljust(15), i := pt.to_quarter(2, 2, 32))
    print("is_quarter".ljust(15), i := pt.is_quarter(1))
    print()

    # Year manipulation
    print(" Year manipulation ".center(80, "="))
    print("pydt:".ljust(15), pt)
    print("year_1st:".ljust(15), i := pt.year_1st)
    print("is_year_1st:".ljust(15), i := pt.year_1st.is_year_1st())
    print("year_lst:".ljust(15), i := pt.year_lst)
    print("is_year_lst:".ljust(15), i := pt.year_lst.is_year_lst())
    print("curr_year:".ljust(15), i := pt.curr_year())
    print("curr_year:".ljust(15), i := pt.curr_year(day=1))
    print("curr_year:".ljust(15), i := pt.curr_year("feb", 31))
    print("next_year:".ljust(15), i := pt.next_year())
    print("next_year:".ljust(15), i := pt.next_year(day=1))
    print("next_year:".ljust(15), i := pt.next_year(2, 31))
    print("last_year:".ljust(15), i := pt.last_year())
    print("last_year:".ljust(15), i := pt.last_year(day=1))
    print("last_year:".ljust(15), i := pt.last_year(2, 31))
    print("to_year:".ljust(15), i := pt.to_year(-2))
    print("to_year:".ljust(15), i := pt.to_year(-2, day=1))
    print("to_year:".ljust(15), i := pt.to_year(-2, "feb", 31))
    print("to_year:".ljust(15), i := pt.to_year(2))
    print("to_year:".ljust(15), i := pt.to_year(2, day=1))
    print("to_year:".ljust(15), i := pt.to_year(2, 2, 31))
    print("is_year".ljust(15), i := pt.is_year(2023))
    print()

    # Timezone manipulation
    print(" Timezone manipulation ".center(80, "="))
    print("pydt:".ljust(15), pt)
    print("tz_available:".ljust(15), pt.tz_available)
    print("tz_localize:".ljust(15), pt_n := pt.tz_localize(None))
    print("tz_localize:".ljust(15), pt_n.tz_localize("CET"))
    print("tz_convert:".ljust(15), pt_n.tz_localize("UTC").tz_convert("CET"))
    print("tz_convert:".ljust(15), pt_n.tz_localize("CET").tz_convert(None))
    print("tz_convert:".ljust(15), pt_n.tz_localize("CET").tz_convert("PST8PDT"))
    print("tz_switch:".ljust(15), pt_n.tz_switch("CET", "UTC"))
    print("tz_switch:".ljust(15), pt_n.tz_switch(None, "CET"))
    print("tz_switch:".ljust(15), pt_n.tz_localize("CET").tz_switch("PST8PDT"))
    print()

    # Frequency manipulation
    print(" Frequency manipulation ".center(80, "="))
    dt = gen_dt(2023, 3, 10, 12, 30, 30, 555555, tzinfo=tzinfo)
    print("pydt:".ljust(15), ptx := pydt(dt))
    print("round D:".ljust(15), ptx.round("D"))
    print("round h:".ljust(15), ptx.round("h"))
    print("round m:".ljust(15), ptx.round("m"))
    print("round s:".ljust(15), ptx.round("s"))
    print("round ms:".ljust(15), ptx.round("ms"))
    print("round us:".ljust(15), ptx.round("us"))
    print("ceil D:".ljust(15), ptx.ceil("D"))
    print("ceil h:".ljust(15), ptx.ceil("h"))
    print("ceil m:".ljust(15), ptx.ceil("m"))
    print("ceil s:".ljust(15), ptx.ceil("s"))
    print("ceil ms:".ljust(15), ptx.ceil("ms"))
    print("ceil us:".ljust(15), ptx.ceil("us"))
    print("floor D:".ljust(15), ptx.floor("D"))
    print("floor h:".ljust(15), ptx.floor("h"))
    print("floor m:".ljust(15), ptx.floor("m"))
    print("floor s:".ljust(15), ptx.floor("s"))
    print("floor ms:".ljust(15), ptx.floor("ms"))
    print("floor us:".ljust(15), ptx.floor("us"))
    print()

    # Delta adjustment
    print(" Delta adjustment ".center(80, "="))
    print("pydt:".ljust(15), pt)
    print("delta1:".ljust(15), pt.delta(1, 1, 1, 1, 1, 1, 1, 1))
    print("delta2:".ljust(15), pt.delta(-1, -1, -1, -1, -1, -1, -1, -1))
    print("delta3:".ljust(15), pt.delta(1, -1, 1, -1, 1, -1, 1, -1))
    print("delta4:".ljust(15), pt.delta(-1, 1, -1, 1, -1, 1, -1, 1))
    print()

    # Replace adjustment
    print(" Replace adjustment ".center(80, "="))
    print("pydt:".ljust(15), pt)
    print("replace1:".ljust(15), pt.replace(2024, 2, 2, 2, 2, 2, 2, 2))
    print("replace2:".ljust(15), pt.replace(tzinfo=datetime.UTC))
    print()

    # Between calculation
    print(" Between calculation ".center(80, "="))
    print("pydt:".ljust(15), pt)
    print("other:".ljust(15), o := pt.delta(1, 1, 1, 1, 1, 1, 1, 1))
    print("between Y:".ljust(15), pt.between(o, "Y"))
    print("between M:".ljust(15), pt.between(o, "M"))
    print("between W:".ljust(15), pt.between(o, "W"))
    print("between D:".ljust(15), pt.between(o, "D"))
    print("between h:".ljust(15), pt.between(o, "h"))
    print("between m:".ljust(15), pt.between(o, "m"))
    print("between s:".ljust(15), pt.between(o, "s"))
    print("between ms:".ljust(15), pt.between(o, "ms"))
    print("between us:".ljust(15), pt.between(o, "us"))
    print()

    # Speical methods - addition
    print(" Special methods - Addition ".center(80, "="))
    dl = gen_delta(0, 0, 112903812)
    tl = pd.Timedelta(112903812, "us")
    cl = cytimedelta(microseconds=112903812)
    rl = relativedelta(microseconds=112903812)
    dl64 = np.timedelta64(112903812, "us")
    print("reference:".ljust(28), a := pt.dt + dl, type(a))
    print("pydt + timedelta:".ljust(28), b := pt + dl, type(b))
    print("timedelta + pydt:".ljust(28), b := dl + pt, type(b))
    print("pydt + pd.Timedelta:".ljust(28), b := pt + tl, type(b))
    print("pd.Timedelta + pydt:".ljust(28), b := tl + pt, type(b))
    print("pydt + cytimedelta:".ljust(28), d := pt + cl, type(d))
    print("cytimedelta + pydt:".ljust(28), d := cl + pt, type(d))
    print("pydt + relativedelata:".ljust(28), f := pt + rl, type(f))
    print("relativedelata + pydt:".ljust(28), f := rl + pt, type(f))
    print("pydt + np.timedelta64:".ljust(28), h := pt + dl64, type(h))
    print()

    # Speical methods - subtraction
    print(" Special methods - Subtraction ".center(80, "="))
    pt1 = pt
    pt2 = pt.delta(1, 1, 1, 1, 1, 1, 1, 1)
    dt1 = pt1.dt
    dt2 = pt2.dt
    dt3 = pd.Timestamp(dt2)
    dt4 = np.datetime64(dt2)
    print("reference:".ljust(28), a := dt1 - dt2, type(a))
    print("pydt - pydt:".ljust(28), b := pt1 - pt2, type(b))
    print("pydt - datetime:".ljust(28), b := pt1 - dt2, type(b))
    print("pydt - pd.Timestamp:".ljust(28), b := pt1 - dt3, type(b))
    print("pydt - np.datetime64:".ljust(28), b := pt1 - dt4, type(b))
    print("pydt - timestr:".ljust(28), b := pt1 - pt2.dtisotz, type(b))
    print()

    print("reference:".ljust(28), a := dt2 - dt1, type(a))
    print("pydt - pydt:".ljust(28), b := pt2 - pt1, type(b))
    print("datetime - pydt:".ljust(28), b := dt2 - pt1, type(b))
    print("pd.Timestamp - pydt:".ljust(28), b := dt3 - pt1, type(b))
    print("timestr - pydt:".ljust(28), b := pt2.dtisotz - pt1, type(b))
    print()

    print("reference:".ljust(28), c := dt1 - dl, type(c))
    print("pydt - timedelta:".ljust(28), d := pt1 - dl, type(d))
    print("pydt - pd.Timedelta:".ljust(28), d := pt1 - tl, type(d))
    print("pydt - cytimedelta:".ljust(28), d := pt1 - cl, type(d))
    print("pydt - relativedelata:".ljust(28), f := pt1 - rl, type(f))
    print("pydt - np.timedelta64:".ljust(28), h := pt1 - dl64, type(h))
    print()

    # Speical methods - comparison
    print(" Special methods - Comparison ".center(80, "="))
    # fmt: off
    dt1 = datetime.datetime(2023, 8, 1, 12, 0, 0)
    dt2 = datetime.datetime(2023, 8, 1, 12, 0, 0, fold=1)
    dt3 = datetime.datetime(2023, 8, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    dt4 = datetime.datetime(2023, 8, 1, 12, 0, 0, tzinfo=datetime.timezone.utc, fold=1)
    dt5 = datetime.datetime(2023, 8, 1, 12, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=-3)))
    dt6 = datetime.datetime(2023, 8, 1, 12, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=+3)))
    dt7 = datetime.datetime(2023, 8, 1, 12, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=-3)), fold=1)
    dt8 = datetime.datetime(2023, 8, 1, 12, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=+3)), fold=1)
    pt1, pt2, pt3, pt4 = pydt(dt1), pydt(dt2), pydt(dt3), pydt(dt4)
    pt5, pt6, pt7, pt8 = pydt(dt5), pydt(dt6), pydt(dt7), pydt(dt8)
    # fmt: on
    print("pydt1 & hash:".ljust(15), pt1, hash(pt1))
    print("pydt2 & hash:".ljust(15), pt2, hash(pt2))
    print("pydt3 & hash:".ljust(15), pt3, hash(pt3))
    print("pydt4 & hash:".ljust(15), pt4, hash(pt4))
    print("pydt5 & hash:".ljust(15), pt5, hash(pt5))
    print("pydt6 & hash:".ljust(15), pt6, hash(pt6))
    print("pydt7 & hash:".ljust(15), pt7, hash(pt7))
    print("pydt8 & hash:".ljust(15), pt8, hash(pt8))
    print()

    print("reference:".ljust(28), dt1 == dt2)
    print("pydt1 == pydt2:".ljust(28), pt1 == pt2)
    print("datetime1 == pydt1:".ljust(28), dt1 == pt1)
    print("datetime1 == pydt2:".ljust(28), dt1 == pt2)
    print("datetime2 == pydt1:".ljust(28), dt2 == pt1)
    print("datetime2 == pydt2:".ljust(28), dt2 == pt2)
    print("pydt1 == datetime1:".ljust(28), pt1 == dt1)
    print("pydt1 == datetime2:".ljust(28), pt1 == dt2)
    print("pydt2 == datetime1:".ljust(28), pt2 == dt1)
    print("pydt2 == datetime2:".ljust(28), pt2 == dt2)
    print()

    print("reference".ljust(28), dt1 == dt3)
    print("pydt1 == pydt3:".ljust(28), pt1 == pt3)
    print("datetime3 == pydt1:".ljust(28), dt3 == pt1)
    print("datetime1 == pydt3:".ljust(28), dt1 == pt3)
    print("pydt1 == datetime3:".ljust(28), pt1 == dt3)
    print("pydt3 == datetime1:".ljust(28), pt3 == dt1)
    print()

    print("reference:".ljust(28), dt1 == dt4)
    print("pydt1 == pydt4:".ljust(28), pt1 == dt4)
    print("datetime1 == pydt4:".ljust(28), dt1 == pt4)
    print("datetime4 == pydt1:".ljust(28), dt4 == pt1)
    print("pydt1 == datetime4:".ljust(28), pt1 == dt4)
    print("pydt4 == datetime1:".ljust(28), pt4 == dt1)
    print()

    print("reference:".ljust(28), dt3 == dt4)
    print("pydt3 == pydt4:".ljust(28), pt3 == pt4)
    print("datetime3 == pydt4:".ljust(28), dt3 == pt4)
    print("pydt3 == datetime4:".ljust(28), pt3 == dt4)
    print("datetime4 == pydt3:".ljust(28), dt4 == pt3)
    print("pydt4 == datetime3:".ljust(28), pt4 == dt3)
    print()

    print("reference:".ljust(28), dt1 == dt5)
    print("pydt1 == pydt5:".ljust(28), pt1 == pt5)
    print("datetime1 == pydt5:".ljust(28), dt1 == pt5)
    print("datetime5 == pydt1:".ljust(28), dt5 == pt1)
    print("pydt1 == datetime5:".ljust(28), pt1 == dt5)
    print("pydt5 == datetime1:".ljust(28), pt5 == dt1)
    print()

    print("reference:".ljust(28), dt1 == dt6)
    print("pydt1 == pydt6:".ljust(28), pt1 == dt6)
    print("datetime1 == pydt6:".ljust(28), dt1 == pt6)
    print("datetime6 == pydt1:".ljust(28), dt6 == pt1)
    print("pydt1 == datetime6:".ljust(28), pt1 == dt6)
    print("pydt6 == datetime1:".ljust(28), pt6 == dt1)
    print()

    print("reference:".ljust(28), dt5 == dt6)
    print("pydt5 == pydt6:".ljust(28), pt5 == pt6)
    print("datetime5 == pydt6:".ljust(28), dt5 == pt6)
    print("datetime6 == pydt5:".ljust(28), dt6 == pt5)
    print("pydt5 == datetime6:".ljust(28), pt5 == dt6)
    print("pydt6 == datetime5:".ljust(28), pt6 == dt5)
    print()

    print("refereence:".ljust(28), dt5 == dt7)
    print("pydt5 == pydt7:".ljust(28), pt5 == pt7)
    print("datetime5 == pydt7:".ljust(28), dt5 == pt7)
    print("datetime7 == pydt5:".ljust(28), dt7 == pt5)
    print("pydt5 == datetime7:".ljust(28), pt5 == dt7)
    print("pydt7 == datetime5:".ljust(28), pt7 == dt5)
    print()

    print("reference:".ljust(28), dt5 == dt8)
    print("pydt5 == pydt8:".ljust(28), pt5 == dt8)
    print("datetime5 == pydt8:".ljust(28), dt5 == pt8)
    print("datetime8 == pydt5:".ljust(28), dt8 == pt5)
    print("pydt5 == datetime8:".ljust(28), pt5 == dt8)
    print("pydt8 == datetime5:".ljust(28), pt8 == dt5)
    print()

    print("reference:".ljust(28), dt6 == dt7)
    print("pydt6 == pydt7:".ljust(28), pt6 == pt7)
    print("datetime6 == pydt7:".ljust(28), dt6 == pt7)
    print("datetime7 == pydt6:".ljust(28), dt7 == pt6)
    print("pydt6 == datetime7:".ljust(28), pt6 == dt7)
    print("pydt7 == datetime6:".ljust(28), pt7 == dt6)
    print()

    print("reference:".ljust(28), pt6 == dt8)
    print("pydt6 == pydt8:".ljust(28), pt6 == dt8)
    print("datetime6 == pydt8:".ljust(28), dt6 == pt8)
    print("datetime8 == pydt6:".ljust(28), dt8 == pt6)
    print("pydt6 == datetime8:".ljust(28), pt6 == dt8)
    print("pydt8 == datetime6:".ljust(28), pt8 == dt6)
    print()

    print("reference:".ljust(28), dt7 == dt8)
    print("pydt7 == pydt8:".ljust(28), pt7 == pt8)
    print("datetime7 == pydt8:".ljust(28), dt7 == pt8)
    print("datetime8 == pydt7:".ljust(28), dt8 == pt7)
    print("pydt7 == datetime8:".ljust(28), pt7 == dt8)
    print("pydt8 == datetime7:".ljust(28), pt8 == dt7)
    print()

    print("reference:".ljust(28), dt7 != dt8)
    print("pydt7 != pydt8:".ljust(28), pt7 != pt8)
    print("datetime7 != pydt8:".ljust(28), dt7 != pt8)
    print("datetime8 != pydt7:".ljust(28), dt8 != pt7)
    print("pydt7 != datetime8:".ljust(28), pt7 != dt8)
    print("pydt8 != datetime7:".ljust(28), pt8 != dt7)
    print()


def pddt_demo() -> None:
    # fmt: off
    tzinfo = datetime.timezone(datetime.timedelta(hours=+3))
    s = pd.Series([gen_dt(i, 3, 10, 12, 0, 1, 1) for i in range(2023, 2250)])
    s = pd.Series([gen_dt(i, 3, 10, 12, 0, 1, 1, tzinfo=tzinfo) for i in range(2023, 2250)])

    # fmt: on
    pt = pddt(s)

    # Access
    print(" Access ".center(80, "="))
    print("pddt:\n", pt)
    print("dtiso:\n", pt.dtiso)
    print("dtisotz:\n", pt.dtisotz)
    print("date:\n", pt.date)
    print("dateiso:\n", pt.dateiso)
    print("time:\n", pt.time)
    print("timeiso:\n", pt.timeiso)
    print("timetz:\n", pt.timetz)
    print("timeisotz:\n", pt.timeisotz)
    print("dtpy:\n", pt.dtpy)
    print("dt64:\n", pt.dt64)
    print("ordinal:\n", pt.ordinal)
    print("seconds:\n", pt.seconds)
    print("seconds_utc:\n", pt.seconds_utc)
    print("microseconds:\n", pt.microseconds)
    print("microseconds_utc:\n", pt.microseconds_utc)
    print("timestamp:\n", pt.timestamp)
    print()

    # Absolute
    print(" Absolute ".center(80, "="))
    print("pddt:\n", pt)
    print("year:\n", pt.year)
    print("month:\n", pt.month)
    print("day:\n", pt.day)
    print("hour:\n", pt.hour)
    print("minute:\n", pt.minute)
    print("second:\n", pt.second)
    print("microsecond:\n", pt.microsecond)
    print("tzinfo:".ljust(15), pt.tzinfo)
    print()

    print(" Calendar ".center(80, "="))
    print("pddt:\n", pt)
    print("is_leapyear:\n", pt.is_leapyear)
    print("days_in_year:\n", pt.days_in_year)
    print("days_bf_year:\n", pt.days_bf_year)
    print("days_of_year:\n", pt.days_of_year)
    print("quarter:\n", i := pt.quarter, type(i))
    print("days_in_quarter:\n", i := pt.days_in_quarter, type(i))
    print("days_bf_quarter:\n", i := pt.days_bf_quarter, type(i))
    print("days_of_quarter:\n", i := pt.days_of_quarter, type(i))
    print("days_in_month:\n", pt.days_in_month)
    print("days_bf_month:\n", pt.days_bf_month)
    print("weekday:\n", pt.weekday)
    print("isoweekday:\n", pt.isoweekday)
    print("isoweek:\n", i := pt.isoweek, type(i))
    print("isoyear:\n", i := pt.isoyear, type(i))
    print("isocalendar:\n", pt.isocalendar)
    print()

    print(" Time manipulation ".center(80, "="))
    print("pddt:\n", pt)
    print("start_time:\n", pt.start_time)
    print("end_time:\n", pt.end_time)
    print()

    print(" Day manipulation ".center(80, "="))
    print("pddt:\n", pt)
    print("tomorrow:\n", pt.tomorrow)
    print("yesterday:\n", pt.yesterday)
    print()

    # Week manipulation
    print(" Week manipulation ".center(80, "="))
    print("pddt:\n", pt)
    print("monday:\n", pt.monday)
    print("tuesday:\n", pt.tuesday)
    print("wednesday:\n", pt.wednesday)
    print("thursday:\n", pt.thursday)
    print("friday:\n", pt.friday)
    print("saturday:\n", pt.saturday)
    print("sunday:\n", pt.sunday)
    print("curr_week:\n", pt.curr_week(None))
    print("curr_week:\n", pt.curr_week("Thu"))
    print("next_week:\n", pt.next_week("Thu"))
    print("next_week:\n", pt.next_week())
    print("last_week:\n", pt.last_week("mon"))
    print("last_week:\n", pt.last_week())
    print("to_week:\n", pt.to_week(2, "Wed"))
    print("to_week:\n", pt.to_week(2))
    print("to_week:\n", pt.to_week(-2, "Wed"))
    print("to_week:\n", pt.to_week(-2))
    print("is_weekday:\n", pt.is_weekday("fri"))
    print()

    # Month manipulation
    print(" Month manipulation ".center(80, "="))
    print("pddt:\n", pt)
    print("month_1st:\n", pt.month_1st)
    print("is_month_1st:\n", pt.month_1st.is_month_1st())
    print("month_lst:\n", pt.month_lst)
    print("is_month_lst:\n", pt.month_lst.is_month_lst())
    print("curr_month:\n", pt.curr_month(0))
    print("curr_month:\n", pt.curr_month(1))
    print("curr_month:\n", pt.curr_month(28))
    print("curr_month:\n", pt.curr_month(30))
    print("curr_month:\n", pt.curr_month(31))
    print("next_month:\n", pt.next_month())
    print("next_month:\n", pt.next_month(1))
    print("next_month:\n", pt.next_month(28))
    print("next_month:\n", pt.next_month(30))
    print("next_month:\n", pt.next_month(31))
    print("last_month:\n", pt.last_month())
    print("last_month:\n", pt.last_month(1))
    print("last_month:\n", pt.last_month(28))
    print("last_month:\n", pt.last_month(30))
    print("last_month:\n", pt.last_month(31))
    print("to_month:\n", pt.to_month(2))
    print("to_month:\n", pt.to_month(2, 1))
    print("to_month:\n", pt.to_month(2, 28))
    print("to_month:\n", pt.to_month(2, 30))
    print("to_month:\n", pt.to_month(2, 31))
    print("is_month:\n", pt.is_month("Mar"))
    print()

    # Quarter manipulation
    print(" Quarter manipulation ".center(80, "="))
    print("pddt:\n", pt)
    print("quarter_1st:\n", pt.quarter_1st)
    print("is_quarter_1st:\n", pt.quarter_1st.is_quarter_1st())
    print("quarter_lst:\n", pt.quarter_lst)
    print("is_quarter_lst:\n", pt.quarter_lst.is_quarter_lst())
    print("curr_quarter:\n", pt.curr_quarter(2))
    print("curr_quarter:\n", pt.curr_quarter(2, 1))
    print("curr_quarter:\n", pt.curr_quarter(2, 28))
    print("curr_quarter:\n", pt.curr_quarter(2, 30))
    print("curr_quarter:\n", pt.curr_quarter(2, 31))
    print("next_quarter:\n", pt.next_quarter(2))
    print("next_quarter:\n", pt.next_quarter(2, 1))
    print("next_quarter:\n", pt.next_quarter(2, 28))
    print("next_quarter:\n", pt.next_quarter(2, 30))
    print("next_quarter:\n", pt.next_quarter(2, 31))
    print("last_quarter:\n", pt.last_quarter(2))
    print("last_quarter:\n", pt.last_quarter(2, 1))
    print("last_quarter:\n", pt.last_quarter(2, 28))
    print("last_quarter:\n", pt.last_quarter(2, 30))
    print("last_quarter:\n", pt.last_quarter(2, 31))
    print("to_quarter:\n", pt.to_quarter(2, 2))
    print("to_quarter:\n", pt.to_quarter(2, 2, 1))
    print("to_quarter:\n", pt.to_quarter(2, 2, 28))
    print("to_quarter:\n", pt.to_quarter(2, 2, 30))
    print("to_quarter:\n", pt.to_quarter(2, 2, 31))
    print("is_quarter:\n", pt.is_quarter(1))
    print()

    # Year manipulation
    print(" Year manipulation ".center(80, "="))
    print("pddt:\n", pt)
    print("year_1st:\n", pt.year_1st)
    print("is_year_1st:\n", pt.year_1st.is_year_1st())
    print("year_lst:\n", pt.year_lst)
    print("is_year_lst:\n", pt.year_lst.is_year_lst())
    print("curr_year:\n", pt.curr_year())
    print("curr_year:\n", pt.curr_year(day=1))
    print("curr_year:\n", pt.curr_year(2))
    print("curr_year:\n", pt.curr_year(2, 1))
    print("curr_year:\n", pt.curr_year(2, 28))
    print("curr_year:\n", pt.curr_year(2, 30))
    print("curr_year:\n", pt.curr_year(2, 31))
    print("next_year:\n", pt.next_year())
    print("next_year:\n", pt.next_year(day=1))
    print("next_year:\n", pt.next_year(2))
    print("next_year:\n", pt.next_year(2, 1))
    print("next_year:\n", pt.next_year(2, 28))
    print("next_year:\n", pt.next_year(2, 30))
    print("next_year:\n", pt.next_year(2, 31))
    print("last_year:\n", pt.last_year())
    print("last_year:\n", pt.last_year(day=1))
    print("last_year:\n", pt.last_year(2))
    print("last_year:\n", pt.last_year(2, 1))
    print("last_year:\n", pt.last_year(2, 28))
    print("last_year:\n", pt.last_year(2, 30))
    print("last_year:\n", pt.last_year(2, 31))
    print("to_year:\n", pt.to_year(2))
    print("to_year:\n", pt.to_year(2, day=1))
    print("to_year:\n", pt.to_year(2, 2))
    print("to_year:\n", pt.to_year(2, 2, 1))
    print("to_year:\n", pt.to_year(2, 2, 28))
    print("to_year:\n", pt.to_year(2, 2, 30))
    print("to_year:\n", pt.to_year(2, 2, 31))
    print("is_year:\n", pt.is_year(2023))
    print()

    # Timezone manipulation
    print(" Timezone manipulation ".center(80, "="))
    print("pddt:\n", pt)
    print("tz_available:".ljust(15), pt.tz_available)
    print("tz_localize:\n", pt_n := pt.tz_localize(None))
    print("tz_localize:\n", pt_n.tz_localize("CET"))
    print("tz_convert:\n", pt_n.tz_localize("UTC").tz_convert("CET"))
    print("tz_convert:\n", pt_n.tz_localize("CET").tz_convert(None))
    print("tz_convert:\n", pt_n.tz_localize("CET").tz_convert("PST8PDT"))
    print("tz_switch:\n", pt_n.tz_switch("CET", "UTC"))
    print("tz_switch:\n", pt_n.tz_switch(None, "CET"))
    print("tz_switch:\n", pt_n.tz_localize("CET").tz_switch("PST8PDT"))

    # Frequency manipulation
    print(" Frequency manipulation ".center(80, "="))
    # fmt: off
    s = pd.Series([gen_dt(i, 3, 10, 12, 30, 30, 555555, tzinfo=tzinfo) for i in range(2023, 2250)])
    # fmt: on
    print("pddt:\n", ptx := pddt(s))
    print("round:\n", ptx.round("D"))
    print("round:\n", ptx.round("h"))
    print("round:\n", ptx.round("m"))
    print("round:\n", ptx.round("s"))
    print("round:\n", ptx.round("ms"))
    print("round:\n", ptx.round("us"))
    print("ceil:\n", ptx.ceil("D"))
    print("ceil:\n", ptx.ceil("h"))
    print("ceil:\n", ptx.ceil("m"))
    print("ceil:\n", ptx.ceil("s"))
    print("ceil:\n", ptx.ceil("ms"))
    print("ceil:\n", ptx.ceil("us"))
    print("floor:\n", ptx.floor("D"))
    print("floor:\n", ptx.floor("h"))
    print("floor:\n", ptx.floor("m"))
    print("floor:\n", ptx.floor("s"))
    print("floor:\n", ptx.floor("ms"))
    print("floor:\n", ptx.floor("us"))

    # Delta adjustment
    print(" Delta adjustment ".center(80, "="))
    print("pddt:\n", pt)
    print("delta1:\n", pt.delta(1, 1, 1, 1, 1, 1, 1, 1))
    print("delta2:\n", pt.delta(-1, -1, -1, -1, -1, -1, -1, -1))
    print("delta3:\n", pt.delta(1, -1, 1, -1, 1, -1, 1, -1))
    print("delta3:\n", pt.delta(-1, 1, -1, 1, -1, 1, -1, 1))
    print()

    # Replace adjustment
    print(" Replace adjustment ".center(80, "="))
    print("pddt:\n", pt)
    print("replace1:\n", pt.replace(2024, 2, 2, 2, 2, 2, 2, 2))
    print("replace2:\n", pt.replace(tzinfo=datetime.UTC))
    print()

    # Between calculation
    print(" Between calculation ".center(80, "="))
    print("pddt:\n", pt)
    print("other:\n", o := pt.delta(1, 1, 1, 1, 1, 1, 1, 1))
    print("Between Y:\n", pt.between(o, "Y"))
    print("Between M:\n", pt.between(o, "M"))
    print("Between W:\n", pt.between(o, "W"))
    print("Between D:\n", pt.between(o, "D"))
    print("Between h:\n", pt.between(o, "h"))
    print("Between m:\n", pt.between(o, "m"))
    print("Between s:\n", pt.between(o, "s"))
    print("Between ms:\n", pt.between(o, "ms"))
    print("Between us:\n", pt.between(o, "us"))
    print()

    # Special methods
    print(" Special methods".center(80, "="))
    print("pddt:\n", pt)
    print("len:".ljust(15), len(pt))
    print("contains".ljust(15), 1 in pt)
    print("copy:", a := id(pt), b := id(pt.copy()), a == b)
    print("array:\n", np.asarray(pt))
    print("iter:")
    [print(i) for i in pt]
    print()

    pt1 = pt.copy()
    pt2 = pt1.delta(1, 1, 1, 1, 1, 1, 1, 1)
    print("sub referance:\n", b := pt1.dt - pt2.dt, type(b))
    print("sub pddt - pddt:\n", x := pt1 - pt2, type(x), x.equals(b))
    print("sub pddt - series:\n", x := pt1 - pt2.dt, type(x), x.equals(b))
    print("sub series - pddt:\n", x := pt1.dt - pt2, type(x), x.equals(b))
    delta = pd.Timedelta(1, "D")
    print("sub referance:\n", b := pt1.dt - delta, type(b))
    print("sub pddt - delta:\n", x := pt1 - delta, type(x), x.equals(b))
    delta = pd.Series([datetime.timedelta(1) for _ in range(len(pt1))])
    print("sub referance:\n", b := pt1.dt - delta, type(b))
    print("sub pddt - series:\n", x := pt1 - delta, type(x), x.equals(b))
    print()

    delta = pd.Timedelta(1, "D")
    print("add referance:\n", b := pt1.dt + delta, type(b))
    print("add pddt + delta:\n", x := pt1 + delta, type(x), x.equals(b))
    print("add delta + pddt:\n", x := delta + pt1, type(x), x.equals(b))
    print()

    print("eq referance:\n", b := pt1.dt == pt1.dt, type(b))
    print("eq pddt == pddt:\n", x := pt1 == pt1, type(x), x.equals(b))
    print("eq pddt == series:\n", x := pt1 == pt1.dt, type(x), x.equals(b))
    print("eq series == pddt:\n", x := pt1.dt == pt1, type(x), x.equals(b))
    print()

    print("ne referance:\n", b := pt1.dt != pt1.dt, type(b))
    print("ne pddt != pddt:\n", x := pt1 != pt1, type(x), x.equals(b))
    print("ne pddt != series:\n", x := pt1 != pt1.dt, type(x), x.equals(b))
    print("ne series != pddt:\n", x := pt1.dt != pt1, type(x), x.equals(b))
    print()

    print("gt referance:\n", b := pt1.dt > pt2.dt, type(b))
    print("gt pddt > pddt:\n", x := pt1 > pt2, type(x), x.equals(b))
    print("gt pddt > series:\n", x := pt1 > pt2.dt, type(x), x.equals(b))
    print("gt series > pddt:\n", x := pt1.dt > pt2, type(x), x.equals(b))
    print()

    print("ge referance:\n", b := pt1.dt >= pt2.dt, type(b))
    print("ge pddt >= pddt:\n", x := pt1 >= pt2, type(x), x.equals(b))
    print("ge pddt >= series:\n", x := pt1 >= pt2.dt, type(x), x.equals(b))
    print("ge series >= pddt:\n", x := pt1.dt >= pt2, type(x), x.equals(b))
    print()

    print("lt referance:\n", b := pt1.dt < pt2.dt, type(b))
    print("lt pddt < pddt:\n", x := pt1 < pt2, type(x), x.equals(b))
    print("lt pddt < series:\n", x := pt1 < pt2.dt, type(x), x.equals(b))
    print("lt series < pddt:\n", x := pt1.dt < pt2, type(x), x.equals(b))
    print()

    print("le referance:\n", b := pt1.dt <= pt2.dt, type(b))
    print("le pddt <= pddt:\n", x := pt1 <= pt2, type(x), x.equals(b))
    print("le pddt <= series:\n", x := pt1 <= pt2.dt, type(x), x.equals(b))
    print("le series <= pddt:\n", x := pt1.dt <= pt2, type(x), x.equals(b))
    print()

    # Difference time unit
    print(" Frequency manipulation ".center(80, "="))
    # fmt: off
    s = pd.Series([gen_dt(i, 3, 10, 12, 30, 30, 555555) for i in range(2023, 2250)])
    s = s.astype("<M8[s]")
    print("Reference:\n", s)
    print("Series[s]:\n", pddt(s))
    s = s.dt.tz_localize(tzinfo)
    print("Series[s+tz]:\n", pddt(s))
    # fmt: on


if __name__ == "__main__":
    # validate_cytimedelta_relative()
    # validate_cytimedelta_absolute()
    # cytimedelta_performance()
    # cytimedelta_demo()
    # cyparser_validate()
    # cyparser_performance()
    # pydt_demo()
    # pddt_demo()

    from cytimes.cydatetime import test

    print(test(1000))
