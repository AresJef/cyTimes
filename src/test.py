from timeit import timeit
from dateutil.parser import parserinfo
import warnings

warnings.filterwarnings("ignore")


class CustomParserInfo(parserinfo):
    # m from a.m/p.m, t from ISO T separator
    # fmt: off
    PERTAIN = ["of"]
    JUMP = [
        " ", ".", ",", ";", "-", "/", "'",
        "at", "on", "and", "ad", "m", "t", "of", 
        "st", "nd", "rd", "th", "年" ,"月", "日" ]
    UTCZONE = ["UTC", "GMT", "Z", "z"]
    MONTHS = [("Jan", "January"),
            ("Feb", "February"),      # TODO: "Febr"
            ("Mar", "March"),
            ("Apr", "April"),
            ("May", "May"),
            ("Jun", "June"),
            ("Jul", "July"),
            ("Aug", "August"),
            ("Sep", "Sept", "September"),
            ("Oct", "October"),
            ("Nov", "November"),
            ("Dec", "December")]
    WEEKDAYS = [("Mon", "Monday"),
                ("Tue", "Tuesday"),     # TODO: "Tues"
                ("Wed", "Wednesday"),
                ("Thu", "Thursday"),    # TODO: "Thurs"
                ("Fri", "Friday"),
                ("Sat", "Saturday"),
                ("Sun", "Sunday")]
    HMS = [("h", "hour", "hours"),
        ("m", "minute", "minutes"),
        ("s", "second", "seconds")]
    AMPM = [("am", "a"),
            ("pm", "p")]
    # fmt: on


def diff(base_t: float, comp_t: float) -> str:
    """Calculate performance difference."""
    if base_t < comp_t:
        res = (comp_t - base_t) / base_t
    else:
        res = -(base_t - comp_t) / comp_t
    return ("" if res < 0 else "+") + f"{res:.6f}x"


def gen_test_datetimes() -> str:
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
        "2023-08-01 12:00:00.000000-12:00",
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
        # "Thu Sep 25 10:36:28 2003",
        # "Thu Sep 25 2003",
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
        "1996.07.10 AD at 15:08:56",
        "12, 1952 AD 3:30:42pm",
        "November 5, 1994, 8:15:30 am EST+23:59",
        "November 5, 1994, 8:15:30 am EST",
        "November 5, 1994, 8:15:30 am +23:59(EST)",
        "November 5, 1994, 8:15:30 am +23:59 (EST)",
        "1994-11-05T08:15:30-05:00",
        "1994-11-05T08:15:30Z",
        "1976-07-04T00:01:02Z",
        "1986-07-05T08:15:30z",
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
        "0:00 PM",
        "5:50 A.M. on June 13, 1990",
        "April 2009",
        "00:11:25.01",
        "00:12:10.01",
        "090107",
        "2015 09 25",
        "2015-15-May",
        "02:17NOV2017",
        "2023年08月01",
        "2023年08月01日 12时00分00秒",
        "Today is 25 of September of 2003, exactly at 10:49:41 with timezone -03:00.",
        "I have a meeting on March 1, 1974.",
        "On June 8th, 2020, I am going to be the first man on Mars",
        "Meet me at the AM/PM on Sunset at 3:00 PM on December 3rd, 2003",
        "Meet me at 3:00AM on December 3rd, 2003 at the AM/PM on Sunset",
        "Jan 29, 1945 14:45 AM I going to see you there?",
        "2017-07-17 06:15:",
    ]


def test_cytimedelta(rounds: int) -> None:
    from typing import Literal
    from random import randint
    from datetime import datetime, date, timedelta
    from dateutil.relativedelta import relativedelta
    from cytimes.cydelta import cytimedelta

    def test_add(
        obj: object,
        mode: Literal["Relative", "Absolute", "Mixed"],
        rounds: int,
    ) -> None:
        print(f"* Addition with {type(obj)} [{mode}]".ljust(100, "-"))
        print(f"object:\t\t\t{obj} {type(obj)}")
        for _ in range(rounds):
            # Random args
            years = randint(-1000, 1000)
            months = randint(-1000, 1000)
            days = randint(-1000, 1000)
            weeks = randint(-1000, 1000)
            hours = randint(-1000, 1000)
            minutes = randint(-1000, 1000)
            seconds = randint(-1000, 1000)
            microseconds = randint(-1_000_000, 1_000_000)
            year = randint(3000, 6000)
            month = randint(1, 12)
            day = randint(1, 28)
            hour = randint(0, 23)
            minute = randint(0, 59)
            second = randint(0, 59)
            microsecond = randint(0, 999_999)
            if mode == "Relative":
                year, month, day = -1, -1, -1
                hour, minute, second, microsecond = -1, -1, -1, -1
            elif mode == "Absolute":
                years, months, days, weeks = 0, 0, 0, 0
                hours, minutes, seconds, microseconds = 0, 0, 0, 0
            # cytimedelta
            ctd = cytimedelta(
                years=years,
                months=months,
                days=days,
                weeks=weeks,
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                microseconds=microseconds,
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=second,
                microsecond=microsecond,
            )
            cval = ctd + obj
            # relativedelta
            rtd = relativedelta(
                years=years,
                months=months,
                days=days,
                weeks=weeks,
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                microseconds=microseconds,
                year=None if year == -1 else year,
                month=None if month == -1 else month,
                day=None if day == -1 else day,
                hour=None if hour == -1 else hour,
                minute=None if minute == -1 else minute,
                second=None if second == -1 else second,
                microsecond=None if microsecond == -1 else microsecond,
            )
            try:
                rval = rtd + obj

            except Exception:
                rtd_compatible = False
            else:
                rtd_compatible = False
                # comparison
                eq_tdl = ctd == rtd
                if type(rval) is date:
                    eq_val = cval.date() == rval
                else:
                    eq_val = cval == rval
                eq = eq_tdl and eq_val
                if not eq_val:
                    print(f"*cytimedelta:\t\t{ctd}")
                    print(f"*relativedelta:\t\t{rtd}")
                    print(f"- Equal delta:\t\t{eq_tdl}")
                    print(f"*cytimedelta result:\t{cval}")
                    print(f"*relativedelta result:\t{rval}")
                    print(f"- Equal result:\t\t{eq_val}")
                    assert eq

        # Test Performance
        ctd_t = timeit(lambda: ctd + obj, number=rounds)
        print("*cytimedelta perf:\t", ctd_t)
        if rtd_compatible:
            rtd_t = timeit(lambda: rtd + obj, number=rounds)
            print("*relativedelta perf:\t", rtd_t, diff(ctd_t, rtd_t))

        # Test Finished
        print("- Test Complete & All Equals".ljust(100))

    def test_sub(
        obj: object,
        mode: Literal["Relative", "Absolute", "Mixed"],
        rounds: int,
    ) -> None:
        print(f"* Substraction (Left) with {type(obj)} [{mode}]".ljust(100, "-"))
        print(f"object:\t\t\t{obj} {type(obj)}")
        for _ in range(rounds):
            # Random args
            years = randint(-1000, 1000)
            months = randint(-1000, 1000)
            days = randint(-1000, 1000)
            weeks = randint(-1000, 1000)
            hours = randint(-1000, 1000)
            minutes = randint(-1000, 1000)
            seconds = randint(-1000, 1000)
            microseconds = randint(-1_000_000, 1_000_000)
            year = randint(3000, 6000)
            month = randint(1, 12)
            day = randint(1, 28)
            hour = randint(0, 23)
            minute = randint(0, 59)
            second = randint(0, 59)
            microsecond = randint(0, 999_999)
            if mode == "Relative":
                year, month, day = -1, -1, -1
                hour, minute, second, microsecond = -1, -1, -1, -1
            elif mode == "Absolute":
                years, months, days, weeks = 0, 0, 0, 0
                hours, minutes, seconds, microseconds = 0, 0, 0, 0
            # cytimedelta
            ctd = cytimedelta(
                years=years,
                months=months,
                days=days,
                weeks=weeks,
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                microseconds=microseconds,
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=second,
                microsecond=microsecond,
            )
            cval = ctd - obj
            # relativedelta
            rtd = relativedelta(
                years=years,
                months=months,
                days=days,
                weeks=weeks,
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                microseconds=microseconds,
                year=None if year == -1 else year,
                month=None if month == -1 else month,
                day=None if day == -1 else day,
                hour=None if hour == -1 else hour,
                minute=None if minute == -1 else minute,
                second=None if second == -1 else second,
                microsecond=None if microsecond == -1 else microsecond,
            )
            try:
                rval = rtd - obj
                rtd_compatible = True
            except Exception:
                rtd_compatible = False
            else:
                # comparison
                eq_tdl = ctd == rtd
                eq_val = cval == rval
                eq = eq_tdl and eq_val
                if not eq_val:
                    print(f"*cytimedelta:\t\t{ctd}")
                    print(f"*relativedelta:\t\t{rtd}")
                    print(f"- Equal delta:\t\t{eq_tdl}")
                    print(f"*cytimedelta result:\t{cval}")
                    print(f"*relativedelta result:\t{rval}")
                    print(f"- Equal result:\t\t{eq_val}")
                    assert eq

        # Test Performance
        ctd_t = timeit(lambda: ctd + obj, number=rounds)
        print("*cytimedelta perf:\t", ctd_t)
        if rtd_compatible:
            rtd_t = timeit(lambda: rtd + obj, number=rounds)
            print("*relativedelta perf:\t", rtd_t, diff(ctd_t, rtd_t))

        # Test Finished
        print("- Test Complete & All Equals".ljust(100))

    def test_rsub(
        obj: object,
        mode: Literal["Relative", "Absolute", "Mixed"],
        rounds: int,
    ) -> None:
        print(f"* Substraction (Right) with {type(obj)} [{mode}]".ljust(100, "-"))
        print(f"object:\t\t\t{obj} {type(obj)}")
        for _ in range(rounds):
            # Random args
            years = randint(-1000, 1000)
            months = randint(-1000, 1000)
            days = randint(-1000, 1000)
            weeks = randint(-1000, 1000)
            hours = randint(-1000, 1000)
            minutes = randint(-1000, 1000)
            seconds = randint(-1000, 1000)
            microseconds = randint(-1_000_000, 1_000_000)
            year = randint(3000, 6000)
            month = randint(1, 12)
            day = randint(1, 28)
            hour = randint(0, 23)
            minute = randint(0, 59)
            second = randint(0, 59)
            microsecond = randint(0, 999_999)
            if mode == "Relative":
                year, month, day = -1, -1, -1
                hour, minute, second, microsecond = -1, -1, -1, -1
            elif mode == "Absolute":
                years, months, days, weeks = 0, 0, 0, 0
                hours, minutes, seconds, microseconds = 0, 0, 0, 0
            # cytimedelta
            ctd = cytimedelta(
                years=years,
                months=months,
                days=days,
                weeks=weeks,
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                microseconds=microseconds,
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=second,
                microsecond=microsecond,
            )
            cval = obj - ctd
            # relativedelta
            rtd = relativedelta(
                years=years,
                months=months,
                days=days,
                weeks=weeks,
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                microseconds=microseconds,
                year=None if year == -1 else year,
                month=None if month == -1 else month,
                day=None if day == -1 else day,
                hour=None if hour == -1 else hour,
                minute=None if minute == -1 else minute,
                second=None if second == -1 else second,
                microsecond=None if microsecond == -1 else microsecond,
            )
            try:
                rval = obj - rtd
                rtd_compatible = True
            except Exception:
                rtd_compatible = False
            else:
                # comparison
                eq_tdl = ctd == rtd
                if type(rval) is date:
                    eq_val = cval.date() == rval
                else:
                    eq_val = cval == rval
                eq = eq_tdl and eq_val
                if not eq_val:
                    print(f"*cytimedelta:\t\t{ctd}")
                    print(f"*relativedelta:\t\t{rtd}")
                    print(f"- Equal delta:\t\t{eq_tdl}")
                    print(f"*cytimedelta result:\t{cval}")
                    print(f"*relativedelta result:\t{rval}")
                    print(f"- Equal result:\t\t{eq_val}")
                    assert eq

        # Test Performance
        ctd_t = timeit(lambda: ctd + obj, number=rounds)
        print("*cytimedelta perf:\t", ctd_t)
        if rtd_compatible:
            rtd_t = timeit(lambda: rtd + obj, number=rounds)
            print("*relativedelta perf:\t", rtd_t, diff(ctd_t, rtd_t))

        # Test Finished
        print("- Test Complete & All Equals".ljust(100))

    run_date = 1
    run_datetime = 1
    run_timedelta = 1
    run_cytimedelta = 1
    run_relativedelta = 1
    run_numeric = 1

    # Addition & Substraction with datetime.date
    if run_date:
        obj = date(2023, 1, 1)
        test_add(obj, "Relative", rounds)
        test_add(obj, "Absolute", rounds)
        test_add(obj, "Mixed", rounds)
        test_rsub(obj, "Relative", rounds)
        test_rsub(obj, "Absolute", rounds)
        test_rsub(obj, "Mixed", rounds)

    # Addition & Substraction with datetime.datetime
    if run_datetime:
        obj = datetime(2023, 12, 31, 23, 59, 59, 999999)
        test_add(obj, "Relative", rounds)
        test_add(obj, "Absolute", rounds)
        test_add(obj, "Mixed", rounds)
        test_rsub(obj, "Relative", rounds)
        test_rsub(obj, "Absolute", rounds)
        test_rsub(obj, "Mixed", rounds)

    # Addition & Substraction with timedelta
    if run_timedelta:
        obj = timedelta(1, 1, 1)
        test_add(obj, "Relative", rounds)
        test_add(obj, "Absolute", rounds)
        test_add(obj, "Mixed", rounds)
        test_sub(obj, "Relative", rounds)
        test_sub(obj, "Absolute", rounds)
        test_sub(obj, "Mixed", rounds)
        test_rsub(obj, "Relative", rounds)
        test_rsub(obj, "Absolute", rounds)
        test_rsub(obj, "Mixed", rounds)

    # Addition & Substraction with cytimedelta
    if run_cytimedelta:
        obj = cytimedelta(
            years=2023,
            months=12,
            days=31,
            hours=23,
            minutes=59,
            seconds=59,
            microseconds=999999,
        )
        test_add(obj, "Relative", rounds)
        test_add(obj, "Absolute", rounds)
        test_add(obj, "Mixed", rounds)
        test_sub(obj, "Relative", rounds)
        test_sub(obj, "Absolute", rounds)
        test_sub(obj, "Mixed", rounds)
        obj = cytimedelta(
            years=2023,
            months=12,
            days=31,
            hours=23,
            minutes=59,
            seconds=59,
            microseconds=999999,
            year=1970,
            month=12,
            day=31,
            hour=23,
            minute=59,
            second=59,
            microsecond=999999,
        )
        test_add(obj, "Relative", rounds)
        test_add(obj, "Absolute", rounds)
        test_add(obj, "Mixed", rounds)
        test_sub(obj, "Relative", rounds)
        test_sub(obj, "Absolute", rounds)
        test_sub(obj, "Mixed", rounds)

    # Addition & Substraction with relativedelta
    if run_relativedelta:
        obj = relativedelta(
            years=2023,
            months=12,
            days=31,
            hours=23,
            minutes=59,
            seconds=59,
            microseconds=999999,
        )
        test_add(obj, "Relative", rounds)
        test_add(obj, "Absolute", rounds)
        test_add(obj, "Mixed", rounds)
        test_sub(obj, "Relative", rounds)
        test_sub(obj, "Absolute", rounds)
        test_sub(obj, "Mixed", rounds)
        test_rsub(obj, "Relative", rounds)
        test_rsub(obj, "Absolute", rounds)
        test_rsub(obj, "Mixed", rounds)
        obj = relativedelta(
            years=2023,
            months=12,
            days=31,
            hours=23,
            minutes=59,
            seconds=59,
            microseconds=999999,
            year=1970,
            month=12,
            day=31,
            hour=23,
            minute=59,
            second=59,
            microsecond=999999,
        )
        test_add(obj, "Relative", rounds)
        test_add(obj, "Absolute", rounds)
        test_add(obj, "Mixed", rounds)
        test_sub(obj, "Relative", rounds)
        test_sub(obj, "Absolute", rounds)
        test_sub(obj, "Mixed", rounds)
        test_rsub(obj, "Relative", rounds)
        test_rsub(obj, "Absolute", rounds)
        test_rsub(obj, "Mixed", rounds)

    # Numeric
    if run_numeric:
        years = -2024
        months = -8
        days = -16
        hours = -20
        minutes = -30
        seconds = -30
        microseconds = -666666
        ctd = cytimedelta(
            years=years,
            months=months,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            microseconds=microseconds,
        )

        obj = 2
        print(f"* Addition with {type(obj)} ".ljust(100, "-"))
        print("cytimedelta:\t\t", ctd)
        print("value:\t\t\t", obj, type(obj))
        res1 = ctd + obj
        print("cytimedelta (Left):\t", res1)
        res2 = obj + ctd
        print("cytimedelta (Right):\t", res1)
        success = (
            res1
            == res2
            == cytimedelta(
                years=years + obj,
                months=months + obj,
                days=days + obj,
                hours=hours + obj,
                minutes=minutes + obj,
                seconds=seconds + obj,
                microseconds=microseconds + obj,
            )
        )
        print("- EQUAL:\t\t", success)
        assert success

        obj = 1.2
        print(f"* Addition with {type(obj)} ".ljust(100, "-"))
        print("cytimedelta:\t\t", ctd)
        print("value:\t\t\t", obj, type(obj))
        res1 = ctd + obj
        print("cytimedelta (Left):\t", res1)
        res2 = obj + ctd
        print("cytimedelta (Right):\t", res1)
        success = res1 == res2
        print("- EQUAL:\t\t", success)
        assert success

        obj = 2
        print(f"* Substraction with {type(obj)} ".ljust(100, "-"))
        print("cytimedelta:\t\t", ctd)
        print("value:\t\t\t", obj, type(obj))
        res1 = ctd - obj
        print("cytimedelta (Left):\t", res1)
        res2 = obj - ctd
        print("cytimedelta (Right):\t", res2)
        success = (
            res1
            == -res2
            == cytimedelta(
                years=years - obj,
                months=months - obj,
                days=days - obj,
                hours=hours - obj,
                minutes=minutes - obj,
                seconds=seconds - obj,
                microseconds=microseconds - obj,
            )
        )
        print("- EQUAL:\t\t", success)
        assert success

        obj = 1.2
        print(f"* Substraction with {type(obj)} ".ljust(100, "-"))
        print("cytimedelta:\t\t", ctd)
        print("value:\t\t\t", obj, type(obj))
        res1 = ctd - obj
        print("cytimedelta (Left):\t", res1)
        res2 = obj - ctd
        print("cytimedelta (Right):\t", res2)
        success = res1 == -res2
        print("- EQUAL:\t\t", success)
        assert success

        obj = 2
        print(f"* Multiplication with {type(obj)} ".ljust(100, "-"))
        print("cytimedelta:\t\t", ctd)
        print("value:\t\t\t", obj, type(obj))
        res1 = ctd * obj
        print("cytimedelta (Left):\t", res1)
        res2 = obj * ctd
        print("cytimedelta (Right):\t", res2)
        success = (
            res1
            == res2
            == cytimedelta(
                years=years * obj,
                months=months * obj,
                days=days * obj,
                hours=hours * obj,
                minutes=minutes * obj,
                seconds=seconds * obj,
                microseconds=microseconds * obj,
            )
        )
        print("- EQUAL:\t\t", success)

        obj = 1.2
        print(f"* Multiplication with {type(obj)} ".ljust(100, "-"))
        print("cytimedelta:\t\t", ctd)
        print("value:\t\t\t", obj, type(obj))
        res1 = ctd * obj
        print("cytimedelta (Left):\t", res1)
        res2 = obj * ctd
        print("cytimedelta (Right):\t", res2)
        success = res1 == res2
        print("- EQUAL:\t\t", success)

        obj = 2
        print(f"* Division with {type(obj)} ".ljust(100, "-"))
        print("cytimedelta:\t\t", ctd)
        print("value:\t\t\t", obj, type(obj))
        res1 = ctd / obj
        print("cytimedelta (Right):\t", res1)
        success = res1 == cytimedelta(
            years=years / obj,
            months=months / obj,
            days=days / obj,
            hours=hours / obj,
            minutes=minutes / obj,
            seconds=seconds / obj,
            microseconds=microseconds / obj,
        )
        print("- EQUAL:\t\t", success)

        obj = 1.2
        print(f"* Division with {type(obj)} ".ljust(100, "-"))
        print("cytimedelta:\t\t", ctd)
        print("value:\t\t\t", obj, type(obj))
        res1 = ctd / obj
        print("cytimedelta (Right):\t", res1)

        print(f"* Negation ".ljust(100, "-"))
        print("cytimedelta:\t\t", ctd)
        res1 = -ctd
        print("cytimedelta (Negate):\t", res1)
        success = res1 == cytimedelta(
            years=-years,
            months=-months,
            days=-days,
            hours=-hours,
            minutes=-minutes,
            seconds=-seconds,
            microseconds=-microseconds,
        )
        print("- EQUAL:\t\t", success)
        assert success

        print(f"* Absolute ".ljust(100, "-"))
        print("cytimedelta:\t\t", ctd)
        res1 = abs(ctd)
        print("cytimedelta (Absolute):\t", res1)
        success = res1 == cytimedelta(
            years=abs(years),
            months=abs(months),
            days=abs(days),
            hours=abs(hours),
            minutes=abs(minutes),
            seconds=abs(seconds),
            microseconds=abs(microseconds),
        )
        print("- EQUAL:\t\t", success)
        assert success

        print(f"* Boolean ".ljust(100, "-"))
        print("cytimedelta:\t\t", ctd)
        res1 = bool(ctd)
        print("cytimedelta (Boolean):\t", res1)
        ctd0 = cytimedelta()
        print("cytimedelta:\t\t", ctd0)
        res2 = bool(ctd0)
        print("cytimedelta (Boolean):\t", res2)
        success = res1 != res2
        print("- EQUAL:\t\t", success)
        assert success

        print(f"* Hash ".ljust(100, "-"))
        print("cytimedelta:\t\t", ctd)
        res1 = hash(ctd)
        print("cytimedelta (Hash):\t", res1)


def test_parser(rounds: int) -> None:
    from cytimes.cyparser import Config
    from dateutil.parser import parserinfo
    from cytimes.cyparser import Parser
    from dateutil.parser import parser
    from datetime import datetime

    print(" Default <'Config'> ".center(80, "-"))
    cfg = Config()
    print(cfg)
    print()
    print(" Import <'Conifg'> ".center(80, "-"))
    info = parserinfo()
    cfg = Config.from_parserinfo(info)
    print(cfg)
    print()

    info = CustomParserInfo()
    default = datetime(1970, 1, 1)

    for text in gen_test_datetimes():
        # fmt: off
        print("-" * 80)
        print(f"Parse text:\t{repr(text)}")
        res1 = Parser().parse(text, default)
        print(f"- cytimes:\t{res1}")
        res2 = parser(info).parse(text, default, fuzzy=True)
        print(f"- dateutil:\t{res2}")
        print(f"- EQUAL:\t{res1 == res2}")
        assert res2 == res1

        print("Performance:")
        tim1 = timeit(lambda: Parser().parse(text, default), number=rounds)
        print(f"- cytimes:\t{tim1}s")
        tim2 = timeit(lambda: parser(info).parse(text, default, fuzzy=True), number=rounds)
        print(f"- dateutil:\t{tim2}s\t{diff(tim1, tim2)}")
        print()
        # fmt: on


def test_pydt_pddt(unit: str) -> None:
    import pandas as pd
    from datetime import timedelta
    from cytimes.pydatetime import pydt
    from cytimes.pddatetime import pddt

    pd.options.display.max_rows = 4

    def print_1line(msg: object, value: object, pad: int = 50) -> None:
        print(str(msg).ljust(pad), ":", str(value), sep="")

    def compare_value(
        msg: object,
        pydt_val: object,
        pddt_val: object,
        pad: int = 50,
        verify: bool = True,
    ) -> None:
        print(f"{'#' * pad} {msg}:")
        print(f"=> pddt: {type(pddt_val)}\n", pddt_val, sep="")
        print(f"=> pydt:\t{pydt_val}\t{type(pydt_val)}")
        try:
            equal = pddt_val[len(pddt_val) - 1] == pydt_val
        except TypeError:
            equal = pddt_val == pydt_val
        print("=> EQUAL:\t", equal, "\t", sep="")
        if verify:
            assert equal == True

    # Datetimes
    dts = []
    for year in range(2000, 2025 + 1):
        for month in range(1, 12 + 1):
            dts.append("%d-%02d-02 03:04:05.000006" % (year, month))
    # other to compare delta
    odt = "2100-01-01 04:05:06.000007"
    otd = timedelta(1, 1, 1, 1)

    # Create
    # fmt: off
    print("Create".center(80, "-"))
    print_1line("cls.now()", pydt.now())
    print_1line("cls.now('UTC')", pydt.now("UTC"))
    print_1line("cls.now('CET')", pydt.now("CET"))
    compare_value("cls.from_datetime(2023, 1, 1)", 
                  pydt.from_datetime(2023, 1, 1), pddt.from_datetime(10, 2023, 1, 1))
    compare_value("cls.from_datetime(2023, 1, 1, tz='CET')", 
                  pydt.from_datetime(2023, 1, 1, tz='CET'), pddt.from_datetime(10, 2023, 1, 1, tz='CET'))
    print_1line("cls.from_ordinal(1)", pydt.from_ordinal(1))
    print_1line("cls.from_ordinal(1, tz='CET')", pydt.from_ordinal(1, tz="CET"))
    print_1line("cls.from_timestamp(1)", pydt.from_timestamp(1))
    print_1line("cls.from_timestamp(1, tz='CET')", pydt.from_timestamp(1, tz="CET"))
    print_1line("cls.from_seconds(1)", pydt.from_seconds(1))
    print_1line("cls.from_seconds(1, tz='CET')", pydt.from_seconds(1, tz="CET"))
    print_1line("cls.from_microseconds(1)", pydt.from_microseconds(1))
    print_1line("cls.from_microseconds(1, tz='CET')", pydt.from_microseconds(1, tz="CET"))
    pyt = pydt(dts[-1])
    if unit in ("s", "ms"):
        verify = False
        pyt = pyt.replace(microsecond=0)
    else:
        verify = True
    pdt = pddt(dts, name="pddt", unit=unit)
    compare_value("original", pyt, pdt)
    # fmt: on

    # Access
    print(" Access ".center(80, "-"))
    compare_value("year", pyt.year, pdt.year)
    compare_value("quarter", pyt.quarter, pdt.quarter)
    compare_value("month", pyt.month, pdt.month)
    compare_value("day", pyt.day, pdt.day)
    compare_value("hour", pyt.hour, pdt.hour)
    compare_value("minute", pyt.minute, pdt.minute)
    compare_value("second", pyt.second, pdt.second)
    compare_value("microsecond", pyt.microsecond, pdt.microsecond)
    compare_value("tzinfo", pyt.tzinfo, pdt.tzinfo)
    print_1line("pydt.fold", pyt.fold)
    compare_value("dt", pyt.dt, pdt.dt)
    compare_value("dt_str", pyt.dt_str, pdt.dt_str, verify=verify)
    compare_value("dt_iso", pyt.dt_iso, pdt.dt_iso, verify=verify)
    compare_value("dt_isotz", pyt.dt_isotz, pdt.dt_isotz, verify=verify)
    compare_value("date", pyt.date, pdt.date)
    compare_value("date_iso", pyt.date_iso, pdt.date_iso)
    compare_value("time", pyt.time, pdt.time)
    compare_value("time_tz", pyt.timetz, pdt.timetz)
    compare_value("time_iso", pyt.time_iso, pdt.time_iso, verify=verify)
    compare_value("dt64", pyt.dt64, pdt.dt64)
    compare_value("ordinal", pyt.ordinal, pdt.ordinal)
    compare_value("seconds", pyt.seconds, pdt.seconds, verify=False)
    compare_value("seconds_utc", pyt.seconds_utc, pdt.seconds_utc, verify=False)
    compare_value("microseconds", pyt.microseconds, pdt.microseconds)
    compare_value("microseconds_utc", pyt.microseconds_utc, pdt.microseconds_utc)
    compare_value("timestamp", pyt.timestamp, pdt.timestamp, verify=False)
    print()

    # Year
    # fmt: off
    print(" Year ".center(80, "-"))
    compare_value("is_leapyear", pyt.is_leapyear(), pdt.is_leapyear())
    compare_value("leap_bt_years(2024)", pyt.leap_bt_years(2024), pdt.leap_bt_years(2024))
    compare_value("days_in_year", pyt.days_in_year, pdt.days_in_year)
    compare_value("days_bf_year", pyt.days_bf_year, pdt.days_bf_year)
    compare_value("days_of_year", pyt.days_of_year, pdt.days_of_year)
    compare_value("is_year(1999)", pyt.is_year(2300), pdt.is_year(2300))
    compare_value("is_year_1st", pyt.is_year_1st(), pdt.is_year_1st())
    compare_value("is_year_lst", pyt.is_year_lst(), pdt.is_year_lst())
    compare_value("to_year_1st", pyt.to_year_1st(), pdt.to_year_1st())
    compare_value("to_year_lst", pyt.to_year_lst(), pdt.to_year_lst())
    compare_value("to_curr_year('Feb, 0)", pyt.to_curr_year("Feb", 0), pdt.to_curr_year("Feb", 0))
    compare_value("to_curr_year('Feb, 1)", pyt.to_curr_year("Feb", 1), pdt.to_curr_year("Feb", 1))
    compare_value("to_curr_year('Feb, 28)", pyt.to_curr_year("Feb", 28), pdt.to_curr_year("Feb", 28))
    compare_value("to_curr_year('Feb, 30)", pyt.to_curr_year("Feb", 30), pdt.to_curr_year("Feb", 30))
    compare_value("to_curr_year('Feb, 100)", pyt.to_curr_year("Feb", 100), pdt.to_curr_year("Feb", 100))
    compare_value("to_next_year('Feb, 0)", pyt.to_next_year("Feb", 0), pdt.to_next_year("Feb", 0))
    compare_value("to_next_year('Feb, 1)", pyt.to_next_year("Feb", 1), pdt.to_next_year("Feb", 1))
    compare_value("to_next_year('Feb, 28)", pyt.to_next_year("Feb", 28), pdt.to_next_year("Feb", 28))
    compare_value("to_next_year('Feb, 30)", pyt.to_next_year("Feb", 30), pdt.to_next_year("Feb", 30))
    compare_value("to_next_year('Feb, 100)", pyt.to_next_year("Feb", 100), pdt.to_next_year("Feb", 100))
    compare_value("to_prev_year('Feb, 0)", pyt.to_prev_year("Feb", 0), pdt.to_prev_year("Feb", 0))
    compare_value("to_prev_year('Feb, 1)", pyt.to_prev_year("Feb", 1), pdt.to_prev_year("Feb", 1))
    compare_value("to_prev_year('Feb, 28)", pyt.to_prev_year("Feb", 28), pdt.to_prev_year("Feb", 28))
    compare_value("to_prev_year('Feb, 30)", pyt.to_prev_year("Feb", 30), pdt.to_prev_year("Feb", 30))
    compare_value("to_prev_year('Feb, 100)", pyt.to_prev_year("Feb", 100), pdt.to_prev_year("Feb", 100))
    compare_value("to_year(-2, 'Feb, 0)", pyt.to_year(-2, "Feb", 0), pdt.to_year(-2, "Feb", 0))
    compare_value("to_year(-2, 'Feb, 1)", pyt.to_year(-2, "Feb", 1), pdt.to_year(-2, "Feb", 1))
    compare_value("to_year(-2, 'Feb, 28)", pyt.to_year(-2, "Feb", 28), pdt.to_year(-2, "Feb", 28))
    compare_value("to_year(-2, 'Feb, 30)", pyt.to_year(-2, "Feb", 30), pdt.to_year(-2, "Feb", 30))
    compare_value("to_year(-2, 'Feb, 100)", pyt.to_year(-2, "Feb", 100), pdt.to_year(-2, "Feb", 100))
    compare_value("to_year(2, 'Feb, 0)", pyt.to_year(2, "Feb", 0), pdt.to_year(2, "Feb", 0))
    compare_value("to_year(2, 'Feb, 1)", pyt.to_year(2, "Feb", 1), pdt.to_year(2, "Feb", 1))
    compare_value("to_year(2, 'Feb, 28)", pyt.to_year(2, "Feb", 28), pdt.to_year(2, "Feb", 28))
    compare_value("to_year(2, 'Feb, 30)", pyt.to_year(2, "Feb", 30), pdt.to_year(2, "Feb", 30))
    compare_value("to_year(2, 'Feb, 100)", pyt.to_year(2, "Feb", 100), pdt.to_year(2, "Feb", 100))
    print()
    # fmt: on

    # Quarter
    # fmt: off
    print(" Quarter ".center(80, "-"))
    compare_value("days_in_quarter", pyt.days_in_quarter, pdt.days_in_quarter)
    compare_value("days_bf_quarter", pyt.days_bf_quarter, pdt.days_bf_quarter)
    compare_value("days_of_quarter", pyt.days_of_quarter, pdt.days_of_quarter)
    compare_value("quarter_1st_month", pyt.quarter_1st_month, pdt.quarter_1st_month)
    compare_value("quarter_lst_month", pyt.quarter_lst_month, pdt.quarter_lst_month)
    compare_value("is_quarter(1)", pyt.is_quarter(1), pdt.is_quarter(1))
    compare_value("is_quarter_1st", pyt.is_quarter_1st(), pdt.is_quarter_1st())
    compare_value("is_quarter_lst", pyt.is_quarter_lst(), pdt.is_quarter_lst())
    compare_value("to_quarter_1st", pyt.to_quarter_1st(), pdt.to_quarter_1st())
    compare_value("to_quarter_lst", pyt.to_quarter_lst(), pdt.to_quarter_lst())
    compare_value("to_curr_quarter(2, 0)", pyt.to_curr_quarter(2, 0), pdt.to_curr_quarter(2, 0))
    compare_value("to_curr_quarter(2, 1)", pyt.to_curr_quarter(2, 1), pdt.to_curr_quarter(2, 1))
    compare_value("to_curr_quarter(2, 28)", pyt.to_curr_quarter(2, 28), pdt.to_curr_quarter(2, 28))
    compare_value("to_curr_quarter(2, 30)", pyt.to_curr_quarter(2, 30), pdt.to_curr_quarter(2, 30))
    compare_value("to_curr_quarter(2, 100)", pyt.to_curr_quarter(2, 100), pdt.to_curr_quarter(2, 100))
    compare_value("to_next_quarter(2, 0)", pyt.to_next_quarter(2, 0), pdt.to_next_quarter(2, 0))
    compare_value("to_next_quarter(2, 1)", pyt.to_next_quarter(2, 1), pdt.to_next_quarter(2, 1))
    compare_value("to_next_quarter(2, 28)", pyt.to_next_quarter(2, 28), pdt.to_next_quarter(2, 28))
    compare_value("to_next_quarter(2, 30)", pyt.to_next_quarter(2, 30), pdt.to_next_quarter(2, 30))
    compare_value("to_next_quarter(2, 100)", pyt.to_next_quarter(2, 100), pdt.to_next_quarter(2, 100))
    compare_value("to_prev_quarter(2, 0)", pyt.to_prev_quarter(2, 0), pdt.to_prev_quarter(2, 0))
    compare_value("to_prev_quarter(2, 1)", pyt.to_prev_quarter(2, 1), pdt.to_prev_quarter(2, 1))
    compare_value("to_prev_quarter(2, 28)", pyt.to_prev_quarter(2, 28), pdt.to_prev_quarter(2, 28))
    compare_value("to_prev_quarter(2, 30)", pyt.to_prev_quarter(2, 30), pdt.to_prev_quarter(2, 30))
    compare_value("to_prev_quarter(2, 100)", pyt.to_prev_quarter(2, 100), pdt.to_prev_quarter(2, 100))
    compare_value("to_quarter(-2, 2, 0)", pyt.to_quarter(-2, 2, 0), pdt.to_quarter(-2, 2, 0))
    compare_value("to_quarter(-2, 2, 1)", pyt.to_quarter(-2, 2, 1), pdt.to_quarter(-2, 2, 1))
    compare_value("to_quarter(-2, 2, 28)", pyt.to_quarter(-2, 2, 28), pdt.to_quarter(-2, 2, 28))
    compare_value("to_quarter(-2, 2, 30)", pyt.to_quarter(-2, 2, 30), pdt.to_quarter(-2, 2, 30))
    compare_value("to_quarter(-2, 2, 100)", pyt.to_quarter(-2, 2, 100), pdt.to_quarter(-2, 2, 100))
    compare_value("to_quarter(2, 2, 0)", pyt.to_quarter(2, 2, 0), pdt.to_quarter(2, 2, 0))
    compare_value("to_quarter(2, 2, 1)", pyt.to_quarter(2, 2, 1), pdt.to_quarter(2, 2, 1))
    compare_value("to_quarter(2, 2, 28)", pyt.to_quarter(2, 2, 28), pdt.to_quarter(2, 2, 28))
    compare_value("to_quarter(2, 2, 30)", pyt.to_quarter(2, 2, 30), pdt.to_quarter(2, 2, 30))
    compare_value("to_quarter(2, 2, 100)", pyt.to_quarter(2, 2, 100), pdt.to_quarter(2, 2, 100))
    print()
    # fmt: on

    # Month
    print(" Month ".center(80, "-"))
    compare_value("days_in_month", pyt.days_in_month, pdt.days_in_month)
    compare_value("days_bf_month", pyt.days_bf_month, pdt.days_bf_month)
    compare_value("is_month('Jan')", pyt.is_month("Jan"), pdt.is_month("Jan"))
    compare_value("is_month_1st", pyt.is_month_1st(), pdt.is_month_1st())
    compare_value("is_month_lst", pyt.is_month_lst(), pdt.is_month_lst())
    compare_value("to_month_1st", pyt.to_month_1st(), pdt.to_month_1st())
    compare_value("to_month_lst", pyt.to_month_lst(), pdt.to_month_lst())
    compare_value("to_curr_month(0)", pyt.to_curr_month(0), pdt.to_curr_month(0))
    compare_value("to_curr_month(1)", pyt.to_curr_month(1), pdt.to_curr_month(1))
    compare_value("to_curr_month(28)", pyt.to_curr_month(28), pdt.to_curr_month(28))
    compare_value("to_curr_month(30)", pyt.to_curr_month(30), pdt.to_curr_month(30))
    compare_value("to_curr_month(100)", pyt.to_curr_month(100), pdt.to_curr_month(100))
    compare_value("to_next_month(0)", pyt.to_next_month(0), pdt.to_next_month(0))
    compare_value("to_next_month(1)", pyt.to_next_month(1), pdt.to_next_month(1))
    compare_value("to_next_month(28)", pyt.to_next_month(28), pdt.to_next_month(28))
    compare_value("to_next_month(30)", pyt.to_next_month(30), pdt.to_next_month(30))
    compare_value("to_next_month(100)", pyt.to_next_month(100), pdt.to_next_month(100))
    compare_value("to_prev_month(0)", pyt.to_prev_month(0), pdt.to_prev_month(0))
    compare_value("to_prev_month(1)", pyt.to_prev_month(1), pdt.to_prev_month(1))
    compare_value("to_prev_month(28)", pyt.to_prev_month(28), pdt.to_prev_month(28))
    compare_value("to_prev_month(30)", pyt.to_prev_month(30), pdt.to_prev_month(30))
    compare_value("to_prev_month(100)", pyt.to_prev_month(100), pdt.to_prev_month(100))
    compare_value("to_month(-2, 0)", pyt.to_month(-2, 0), pdt.to_month(-2, 0))
    compare_value("to_month(-2, 1)", pyt.to_month(-2, 1), pdt.to_month(-2, 1))
    compare_value("to_month(-2, 28)", pyt.to_month(-2, 28), pdt.to_month(-2, 28))
    compare_value("to_month(-2, 30)", pyt.to_month(-2, 30), pdt.to_month(-2, 30))
    compare_value("to_month(-2, 100)", pyt.to_month(-2, 100), pdt.to_month(-2, 100))
    compare_value("to_month(2, 0)", pyt.to_month(2, 0), pdt.to_month(2, 0))
    compare_value("to_month(2, 1)", pyt.to_month(2, 1), pdt.to_month(2, 1))
    compare_value("to_month(2, 28)", pyt.to_month(2, 28), pdt.to_month(2, 28))
    compare_value("to_month(2, 30)", pyt.to_month(2, 30), pdt.to_month(2, 30))
    compare_value("to_month(2, 100)", pyt.to_month(2, 100), pdt.to_month(2, 100))
    print()

    # Weekday
    # fmt: off
    print(" Weekday ".center(80, "-"))
    compare_value("weekday", pyt.weekday, pdt.weekday)
    compare_value("isoweekday", pyt.isoweekday, pdt.isoweekday)
    compare_value("isoweek", pyt.isoweek, pdt.isoweek)
    compare_value("isoyear", pyt.isoyear, pdt.isoyear)
    compare_value("is_weekday('Mon')", pyt.is_weekday("Mon"), pdt.is_weekday("Mon"))
    compare_value("to_monday", pyt.to_monday(), pdt.to_monday())
    compare_value("to_tuesday", pyt.to_tuesday(), pdt.to_tuesday())
    compare_value("to_wednesday", pyt.to_wednesday(), pdt.to_wednesday())
    compare_value("to_thursday", pyt.to_thursday(), pdt.to_thursday())
    compare_value("to_friday", pyt.to_friday(), pdt.to_friday())
    compare_value("to_saturday", pyt.to_saturday(), pdt.to_saturday())
    compare_value("to_sunday", pyt.to_sunday(), pdt.to_sunday())
    compare_value("to_curr_weekday('Sun')", pyt.to_curr_weekday("Sun"), pdt.to_curr_weekday("Sun"))
    compare_value("to_next_weekday('Sun')", pyt.to_next_weekday("Sun"), pdt.to_next_weekday("Sun"))
    compare_value("to_prev_weekday('Sun')", pyt.to_prev_weekday("Sun"), pdt.to_prev_weekday("Sun"))
    compare_value("to_weekday(-2, 'Sun')", pyt.to_weekday(-2, "Sun"), pdt.to_weekday(-2, "Sun"))
    compare_value("to_weekday(2, 'Sun')", pyt.to_weekday(2, "Sun"), pdt.to_weekday(2, "Sun"))
    print()
    # fmt: on

    # Day
    print(" Day ".center(80, "-"))
    compare_value("is_day(2)", pyt.is_day(2), pdt.is_day(2))
    compare_value("to_tomorrow", pyt.to_tomorrow(), pdt.to_tomorrow())
    compare_value("to_yesterday", pyt.to_yesterday(), pdt.to_yesterday())
    compare_value("to_day(-2)", pyt.to_day(-2), pdt.to_day(-2))
    compare_value("to_day(2)", pyt.to_day(2), pdt.to_day(2))
    print()

    # Time
    # fmt: off
    print(" Time ".center(80, "-"))
    compare_value("is_time_start", pyt.is_time_start(), pdt.is_time_start())
    compare_value("is_time_end", pyt.is_time_end(), pdt.is_time_end())
    compare_value("to_time_start", pyt.to_time_start(), pdt.to_time_start())
    compare_value("to_time_end", pyt.to_time_end(), pdt.to_time_end())
    compare_value("to_time()", pyt.to_time(), pdt.to_time(), verify=verify)
    compare_value("to_time(1)", pyt.to_time(1), pdt.to_time(1), verify=verify)
    compare_value("to_time(1, 1)", pyt.to_time(1, 1), pdt.to_time(1, 1), verify=verify)
    compare_value("to_time(1, 1, 1)", pyt.to_time(1, 1, 1), pdt.to_time(1, 1, 1), verify=verify)
    compare_value("to_time(1, 1, 1, 1)", pyt.to_time(1, 1, 1, 1), pdt.to_time(1, 1, 1, 1), verify=verify)
    compare_value("to_time(1, 1, 1, 1, 1)", pyt.to_time(1, 1, 1, 1, 1), pdt.to_time(1, 1, 1, 1, 1), verify=verify)
    print()
    # fmt: on

    # Timezone
    # fmt: off
    print(" Timezone ".center(80, "-"))
    compare_value("tz_localize(None)", pyt.tz_localize(None), pdt.tz_localize(None))
    compare_value("tz_localize('UTC')", pyt.tz_localize("UTC"), pdt.tz_localize("UTC"))
    compare_value("tz_localize('CET')", pyt.tz_localize("CET"), pdt.tz_localize("CET"))
    compare_value("tz_convert(None)", pyt.tz_convert(None), pdt.tz_convert(None))
    compare_value("tz_convert('UTC')", pyt.tz_convert("UTC"), pdt.tz_convert("UTC"))
    compare_value("tz_convert('CET')", pyt.tz_convert("CET"), pdt.tz_convert("CET"))
    compare_value("tz_switch(None, 'UTC')", pyt.tz_switch(None, "UTC"), pdt.tz_switch(None, "UTC"))
    compare_value("tz_switch('UTC', 'CET')", pyt.tz_switch("UTC", "CET"), pdt.tz_switch("UTC", "CET"))
    compare_value("tz_switch('CET', 'UTC')", pyt.tz_switch("CET", "UTC"), pdt.tz_switch("CET", "UTC"))
    print()
    # fmt: on

    # Frequency
    print(" Frequency ".center(80, "-"))
    compare_value("freq_round('D')", pyt.freq_round("D"), pdt.freq_round("D"))
    compare_value("freq_round('h')", pyt.freq_round("h"), pdt.freq_round("h"))
    compare_value("freq_round('m')", pyt.freq_round("m"), pdt.freq_round("m"))
    compare_value("freq_round('s')", pyt.freq_round("s"), pdt.freq_round("s"))
    compare_value("freq_round('ms')", pyt.freq_round("ms"), pdt.freq_round("ms"))
    compare_value("freq_round('us')", pyt.freq_round("us"), pdt.freq_round("us"))
    compare_value("freq_ceil('D')", pyt.freq_ceil("D"), pdt.freq_ceil("D"))
    compare_value("freq_ceil('h')", pyt.freq_ceil("h"), pdt.freq_ceil("h"))
    compare_value("freq_ceil('m')", pyt.freq_ceil("m"), pdt.freq_ceil("m"))
    compare_value("freq_ceil('s')", pyt.freq_ceil("s"), pdt.freq_ceil("s"))
    compare_value("freq_ceil('ms')", pyt.freq_ceil("ms"), pdt.freq_ceil("ms"))
    compare_value("freq_ceil('us')", pyt.freq_ceil("us"), pdt.freq_ceil("us"))
    compare_value("freq_floor('D')", pyt.freq_floor("D"), pdt.freq_floor("D"))
    compare_value("freq_floor('h')", pyt.freq_floor("h"), pdt.freq_floor("h"))
    compare_value("freq_floor('m')", pyt.freq_floor("m"), pdt.freq_floor("m"))
    compare_value("freq_floor('s')", pyt.freq_floor("s"), pdt.freq_floor("s"))
    compare_value("freq_floor('ms')", pyt.freq_floor("ms"), pdt.freq_floor("ms"))
    compare_value("freq_floor('us')", pyt.freq_floor("us"), pdt.freq_floor("us"))
    print()

    # Delta
    # fmt: off
    print(" Delta ".center(80, "-"))
    compare_value("add_delta(1)", pyt.add_delta(1), pdt.add_delta(1))
    compare_value("add_delta(1, 1)", pyt.add_delta(1, 1), pdt.add_delta(1, 1))
    compare_value("add_delta(1, 1, 1)", pyt.add_delta(1, 1, 1), pdt.add_delta(1, 1, 1))
    compare_value("add_delta(1, 1, 1, 1)", pyt.add_delta(1, 1, 1, 1), pdt.add_delta(1, 1, 1, 1))
    compare_value("add_delta(1, 1, 1, 1, 1)", pyt.add_delta(1, 1, 1, 1, 1), pdt.add_delta(1, 1, 1, 1, 1))
    compare_value("add_delta(1, 1, 1, 1, 1, 1)", pyt.add_delta(1, 1, 1, 1, 1, 1), pdt.add_delta(1, 1, 1, 1, 1, 1))
    compare_value("add_delta(1, 1, 1, 1, 1, 1, 1)", pyt.add_delta(1, 1, 1, 1, 1, 1, 1), pdt.add_delta(1, 1, 1, 1, 1, 1, 1))
    compare_value("add_delta(1, 1, 1, 1, 1, 1, 1, 1)", pyt.add_delta(1, 1, 1, 1, 1, 1, 1, 1), pdt.add_delta(1, 1, 1, 1, 1, 1, 1, 1))
    compare_value("add_delta(1, 1, 1, 1, 1, 1, 1, 1, 1)", pyt.add_delta(1, 1, 1, 1, 1, 1, 1, 1, 1), pdt.add_delta(1, 1, 1, 1, 1, 1, 1, 1, 1))
    compare_value("sub_delta(1)", pyt.sub_delta(1), pdt.sub_delta(1))
    compare_value("sub_delta(1, 1)", pyt.sub_delta(1, 1), pdt.sub_delta(1, 1))
    compare_value("sub_delta(1, 1, 1)", pyt.sub_delta(1, 1, 1), pdt.sub_delta(1, 1, 1))
    compare_value("sub_delta(1, 1, 1, 1)", pyt.sub_delta(1, 1, 1, 1), pdt.sub_delta(1, 1, 1, 1))
    compare_value("sub_delta(1, 1, 1, 1, 1)", pyt.sub_delta(1, 1, 1, 1, 1), pdt.sub_delta(1, 1, 1, 1, 1))
    compare_value("sub_delta(1, 1, 1, 1, 1, 1)", pyt.sub_delta(1, 1, 1, 1, 1, 1), pdt.sub_delta(1, 1, 1, 1, 1, 1))
    compare_value("sub_delta(1, 1, 1, 1, 1, 1, 1)", pyt.sub_delta(1, 1, 1, 1, 1, 1, 1), pdt.sub_delta(1, 1, 1, 1, 1, 1, 1))
    compare_value("sub_delta(1, 1, 1, 1, 1, 1, 1, 1)", pyt.sub_delta(1, 1, 1, 1, 1, 1, 1, 1), pdt.sub_delta(1, 1, 1, 1, 1, 1, 1, 1))
    compare_value("sub_delta(1, 1, 1, 1, 1, 1, 1, 1, 1)", pyt.sub_delta(1, 1, 1, 1, 1, 1, 1, 1, 1), pdt.sub_delta(1, 1, 1, 1, 1, 1, 1, 1, 1))
    compare_value("cal_delta(other, 'Y')", pyt.cal_delta(odt, "Y"), pdt.cal_delta(odt, "Y"))
    compare_value("cal_delta(other, 'M')", pyt.cal_delta(odt, "M"), pdt.cal_delta(odt, "M"))
    compare_value("cal_delta(other, 'W')", pyt.cal_delta(odt, "W"), pdt.cal_delta(odt, "W"))
    compare_value("cal_delta(other, 'D')", pyt.cal_delta(odt, "D"), pdt.cal_delta(odt, "D"))
    compare_value("cal_delta(other, 'h')", pyt.cal_delta(odt, "h"), pdt.cal_delta(odt, "h"))
    compare_value("cal_delta(other, 'm')", pyt.cal_delta(odt, "m"), pdt.cal_delta(odt, "m"))
    compare_value("cal_delta(other, 's')", pyt.cal_delta(odt, "s"), pdt.cal_delta(odt, "s"))
    compare_value("cal_delta(other, 'ms')", pyt.cal_delta(odt, "ms"), pdt.cal_delta(odt, "ms"))
    compare_value("cal_delta(other, 'us')", pyt.cal_delta(odt, "us"), pdt.cal_delta(odt, "us"))
    print()
    # fmt: on

    # Replace
    # fmt: off
    print(" Replace ".center(80, "-"))
    compare_value("replace(1)", 
        pyt.replace(1), pdt.replace(1), verify=verify)
    compare_value("replace(1, 1)", 
        pyt.replace(1, 1), pdt.replace(1, 1), verify=verify)
    compare_value("replace(1, 1, 1)", 
        pyt.replace(1, 1, 1), pdt.replace(1, 1, 1), verify=verify)
    compare_value("replace(1, 1, 1, 1)", 
        pyt.replace(1, 1, 1, 1), pdt.replace(1, 1, 1, 1), verify=verify)
    compare_value("replace(1, 1, 1, 1, 1)", 
        pyt.replace(1, 1, 1, 1, 1), pdt.replace(1, 1, 1, 1, 1), verify=verify)
    compare_value("replace(1, 1, 1, 1, 1, 1)", 
        pyt.replace(1, 1, 1, 1, 1, 1), pdt.replace(1, 1, 1, 1, 1, 1), verify=verify)
    compare_value("replace(1, 1, 1, 1, 1, 1, 1)", 
        pyt.replace(1, 1, 1, 1, 1, 1, 1), pdt.replace(1, 1, 1, 1, 1, 1, 1), verify=verify)
    compare_value("replace(1, 1, 1, 1, 1, 1, 1, 1)", 
        pyt.replace(1, 1, 1, 1, 1, 1, 1, 1), pdt.replace(1, 1, 1, 1, 1, 1, 1, 1), verify=verify)
    compare_value("replace(-1, -1, -1, 1, 1, 1, 1, 1)", 
        pyt.replace(-1, -1, -1, 1, 1, 1, 1, 1), pdt.replace(-1, -1, -1, 1, 1, 1, 1, 1), verify=verify)
    print()
    # fmt: on

    # Addition
    print(" Addition ".center(80, "-"))
    compare_value("dt + timedelta", pyt + otd, pdt + otd)
    compare_value("timedelta + dt", otd + pyt, otd + pdt)
    print()

    # Substraction
    print(" Substraction ".center(80, "-"))
    compare_value("dt - timedelta", pyt - otd, pdt - otd)
    compare_value("dt - dt", pyt - odt, pdt - odt)
    compare_value("dt - pydt", pyt - pyt, pdt - pyt)
    compare_value("dt - datetime64", pyt - pyt.dt64, pdt - pyt.dt64)
    compare_value("dt - Timestamp", pyt - pyt.ts, pdt - pyt.ts)
    print()


def parse_performance(rounds: int) -> None:
    from zoneinfo import ZoneInfo
    from datetime import datetime
    from cytimes.pydatetime import pydt
    from pendulum import parse as plparse
    from dateutil.parser import parse, isoparse

    # Strict ISO Without timezone
    print(" Strict Isoformat w/o Timezone ".center(80, "-"))
    text = "2023-08-01 12:00:00.000001"
    print(f"Text: {repr(text)}\tRounds: {rounds:,}")
    t1 = timeit(lambda: pydt(text), number=rounds)
    print(f"- pydt():\t\t{t1:.6f}s")
    t2 = timeit(lambda: datetime(2023, 8, 1, 12, 0, 0, 1), number=rounds)
    print(f"- direct create:\t{t2:.6f}s\tPerf Diff: {diff(t1, t2)}")
    t3 = timeit(lambda: datetime.fromisoformat(text), number=rounds)
    print(f"- dt.fromisoformat():\t{t3:.6f}s\tPerf Diff: {diff(t1, t3)}")
    t4 = timeit(lambda: plparse(text), number=rounds)
    print(f"- pendulum.parse():\t{t4:.6f}s\tPerf Diff: {diff(t1, t4)}")
    t5 = timeit(lambda: isoparse(text), number=rounds)
    print(f"- dateutil.isoparse():\t{t5:.6f}s\tPerf Diff: {diff(t1, t5)}")
    t6 = timeit(lambda: parse(text, ignoretz=True), number=rounds)
    print(f"- dateutil.parse():\t{t6:.6f}s\tPerf Diff: {diff(t1, t6)}")
    print()

    # Strict ISO With timezone
    print(" Strict Isoformat w/t Timezone ".center(80, "-"))
    text = "2023-08-01 12:00:00.000001+02:00"
    tz = ZoneInfo("CET")
    print(f"Text: {repr(text)}\tRounds: {rounds:,}")
    t1 = timeit(lambda: pydt(text), number=rounds)
    print(f"- pydt():\t\t{t1:.6f}s")
    t2 = timeit(lambda: datetime(2023, 8, 1, 12, 0, 0, 1, tz), number=rounds)
    print(f"- direct create:\t{t2:.6f}s\tPerf Diff: {diff(t1, t2)}")
    t3 = timeit(lambda: datetime.fromisoformat(text), number=rounds)
    print(f"- dt.fromisoformat():\t{t3:.6f}s\tPerf Diff: {diff(t1, t3)}")
    t4 = timeit(lambda: plparse(text), number=rounds)
    print(f"- pendulum.parse():\t{t4:.6f}s\tPerf Diff: {diff(t1, t4)}")
    t5 = timeit(lambda: isoparse(text), number=rounds)
    print(f"- dateutil.isoparse():\t{t5:.6f}s\tPerf Diff: {diff(t1, t5)}")
    t6 = timeit(lambda: parse(text), number=rounds)
    print(f"- dateutil.parse():\t{t6:.6f}s\tPerf Diff: {diff(t1, t6)}")
    print()

    # Loose ISO Without timezone
    print(" Loose Isoformat w/o Timezone ".center(80, "-"))
    text = "2023/08/01 12:00:00.000001"
    print(f"Text: {repr(text)}\tRounds: {rounds:,}")
    t1 = timeit(lambda: pydt(text), number=rounds)
    print(f"- pydt():\t\t{t1:.6f}s")
    t4 = timeit(lambda: plparse(text), number=rounds)
    print(f"- pendulum.parse():\t{t4:.6f}s\tPerf Diff: {diff(t1, t4)}")
    t6 = timeit(lambda: parse(text, ignoretz=True), number=rounds)
    print(f"- dateutil.parse():\t{t6:.6f}s\tPerf Diff: {diff(t1, t6)}")
    print()

    # Loose ISO With timezone
    print(" Loose Isoformat w/t Timezone ".center(80, "-"))
    text = "2023/08/01 12:00:00.000001+02:00"
    print(f"Text: {repr(text)}\tRounds: {rounds:,}")
    t1 = timeit(lambda: pydt(text), number=rounds)
    print(f"- pydt():\t\t{t1:.6f}s")
    t6 = timeit(lambda: parse(text), number=rounds)
    print(f"- dateutil.parse():\t{t6:.6f}s\tPerf Diff: {diff(t1, t6)}")
    print()

    # Datetime Strings Parsing
    # fmt: off
    rounds  = max(1, int(rounds / 100))
    default = datetime(1970, 1, 1)
    print(" Parse Datetime Strings ".center(80, "-"))
    texts = gen_test_datetimes()
    print(f"Total datetime strings: #{len(texts)}\tRounds: {rounds:,}")
    t1 = timeit("for text in texts: pydt(text, default=default)", number=rounds, globals=locals())
    print(f"- pydt():\t\t{t1:.6f}s")
    t2 = timeit("for text in texts: parse(text, default=default, fuzzy=True)", number=rounds, globals=locals())
    print(f"- dateutil.parse():\t{t2:.6f}s\tPerf Diff: {diff(t1, t2)}")
    print()
    # fmt: on


if __name__ == "__main__":
    test_cytimedelta(100_000)
    test_parser(100_000)
    parse_performance(100_000)
    test_pydt_pddt("ns")
    test_pydt_pddt("us")
    test_pydt_pddt("ms")
    test_pydt_pddt("s")
