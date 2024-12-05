import warnings
from zoneinfo import ZoneInfo
import time, unittest, datetime
import numpy as np, pandas as pd, pendulum as pl
from cytimes.pydt import Pydt
from cytimes.pddt import Pddt

warnings.filterwarnings("ignore")


class TestCase(unittest.TestCase):
    name: str = "Case"

    def test_all(self) -> None:
        pass

    # Utils
    def assertTimeUnit(self, pt: Pddt, unit: str):
        self.assertTrue(pt.unit == unit)

    def assertEqualPtDt(self, pt: Pddt, dt: datetime.datetime) -> None:
        self.assertEqual(dt, pt[0])

    def assertEqualPtDtMS(self, pt: Pddt, dt: datetime.datetime) -> None:
        self.assertTrue((dt - pt[0]).total_seconds() < 0.1)

    def assertEqualPtDt_unit(self, pt: Pddt, dt: Pydt, unit: str) -> None:
        if unit == "ns" or unit == "us":
            self.assertEqual(dt, pt[0])
        elif pt[0].microsecond % 1000 > 0:
            self.assertEqual(dt, pt[0])
        else:
            if not isinstance(dt, Pydt):
                dt = Pydt.fromdatetime(dt)
            self.assertEqual(dt.to_start_of(unit), pt[0])

    def log_start(self, msg: str) -> None:
        msg = "START TEST '%s': %s" % (self.name, msg)
        print(msg.ljust(60), end="\r")
        self._start_time = time.perf_counter()

    def log_ended(self, msg: str, skip: bool = False) -> None:
        self._ended_time = time.perf_counter()
        msg = "%s TEST '%s': %s" % ("SKIP" if skip else "PASS", self.name, msg)
        if self._start_time is not None:
            msg += " (%.6fs)" % (self._ended_time - self._start_time)
        print(msg.ljust(60))


class Test_Pddt(TestCase):
    name = "Pddt"

    def test_all(self) -> None:
        self.test_parse()
        self.test_constructor()
        self.test_converter()
        self.test_manipulator()
        self.test_calendar()
        self.test_timezone()
        self.test_arithmetic()
        self.test_comparision()

    def test_parse(self) -> None:
        test = "Parse"
        self.log_start(test)

        # nanoseconds parse timezone-naive
        data = [
            "2023-01-01 00:00:00",
            "2023-01-02 00:00:00",
            "2023-01-03 00:00:00",
        ]
        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = pd.DatetimeIndex(data, tz=tz)
            pt = Pddt(data, tz=tz)
            self.assertTrue(dt.equals(pt))
            self.assertTimeUnit(pt, "ns")
            self.assertEqual(pt.tzinfo is None, tz is None)
        data = [pd.Timestamp.min, pd.Timestamp.max]
        dt = pd.DatetimeIndex(data)
        pt = Pddt(data)
        self.assertTrue(dt.equals(pt))
        self.assertTimeUnit(pt, "ns")
        self.assertTrue(pt.tzinfo is None)

        # microseconds parse timezone-naive
        data = [
            "9999-01-01 00:00:00",
            "9999-01-02 00:00:00",
            "9999-01-03 00:00:00",
        ]
        for tz in (None, ZoneInfo("CET"), datetime.UTC):
            pt = Pddt(data, tz=tz)
            dt = pd.DatetimeIndex(data, dtype="datetime64[us]")
            if tz is not None:
                dt = dt.tz_localize(tz)
            self.assertTrue(dt.equals(pt))
            self.assertTimeUnit(pt, "us")
            if tz is None:
                self.assertTrue(pt.tzinfo is None)
            else:
                self.assertTrue(pt.tzinfo is not None)

        # nanoseconds parse timezone-aware
        data = [
            "2023-01-01 00:00:00+00:00",
            "2023-01-02 00:00:00+00:00",
            "2023-01-03 00:00:00+00:00",
        ]
        for tz in (None, ZoneInfo("CET"), datetime.UTC):
            pt = Pddt(data, tz=tz)
            if tz is None:
                dt = pd.DatetimeIndex(data)
            else:
                dt = pd.DatetimeIndex(data, tz=tz)
            self.assertTrue(dt.equals(pt))
            self.assertTimeUnit(pt, "ns")
            self.assertTrue(pt.tzinfo is not None)

        # microseconds parse timezone-aware
        data = [
            "9999-01-01 00:00:00+00:00",
            "9999-01-02 00:00:00+00:00",
            "9999-01-03 00:00:00+00:00",
        ]
        comp = (
            (
                None,
                ["9999-01-01 00:00:00", "9999-01-02 00:00:00", "9999-01-03 00:00:00"],
            ),
            (
                ZoneInfo("CET"),
                ["9999-01-01 01:00:00", "9999-01-02 01:00:00", "9999-01-03 01:00:00"],
            ),
            (
                datetime.UTC,
                ["9999-01-01 00:00:00", "9999-01-02 00:00:00", "9999-01-03 00:00:00"],
            ),
        )
        for tz, dts in comp:
            pt = Pddt(data, tz=tz)
            dt = pd.DatetimeIndex(dts, dtype="datetime64[us]")
            dt = dt.tz_localize(datetime.UTC if tz is None else tz)
            self.assertTrue(dt.equals(pt))
            self.assertTimeUnit(pt, "us")
            self.assertTrue(pt.tzinfo is not None)

        # nanoseconds parse mixed-timezone
        data = [
            "2023-01-01 00:00:00",
            "2023-01-02 00:00:00+00:00",
            "2023-01-03 00:00:00+01:00",
        ]
        comp = (
            (
                None,
                ["2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-02 23:00:00"],
            ),
            (
                ZoneInfo("CET"),
                ["2023-01-01 00:00:00", "2023-01-02 01:00:00", "2023-01-03 00:00:00"],
            ),
            (
                datetime.UTC,
                ["2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-02 23:00:00"],
            ),
        )
        for tz, dts in comp:
            pt = Pddt(data, tz=tz)
            dt = pd.DatetimeIndex(dts, tz=datetime.UTC if tz is None else tz)
            self.assertTrue(dt.equals(pt))
            self.assertTimeUnit(pt, "ns")
            self.assertTrue(pt.tzinfo is not None)

        # microseconds parse mixed-timezone
        data = [
            "9999-01-01 00:00:00",
            "9999-01-02 00:00:00+00:00",
            "9999-01-03 00:00:00+01:00",
        ]
        comp = (
            (
                None,
                ["9999-01-01 00:00:00", "9999-01-02 00:00:00", "9999-01-02 23:00:00"],
            ),
            (
                ZoneInfo("CET"),
                ["9999-01-01 00:00:00", "9999-01-02 01:00:00", "9999-01-03 00:00:00"],
            ),
            (
                datetime.UTC,
                ["9999-01-01 00:00:00", "9999-01-02 00:00:00", "9999-01-02 23:00:00"],
            ),
        )
        for tz, dts in comp:
            pt = Pddt(data, tz=tz)
            dt = pd.DatetimeIndex(dts, dtype="datetime64[us]")
            dt = dt.tz_localize(datetime.UTC if tz is None else tz)
            self.assertTrue(dt.equals(pt))
            self.assertTimeUnit(pt, "us")
            self.assertTrue(pt.tzinfo is not None)

        self.log_ended(test)

    def test_constructor(self) -> None:
        test = "Constructor"
        self.log_start(test)

        # now()
        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            pt = Pddt.now(tz=tz)
            dt = Pydt.now(tz=tz)
            self.assertEqualPtDtMS(pt, dt)

        # utcnow()
        self.assertEqualPtDtMS(Pddt.utcnow(), Pydt.utcnow())

        # today()
        self.assertEqualPtDtMS(Pddt.today(), Pydt.today())

        # combine
        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = Pydt.combine(datetime.date(1970, 1, 2), datetime.time(3, 4, 5, 6, tz))
            pt = Pddt.combine(datetime.date(1970, 1, 2), datetime.time(3, 4, 5, 6), tz)
            self.assertEqualPtDt(pt, dt)
            dt = Pydt.combine(datetime.date(1970, 1, 2), datetime.time(3, 4, 5, 6), tz)
            pt = Pddt.combine(datetime.date(1970, 1, 2), datetime.time(3, 4, 5, 6), tz)
            self.assertEqualPtDt(pt, dt)
            dt = Pydt.combine("1970-01-02", datetime.time(3, 4, 5, 6, tz))
            pt = Pddt.combine("1970-01-02", datetime.time(3, 4, 5, 6), tz)
            self.assertEqualPtDt(pt, dt)
            dt = Pydt.combine("1970-01-02", datetime.time(3, 4, 5, 6), tz)
            pt = Pddt.combine("1970-01-02", datetime.time(3, 4, 5, 6), tz)
            self.assertEqualPtDt(pt, dt)
            dt = Pydt.combine(datetime.date(1970, 1, 2), "03:04:05.000006", tz)
            pt = Pddt.combine(datetime.date(1970, 1, 2), "03:04:05.000006", tz)
            self.assertEqualPtDt(pt, dt)
            dt = Pydt.combine("1970-01-02", "03:04:05.000006", tz)
            pt = Pddt.combine("1970-01-02", "03:04:05.000006", tz)
            self.assertEqualPtDt(pt, dt)

        # fromordinal
        for ordinal in (2, 100, 400, 800000):
            for tz in (None, datetime.UTC, ZoneInfo("CET")):
                dt = Pydt.fromordinal(ordinal, tz=tz)
                for pt in (
                    Pddt.fromordinal(ordinal, tz=tz),
                    Pddt.fromordinal(np.array([ordinal], dtype="int64"), tz=tz),
                    Pddt.fromordinal(np.array([ordinal], dtype="float64"), tz=tz),
                    Pddt.fromordinal(pd.Series([ordinal], dtype="int32"), tz=tz),
                    Pddt.fromordinal(pd.Series([ordinal], dtype="float32"), tz=tz),
                    Pddt.fromordinal([ordinal], tz=tz),
                    Pddt.fromordinal((ordinal,), tz=tz),
                ):
                    self.assertEqualPtDt(pt, dt)

        # fromseconds()
        for seconds in (-78914791.123, -1, 0, 1, 78914791.123):
            for tz in (None, datetime.UTC, ZoneInfo("CET")):
                dt = Pydt.fromseconds(seconds, tz=tz)
                for pt in (
                    Pddt.fromseconds(seconds, tz=tz),
                    Pddt.fromseconds(np.array([seconds], dtype="float64"), tz=tz),
                    Pddt.fromseconds(pd.Series([seconds], dtype="float64"), tz=tz),
                    Pddt.fromseconds([seconds], tz=tz),
                    Pddt.fromseconds((seconds,), tz=tz),
                ):
                    self.assertEqualPtDt(pt, dt)

        # fromicroseconds()
        for seconds in (-78914791.123, -1, 0, 1, 78914791.123):
            us = int(seconds * 1_000_000)
            for tz in (None, datetime.UTC, ZoneInfo("CET")):
                dt = Pydt.fromicroseconds(us, tz=tz)
                for pt in (
                    Pddt.fromicroseconds(us, tz=tz),
                    Pddt.fromicroseconds(np.array([us], dtype="int64"), tz=tz),
                    Pddt.fromicroseconds(np.array([us], dtype="float64"), tz=tz),
                    Pddt.fromicroseconds(pd.Series([us], dtype="int64"), tz=tz),
                    Pddt.fromicroseconds(pd.Series([us], dtype="float64"), tz=tz),
                    Pddt.fromicroseconds([us], tz=tz),
                    Pddt.fromicroseconds((us,), tz=tz),
                ):
                    self.assertEqualPtDt(pt, dt)

        # fromtimestamp()
        for ts in (-78914791.123, -1, 0, 1, 78914791.123):
            for tz in (None, datetime.UTC, ZoneInfo("CET")):
                dt = Pydt.fromtimestamp(ts, tz=tz)
                for pt in (
                    Pddt.fromtimestamp(ts, tz=tz),
                    Pddt.fromtimestamp(np.array([ts], dtype="float64"), tz=tz),
                    Pddt.fromtimestamp(pd.Series([ts], dtype="float64"), tz=tz),
                    Pddt.fromtimestamp([ts], tz=tz),
                    Pddt.fromtimestamp((ts,), tz=tz),
                ):
                    self.assertEqualPtDt(pt, dt)

        # utcfromtimestamp()
        for ts in (-78914791.123, -1, 0, 1, 78914791.123):
            dt = Pydt.utcfromtimestamp(ts)
            for pt in (
                Pddt.utcfromtimestamp(ts),
                Pddt.utcfromtimestamp(np.array([ts], dtype="float64")),
                Pddt.utcfromtimestamp(pd.Series([ts], dtype="float64")),
                Pddt.utcfromtimestamp([ts]),
                Pddt.utcfromtimestamp((ts,)),
            ):
                self.assertEqualPtDt(pt, dt)

        # fromisoformat()
        for iso in ("1970-01-02T03:04:05.000006", "1970-01-02T03:04:05+01:00"):
            dt = Pydt.fromisoformat(iso)
            for pt in (
                Pddt.fromisoformat(iso),
                Pddt.fromisoformat(np.array([iso])),
                Pddt.fromisoformat(pd.Series([iso])),
                Pddt.fromisoformat([iso]),
                Pddt.fromisoformat((iso,)),
            ):
                self.assertEqualPtDt(pt, dt)

        # fromisocalendar()
        ywd_dict = {"year": 2024, "week": 43, "day": 2}
        ywd_list = list(ywd_dict.values())
        ywd_tupl = tuple(ywd_dict.values())
        for iso in (
            ywd_dict,
            ywd_dict | {"weekday": 2},
            ywd_tupl,
            ywd_list,
            (ywd_dict, ywd_dict | {"weekday": 2}),
            [ywd_dict, ywd_dict | {"weekday": 2}],
            (ywd_list, ywd_tupl),
            [ywd_list, ywd_tupl],
            Pddt.fromdatetime(Pydt.fromisocalendar(*ywd_tupl), size=2).isocalendar(),
        ):
            for tz in (None, datetime.UTC, ZoneInfo("CET")):
                dt = Pydt.fromisocalendar(*ywd_tupl, tz)
                pt = Pddt.fromisocalendar(iso, tz=tz)
                self.assertEqualPtDt(pt, dt)

        # fromdate()
        for date in (datetime.date(1970, 1, 2), pl.date(1970, 1, 2)):
            for tz in (None, datetime.UTC, ZoneInfo("CET")):
                dt = Pydt.fromdate(date, tz)
                for pt in (
                    Pddt.fromdate(date, tz=tz),
                    Pddt.fromdate(np.array([date]), tz=tz),
                    Pddt.fromdate(pd.Series([date]), tz=tz),
                    Pddt.fromdate(pd.DatetimeIndex([date]), tz=tz),
                    Pddt.fromdate([date], tz=tz),
                    Pddt.fromdate((date,), tz=tz),
                ):
                    self.assertEqualPtDt(pt, dt)

        # fromdatetime()
        for dt in (
            datetime.datetime(1970, 1, 2, 3, 4, 5, 6),
            pl.datetime(1970, 1, 2, 3, 4, 5, 6),
            pd.Timestamp("1970-01-02 03:04:05.000006"),
            Pydt(1970, 1, 2, 3, 4, 5, 6),
        ):
            for tz in (None, datetime.UTC, ZoneInfo("CET")):
                dt = dt.replace(tzinfo=tz)
                for pt in (
                    Pddt.fromdatetime(dt),
                    Pddt.fromdatetime(np.array([dt])),
                    Pddt.fromdatetime(pd.Series([dt])),
                    Pddt.fromdatetime(pd.DatetimeIndex([dt])),
                    Pddt.fromdatetime([dt]),
                    Pddt.fromdatetime((dt,)),
                ):
                    self.assertEqualPtDt(pt, dt)

        # fromdatetime64()
        for unit in ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"):
            for tz in (None, datetime.UTC, ZoneInfo("CET")):
                dt64 = np.datetime64(1, unit)
                dt = Pydt.fromdatetime64(dt64, tz=tz)
                for pt in (
                    Pddt.fromdatetime64(dt64, tz=tz),
                    Pddt.fromdatetime64(np.array([dt64]), tz=tz),
                    Pddt.fromdatetime64(pd.Series([dt64]), tz=tz),
                    Pddt.fromdatetime64(pd.DatetimeIndex([dt64]), tz=tz),
                    Pddt.fromdatetime64([dt64], tz=tz),
                    Pddt.fromdatetime64((dt64,), tz=tz),
                ):
                    self.assertEqualPtDt(pt, dt)

        # strptime()
        dt_str = "03:04:05.000006 1970-01-02 +01:00"
        fmt = "%H:%M:%S.%f %Y-%m-%d %z"
        dt = Pydt.strptime(dt_str, fmt)
        for pt in (
            Pddt.strptime(dt_str, fmt),
            Pddt.strptime(np.array([dt_str]), fmt),
            Pddt.strptime(pd.Series([dt_str]), fmt),
            Pddt.strptime([dt_str], fmt),
            Pddt.strptime((dt_str,), fmt),
        ):
            self.assertEqualPtDt(pt, dt)

        self.log_ended(test)

    def test_converter(self) -> None:
        test = "Converter"
        self.log_start(test)

        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = Pydt(2023, 1, 31, 3, 4, 5, 6, tz)
            pt = Pddt.fromdatetime(dt)
            # ctime()
            self.assertEqualPtDt(pt.ctime(), dt.ctime())
            # strftime()
            for fmt in (
                "%d/%m/%Y, %H:%M:%S",
                "%d/%m/%Y, %H:%M:%S%z",
                "%d/%m/%Y, %H:%M:%S%Z",
            ):
                self.assertEqualPtDt(pt.strftime(fmt), dt.strftime(fmt))
            # isoformat()
            for sep in (" ", "T", "x"):
                self.assertEqualPtDt(pt.isoformat(sep), dt.isoformat(sep))
            # timedf()
            self.assertEqual(
                dt.timetuple()[:8],
                tuple(pt.timedf().iloc[0].tolist()),
            )
            # utctimedf()
            self.assertEqual(
                dt.utctimetuple()[:8],
                tuple(pt.utctimedf().iloc[0].tolist()),
            )
            # toordinal()
            self.assertEqualPtDt(pt.toordinal(), dt.toordinal())
            # seconds()
            self.assertEqual(pt.seconds(False), dt.seconds(False))
            self.assertEqual(pt.seconds(True), dt.seconds(True))
            # microseconds()
            self.assertEqual(pt.microseconds(False), dt.microseconds(False))
            self.assertEqual(pt.microseconds(True), dt.microseconds(True))
            # timestamp()
            self.assertEqual(pt.timestamp(), dt.timestamp())
            # date()
            self.assertEqual(pt.date(), dt.date())
            # time()
            self.assertEqual(pt.time(), dt.time())
            # timetz()
            self.assertEqual(pt.timetz(), dt.timetz())

        self.log_ended(test)

    def test_manipulator(self) -> None:
        test = "Manipulator"
        self.log_start(test)

        base_dts = [
            Pydt(2023, 8, 17, 13, 31, 31, 666666),
            Pydt(2023, 5, 14, 11, 29, 28, 444444),
            Pydt(1800, 8, 17, 13, 31, 31, 666666),
            Pydt(1800, 5, 14, 11, 29, 28, 444444),
            Pydt(2023, 5, 14, 11, 29, 28, 444444, ZoneInfo("CET")),
            Pydt(1800, 6, 15, 12, 30, 30, 500000, datetime.UTC),
            pd.Timestamp.min,
            pd.Timestamp.max,
        ]
        my_units = ("ns", "us", "ms", "s")
        # replace()
        for base_dt in base_dts:
            for unit in my_units:
                for targ_tz in (None, datetime.UTC, ZoneInfo("CET")):
                    dt = Pydt.fromdatetime(base_dt)
                    pt = Pddt.fromdatetime(base_dt, unit=unit)
                    for args in (
                        # dates only
                        [2000, -1, -1, -1, -1, -1, -1, -1, -1],
                        [2000, 3, -1, -1, -1, -1, -1, -1, -1],
                        [2000, 3, 31, -1, -1, -1, -1, -1, -1],
                        [-1, 3, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 3, 31, -1, -1, -1, -1, -1, -1],
                        [-1, -1, 31, -1, -1, -1, -1, -1, -1],
                        # times only
                        [-1, -1, -1, 23, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, 59, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, 59, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, 999999, -1, -1],
                        [-1, -1, -1, 23, 59, -1, -1, -1, -1],
                        [-1, -1, -1, 23, 59, 59, -1, -1, -1],
                        [-1, -1, -1, 23, 59, 59, 999999, -1, -1],
                        # dates & times
                        [2000, -1, -1, 23, -1, -1, -1, -1, -1],
                        [2000, 3, -1, 23, -1, -1, -1, -1, -1],
                        [2000, 3, 31, 23, -1, -1, -1, -1, -1],
                        [-1, 3, -1, 23, -1, -1, -1, -1, -1],
                        [-1, 3, 31, 23, -1, -1, -1, -1, -1],
                        [-1, -1, 31, 23, -1, -1, -1, -1, -1],
                        [2000, -1, -1, 23, -1, -1, -1, -1, -1],
                        [2000, -1, -1, -1, 59, -1, -1, -1, -1],
                        [2000, -1, -1, -1, -1, 59, -1, -1, -1],
                        [2000, -1, -1, -1, -1, -1, 999999, -1, -1],
                        [2000, -1, -1, 23, 59, -1, -1, -1, -1],
                        [2000, -1, -1, 23, 59, 59, -1, -1, -1],
                        [2000, -1, -1, 23, 59, 59, 999999, -1, -1],
                        [2000, 3, 31, 23, 59, 59, 999999, -1, -1],
                        # Timezone
                        [2000, 3, 31, 23, 59, 59, 999999, targ_tz, -1],
                    ):
                        kwargs = {
                            "year": args[0],
                            "month": args[1],
                            "day": args[2],
                            "hour": args[3],
                            "minute": args[4],
                            "second": args[5],
                            "microsecond": args[6],
                            "tzinfo": args[7],
                        }
                        self.assertEqualPtDt_unit(
                            pt.replace(**kwargs), dt.replace(**kwargs), unit
                        )

        # to_*()
        for base_dt in base_dts:
            for unit in my_units:
                dt = Pydt.fromdatetime(base_dt)
                pt = Pddt.fromdatetime(base_dt, unit=unit)
                for offset in range(-3, 4):
                    # to_curr_year() / to_year()
                    for month in range(1, 13):
                        for day in range(0, 33):
                            self.assertEqualPtDt_unit(
                                pt.to_curr_year(month, day),
                                dt.to_curr_year(month, day),
                                unit,
                            )
                            self.assertEqualPtDt_unit(
                                pt.to_year(offset, month, day),
                                dt.to_year(offset, month, day),
                                unit,
                            )
                    # to_curr_quarter() / to_quarter()
                    for month in range(5):
                        for day in range(0, 33):
                            self.assertEqualPtDt_unit(
                                pt.to_curr_quarter(month, day),
                                dt.to_curr_quarter(month, day),
                                unit,
                            )
                            self.assertEqualPtDt_unit(
                                pt.to_quarter(offset, month, day),
                                dt.to_quarter(offset, month, day),
                                unit,
                            )
                    # to_curr_month() / to_month()
                    for day in range(0, 33):
                        self.assertEqualPtDt_unit(
                            pt.to_curr_month(day), dt.to_curr_month(day), unit
                        )
                        self.assertEqualPtDt_unit(
                            pt.to_month(offset, day), dt.to_month(offset, day), unit
                        )
                    # to_weekday()
                    for weekday in range(7):
                        self.assertEqualPtDt_unit(
                            pt.to_weekday(offset, weekday),
                            dt.to_weekday(offset, weekday),
                            unit,
                        )
                    # to_day()
                    for day in range(-32, 33):
                        self.assertEqualPtDt_unit(pt.to_day(day), dt.to_day(day), unit)

        # to_datetime()
        for base_dt in base_dts:
            for unit in my_units:
                dt = Pydt.fromdatetime(base_dt)
                pt = Pddt.fromdatetime(base_dt, unit=unit)
                for args in (
                    # dates only
                    [2000, -1, -1, -1, -1, -1, -1, -1, -1],
                    [2000, 3, -1, -1, -1, -1, -1, -1, -1],
                    [2000, 3, 31, -1, -1, -1, -1, -1, -1],
                    [-1, 3, -1, -1, -1, -1, -1, -1, -1],
                    [-1, 3, 31, -1, -1, -1, -1, -1, -1],
                    [-1, -1, 31, -1, -1, -1, -1, -1, -1],
                    # times only
                    [-1, -1, -1, 23, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, 59, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, 59, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, 999999, -1, -1],
                    [-1, -1, -1, 23, 59, -1, -1, -1, -1],
                    [-1, -1, -1, 23, 59, 59, -1, -1, -1],
                    [-1, -1, -1, 23, 59, 59, 999999, -1, -1],
                    # dates & times
                    [2000, -1, -1, 23, -1, -1, -1, -1, -1],
                    [2000, 3, -1, 23, -1, -1, -1, -1, -1],
                    [2000, 3, 31, 23, -1, -1, -1, -1, -1],
                    [-1, 3, -1, 23, -1, -1, -1, -1, -1],
                    [-1, 3, 31, 23, -1, -1, -1, -1, -1],
                    [-1, -1, 31, 23, -1, -1, -1, -1, -1],
                    [2000, -1, -1, 23, -1, -1, -1, -1, -1],
                    [2000, -1, -1, -1, 59, -1, -1, -1, -1],
                    [2000, -1, -1, -1, -1, 59, -1, -1, -1],
                    [2000, -1, -1, -1, -1, -1, 999999, -1, -1],
                    [2000, -1, -1, 23, 59, -1, -1, -1, -1],
                    [2000, -1, -1, 23, 59, 59, -1, -1, -1],
                    [2000, -1, -1, 23, 59, 59, 999999, -1, -1],
                    [2000, 3, 31, 23, 59, 59, 999999, -1, -1],
                ):
                    kwargs = {
                        "year": args[0],
                        "month": args[1],
                        "day": args[2],
                        "hour": args[3],
                        "minute": args[4],
                        "second": args[5],
                        "microsecond": args[6],
                    }
                    self.assertEqualPtDt_unit(
                        pt.to_datetime(**kwargs), dt.to_datetime(**kwargs), unit
                    )

        # to_date()
        for base_dt in base_dts:
            for unit in my_units:
                dt = Pydt.fromdatetime(base_dt)
                pt = Pddt.fromdatetime(base_dt, unit=unit)
                for args in (
                    [2000, -1, -1],
                    [2000, 3, -1],
                    [2000, 3, 31],
                    [-1, 3, -1],
                    [-1, 3, 31],
                    [-1, -1, 31],
                ):
                    kwargs = {
                        "year": args[0],
                        "month": args[1],
                        "day": args[2],
                    }
                    self.assertEqualPtDt_unit(
                        pt.to_date(**kwargs), dt.to_date(**kwargs), unit
                    )

        # to_time()
        for base_dt in base_dts:
            for unit in my_units:
                dt = Pydt.fromdatetime(base_dt)
                pt = Pddt.fromdatetime(base_dt, unit=unit)
                for args in (
                    [23, -1, -1, -1],
                    [-1, 59, -1, -1],
                    [-1, -1, 59, -1],
                    [-1, -1, -1, 999999],
                    [23, 59, -1, -1],
                    [23, 59, 59, -1],
                    [23, 59, 59, 999999],
                ):
                    kwargs = {
                        "hour": args[0],
                        "minute": args[1],
                        "second": args[2],
                        "microsecond": args[3],
                    }
                    self.assertEqualPtDt_unit(
                        pt.to_time(**kwargs), dt.to_time(**kwargs), unit
                    )

        # to_first_of / to_last_of
        to_units = ("Y", "Q", "M", "W", "Feb", "Sep")
        for base_dt in base_dts:
            for unit in my_units:
                dt = Pydt.fromdatetime(base_dt)
                pt = Pddt.fromdatetime(base_dt, unit=unit)
                for to_unit in ("Y", "Q", "M", "W", "Feb", "Sep"):
                    self.assertEqualPtDt_unit(
                        pt.to_first_of(to_unit), dt.to_first_of(to_unit), unit
                    )
                    self.assertEqualPtDt_unit(
                        pt.to_last_of(to_unit), dt.to_last_of(to_unit), unit
                    )

        # to_start_of / to_end_of
        to_units += ("D", "h", "m", "s", "ms", "us")
        for base_dt in base_dts:
            for unit in my_units:
                dt = Pydt.fromdatetime(base_dt)
                pt = Pddt.fromdatetime(base_dt, unit=unit)
                for to_unit in to_units:
                    self.assertEqualPtDt_unit(
                        pt.to_start_of(to_unit), dt.to_start_of(to_unit), unit
                    )
                    pt_res = pt.to_end_of(to_unit)
                    dt_res = dt.to_end_of(to_unit)
                    self.assertEqualPtDt_unit(pt_res, dt_res, unit)

        # round()
        base_dts = [
            Pydt(2023, 8, 17, 13, 31, 31, 666666),
            Pydt(2023, 5, 14, 11, 29, 28, 444444),
            Pydt(1800, 8, 17, 13, 31, 31, 666666),
            Pydt(1800, 5, 14, 11, 29, 28, 444444),
            Pydt(1800, 6, 15, 12, 30, 30, 500000, datetime.UTC),
        ]
        to_units = ("ns", "us", "ms", "s", "min", "h", "D")
        for base_dt in base_dts:
            for unit in my_units:
                for to_unit in to_units:
                    dt = pd.DatetimeIndex([base_dt]).as_unit(unit)
                    pt = Pddt.fromdatetime(base_dt, unit=unit)
                    dt_res = dt.round(to_unit)
                    pt_res = pt.round(to_unit)
                    self.assertEqual(dt_res[0], pt_res[0])

        # ceil()
        for base_dt in base_dts + [pd.Timestamp.min]:
            for unit in my_units:
                for to_unit in to_units:
                    dt = pd.DatetimeIndex([base_dt]).as_unit(unit)
                    pt = Pddt.fromdatetime(base_dt, unit=unit)
                    dt_res = dt.ceil(to_unit)
                    pt_res = pt.ceil(to_unit)
                    self.assertEqual(dt_res[0], pt_res[0])

        # floor()
        for base_dt in base_dts + [pd.Timestamp.max]:
            for unit in my_units:
                for to_unit in to_units:
                    dt = pd.DatetimeIndex([base_dt]).as_unit(unit)
                    pt = Pddt.fromdatetime(base_dt, unit=unit)
                    dt_res = dt.floor(to_unit)
                    pt_res = pt.floor(to_unit)
                    self.assertEqual(dt_res[0], pt_res[0])

        self.log_ended(test)

    def test_calendar(self) -> None:
        test = "Calendar"
        self.log_start(test)

        base_dts = [
            Pydt(2023, 8, 17, 13, 31, 31, 666666),
            Pydt(2023, 5, 14, 11, 29, 28, 444444),
            Pydt(1800, 8, 17, 13, 31, 31, 666666),
            Pydt(1800, 5, 14, 11, 29, 28, 444444),
            Pydt(2023, 5, 14, 11, 29, 28, 444444, ZoneInfo("CET")),
            Pydt(1800, 6, 15, 12, 30, 30, 500000, datetime.UTC),
            pd.Timestamp.min,
            pd.Timestamp.max,
        ]
        my_units = ("ns", "us", "ms", "s")
        for base_dt in base_dts:
            for unit in my_units:
                dt = Pydt.fromdatetime(base_dt)
                pt = Pddt.fromdatetime(base_dt, unit=unit)
                # ISO ------------------------------------------------------------
                # isocalendar()
                dt_iso = dt.isocalendar()
                pt_iso = pt.isocalendar().iloc[0]
                self.assertEqual(
                    (dt_iso["year"], dt_iso["week"], dt_iso["weekday"]),
                    (pt_iso["year"], pt_iso["week"], pt_iso["weekday"]),
                )
                # isoyear()
                self.assertEqual(dt.isoyear(), pt.isoyear()[0])
                # isoweek()
                self.assertEqual(dt.isoweek(), pt.isoweek()[0])
                # isoweekday()
                self.assertEqual(dt.isoweekday(), pt.isoweekday()[0])

                # Year -----------------------------------------------------------
                self.assertEqual(dt.year, pt.year[0])
                self.assertEqual(dt.is_leap_year(), pt.is_leap_year()[0])
                self.assertEqual(dt.is_long_year(), pt.is_long_year()[0])
                self.assertEqual(dt.leap_bt_year(1970), pt.leap_bt_year(1970)[0])
                self.assertEqual(dt.days_in_year(), pt.days_in_year()[0])
                self.assertEqual(dt.days_bf_year(), pt.days_bf_year()[0])
                self.assertEqual(dt.days_of_year(), pt.days_of_year()[0])
                self.assertEqual(dt.is_year(2023), pt.is_year(2023)[0])

                # Quarter --------------------------------------------------------
                self.assertEqual(dt.quarter, pt.quarter[0])
                self.assertEqual(dt.days_in_quarter(), pt.days_in_quarter()[0])
                self.assertEqual(dt.days_bf_quarter(), pt.days_bf_quarter()[0])
                self.assertEqual(dt.days_of_quarter(), pt.days_of_quarter()[0])
                self.assertEqual(dt.is_quarter(3), pt.is_quarter(3)[0])

                # Month ----------------------------------------------------------
                self.assertEqual(dt.month, pt.month[0])
                self.assertEqual(dt.days_in_month(), pt.days_in_month()[0])
                self.assertEqual(dt.days_bf_month(), pt.days_bf_month()[0])
                self.assertEqual(dt.days_of_month(), pt.days_of_month()[0])
                self.assertEqual(dt.is_month("Aug"), pt.is_month("Aug")[0])

                # Weekday --------------------------------------------------------
                self.assertEqual(dt.weekday, pt.weekday[0])
                for wkd in (3, "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"):
                    self.assertEqual(dt.is_weekday(wkd), pt.is_weekday(wkd)[0])

                # Day ------------------------------------------------------------
                self.assertEqual(dt.day, pt.day[0])
                self.assertEqual(dt.is_day(14), pt.is_day(14)[0])

                # Time -----------------------------------------------------------
                if unit in ("ns", "us"):
                    self.assertEqual(
                        (dt.hour, dt.minute, dt.second, dt.millisecond, dt.microsecond),
                        (
                            pt.hour[0],
                            pt.minute[0],
                            pt.second[0],
                            pt.millisecond[0],
                            pt.microsecond[0],
                        ),
                    )

                # Date&Time ------------------------------------------------------
                # is_first_of / is_last_of
                units = ["W", "M", "Q", "Y", "Jan", "Feb", "Dec"]
                for unit in units:
                    self.assertEqual(
                        dt.to_first_of(unit).is_first_of(unit),
                        pt.to_first_of(unit).is_first_of(unit)[0],
                    )
                    self.assertEqual(
                        dt.to_last_of(unit).is_last_of(unit),
                        pt.to_last_of(unit).is_last_of(unit)[0],
                    )

                # is_start_of / is_end_of
                units += ["us", "ms", "s", "m", "h", "D", "Mon", "Sun"]
                for unit in units:
                    self.assertEqual(
                        dt.to_start_of(unit).is_start_of(unit),
                        pt.to_start_of(unit).is_start_of(unit)[0],
                    )
                    self.assertEqual(
                        dt.to_end_of(unit).is_end_of(unit),
                        pt.to_end_of(unit).is_end_of(unit)[0],
                    )

        self.log_ended(test)

    def test_timezone(self) -> None:
        test = "Timezone"
        self.log_start(test)

        # Compare to DatetimeIndex
        data = [
            "2023-01-01 01:01:01",
            "2023-01-02 01:01:01",
            "2023-01-03 01:01:01",
        ]
        for tz in (datetime.UTC, ZoneInfo("CET")):
            # tz_localize(): naive -> aware
            dt = pd.DatetimeIndex(data).tz_localize(tz)
            pt = Pddt(data).tz_localize(tz)
            self.assertTrue(np.equal(dt, pt).all())
            # tz_localize(): aware -> naive
            dt = pd.DatetimeIndex(data, tz=tz).tz_localize(None)
            pt = Pddt(data, tz=tz).tz_localize(None)
            self.assertTrue(np.equal(dt, pt).all())
            # tz_convert()
            for targ_tz in (datetime.UTC, ZoneInfo("CET")):
                dt = pd.DatetimeIndex(data, tz=tz).tz_convert(targ_tz)
                pt = Pddt(data, tz=tz).tz_convert(targ_tz)
                self.assertTrue(np.equal(dt, pt).all())

        # Test out of bounds
        for tz in (ZoneInfo("CET"), ZoneInfo("PST8PDT")):
            pt = Pddt([pd.Timestamp.min, pd.Timestamp.max]).tz_localize(tz)
            self.assertTrue(pt.unit == "us")  # auto convert to us to avoid overflow

        # Compare to Pydt
        base_dts = [
            Pydt(2023, 8, 17, 13, 31, 31, 666666),
            Pydt(2023, 5, 14, 11, 29, 28, 444444),
            pd.Timestamp.max,
        ]
        my_units = ("ns", "us", "ms", "s")
        for base_dt in base_dts:
            for unit in my_units:
                for tz in (None, datetime.UTC, ZoneInfo("CET")):
                    dt = Pydt.fromdatetime(base_dt).replace(tzinfo=tz)
                    pt = Pddt.fromdatetime(base_dt, unit=unit).replace(tzinfo=tz)
                    # . is_utc()
                    self.assertEqual(dt.is_utc(), pt.is_utc())
                    # . tz_name()
                    self.assertEqual(dt.tzname(), pt.tzname())
                    for targ_tz in (
                        None,
                        datetime.UTC,
                        ZoneInfo("CET"),
                        datetime.timezone(datetime.timedelta(hours=-5)),
                    ):
                        # astimezone()
                        dt = Pydt.fromdatetime(base_dt).replace(tzinfo=tz)
                        dt = dt.astimezone(targ_tz)
                        pt = Pddt.fromdatetime(base_dt, unit=unit).replace(tzinfo=tz)
                        pt = pt.astimezone(targ_tz)
                        self.assertEqualPtDt_unit(pt, dt, unit)
                        # tz_switch()
                        for naive in (True, False):
                            dt = Pydt.fromdatetime(base_dt).replace(tzinfo=tz)
                            pt = Pddt.fromdatetime(base_dt, unit=unit).replace(
                                tzinfo=tz
                            )
                            dt = dt.tz_switch(
                                targ_tz, datetime.UTC if tz is None else tz, naive
                            )
                            pt = pt.tz_switch(
                                targ_tz, datetime.UTC if tz is None else tz, naive
                            )
                            self.assertEqualPtDt_unit(pt, dt, unit)

        base_dts += [
            Pydt(1800, 8, 17, 13, 31, 31, 666666),
            Pydt(1800, 5, 14, 11, 29, 28, 444444),
            pd.Timestamp.min,
        ]
        for base_dt in base_dts:
            for unit in my_units:
                for tz in (datetime.UTC, ZoneInfo("CET")):
                    # tz_localize(): naive -> aware
                    dt = Pydt.fromdatetime(base_dt).tz_localize(tz)
                    pt = Pddt.fromdatetime(base_dt, unit=unit).tz_localize(tz)
                    self.assertEqualPtDt_unit(pt, dt, unit)
                    # tz_localize(): aware -> naive
                    dt = dt.tz_localize(None)
                    pt = pt.tz_localize(None)
                    self.assertEqualPtDt_unit(pt, dt, unit)
                    # tz_convert()
                    for targ_tz in (datetime.UTC, ZoneInfo("CET")):
                        dt = (
                            Pydt.fromdatetime(base_dt)
                            .tz_localize(tz)
                            .tz_convert(targ_tz)
                        )
                        pt = (
                            Pddt.fromdatetime(base_dt, unit=unit)
                            .tz_localize(tz)
                            .tz_convert(targ_tz)
                        )
                        self.assertEqualPtDt_unit(pt, dt, unit)

        self.log_ended(test)

    def test_arithmetic(self) -> None:
        test = "Arithmetic"
        self.log_start(test)

        base_dts = [
            Pydt(2023, 8, 17, 13, 31, 31, 666666),
            Pydt(2023, 5, 14, 11, 29, 28, 444444),
            Pydt(1800, 8, 17, 13, 31, 31, 666666),
            Pydt(1800, 5, 14, 11, 29, 28, 444444),
            pd.Timestamp.min,
            pd.Timestamp.max,
        ]
        my_units = ("ns", "us", "ms", "s")
        for base_dt in base_dts:
            for unit in my_units:
                for tz in (None, datetime.UTC, ZoneInfo("CET")):
                    for args in [
                        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
                        (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
                        (1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
                        (1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
                        (1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
                        (1, 1, 1, 1, 1, 0, 0, 0, 0, 0),
                        (1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
                        (1, 1, 1, 1, 1, 1, 1, 0, 0, 0),
                        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                        (1, 1, 1, 1, 1, 23, 1, 1, 1, 1),
                        (1, 1, 1, 1, 1, 23, 59, 1, 1, 1),
                        (1, 1, 1, 1, 1, 23, 59, 59, 1, 1),
                        (1, 1, 1, 1, 1, 23, 59, 59, 999, 1),
                        (1, 1, 1, 1, 1, 23, 59, 59, 999, 999),
                        (1, 1, 1, 1, 1, 24, 1, 1, 1, 1),
                        (1, 1, 1, 1, 1, 24, 60, 1, 1, 1),
                        (1, 1, 1, 1, 1, 24, 60, 60, 1, 1),
                        (1, 1, 1, 1, 1, 24, 60, 60, 1000, 1),
                        (1, 1, 1, 1, 1, 24, 60, 60, 1000, 1000000),
                        (1000, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                        (1000, 1, 1, 1, 1, 23, 1, 1, 1, 1),
                        (1000, 1, 1, 1, 1, 23, 59, 1, 1, 1),
                        (1000, 1, 1, 1, 1, 23, 59, 59, 1, 1),
                        (1000, 1, 1, 1, 1, 23, 59, 59, 999, 1),
                        (1000, 1, 1, 1, 1, 23, 59, 59, 999, 999),
                        (1000, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                        (1000, 1, 1, 1, 1, 24, 1, 1, 1, 1),
                        (1000, 1, 1, 1, 1, 24, 60, 1, 1, 1),
                        (1000, 1, 1, 1, 1, 24, 60, 60, 1, 1),
                        (1000, 1, 1, 1, 1, 24, 60, 60, 1000, 1),
                        (1000, 1, 1, 1, 1, 24, 60, 60, 1000, 1000000),
                    ]:
                        # add()
                        kwargs = {
                            "years": args[0],
                            "quarters": args[1],
                            "months": args[2],
                            "weeks": args[3],
                            "days": args[4],
                            "hours": args[5],
                            "minutes": args[6],
                            "seconds": args[7],
                            "milliseconds": args[8],
                            "microseconds": args[9],
                        }
                        dt = Pydt.fromdatetime(base_dt).replace(tzinfo=tz)
                        pt = Pddt.fromdatetime(base_dt, unit=unit).replace(tzinfo=tz)
                        # add()
                        self.assertEqualPtDt_unit(
                            pt.add(**kwargs), dt.add(**kwargs), unit
                        )
                        # sub()
                        self.assertEqualPtDt_unit(
                            pt.sub(**kwargs), dt.sub(**kwargs), unit
                        )

        # timezone-naive
        diff_units = ("Y", "Q", "M", "W", "D", "h", "m", "s", "ms", "us")
        for base_dt in base_dts:
            for unit in ("us", "ns"):
                for comp_dt in (
                    Pydt(2023, 8, 17, 13, 31, 31, 666666),
                    Pydt(2023, 5, 14, 11, 29, 28, 444444),
                    Pydt(1800, 8, 17, 13, 31, 31, 666666),
                    Pydt(1800, 5, 14, 11, 29, 28, 444444),
                ):
                    dt = Pydt.fromdatetime(base_dt)
                    pt = Pddt.fromdatetime(base_dt, unit=unit)
                    for diff_unit in diff_units:
                        for incl in ("neither", "one", "both"):
                            for absl in (True, False):
                                self.assertEqual(
                                    dt.diff(comp_dt, diff_unit, absl, incl),
                                    pt.diff(comp_dt, diff_unit, absl, incl)[0],
                                )

        # timezone-aware
        diff_units = ("Y", "Q", "M", "W", "D", "h", "m", "s", "ms", "us")
        tzinfos = (datetime.UTC, ZoneInfo("Asia/Shanghai"))
        for base_dt in base_dts:
            for unit in ("us", "ns"):
                for comp_dt in (
                    Pydt(2023, 8, 17, 13, 31, 31, 666666),
                    Pydt(2023, 5, 14, 11, 29, 28, 444444),
                    Pydt(1800, 8, 17, 13, 31, 31, 666666),
                    Pydt(1800, 5, 14, 11, 29, 28, 444444),
                ):
                    for my_tz in tzinfos:
                        for to_tz in tzinfos:
                            dt = Pydt.fromdatetime(base_dt).replace(tzinfo=my_tz)
                            pt = Pddt.fromdatetime(base_dt, unit=unit).replace(
                                tzinfo=my_tz
                            )
                            for diff_unit in diff_units:
                                for incl in ("neither", "one", "both"):
                                    comp_dt = comp_dt.replace(tzinfo=to_tz)
                                    for absl in (True, False):
                                        self.assertEqual(
                                            dt.diff(comp_dt, diff_unit, absl, incl),
                                            pt.diff(comp_dt, diff_unit, absl, incl)[0],
                                        )

        self.log_ended(test)

    def test_comparision(self) -> None:
        test = "Comparision"
        self.log_start(test)

        base_dts = [
            Pydt(2023, 8, 17, 13, 31, 31, 666666),
            Pydt(2023, 5, 14, 11, 29, 28, 444444),
            Pydt(1800, 8, 17, 13, 31, 31, 666666),
            Pydt(1800, 5, 14, 11, 29, 28, 444444),
            pd.Timestamp.min,
            pd.Timestamp.max,
        ]
        my_units = ("ns", "us", "ms", "s")
        for base_dt in base_dts:
            for unit in my_units:
                for tz in (None, datetime.UTC, ZoneInfo("CET"), ZoneInfo("PST8PDT")):
                    dt = Pydt.fromdatetime(base_dt).replace(tzinfo=tz)
                    pt = Pddt.fromdatetime(base_dt, unit=unit).replace(tzinfo=tz)
                    # is_past()
                    self.assertEqual(dt.is_past(), pt.is_past()[0])
                    # is_future()
                    self.assertEqual(dt.is_future(), pt.is_future()[0])

        self.log_ended(test)


if __name__ == "__main__":
    Test_Pddt().test_all()
