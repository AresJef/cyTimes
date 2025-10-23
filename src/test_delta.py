from typing import Literal
from random import randint
import time, unittest, datetime
import numpy as np, pandas as pd, pendulum as pl
from cytimes.pydt import Pydt
from cytimes.delta import Delta
from dateutil.relativedelta import relativedelta


class TestCase(unittest.TestCase):
    name: str = "Case"

    def test_all(self) -> None:
        pass

    # Utils
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


class TestDelta(TestCase):
    name = "Delta"
    date: datetime.date = datetime.date(5000, 1, 2)
    dt: datetime.datetime = datetime.datetime(5000, 1, 2, 3, 4, 5, 6)
    td: datetime.timedelta = datetime.timedelta(1, 1, 1, 1, 1, 1, 1)
    ctd: Delta = Delta(1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    # fmt: off
    rtd = relativedelta(
        years=1, months=1, days=1, weeks=1, hours=1, minutes=1, seconds=1, microseconds=1, 
        year=1, month=1, day=1, hour=1, minute=1, second=1, microsecond=1,
    )
    # fmt: on

    def test_all(self, rounds: int) -> None:
        self.test_from_relativedelta()
        self.test_addition(rounds, "Relative")
        self.test_addition(rounds, "Absolute")
        self.test_addition(rounds, "Mixed")
        self.test_subtraction(rounds, "Relative")
        self.test_subtraction(rounds, "Absolute")
        self.test_subtraction(rounds, "Mixed")
        self.test_negate_n_absolute(rounds)
        self.test_equal_timedelta(rounds)
        self.test_equal_relativedelta(rounds, "Relative")
        self.test_equal_relativedelta(rounds, "Absolute")
        self.test_equal_relativedelta(rounds, "Mixed")
        self.test_boolean()
        self.test_typing()

    def test_from_relativedelta(self) -> None:
        test = "From Relativedelta"
        self.log_start(test)

        rtd = relativedelta(days=1)
        ctd = Delta.from_relativedelta(rtd)
        self.assertIsInstance(ctd, Delta)

        self.log_ended(test)

    def test_addition(
        self,
        rounds: int,
        mode: Literal["Relative", "Absolute", "Mixed"],
    ) -> None:
        test = f"Addition ({mode})"
        self.log_start(test)

        for _ in range(rounds):
            ctd, rtd = self._random_deltas(mode)
            # Left Addition ------------------------------------------------------
            # . Delta + date
            try:
                r_res = rtd + self.date
            except (ValueError, OverflowError):
                pass
            else:
                self.assertEqual(ctd + self.date, r_res.date())

            # . Delta + datetime
            try:
                r_res = rtd + self.dt
            except (ValueError, OverflowError):
                pass
            else:
                self.assertEqual(ctd + self.dt, r_res)

            # . Delta + timedelta
            try:
                r_res = rtd + self.td
            except (ValueError, OverflowError):
                pass
            else:
                self.assertEqual(ctd + self.td, r_res)

            # . Delta + relativedelta
            self.assertEqual(ctd + self.rtd, rtd + self.rtd)

            # Right Addition -----------------------------------------------------
            # . date + Delta
            try:
                r_res = self.date + rtd
            except (ValueError, OverflowError):
                pass
            else:
                self.assertEqual(self.date + ctd, r_res.date())

            # . datetime + Delta
            try:
                r_res = self.dt + rtd
            except (ValueError, OverflowError):
                pass
            else:
                self.assertEqual(self.dt + ctd, r_res)

            # . timedelta + Delta
            try:
                r_res = self.td + rtd
            except (ValueError, OverflowError):
                pass
            else:
                self.assertEqual(self.td + ctd, r_res)

            # . relativedelta + Delta
            self.assertEqual(self.rtd + ctd, self.rtd + rtd)

            # Left & Right Addition ----------------------------------------------
            # . Delta + Delta
            self.assertEqual(ctd + self.ctd, rtd + self.ctd)

        self.log_ended(test)

    def test_subtraction(
        self,
        rounds: int,
        mode: Literal["Relative", "Absolute", "Mixed"],
    ) -> None:
        test = f"Subtraction ({mode})"
        self.log_start(test)

        for _ in range(rounds):
            ctd, rtd = self._random_deltas(mode)
            # Left Subtraction ------------------------------------------------------
            # . Delta - timedelta
            try:
                r_res = rtd + (-self.td)
            except (ValueError, OverflowError):
                pass
            else:
                self.assertEqual(ctd - self.td, r_res)

            # . Delta - relativedelta
            self.assertEqual(ctd - self.rtd, rtd - self.rtd)

            # Right Subtraction -----------------------------------------------------
            # . date - Delta
            try:
                r_res = self.date - rtd
            except (ValueError, OverflowError):
                pass
            else:
                self.assertEqual(self.date - ctd, r_res.date())

            # . datetime - Delta
            try:
                r_res = self.dt - rtd
            except (ValueError, OverflowError):
                pass
            else:
                self.assertEqual(self.dt - ctd, r_res)

            # . timedelta - Delta
            try:
                r_res = self.td - rtd
            except (ValueError, OverflowError):
                pass
            else:
                self.assertEqual(self.td - ctd, r_res)

            # . relativedelta - Delta
            self.assertEqual(self.rtd - ctd, self.rtd - rtd)

            # Left & Right Subtraction ----------------------------------------------
            self.assertEqual(ctd - self.ctd, rtd - self.ctd)

        self.log_ended(test)

    def test_negate_n_absolute(self, rounds: int) -> None:
        test = "Negate & Absolute"
        self.log_start(test)

        for _ in range(rounds):
            ctd, _ = self._random_deltas("Relative")
            ctd_neg = Delta(
                years=-ctd.years,
                months=-ctd.months,
                days=-ctd.days,
                hours=-ctd.hours,
                minutes=-ctd.minutes,
                seconds=-ctd.seconds,
                microseconds=-ctd.microseconds,
            )
            # Negate
            self.assertEqual(-ctd, ctd_neg)
            # Absolute
            if ctd_neg.years < 0 and ctd_neg.days < 0:
                self.assertEqual(abs(ctd_neg), ctd)

        # Negate
        args = {
            "years": 1000,
            "months": 1000,
            "days": 1000,
            "hours": 1000,
            "minutes": 1000,
            "seconds": 1000,
            "microseconds": 1000,
        }
        ctd = Delta(**args)
        ctd_neg = Delta(**{k: -v for k, v in args.items()})
        self.assertEqual(-ctd, ctd_neg)
        # Absolute
        self.assertEqual(abs(ctd_neg), ctd)

        self.log_ended(test)

    def test_equal_timedelta(self, rounds: int) -> None:
        test = "Equal Timedelta"
        self.log_start(test)

        for _ in range(rounds):
            days = randint(-100_000, 100_000)
            seconds = randint(-1_000_000, 1_000_000)
            microseconds = randint(-10_000_000, 10_000_000)

            td = datetime.timedelta(days, seconds, microseconds)
            ctd = Delta(days=days, seconds=seconds, microseconds=microseconds)

            # . Delta == timedelta
            self.assertEqual(ctd, td)

            # . datetime + Delta/timedelta
            try:
                dt = self.dt + td
            except (ValueError, OverflowError):
                pass
            else:
                self.assertEqual(self.dt + ctd, dt)

            # . Delta/timedelta + datetime
            try:
                dt = td + self.dt
            except (ValueError, OverflowError):
                pass
            else:
                self.assertEqual(ctd + self.dt, dt)

        self.log_ended(test)

    def test_equal_relativedelta(
        self,
        rounds: int,
        mode: Literal["Relative", "Absolute", "Mixed"],
    ) -> None:
        test = "Equal Relativedelta"
        self.log_start(test)

        for _ in range(rounds):
            ctd, rtd = self._random_deltas(mode)
            self.assertEqual(ctd, rtd)

        self.log_ended(test)

    def test_boolean(self) -> None:
        test = "Boolean"
        self.log_start(test)

        self.assertFalse(Delta())
        self.assertTrue(Delta(seconds=1))
        self.assertTrue(Delta(second=1))

        self.log_ended(test)

    def test_typing(self) -> None:
        test = "Typing"
        self.log_start(test)

        ctd = Delta(microseconds=1)

        # return datetime.datetime (instance & subclass)
        for obj in (datetime.datetime.now(), Pydt.now(), pd.Timestamp.now(), pl.now()):
            self.assertTrue(isinstance(ctd + obj, obj.__class__))
            self.assertTrue(isinstance(obj + ctd, obj.__class__))
            self.assertTrue(isinstance(obj - ctd, obj.__class__))
            with self.assertRaises(TypeError):
                ctd - obj
        # . Delta + numpy.datetime64
        self.assertTrue(isinstance(ctd + np.datetime64(1, "us"), datetime.datetime))
        self.assertTrue(isinstance(ctd + np.datetime64(1, "ns"), datetime.datetime))
        # . np.datetime64 + Delta
        self.assertTrue(isinstance(np.datetime64(1, "us") + ctd, datetime.datetime))
        with self.assertRaises(TypeError):
            np.datetime64(1, "ns") + ctd
        # . numpy.datetime64 - Delta
        self.assertTrue(isinstance(np.datetime64(1, "us") - ctd, datetime.datetime))
        with self.assertRaises(TypeError):
            np.datetime64(1, "ns") - ctd

        # return datetime.date (instance & subclass)
        for obj in (datetime.date.today(), pl.Date.today()):
            self.assertTrue(isinstance(ctd + obj, obj.__class__))
            self.assertTrue(isinstance(obj + ctd, obj.__class__))
            self.assertTrue(isinstance(obj - ctd, obj.__class__))
            with self.assertRaises(TypeError):
                ctd - obj

        # return Delta
        for obj in (
            Delta(1),
            datetime.timedelta(1),
            pd.Timedelta(1, "D"),
            relativedelta(days=1),
        ):
            self.assertTrue(isinstance(ctd + obj, Delta))
            self.assertTrue(isinstance(obj + ctd, Delta))
            self.assertTrue(isinstance(obj - ctd, Delta))
            self.assertTrue(isinstance(ctd - obj, Delta))
        # . Delta + numpy.timedelta64
        self.assertEqual(ctd + np.timedelta64(1, "us"), Delta(microseconds=2))
        self.assertEqual(ctd + np.timedelta64(1, "ns"), ctd)
        # . Delta - numpy.timedelta64
        self.assertEqual(ctd - np.timedelta64(1, "us"), Delta(microseconds=0))
        self.assertEqual(ctd - np.timedelta64(1, "ns"), ctd)
        # . numpy.timedelta64 + Delta
        self.assertEqual(np.timedelta64(1, "us") + ctd, Delta(microseconds=2))
        with self.assertRaises(TypeError):
            np.timedelta64(1, "ns") + ctd
        # . numpy.timedelta64 - Delta
        self.assertEqual(np.timedelta64(1, "us") - ctd, Delta(microseconds=0))
        with self.assertRaises(TypeError):
            np.timedelta64(1, "ns") - ctd

        self.log_ended(test)

    # Utils
    def _random_deltas(
        self,
        mode: Literal["Relative", "Absolute", "Mixed"],
    ) -> tuple[Delta, relativedelta]:
        args = {}
        if mode == "Relative" or mode == "Mixed":
            args |= {
                "years": randint(-9999, 9999),
                "months": randint(-1000, 1000),
                "days": randint(-1000, 1000),
                "weeks": randint(-1000, 1000),
                "hours": randint(-1000, 1000),
                "minutes": randint(-1000, 1000),
                "seconds": randint(-10_000, 10_000),
                "microseconds": randint(-10_000_000, 10_000_000),
            }
        if mode == "Absolute" or mode == "Mixed":
            args |= {
                "year": randint(1, 9999),
                "month": randint(1, 12),
                "day": randint(1, 31),
                "hour": randint(0, 23),
                "minute": randint(0, 59),
                "second": randint(0, 59),
                "microsecond": randint(0, 999_999),
            }
        return Delta(**args), relativedelta(**args)


if __name__ == "__main__":
    TestDelta().test_all(100_000)
