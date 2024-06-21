import datetime
import unittest

import pytz

from quantum_inferno.utilities import date_time as dat


class TimeUnitTest(unittest.TestCase):
    def test_picos_to_seconds(self):
        picos = 100. * dat.time_unit_dict["ps"]
        self.assertAlmostEqual(picos, 1e-10, 2)

    def test_nanos_to_seconds(self):
        nanos = 100. * dat.time_unit_dict["ns"]
        self.assertAlmostEqual(nanos, 1e-7, 2)

    def test_micros_to_seconds(self):
        micros = 100. * dat.time_unit_dict["us"]
        self.assertAlmostEqual(micros, 1e-5, 2)

    def test_millis_to_seconds(self):
        millis = 100. * dat.time_unit_dict["ms"]
        self.assertAlmostEqual(millis, .1, 2)

    def test_seconds_to_seconds(self):
        seconds = 100. * dat.time_unit_dict["s"]
        self.assertAlmostEqual(seconds, 100, 2)

    def test_minutes_to_seconds(self):
        minutes = 100. * dat.time_unit_dict["m"]
        self.assertAlmostEqual(minutes, 6000, 2)

    def test_hours_to_seconds(self):
        hours = 100. * dat.time_unit_dict["h"]
        self.assertAlmostEqual(hours, 360000, 2)

    def test_days_to_seconds(self):
        days = 100. * dat.time_unit_dict["d"]
        self.assertAlmostEqual(days, 8640000, 2)

    def test_weeks_to_seconds(self):
        weeks = 100. * dat.time_unit_dict["weeks"]
        self.assertAlmostEqual(weeks, 60480000, 2)

    def test_months_to_seconds(self):
        months = 100. * dat.time_unit_dict["months"]
        self.assertAlmostEqual(months, 262800000, 2)

    def test_years_to_seconds(self):
        years = 100. * dat.time_unit_dict["years"]
        self.assertAlmostEqual(years, 3153600000, 2)


class ConvertTimeTest(unittest.TestCase):
    def test_convert_time_to_seconds(self):
        time_seconds = dat.convert_time_unit(100., "m", "s")
        self.assertAlmostEqual(time_seconds, 6000, 2)

    def test_fail_convert(self):
        self.assertRaises(ValueError, dat.convert_time_unit, 0, "x", "y")


class DatetimeToTimestampTest(unittest.TestCase):
    def test_datetime_to_timestamp(self):
        datetime_obj = dat.datetime(2021, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        timestamp = dat.utc_datetime_to_utc_timestamp(datetime_obj, "s")
        self.assertEqual(timestamp, 1609459200.0)

    def test_fail_timestamp(self):
        self.assertRaises(ValueError, dat.utc_datetime_to_utc_timestamp, 0, "x")


class TimestampToDatetimeTest(unittest.TestCase):
    def test_timestamp_to_datetime(self):
        datetime_obj = dat.utc_timestamp_to_utc_datetime(1609459200.0, "s")
        self.assertEqual(datetime_obj, dat.datetime(2021, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc))

    def test_fail_datetime(self):
        self.assertRaises(ValueError, dat.utc_timestamp_to_utc_datetime, 0, "x")


class SetDatetimeToUtcTest(unittest.TestCase):
    def test_set_datetime_to_utc(self):
        datetime_obj = dat.datetime(2021, 1, 1, 0, 0, 0, tzinfo=pytz.timezone("HST"))
        utc_datetime_obj = dat.set_datetime_to_utc(datetime_obj)
        self.assertEqual(utc_datetime_obj, dat.datetime(2021, 1, 1, 10, 0, 0, tzinfo=datetime.timezone.utc))

    def test_set_datetime_to_utc_warning(self):
        datetime_obj = dat.datetime(2021, 1, 1, 0, 0, 0)
        utc_datetime_obj = dat.set_datetime_to_utc(datetime_obj, tzinfo_warning=False)
        self.assertEqual(utc_datetime_obj, dat.datetime(2021, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc))


class SetTimestampToUtcTest(unittest.TestCase):
    def test_set_timestamp_to_utc(self):
        utc_timestamp = dat.set_timestamp_to_utc(1609462800., 1, "s")
        self.assertEqual(utc_timestamp, 1609459200)

        utc_timestamp = dat.set_timestamp_to_utc(1609455600., -1, "s")
        self.assertEqual(utc_timestamp, 1609459200)
