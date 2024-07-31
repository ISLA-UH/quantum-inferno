import datetime
import unittest

import pytz

from quantum_inferno.utilities import date_time as dat


class TimeUnitTest(unittest.TestCase):
    def test_picos_to_seconds(self):  # use large number to avoid floating point error
        picos = 1e12 * dat.time_unit_dict["ps"]
        self.assertEqual(picos, 1.0)

    def test_nanos_to_seconds(self):  # use large number to avoid floating point error
        nanos = 1e9 * dat.time_unit_dict["ns"]
        self.assertEqual(nanos, 1.0)

    def test_micros_to_seconds(self):  # use large number to avoid floating point error
        micros = 1e6 * dat.time_unit_dict["us"]
        self.assertEqual(micros, 1.0)

    def test_millis_to_seconds(self):
        millis = 100.0 * dat.time_unit_dict["ms"]
        self.assertEqual(millis, 0.1)

    def test_seconds_to_seconds(self):
        seconds = 100.0 * dat.time_unit_dict["s"]
        self.assertEqual(seconds, 100)

    def test_minutes_to_seconds(self):
        minutes = 100.0 * dat.time_unit_dict["m"]
        self.assertEqual(minutes, 6000)

    def test_hours_to_seconds(self):
        hours = 100.0 * dat.time_unit_dict["h"]
        self.assertEqual(hours, 360000)

    def test_days_to_seconds(self):
        days = 100.0 * dat.time_unit_dict["d"]
        self.assertEqual(days, 8640000)

    def test_weeks_to_seconds(self):
        weeks = 100.0 * dat.time_unit_dict["weeks"]
        self.assertEqual(weeks, 60480000)

    def test_months_to_seconds(self):
        months = 100.0 * dat.time_unit_dict["months"]
        self.assertEqual(months, 262800000)

    def test_years_to_seconds(self):
        years = 100.0 * dat.time_unit_dict["years"]
        self.assertEqual(years, 3153600000)


class ConvertTimeTest(unittest.TestCase):
    def test_convert_time_to_seconds(self):
        time_seconds = dat.convert_time_unit(100.0, "m", "s")
        self.assertEqual(time_seconds, 6000)

    def test_fail_convert(self):
        self.assertRaises(ValueError, dat.convert_time_unit, 0, "x", "y")


class DatetimeToTimestampTest(unittest.TestCase):
    def test_datetime_to_timestamp(self):
        datetime_obj = dat.datetime(2021, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        timestamp = dat.utc_datetime_to_utc_timestamp(datetime_obj, "s")
        self.assertEqual(timestamp, 1609459200.0)

    def test_datetime_to_timestamp_no_tz(self):
        datetime_obj = dat.datetime(2021, 1, 1, 0, 0, 0)
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
        utc_timestamp = dat.set_timestamp_to_utc(1609462800.0, 1, "s")
        self.assertEqual(utc_timestamp, 1609459200)

        utc_timestamp = dat.set_timestamp_to_utc(1609455600.0, -1, "s")
        self.assertEqual(utc_timestamp, 1609459200)

    def test_fail_timestamp(self):
        self.assertRaises(ValueError, dat.set_timestamp_to_utc, 0, 0, "x")


class GetDatetimeFromTimestampToUtcTest(unittest.TestCase):
    def test_get_datetime_from_timestamp_to_utc(self):
        datetime_obj = dat.get_datetime_from_timestamp_to_utc(1609459200.0, -1.0, "s")
        # print(datetime_obj)
        self.assertEqual(datetime_obj, dat.datetime(2021, 1, 1, 1, 0, 0, tzinfo=datetime.timezone.utc))

    def test_fail_get_datetime(self):
        self.assertRaises(ValueError, dat.get_datetime_from_timestamp_to_utc, 1609459200.0, -1.0, "x")
