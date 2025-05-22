import unittest
from unittest.mock import patch, MagicMock
import datetime
import pytz
import tensorflow as tf # For tf.test.TestCase, or from absl import app, flags, logging / from absl.testing import absltest

from smart_control.reinforcement_learning.utils import time_utils


class TimeUtilsTest(tf.test.TestCase): # Or absltest.TestCase

    def test_get_local_time_str_specific_datetime_and_timezone(self):
        # Scenario 1: dt_obj is naive, representing the target local time
        dt_la = datetime.datetime(2023, 10, 26, 7, 30, 15)
        time_str_la = time_utils.get_local_time_str(dt_la, time_zone_str='America/Los_Angeles')
        self.assertEqual(time_str_la, '2023-10-26_07-30-15')

        dt_berlin = datetime.datetime(2023, 10, 26, 16, 30, 15)
        time_str_berlin = time_utils.get_local_time_str(dt_berlin, time_zone_str='Europe/Berlin')
        self.assertEqual(time_str_berlin, '2023-10-26_16-30-15')

    @patch('smart_control.reinforcement_learning.utils.time_utils.datetime.datetime')
    def test_get_local_time_str_dt_obj_none(self, MockDateTime):
        # Scenario 2: dt_obj is None, uses datetime.datetime.now()
        # Test with UTC
        mock_now_utc = datetime.datetime(2023, 1, 1, 10, 0, 0) # Naive, intended as UTC time
        MockDateTime.now.return_value = mock_now_utc
        time_str_utc = time_utils.get_local_time_str(dt_obj=None, time_zone_str='UTC')
        # pytz.timezone('UTC').localize(mock_now_utc) will attach UTC tz to 10:00
        self.assertEqual(time_str_utc, '2023-01-01_10-00-00')

        # Test with America/New_York
        # Mock now() to return a naive datetime that we intend to be localized to NY
        mock_now_ny = datetime.datetime(2023, 1, 1, 5, 0, 0) # 5 AM
        MockDateTime.now.return_value = mock_now_ny
        time_str_ny = time_utils.get_local_time_str(dt_obj=None, time_zone_str='America/New_York')
        # pytz.timezone('America/New_York').localize(mock_now_ny) attaches NY tz to 5:00
        self.assertEqual(time_str_ny, '2023-01-01_05-00-00')

    def test_get_local_time_str_invalid_timezone(self):
        # Scenario 3: Invalid timezone string
        dt_now = datetime.datetime(2023, 1, 1, 12, 0, 0) # Some datetime
        with self.assertRaises(pytz.exceptions.UnknownTimeZoneError):
            time_utils.get_local_time_str(dt_now, time_zone_str='Invalid/Timezone')

    def test_get_time_zone_valid(self):
        # Scenario 1: Valid timezone string
        tz_ny = time_utils.get_time_zone('America/New_York')
        self.assertIsInstance(tz_ny, datetime.tzinfo) # General check
        self.assertTrue(isinstance(tz_ny, pytz.tzinfo.BaseTzInfo)) # More specific Pytz check
        self.assertEqual(str(tz_ny), 'America/New_York')

        # Scenario 2: Another valid timezone string
        tz_paris = time_utils.get_time_zone('Europe/Paris')
        self.assertIsInstance(tz_paris, datetime.tzinfo)
        self.assertTrue(isinstance(tz_paris, pytz.tzinfo.BaseTzInfo))
        self.assertEqual(str(tz_paris), 'Europe/Paris')

    def test_get_time_zone_invalid(self):
        # Scenario 3: Invalid timezone string
        with self.assertRaises(pytz.exceptions.UnknownTimeZoneError):
            time_utils.get_time_zone('Invalid/Timezone')

    def test_get_time_zone_none(self):
        # Scenario 4: time_zone_str is None
        tz_none = time_utils.get_time_zone(None)
        self.assertIsNone(tz_none)


if __name__ == '__main__':
    tf.test.main() # Or absltest.main() if using absltest
