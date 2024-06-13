"""
Collection of functions to convert between time bases
"""

from datetime import datetime, tzinfo, timedelta, timezone
from typing import List, Tuple
import numpy as np

# TODO: Add support for time zones?
# TODO: Add utils for graphing time data...


# dictionary of time units and their conversion factors to seconds (can add more units as needed)
time_unit_dict = {
    "ps": 1e-12,  # "picosecond"
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "weeks": 604800,
    "months": 2628000,
    "years": 31536000,
}


def convert_time_unit(input_time: np.ndarray or float, input_unit: str, output_unit: str) -> np.ndarray or float:
    """
    Convert time data from a given time unit to another time unit.
    :param input_time: time data to convert
    :param input_unit: time unit of the input data
    :param output_unit: time unit to convert the input data to
    :return: converted time data
    """
    if input_unit not in time_unit_dict.keys() or output_unit not in time_unit_dict.keys():
        raise ValueError(f"Invalid time unit, please use one of the following: {time_unit_dict.keys()}")
    return input_time * time_unit_dict[input_unit] / time_unit_dict[output_unit]


def utc_datetime_to_utc_timestamp(datetime_obj: datetime, output_unit: str = "s") -> np.ndarray or float:
    """
    Convert a UTC datetime object to a UTC timestamp.
    :param datetime_obj: UTC datetime object to convert
    :param output_unit: time unit to convert the UTC timestamp to (default: seconds)
    :return: converted UTC timestamp
    """
    if output_unit not in time_unit_dict.keys():
        raise ValueError(f"Invalid time unit, please use one of the following: {time_unit_dict.keys()}")
    return convert_time_unit(datetime_obj.timestamp(), "s", output_unit)


def utc_timestamp_to_utc_datetime(timestamp: np.ndarray or float, input_unit: str = "s") -> datetime:
    """
    Convert a UTC timestamp to a UTC datetime object.
    :param timestamp: UTC timestamp to convert
    :param input_unit: time unit of the UTC timestamp (default: seconds)
    :return: converted UTC datetime object
    """
    if input_unit not in time_unit_dict.keys():
        raise ValueError(f"Invalid time unit, please use one of the following: {time_unit_dict.keys()}")
    return datetime.utcfromtimestamp(convert_time_unit(timestamp, input_unit, "s"))
