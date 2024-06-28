"""
Collection of functions to convert between time bases
"""

from datetime import datetime, timezone
from typing import Union

import numpy as np

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


def convert_time_unit(
    input_time: Union[np.ndarray, float], input_unit: str, output_unit: str
) -> Union[np.ndarray, float]:
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


def utc_datetime_to_utc_timestamp(datetime_obj: datetime, output_unit: str = "s") -> float:
    """
    Convert a UTC datetime object to a UTC timestamp.
    If datetime_object is not timezone aware, it will be assumed to be in UTC.
    If datetime_object is timezone aware, it will be converted to UTC.

    :param datetime_obj: UTC datetime object to convert
    :param output_unit: time unit to convert the UTC timestamp to (default: seconds)
    :return: converted UTC timestamp
    """
    if output_unit not in time_unit_dict.keys():
        raise ValueError(f"Invalid time unit, please use one of the following: {time_unit_dict.keys()}")
    if datetime_obj.tzinfo is None:
        datetime_obj = datetime_obj.replace(tzinfo=timezone.utc)
    elif datetime_obj.tzinfo != timezone.utc:
        datetime_obj = datetime_obj.astimezone(timezone.utc)
    return convert_time_unit(datetime_obj.timestamp(), "s", output_unit)


def utc_timestamp_to_utc_datetime(timestamp: float, input_unit: str = "s") -> datetime:
    """
    Convert a UTC timestamp to a UTC datetime object.
    Note: timestamp is assumed to be in UTC.

    :param timestamp: UTC timestamp to convert
    :param input_unit: time unit of the UTC timestamp (default: seconds)
    :return: converted UTC datetime object
    """
    if input_unit not in time_unit_dict.keys():
        raise ValueError(f"Invalid time unit, please use one of the following: {time_unit_dict.keys()}")
    return datetime.utcfromtimestamp(convert_time_unit(timestamp, input_unit, "s")).replace(tzinfo=timezone.utc)


def set_datetime_to_utc(datetime_obj: datetime, tzinfo_warning: bool = False) -> datetime:
    """
    Convert a datetime object to a UTC datetime object.
    If the input datetime object is not timezone-aware, it is assumed to be in UTC.

    :param datetime_obj: datetime object to convert
    :param tzinfo_warning: flag to raise a warning if the input datetime object is not timezone-aware
    :return: converted UTC datetime object
    """
    if datetime_obj.tzinfo is None:
        if tzinfo_warning:
            print("Warning: input datetime object is not timezone-aware, assuming UTC...")
        return datetime_obj.replace(tzinfo=timezone.utc)
    return datetime_obj.astimezone(timezone.utc)


def set_timestamp_to_utc(timestamp: float, utc_offset_h: float, input_unit: str = "s") -> float:
    """
    Convert a timestamp to be in UTC using the UTC offset.

    :param timestamp: timestamp to convert
    :param utc_offset_h: UTC offset of the timestamp in hours
    :param input_unit: time unit of the timestamp (default: seconds)
    :return: converted timestamp in UTC while keeping the same unit
    """
    if input_unit not in time_unit_dict.keys():
        raise ValueError(f"Invalid time unit, please use one of the following: {time_unit_dict.keys()}")
    offset_in_input_unit = utc_offset_h * time_unit_dict["h"] / time_unit_dict[input_unit]
    return timestamp - offset_in_input_unit


def get_datetime_from_timestamp_to_utc(timestamp: float, utc_offset_h: float, input_unit: str = "s") -> datetime:
    """
    Convert a timestamp to a UTC datetime object using the UTC offset.

    :param timestamp: timestamp to convert into UTC time
    :param utc_offset_h: UTC offset of the timestamp in hours
    :param input_unit: time units (default: seconds)
    :return: converted UTC datetime object
    """
    if input_unit not in time_unit_dict.keys():
        raise ValueError(f"Invalid time unit, please use one of the following: {time_unit_dict.keys()}")
    return utc_timestamp_to_utc_datetime(set_timestamp_to_utc(timestamp, utc_offset_h, input_unit))
