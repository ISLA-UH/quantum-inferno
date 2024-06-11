"""
Collection of functions to convert between time bases
"""

from datetime import datetime, tzinfo, timedelta
from typing import List, Tuple
import numpy as np

# TODO: Move over relevant functions from quantum_inferno/utils/date_time.py (some may be specific to graphs)


def convert_time_unit(input_time: np.ndarray or float, input_unit: str, output_unit: str) -> np.ndarray or float:
    """
    Convert time data from a given time unit to another time unit.
    :param input_time: time data to convert
    :param input_unit: time unit of the input data
    :param output_unit: time unit to convert the input data to
    :return: converted time data
    """
    time_units = ["ns", "us", "ms", "s", "m", "h", "d", "w"]
    if input_unit not in time_units or output_unit not in time_units:
        raise ValueError("Invalid time unit, please use one of the following: ns, us, ms, s, m, h, d, or w.")
    time_unit_dict = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    return input_time * time_unit_dict[input_unit] / time_unit_dict[output_unit]


# TODO: What is wanted from this class?
class DateIterator:
    """
    A class to iterate over a range of dates.
    """

    def __init__(self, start_date: datetime, end_date: datetime, step: timedelta):
        """
        :param start_date: start date of the range
        :param end_date: end date of the range
        :param step: time step between dates
        """
        self.start_date = start_date
        self.end_date = end_date
        self.step = step
        self.current_date = start_date

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_date > self.end_date:
            raise StopIteration
        else:
            self.current_date += self.step
            return self.current_date - self.step
