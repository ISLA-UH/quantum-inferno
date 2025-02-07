"""
This module contains custom error definitions and error checking functions for Quantum Inferno
"""
from typing import List


class QuantumInfernoError(Exception):
    """
    Base class for Quantum Inferno errors
    """
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class QuantumInfernoDebugger:
    """
    This class contains all the messages produced by Quantum Inferno error checking

    Properties:
        messages: List[str], list of messages
        debug_mode: bool, flag to enable or disable printing new messages.  Default False
    """
    def __init__(self, debug_mode: bool = False):
        self.messages: List[str] = []
        self.debug_mode = debug_mode

    def get_num_messages(self) -> int:
        """
        Get the number of messages in the debugger
        :return: number of messages
        """
        return len(self.messages)

    def print_messages(self):
        """
        Print all the messages in the debugger
        """
        for message in self.messages:
            print(message)

    def add_message(self, message: str):
        """
        Add a message to the debugger, printing it if debug mode is enabled
        :param message: message to add
        """
        self.messages.append(message)
        if self.debug_mode:
            print(message)

    def add_message_with_print(self, message: str):
        """
        Add a message to the debugger and print it
        :param message: message to add
        """
        self.add_message(message)
        print(message)
