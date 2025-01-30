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

    def add_message(self, message: str):
        """
        Add a message to the debugger
        :param message: message to add
        """
        self.messages.append(message)
        if self.debug_mode:
            print(message)
