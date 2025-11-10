"""
Logging utilities for the wind velocity interpolation project.
"""
import sys


class Logger(object):
    """
    Logger class that writes output to both stdout and a log file.

    Attributes:
        terminal: Original stdout
        log: File handle for log file
    """

    def __init__(self, filename='Default.log'):
        """
        Initialize logger with output file.

        Args:
            filename: Path to log file
        """
        self.terminal = sys.stdout  # Save the original stdout
        self.log = open(filename, 'a')

    def write(self, message):
        """Write message to both terminal and log file."""
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """Flush method for file-like interface."""
        pass
