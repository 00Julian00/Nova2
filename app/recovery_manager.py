"""
Descriptions: Responsible for attempting automated recovery of sub-systems upon Exceptions.
"""
from typing import Callable
from warnings import warn
from time import time

def automated_recovery(max_attempts_per_minute: int = 3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            process = ManagedProcess(func, max_attempts_per_minute, *args, **kwargs)
            return process.start_process()
        return wrapper
    return decorator

class ManagedProcess:
    """
    Represents a process that is managed by the RecoveryManager.
    """
    def __init__(self, function: Callable, max_recoveries_per_minute: int, *args, **kwargs):
        if max_recoveries_per_minute <= 0:
            raise ValueError("max_recoveries_per_minute must be greater than 0")
        
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.total_recoveries = 0
        self.max_recoveries_per_minute = max_recoveries_per_minute

        self.recovery_timestamps = []

    def truncate_recovery_timestamps(self) -> None:
        """
        Truncates the recovery timestamps to the maximum allowed recoveries per minute.
        """
        current_time = time()
        while (len(self.recovery_timestamps) > 0 and self.recovery_timestamps[0] < current_time - 60):
            self.recovery_timestamps.pop(0)

    def start_process(self) -> None:
        """
        Starts the managed process.
        """
        try:
            self.function(*self.args, **self.kwargs)
        except Exception as e:
            warn(f"Exception occurred in managed process: \"{e}\" This is exception number {self.total_recoveries + 1} of {self.max_recoveries_per_minute} allowed recoveries per minute. {self.total_recoveries + 1} recoveries so far.", stacklevel=2)

            self.truncate_recovery_timestamps()

            if len(self.recovery_timestamps) >= self.max_recoveries_per_minute:
                raise Exception(f"Maximum recoveries of {self.max_recoveries_per_minute} per minute surpassed for process: {self.function.__name__}. Total recoveries during lifetime: {self.total_recoveries}.")

            self.total_recoveries += 1
            self.recovery_timestamps.append(time())

            self.start_process()