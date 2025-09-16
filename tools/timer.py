import time


class TimerError(Exception):
    """timer class error reporting"""


class Timer:
    def __init__(self):
        self._start_time = None
        self._lap_time = None

    def start(self):
        """start new timer"""
        if self._start_time is not None:
            raise TimerError(f"timer is running")

        self._start_time = time.perf_counter()
        self._lap_time = None

    def stop(self):
        """stop running timer and report elapsed"""
        if self._start_time is None:
            raise TimerError(f"timer not running")

        elapsed = time.perf_counter() - self._start_time
        self._start_time = None
        self._lap_time = None

        print(f"timer stopped after {elapsed} s")

    def report(self, achieved: str):
        """report currently running time"""
        print(f"{time.perf_counter() - self._start_time:0.2f} s - {achieved}")

    def report_lap(self, achieved: str) -> None:
        """report currently running time and reset lap time"""
        if self._lap_time is None:
            self._lap_time = time.perf_counter()
            lap = self._lap_time - self._start_time
            print(f"{lap:0.2f} s - {achieved}")
        else:
            lap = time.perf_counter() - self._lap_time
            self._lap_time = time.perf_counter()
            print(f"{lap:0.2f} s - {achieved}")
