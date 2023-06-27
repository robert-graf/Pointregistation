from __future__ import annotations

import warnings

from BIDS import *
from typing import Protocol
import weakref
from pathlib import Path
import os
import time
from time import struct_time
import datetime
import traceback
from BIDS.logger.log_constants import color_log_text, Log_Type, type2bcolors


class Logger_Interface(Protocol):
    def print(
        self,
        *text,
        end="\n",
        type=Log_Type.TEXT,
        verbose: bool | None = True,
        ignore_prefix: bool = False,
    ):
        """
        Args:
            *text: Text to the printed/logged
            end: end char
            type: log type (Text, Warning,...)
            verbose: true/false: prints to terminal (If none, uses default verbose)
            ignore_prefix: If false, will set a prefix character based on Log_Type (e.g. [*], [!], ...)

        Returns:

        """
        if verbose is None:
            verbose = getattr(self, "default_verbose", False)
        if len(text) == 0 or text == [""] or text == "" or text is None:
            ignore_prefix = True

        log_type_in_text = [t for t in text if isinstance(t, Log_Type)]
        text = [str(t) for t in text if not isinstance(t, Log_Type)]

        if len(log_type_in_text) > 0 and type == Log_Type.TEXT:
            type = log_type_in_text[0]

        string: str = preprocess_text(text, type=type, ignore_prefix=ignore_prefix)
        if verbose:
            print_to_terminal(string, end=end, type=type)
        self._log(string, end=end, type=type)

    def _log(self, text: str, end: str = "\n", type=Log_Type.TEXT):
        ...

    def close(self):
        ...

    def flush(self):
        ...

    def print_dict(self, u_dict: dict, name: str = "", **args):
        if name != "":
            self.print(name, "=", end="")
        self.print("{", end="", ignore_prefix=True)
        for key, value in u_dict.items():
            self.print("  ", key, ":", value, end=";  ", ignore_prefix=True)
        self.print("}", end="\n", **args, ignore_prefix=True)

    def flush_sub_logger(self, sublogger: String_Logger, closed=False):
        ...

    def add_sub_logger(
        self, name: str, default_verbose: bool = False
    ) -> String_Logger | No_Logger:
        """Creates a sub-logger that only logs to string. Will be appended in this loggers log file as sub-logger.
        Args:
            name: name of the sub-logger
            default_verbose: default_verbose attribute for the sub-logger

        Returns:
            sub_logger: String_Logger
        """
        if hasattr(self, "sub_loggers"):
            sub_logger = String_Logger.as_sub_logger(
                head_logger=self, default_verbose=default_verbose
            )
            self.sub_loggers.append(sub_logger)  # type: ignore
            sub_logger.print(
                "Sub-logger: ",
                name,
                verbose=False,
                ignore_prefix=False,
                type=Log_Type.LOG,
            )
            return sub_logger
        else:
            return self  # type: ignore

    def print_error(self, **args):
        self.print(traceback.format_exc(), type=Log_Type.FAIL, **args)


class Logger(Logger_Interface):
    """
    Defines a logger object, that automatically creates a logs folder and file in it. Logs logger.print() calls to this file.
    """

    def __init__(
        self,
        path: Path | str,
        log_filename: str | dict[str, str],
        default_verbose: bool = False,
        log_arguments=None,
    ):
        """

        Args:
            path: path to the folder that needs logging (usual dataset with raw/der in it)
            log_filename: the filename or the bids-conform key-value pairs as dict
            default_verbose: default verbose behavior of not specified in calls
            log_arguments: if set, will print the contents in a "run with arguments" section
        """
        path = Path(path)  # ensure pathlib object
        # Get Start time
        self.start_time = get_time()
        start_time_short = format_time_short(self.start_time)

        # Processes log_filename
        log_filename_processed = ""
        if isinstance(log_filename, dict):
            for k, v in log_filename.items():
                log_filename_processed += k + "-" + v + "_"
        else:
            log_filename_processed = log_filename + "_"
        log_filename_full = start_time_short + "_" + log_filename_processed + "log.log"

        # Creates logs folder if not existent
        log_path = Path(path).joinpath("logs")
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        # Open log file
        self.f = open(log_path.joinpath(log_filename_full), "w")
        # calls close() if program terminates
        self._finalizer = weakref.finalize(self.f, self.close)
        self.default_verbose = default_verbose
        # Log file always start with their name and start log time
        self.print(log_filename_processed[:-1], verbose=False, type=Log_Type.LOG)
        self.print(f"Log started at: {start_time_short}\n", type=Log_Type.LOG)

        if log_arguments is not None:
            self.print("Run with arguments", log_arguments, "\n", type=Log_Type.LOG)

        self.sub_loggers: list[String_Logger] = []

    @classmethod
    def create_from_bids(
        cls,
        bids_file: BIDS_FILE,
        log_filename: str | dict[str, str],
        default_verbose: bool = False,
    ):
        path = bids_file.dataset
        return Logger(path, log_filename, default_verbose=default_verbose)

    def _log(self, text: str, end: str = "\n", type=Log_Type.TEXT):
        self.f.write(str(text))
        self.f.write(end)

    def flush(self):
        self.f.flush()

    def flush_sub_logger(self, sublogger: String_Logger, closed=False):
        if sublogger in self.sub_loggers:
            self.sub_loggers.remove(sublogger) if closed else None
            self.print(f"Flushed sub logger:", verbose=False, type=Log_Type.LOG)
            self.print(sublogger.log_content, verbose=False)
            # Clear Sublogger
            sublogger.log_content = ""

    def remove(self):
        self._finalizer()

    def close(self):
        if not self.f.closed:
            self.sub_loggers = [s for s in self.sub_loggers if s.log_content != ""]
            if len(self.sub_loggers) > 0:
                self.print()
                self.print(
                    f"Found {len(self.sub_loggers)} sub logger:",
                    verbose=False,
                    type=Log_Type.LOG,
                )
                for tl in self.sub_loggers:
                    self.print(tl.log_content, verbose=False)

            end_time = get_time()
            duration = time.mktime(end_time) - time.mktime(self.start_time)
            self.print()
            self.print(
                "Program duration:", convert_seconds(duration), type=Log_Type.LOG
            )
            self.print(
                f"Log ended at: {format_time_short(end_time)}",
                verbose=False,
                type=Log_Type.LOG,
            )
            self.f.flush()
            self.f.close()

    @property
    def removed(self):
        return not self._finalizer.alive


class No_Logger(Logger_Interface):
    """
    Does not create any logs, but instead verbose defaults to true, printing calls to the terminal
    """

    def __init__(self):
        self.default_verbose = True
        pass

    def _log(self, text: str, end: str = "\n", type=Log_Type.TEXT):
        pass  # self.print()

    def flush(self):
        pass

    def flush_sub_logger(self, sublogger: String_Logger, closed=False):
        pass

    def close(self):
        pass

    def add_sub_logger(
        self, name: str, default_verbose: bool = False
    ) -> Logger_Interface:
        return self


class Reflection_Logger(No_Logger):
    """
    Does not create any logs, but instead verbose defaults to true, printing calls to the terminal.
    """

    def print(
        self,
        *text,
        end="\n",
        type=Log_Type.NEUTRAL,
        verbose: bool | Logger_Interface | None = True,
        ignore_prefix: bool = False,
    ):
        """
        Args:
            *text: Text to the printed/logged
            end: end char
            type: log type (Text, Warning,...)
            verbose: true/false or other Logger: prints to terminal (If none, uses default verbose)
            ignore_prefix: If false, will set a prefix character based on Log_Type (e.g. [*], [!], ...)

        Returns:

        """
        if isinstance(verbose, bool) or verbose is None:
            super().print(
                *text, end=end, type=type, verbose=verbose, ignore_prefix=ignore_prefix
            )
        else:
            verbose.print(*text, end=end, type=type, ignore_prefix=ignore_prefix)


class String_Logger(Logger_Interface):
    """
    Logger that logs only to a string object "log_content".
    """

    def __init__(self, default_verbose: bool = False, finalize: bool = True):
        self.default_verbose = default_verbose
        self.log_content = ""
        self.log_content_colored = ""
        self.sub_loggers: list[String_Logger] = []
        self.start_time = get_time()
        self.head_logger: Logger_Interface | None = None
        if finalize:
            self._finalizer = weakref.finalize(self, self.close)

    @classmethod
    def as_sub_logger(
        cls, head_logger: Logger_Interface, default_verbose: bool = False
    ) -> String_Logger:
        sub_logger = String_Logger(default_verbose=default_verbose, finalize=False)
        sub_logger.head_logger = head_logger
        return sub_logger

    def _log(self, text: str, end: str = "\n", type=Log_Type.TEXT):
        self.log_content += text
        self.log_content += end
        self.log_content_colored += color_log_text(c=type, text=text + end)

    def flush(self):
        if self.head_logger is not None:
            self.head_logger.flush_sub_logger(self, closed=False)

    def close(self):
        if len(self.sub_loggers) > 0:
            self.print()
            self.print(
                f"Found {len(self.sub_loggers)} sub logger:",
                verbose=None,
                type=Log_Type.LOG,
                ignore_prefix=True,
            )
            for tl in self.sub_loggers:
                self.print(tl.log_content, verbose=None)
                tl.close()

        end_time = get_time()
        duration = time.mktime(end_time) - time.mktime(self.start_time)
        self.print(
            "Sub-process duration:", convert_seconds(duration), type=Log_Type.LOG
        )
        # self.print(f"Log ended at: {format_time_short(end_time)}", verbose=False, type=Log_Type.LOG)
        if self.head_logger is not None:
            self.head_logger.flush_sub_logger(self, closed=True)
            self.head_logger.flush()
        return self.log_content, self.log_content_colored

    def flush_sub_logger(self, sublogger, closed=False):
        pass


#####################################
# Utils
#####################################


def preprocess_text(
    text: list[str], type=Log_Type.TEXT, ignore_prefix: bool = False
) -> str:
    string = str.join(" ", text)
    if "[" not in string[:3] and not ignore_prefix:
        string = type2bcolors[type][1] + " " + string
    return string


def print_to_terminal(text: str, end: str, type=Log_Type.TEXT) -> None:
    if type == Log_Type.WARNING_THROW:
        warnings.warn(color_log_text(c=type, text=text), Warning)
    else:
        print(color_log_text(c=type, text=text), end=end)


def get_time() -> struct_time:
    t = time.localtime()
    return t


def format_time(t: struct_time):
    return time.asctime(t)


def format_time_short(t: struct_time) -> str:
    return (
        "date-"
        + str(t.tm_year)
        + "-"
        + str(t.tm_mon)
        + "-"
        + str(t.tm_mday)
        + "_time-"
        + str(t.tm_hour)
        + "-"
        + str(t.tm_min)
        + "-"
        + str(t.tm_sec)
    )


def convert_seconds(seconds: float):
    return str(datetime.timedelta(seconds=seconds)) + " h:mm:ss"


def sub_log_call_func(name, logger: Logger, function, **kwargs):
    default_verbose = False
    if "default_verbose" in kwargs:
        default_verbose = kwargs["default_verbose"]
    if "verbose" in kwargs:
        default_verbose = kwargs["verbose"]
    sub_logger = logger.add_sub_logger(name, default_verbose=default_verbose)
    return function(name=name, logger=sub_logger, **kwargs)
