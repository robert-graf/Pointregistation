from enum import Enum, auto


class Log_Type(Enum):
    TEXT = auto()
    NEUTRAL = auto()
    SAVE = auto()
    WARNING = auto()
    WARNING_THROW = auto()
    LOG = auto()
    OK = auto()
    FAIL = auto()
    STRANGE = auto()
    UNDERLINE = auto()
    ITALICS = auto()
    DOCKER = auto()


class bcolors:
    # Front Colors
    BLACK = "\033[30m"
    PINK = "\033[95m"  # Misc
    BLUE = "\033[94m"  # reserved for log-related stuff
    CYAN = "\033[96m"  # save
    GREEN = "\033[92m"  # okay/success
    YELLOW = "\033[93m"  # warnings
    RED = "\033[91m"  # failure, error
    ORANGE = "\033[33m"
    GRAY = "\033[37m"
    # Modes (unused)
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DISABLE = "\033[02m"
    STRIKETHROUGH = "\033[09m"
    REVERSE = "\033[07m"
    ITALICS = "\033[3m"
    # Background Colors (unused)
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_ORANGE = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_PURPLE = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_GRAY = "\033[47m"
    # End of line
    ENDC = "\033[0m"


type2bcolors: dict[Log_Type, tuple[str, str]] = {
    Log_Type.TEXT: ("", "[*]"),
    Log_Type.NEUTRAL: ("", "[ ]"),
    Log_Type.SAVE: (bcolors.CYAN, "[*]"),
    Log_Type.WARNING: (bcolors.YELLOW, "[?]"),
    Log_Type.WARNING_THROW: (bcolors.YELLOW, "[?]"),
    Log_Type.LOG: (bcolors.BLUE, "[#]"),
    Log_Type.OK: (bcolors.GREEN, "[+]"),
    Log_Type.FAIL: (bcolors.RED, "[!]"),
    Log_Type.STRANGE: (bcolors.PINK, "[-]"),
    Log_Type.UNDERLINE: (bcolors.UNDERLINE, "[_]"),
    Log_Type.ITALICS: (bcolors.ITALICS, "[ ]"),
    Log_Type.DOCKER: (bcolors.ITALICS, "[Docker]"),
}


def color_log_text(c: Log_Type, text: str):
    return color_text(color_char=type2bcolors[c][0], text=text)


def color_text(text: str, color_char):
    return f"{color_char}{text}{bcolors.ENDC}"
