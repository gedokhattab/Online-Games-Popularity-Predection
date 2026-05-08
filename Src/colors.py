import os

if os.name == 'nt':
    os.system("")

GREEN = "\033[38;2;55;247;18m"
BLUE = "\033[38;2;0;166;244m"
ORANGE = "\033[38;2;254;153;0m"
RED = "\033[38;2;255;33;87m"

PRIMARY = GREEN
INFO = BLUE
WARNING = ORANGE
DANGER = RED
TEXT = "\033[38;2;120;113;107m"

RESET = "\033[0m"
BOLD = "\033[1m"
