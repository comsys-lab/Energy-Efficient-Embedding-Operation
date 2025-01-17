import time
from colorama import init, Fore, Back, Style

init()  # colorama initialization

class Helper:
    def init(self):
        self.start = 0
        self.end = 0
        
    def set_timer(self):
        self.start = time.perf_counter()
    
    def end_timer(self, task):
        self.end = time.perf_counter()
        print('(Time elapsed(s) in {}: {:10.6f}sec)'.format(task, self.end-self.start))

def print_styled_header(title):
    width = 40
    print(f"\n{Back.YELLOW}{Fore.BLACK}{'='*width}")
    print(f"{title.center(width)}")
    print(f"{'='*width}{Style.RESET_ALL}")

def print_styled_box(title, content_lines):
    width = 50
    print(f"\n{Back.YELLOW}{Fore.BLACK}{'='*width}")
    print(f"{title.center(width)}")
    print(f"{'-'*width}")
    for line in content_lines:
        print(f"{line.ljust(width)}")
    print(f"{'='*width}{Style.RESET_ALL}")