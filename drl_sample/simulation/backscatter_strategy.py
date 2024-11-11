from enviroment import Environment

class BackScatterStrategy:

    def __init__(self):
        self.env = Environment()
        self.success_package_num = 0
        self.time = 1_000_000