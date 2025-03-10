# PJ = [5W, 10W, 15W]
class JammerPower:
    def __init__(self, nu, nu_p):
        self.nu = nu
        self.nu_p = nu_p
    
    def calculate_p_avg(self):
        print('Pavg = ' + str(0 * self.nu + (1 - self.nu) * (self.nu_p[0] * 5 + self.nu_p[1] * 10 + self.nu_p[2] * 15)))

list_nu_p = [
    JammerPower(0.85, [0.8, 0.1, 0.1]), # 1W,
    JammerPower(0.75, [0.6, 0.2, 0.2]), # 2W
    JammerPower(0.65, [0.5, 0.3, 0.2]), # 3W,
    JammerPower(0.5, [0.6, 0.2, 0.2]), # 4W
    JammerPower(0.4, [0.55, 0.25, 0.2]), # 5W,
    JammerPower(0.3, [0.5, 0.3, 0.2]), # 6W
    JammerPower(0.2, [0.45, 0.35, 0.2]), # 7W,
    JammerPower(0.15, [0.4, 0.3, 0.3]), # 8W
    JammerPower(0.1, [0.4, 0.2, 0.4]), # 9W
]