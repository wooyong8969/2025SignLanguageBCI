import math

print()

def f():
    for y in range(15, -15, -1):
        line = ''
        for x in range(-30, 30):
            x1 = x * 0.04
            y1 = y * 0.1
            eq = (x1**2 + y1**2 - 1)**3 - x1**2 * y1**3
            line += '*' if eq <= 0 else ' '
        print(line)

f()
