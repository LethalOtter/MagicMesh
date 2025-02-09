x = 1
i = 0
while True:
    x *= 0.1
    if 1 + x == 1:
        print(f'Machine Epsilon is {x}')
        break
