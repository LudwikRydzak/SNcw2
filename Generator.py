import random
def generate(_amount):
    result = []
    for i in range(_amount):
        a = random.random()/100
        b = random.random()/100
        result.append([-1, a - 1, b - 1])
        result.append([-1, (a + b) - 1, 1 - a])
        result.append([-1, 1 - b, a - 1])
        result.append([1, 1 - a, 1 - b])
    return result
