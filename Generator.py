import random
def generate(_amount, _uni0_bi1):
    result = []
    for i in range(_amount):
        a = random.random()/100
        b = random.random()/100
        if _uni0_bi1:
            result.append([-1, a - 1, b - 1])
            result.append([-1, (a + b) - 1, 1 - a])
            result.append([-1, 1 - b, a - 1])
            result.append([1, 1 - a, 1 - b])
        else:
            result.append([0, a - 1, b - 1])
            result.append([0, (a + b) - 1, 1 - a])
            result.append([0, 1 - b, a - 1])
            result.append([1, 1 - a, 1 - b])
    return result
