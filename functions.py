import math

def sigmoid(x, beta):
    if x < -5.0:
        return 1.0 / (1.0 + (math.exp(-beta * -5.0)))
    else:
        return 1.0 / (1.0 + (math.exp(-beta * x)))

def diff_sigmoid(x, beta):
    value = sigmoid(x, beta)
    return value * (1 - value)

def inner_product(vec1, vec2):
    result = 0
    for i in xrange(0, len(vec1)):
        result += (vec1[i] * vec2[i])
    return result
