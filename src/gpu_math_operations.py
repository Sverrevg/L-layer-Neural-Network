import gnumpy as gnp


def sigmoid(Z):
    A = gnp.logistic(Z)  # GNumPY logistic function.
    cache = Z

    return A, cache
