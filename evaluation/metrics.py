def precision(tp, fp):

    if tp + fp == 0:
        return 0

    return tp / (tp + fp)



def recall(tp, fn):

    if tp + fn == 0:
        return 0

    return tp / (tp + fn)



def f1_score(precision, recall):

    if precision + recall == 0:
        return 0

    return (
        2 * precision * recall
    ) / (precision + recall)


def wilson_interval(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion (§4.4); unlike the normal approximation it stays
    inside [0, 1] and keeps its coverage near 0% and 100%."""

    if n == 0:
        return 0.0, 1.0

    p = successes / n
    centre = p + z * z / (2 * n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return (centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)
