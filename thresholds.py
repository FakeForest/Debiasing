import numpy as np


def operating_points(scores, labels):
    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]
    # A tied score enters the positive set as a block.
    selected = np.flatnonzero(scores[:-1] > scores[1:]) + 1
    positives = np.cumsum(labels, dtype=np.int64)[selected - 1]
    correct = len(labels) - labels.sum() + 2 * positives - selected
    return scores[selected - 1], selected / len(labels), correct


def fit_parity_thresholds(scores, labels, groups, tolerance=0.005):
    """Maximize validation accuracy with a gap <= tolerance and interior group rates."""
    points = [operating_points(scores[groups == g], labels[groups == g]) for g in (0, 1)]
    (t0, q0, c0), (t1, q1, c1) = points
    best = None
    thresholds = None
    for threshold, rate, correct in zip(t0, q0, c0):
        start = np.searchsorted(q1, rate - tolerance - 1e-12)
        end = np.searchsorted(q1, rate + tolerance + 1e-12, side="right")
        for j in range(start, end):
            key = (correct + c1[j], -abs(rate - q1[j]),
                   -abs(threshold - 0.5) - abs(t1[j] - 0.5))
            if best is None or key > best:
                best = key
                thresholds = (float(threshold), float(t1[j]))
    if thresholds is None:
        raise ValueError("No nonconstant threshold pair meets the validation DP tolerance.")
    return thresholds
