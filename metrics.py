
import numpy as np
import torch


@torch.no_grad()
def probabilities(predictor, X):
    predictor.eval()
    return torch.sigmoid(predictor(X)).cpu().numpy()


def score_predictions(p, y, z, threshold=0.5):
    y, z = np.asarray(y), np.asarray(z)
    yhat = p >= threshold

    def rate(mask):
        return yhat[mask].mean() if mask.any() else np.nan

    metrics = {"accuracy": (yhat == y).mean()}
    for name, eligible in (("DP", np.ones_like(y, dtype=bool)), ("TPR", y == 1), ("FPR", y == 0)):
        first, second = [rate(eligible & (z == group)) for group in (0, 1)]
        metrics.update({f"{name} gap": abs(second - first), f"{name} signed": second - first,
                        f"{name} z0": first, f"{name} z1": second})
    return metrics


def evaluate(predictor, X, y, z, threshold=0.5):
    return score_predictions(probabilities(predictor, X), y, z, threshold)
