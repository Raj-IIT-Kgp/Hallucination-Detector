import json
import os
import numpy as np


class TemperatureScaler:
    """
    Temperature scaling of the NLI posteriors (Guo et al., 2017).

    The NLI model's softmax probabilities are over-confident on retrieved evidence. Temperature scaling divides
    the logits by a single constant T > 0 before the softmax:
        pi_y(z; T) = exp(z_y / T) / sum_y' exp(z_y' / T)
    T > 1 softens the probabilities, T < 1 sharpens them, and the arg max (the verdict) never changes.

    T is fitted by maximum likelihood. Writing beta = 1/T, the mean negative log-likelihood
        L(beta) = (1/n) sum_i [ log sum_y exp(beta z_iy) - beta z_i,y_i ]
    has derivatives
        L'(beta)  = (1/n) sum_i ( E_pi_i[z_i] - z_i,y_i )
        L''(beta) = (1/n) sum_i Var_pi_i(z_i)  >= 0,
    so L is convex in beta and Newton's method finds its unique minimum in a few steps.
    """

    def __init__(self, calib_file=None):
        self.temperature = 1.0
        if calib_file and os.path.exists(calib_file):
            with open(calib_file) as f:
                self.temperature = float(json.load(f).get("temperature", 1.0))

    @staticmethod
    def softmax(logits, temperature=1.0):
        z = np.asarray(logits, dtype=float) / temperature
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def nll(logits, y, temperature=1.0):
        p = TemperatureScaler.softmax(logits, temperature)
        return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-300, None))))

    @staticmethod
    def fit(logits, y, tol=1e-10, max_iter=100):
        """Maximum-likelihood temperature for logits (n x K) and true class indices y (n). Returns T."""
        z = np.asarray(logits, dtype=float)
        y = np.asarray(y, dtype=int)
        z_true = z[np.arange(len(y)), y]
        loss = lambda b: float(np.mean(np.log(np.exp(b * z - (b * z).max(1, keepdims=True)).sum(1))
                                       + (b * z).max(1) - b * z_true))
        beta = 1.0
        for _ in range(max_iter):
            p = TemperatureScaler.softmax(z, 1.0 / beta)
            mean_z = (p * z).sum(1)
            grad = float(np.mean(mean_z - z_true))
            hess = float(np.mean((p * z * z).sum(1) - mean_z ** 2))
            if hess <= 0:
                break
            step = grad / hess
            # damped Newton: halve the step until the loss decreases and beta stays positive
            while beta - step <= 0 or loss(beta - step) > loss(beta) + 1e-15:
                step /= 2
                if abs(step) < 1e-14:
                    break
            beta -= step
            if abs(step) < tol:
                break
        return 1.0 / beta

    def calibrate(self, logits):
        """Calibrated probability vector for one claim's logits (or None if there are none)."""
        if logits is None:
            return None
        return [float(x) for x in self.softmax(logits, self.temperature)]


def expected_calibration_error(confidence, correct, bins=10):
    """
    Top-label ECE with equal-mass bins: sort the claims by confidence, split them into `bins` groups of (nearly)
    equal size, and average |mean confidence - accuracy| over the groups, weighted by group size.
    Returns (ece, [(n, mean confidence, accuracy), ...]).
    """
    order = np.argsort(confidence, kind="stable")
    conf, corr = np.asarray(confidence)[order], np.asarray(correct, dtype=float)[order]
    table, ece = [], 0.0
    for idx in np.array_split(np.arange(len(conf)), bins):
        if len(idx) == 0:
            continue
        c, a = float(conf[idx].mean()), float(corr[idx].mean())
        table.append((int(len(idx)), c, a))
        ece += len(idx) / len(conf) * abs(c - a)
    return ece, table


def brier_score(probs, y):
    """Multi-class Brier score: mean over claims of sum_k (pi_k - 1[k = y])^2 (0 is perfect, 2 the worst)."""
    p = np.asarray(probs, dtype=float)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(((p - onehot) ** 2).sum(1)))
