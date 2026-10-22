import json
import os
import math
import numpy as np


class ConformalVerifier:
    """
    Split Conformal Prediction Layer for Natural Language Inference & Fact Verification.
    
    Provides finite-sample, distribution-free theoretical coverage guarantees:
        P(Y in C(X)) >= 1 - alpha
        
    IMPORTANT STATISTICAL ASSUMPTION (Exchangeability):
    This guarantee holds under the standard statistical assumption that calibration and 
    test samples (X_i, Y_i) are exchangeable (drawn from the same joint distribution P_{XY}).
    If the deployment environment suffers from severe domain shift (e.g., calibrating on 
    general-knowledge FEVER and deploying on specialised scientific claims), coverage 
    can degrade. In such cases, domain-specific calibration partitions are required.
    """

    def __init__(self, alpha=0.10, domain="general", calib_file=None):
        """
        Args:
            alpha (float): Significance level (e.g. 0.10 means 90% guaranteed coverage).
            domain (str): Domain context ('general' or 'scientific').
            calib_file (str, optional): Path to a pre-saved calibration JSON file.
        """
        self.alpha = float(alpha)
        self.domain = domain
        self.labels = ["contradicted", "unknown", "supported"]
        self.q_hat = None
        self.calibrated_scores = []
        self.n_samples = 0
        
        if calib_file and os.path.exists(calib_file):
            self.load_calibration(calib_file)

    def compute_non_conformity(self, prob_vector, y_true):
        """
        Computes non-conformity score for a given observation:
            s_i = 1 - p(y_true)
        Higher score implies the model was more non-conforming / less confident in the truth.
        """
        if isinstance(y_true, str):
            if y_true not in self.labels:
                raise ValueError(f"Unknown label '{y_true}'. Must be one of {self.labels}")
            y_idx = self.labels.index(y_true)
        else:
            y_idx = int(y_true)
        
        return 1.0 - float(prob_vector[y_idx])

    def calibrate(self, probabilities_list, true_labels, save_path=None):
        """
        Computes the exact finite-sample conformal quantile q_hat on a hold-out calibration set.
        
        Exact Formula (Vovk et al., 2005; Angelopoulos & Bates, 2021):
            k = ceil((n + 1) * (1 - alpha))
            q_hat = s_{(k)} = sorted_scores[k - 1]
            
        MATHEMATICAL AUDIT SAFEGUARD:
        Avoid continuous np.quantile()! Continuous quantile implementations compute virtual
        indices scaled by (n - 1), which causes off-by-one errors or continuous interpolation
        that invalidates the finite-sample distribution-free coverage guarantee. 
        Direct discrete order-statistic indexing sorted_scores[k - 1] is mathematically exact.
        """
        n = len(true_labels)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        scores = []
        for probs, y_true in zip(probabilities_list, true_labels):
            score = self.compute_non_conformity(probs, y_true)
            scores.append(score)

        self.calibrated_scores = np.sort(np.array(scores))
        self.n_samples = n

        # Finite-sample exact rank index (1-based)
        k = int(math.ceil((n + 1) * (1.0 - self.alpha)))

        # Edge Case 1: Insufficient sample size for nominal coverage guarantee
        if k > n:
            import warnings
            warnings.warn(
                f"[ConformalVerifier] Sample size n={n} is insufficient to guarantee 1-alpha={1.0 - self.alpha:.2f} "
                f"coverage (requires n >= {math.ceil(1.0 / self.alpha) - 1}). Setting q_hat=1.0 (conservative maximum)."
            )
            self.q_hat = 1.0
        elif k <= 0:
            self.q_hat = 0.0
        else:
            # Exact k-th order statistic (0-indexed: k - 1)
            self.q_hat = float(self.calibrated_scores[k - 1])

        if save_path:
            self.save_calibration(save_path)

        return self.q_hat

    def predict_set(self, prob_vector):
        """
        Constructs the prediction set C(X) for an incoming test sample:
            C(X) = { y in Y : 1 - p(y) <= q_hat }
            
        Returns:
            dict containing:
                - point_prediction (str)
                - point_confidence (float)
                - conformal_set (list of str)
                - set_size (int)
                - is_escalation_candidate (bool): True if set_size > 1 or contains 'unknown'.
                  NOTE: Multiple labels reflect model posterior dispersion/indecision under 
                  ambiguous evidence, serving as an escalation trigger rather than empirical proof 
                  of conflicting external sources.
                - p_values (dict of str -> float): Conformal p-values for each candidate label
                - coverage_guarantee (str)
        """
        if self.q_hat is None:
            raise ValueError("ConformalVerifier has not been calibrated! Run calibrate() or load a pre-calibrated model.")

        prob_vector = np.array(prob_vector, dtype=float)
        prediction_set = []
        p_values = {}

        n = len(self.calibrated_scores)

        for idx, label in enumerate(self.labels):
            test_score = 1.0 - prob_vector[idx]
            
            # Conformal condition
            if test_score <= self.q_hat:
                prediction_set.append(label)

            # Conformal p-value calculation:
            # p_value(y) = (1 + sum(score_calib >= test_score)) / (n + 1)
            count = np.sum(self.calibrated_scores >= test_score)
            p_val = (1.0 + float(count)) / (float(n) + 1.0)
            p_values[label] = round(p_val, 4)

        # Fallback for empty set (occurs with probability at most alpha)
        argmax_idx = int(np.argmax(prob_vector))
        if not prediction_set:
            prediction_set = [self.labels[argmax_idx]]

        point_pred = self.labels[argmax_idx]
        point_conf = float(prob_vector[argmax_idx])

        # Escalation trigger: set size > 1 or includes 'unknown'
        is_escalation = (len(prediction_set) > 1) or ("unknown" in prediction_set)

        return {
            "point_prediction": point_pred,
            "point_confidence": point_conf,
            "conformal_set": prediction_set,
            "set_size": len(prediction_set),
            "is_escalation_candidate": is_escalation,
            "p_values": p_values,
            "coverage_guarantee": f"{int((1 - self.alpha) * 100)}%",
            "q_hat": self.q_hat
        }

    @staticmethod
    def compute_wilson_ci(k, n, confidence=0.95):
        """
        Calculates the Wilson score confidence interval for a binomial proportion k/n.
        Essential for reporting empirical coverage with rigorous statistical uncertainty.
        """
        if n == 0:
            return (0.0, 0.0)
        z = 1.95996  # For 95% confidence
        p_hat = k / n
        denominator = 1.0 + (z**2) / n
        center = (p_hat + (z**2) / (2.0 * n)) / denominator
        margin = (z / denominator) * math.sqrt((p_hat * (1.0 - p_hat) / n) + ((z**2) / (4.0 * (n**2))))
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        return (lower, upper)

    def save_calibration(self, filepath):
        """Saves calibration parameters and scores to a JSON file."""
        data = {
            "alpha": self.alpha,
            "domain": self.domain,
            "q_hat": self.q_hat,
            "n_samples": self.n_samples,
            "labels": self.labels,
            "calibrated_scores": [float(s) for s in self.calibrated_scores]
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_calibration(self, filepath):
        """Loads calibration parameters from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.alpha = data["alpha"]
        self.domain = data.get("domain", "general")
        self.q_hat = data["q_hat"]
        self.n_samples = data["n_samples"]
        self.labels = data["labels"]
        self.calibrated_scores = np.array(data["calibrated_scores"])
