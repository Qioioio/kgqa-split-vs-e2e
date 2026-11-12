"""Bootstrap statistics over questions. Means and differences are in points (F1 x 100).

Every function draws 10,000 resamples of the questions with a fixed seed, so the
numbers are reproducible; they depend on the order of the questions.
"""
import numpy as np

SEED = 20261030
N_BOOT = 10000


def _resampled_means(x, rng, n_boot=N_BOOT, chunk=1000):
    out = []
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, len(x), size=(min(chunk, n_boot - start), len(x)))
        out.append(x[idx].mean(axis=1))
    return np.concatenate(out)


def mean_ci(scores, seed=SEED):
    """Mean and 95% percentile interval."""
    x = np.asarray(scores, dtype=float)
    m = _resampled_means(x, np.random.default_rng(seed))
    return float(100 * x.mean()), (float(100 * np.quantile(m, 0.025)), float(100 * np.quantile(m, 0.975)))


def paired_test(a, b, seed=SEED):
    """Paired difference a - b on the same questions: mean, 95% and 90% intervals, two-sided p
    (twice the smaller share of resampled differences on either side of zero)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("need two equal-length, non-empty score lists")
    d = a - b
    m = 100 * _resampled_means(d, np.random.default_rng(seed))
    p = min(1.0, 2 * min(float((m <= 0).mean()), float((m >= 0).mean())))
    return {"n": len(d), "diff": float(100 * d.mean()),
            "ci95": (float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))),
            "ci90": (float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))), "p": p}


def noninferiority_p(a, b, margin, seed=SEED):
    """One-sided p for "a is not worse than b by `margin` points or more":
    the share of resampled differences at or below -margin."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    m = 100 * _resampled_means(d, np.random.default_rng(seed))
    return float((m <= -margin).mean())


def holm(pvalues, alpha=0.05):
    """Holm step-down correction. {name: p} -> {name: (adjusted p, rejected)}."""
    order = sorted(pvalues, key=lambda k: (pvalues[k], k))
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[k]))
        out[k] = (running, running < alpha)
    return out


def group_difference(d, groups, g1, g2, seed=SEED):
    """mean(d | g1) - mean(d | g2) with a stratified bootstrap (each group resampled on its own).
    `d` holds per-question differences on the 0-1 scale."""
    d, groups = np.asarray(d, dtype=float), np.asarray(groups)
    x1, x2 = d[groups == g1], d[groups == g2]
    if len(x1) == 0 or len(x2) == 0:
        raise ValueError("both groups need at least one question")
    rng = np.random.default_rng(seed)
    s = 100 * _resampled_means(x1, rng) - 100 * _resampled_means(x2, rng)
    return {"n1": len(x1), "n2": len(x2), "diff": float(100 * (x1.mean() - x2.mean())),
            "ci95": (float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975)))}
