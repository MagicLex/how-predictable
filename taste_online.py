"""Per-user online preference learner. Shared by the app (live session) and the
offline replay evaluation (cold-start curves) -- same code, same numbers.

Model: Bayesian logistic regression on pair features x = emb_A - emb_B,
y = 1 if the user picked A. Diagonal-Gaussian posterior over weights,
initialized at the GLOBAL prior weights (mean) with prior variance sigma0^2.
Each swipe does one assumed-density-filtering step (diagonal Laplace /
extended-Kalman for the logistic likelihood):

    precision += p(1-p) * x^2          (diagonal)
    mean      += var * x * (y - p)

Prediction uses the probit-corrected marginal:  sigmoid(m.x / sqrt(1 + pi*s/8))
with s = x.(var*x), so an uncertain user model predicts closer to 0.5 -- the
accuracy line starts near the global model and sharpens as evidence arrives.

Personalization = the posterior mean drifting away from the global weights where
this user disagrees with the crowd. Shrinkage to the global prior keeps 768-d
sane at 20 observations.

Supervised online learning, NOT RL: nothing optimizes which pairs are shown
based on reward. Pair selection (active vs measure) lives in the app, and the
active choice maximizes information about the user, never engagement.
"""
import numpy as np

_SIGMA0 = 0.3          # prior std around the global weights; tuned by replay eval
_PROBIT = np.pi / 8.0


class UserPosterior:
    def __init__(self, w_global, sigma0=_SIGMA0):
        w = np.asarray(w_global, dtype=np.float64)
        self.mean = w.copy()
        self.var = np.full(w.shape, sigma0 ** 2)
        self.n_obs = 0

    def predict(self, x):
        """P(user picks A) for pair feature x = emb_A - emb_B. Marginalized."""
        x = np.asarray(x, dtype=np.float64)
        m = float(self.mean @ x)
        s = float(x @ (self.var * x))
        return _sigmoid(m / np.sqrt(1.0 + _PROBIT * s))

    def update(self, x, y):
        """One ADF step for outcome y in {0,1}."""
        x = np.asarray(x, dtype=np.float64)
        p = _sigmoid(float(self.mean @ x))
        lam = max(p * (1.0 - p), 1e-6)
        prec = 1.0 / self.var + lam * x * x
        self.var = 1.0 / prec
        self.mean = self.mean + self.var * x * (y - p)
        self.n_obs += 1

    def uncertainty(self, x):
        """Predictive variance term s -- the active-selection signal."""
        x = np.asarray(x, dtype=np.float64)
        return float(x @ (self.var * x))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def global_prob(w_global, x):
    """The frozen global model's P(pick A) -- the baseline curve on the UI."""
    return _sigmoid(float(np.asarray(w_global) @ np.asarray(x)))


def select_pair(posterior, cand_feats, top_frac=0.2, rng=None):
    """Active selection: among candidate pair features, pick one of the most
    informative (highest predictive uncertainty), randomized within the top
    fraction so sessions do not all see the same pairs.
    Returns the chosen index."""
    rng = rng or np.random.default_rng()
    s = np.array([posterior.uncertainty(x) for x in cand_feats])
    k = max(1, int(len(s) * top_frac))
    top = np.argsort(-s)[:k]
    return int(rng.choice(top))


def replay_session(w_global, feats, choices, measure_mask, sigma0=_SIGMA0):
    """Offline replay of one logged session -> per-swipe hit/miss for the
    personalized and global models, on measure pairs only. Train pairs update
    the posterior but never score. Returns dict of arrays."""
    post = UserPosterior(w_global, sigma0)
    hits_p, hits_g, idx = [], [], []
    for i, (x, y) in enumerate(zip(feats, choices)):
        if measure_mask[i]:
            hits_p.append((post.predict(x) > 0.5) == bool(y))
            hits_g.append((global_prob(w_global, x) > 0.5) == bool(y))
            idx.append(i)
        post.update(x, y)
    return {"swipe_idx": np.array(idx), "hit_personal": np.array(hits_p),
            "hit_global": np.array(hits_g)}
