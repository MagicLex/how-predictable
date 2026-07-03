"""Per-user online preference learner. Shared by the app (live session) and the
offline replay evaluation (cold-start curves) -- same code, same numbers.

Design: personalization happens in a LOW-DIM taste subspace, not raw 768-d.
A swipe carries ~1 effective dimension of information; 20-30 swipes cannot pin
768 weights, but they can pin ~25. Feature map for a pair (A, B):

    phi(x) = [ w_global . x,  P . x ]        x = emb_A - emb_B

dim 0 is the frozen crowd model's logit (the prior knowledge), dims 1..K are
the pool's top-K principal components (where pet-to-pet variation actually
lives). The user model is Bayesian logistic regression on phi with prior mean
[1, 0, ..., 0]: "start as the crowd, learn your delta where you disagree".

Posterior: diagonal Gaussian, one assumed-density-filtering step per swipe
(diagonal Laplace / extended-Kalman for the logistic likelihood):

    precision += p(1-p) * phi^2         mean += var * phi * (y - p)

Prediction uses the probit-corrected marginal sigmoid(m.phi / sqrt(1 + pi*s/8)),
s = phi.(var*phi): an uncertain user model predicts near the crowd model and
sharpens as evidence arrives -- that is the climbing line.

The projection P (and the logit scale) ships in the pet_taste model artifact
(taste_space.npz), computed by the training pipeline from pool embeddings.

Supervised online learning, NOT RL: nothing optimizes which pairs are shown
based on reward. Active selection maximizes information about the user, never
engagement.
"""
import numpy as np

SIGMA0_GLOBAL = 0.25    # prior std on the crowd-logit coefficient (dim 0)
SIGMA0_TASTE = 1.0      # prior std on the personal taste dims
VAR_FLOOR = 0.02        # ADF never raises variance; without a floor the
                        # posterior turns arrogant ("100% sure", 53% right).
                        # The floor models user noise/drift and caps confidence.
_PROBIT = np.pi / 8.0


class TasteSpace:
    """phi(x) = [scale * w_global.x, P.x] with unit-variance-ish columns.

    `p_matrix` (K, D): top-K pool PCs, rows scaled to unit response std over
    the pool; `logit_scale` = 1/std of w_global.x over the pool, so dim 0 is
    comparable in magnitude to the PC dims."""

    def __init__(self, w_global, p_matrix, logit_scale=1.0):
        self.w = np.asarray(w_global, dtype=np.float64)
        self.P = np.asarray(p_matrix, dtype=np.float64)
        self.scale = float(logit_scale)

    @property
    def dim(self):
        return 1 + self.P.shape[0]

    def phi(self, x):
        x = np.asarray(x, dtype=np.float64)
        return np.concatenate([[self.scale * (self.w @ x)], self.P @ x])

    @classmethod
    def fit(cls, w_global, pool_emb, k=24):
        """PCA of pool embeddings -> top-k taste axes + logit scale."""
        E = np.asarray(pool_emb, dtype=np.float64)
        E = E - E.mean(axis=0)
        _, _, vt = np.linalg.svd(E, full_matrices=False)
        P = vt[:k]
        g = E @ np.asarray(w_global, dtype=np.float64)
        scale = 1.0 / max(g.std(), 1e-9)
        pc_std = (E @ P.T).std(axis=0)
        P = P / np.maximum(pc_std, 1e-9)[:, None]
        return cls(w_global, P, scale)

    def save(self, path):
        np.savez(path, w=self.w, P=self.P, scale=self.scale)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["w"], z["P"], float(z["scale"]))


class UserPosterior:
    def __init__(self, space, sigma0_global=SIGMA0_GLOBAL,
                 sigma0_taste=SIGMA0_TASTE):
        self.space = space
        self.mean = np.zeros(space.dim)
        self.mean[0] = 1.0                      # start as the crowd
        self.var = np.full(space.dim, sigma0_taste ** 2)
        self.var[0] = sigma0_global ** 2
        self.n_obs = 0

    def predict(self, x):
        """P(user picks A) for x = emb_A - emb_B. Marginalized."""
        phi = self.space.phi(x)
        m = float(self.mean @ phi)
        s = float(phi @ (self.var * phi))
        return _sigmoid(m / np.sqrt(1.0 + _PROBIT * s))

    def update(self, x, y):
        """One ADF step for outcome y in {0,1}."""
        phi = self.space.phi(x)
        p = _sigmoid(float(self.mean @ phi))
        lam = max(p * (1.0 - p), 1e-6)
        prec = 1.0 / self.var + lam * phi * phi
        self.var = np.maximum(1.0 / prec, VAR_FLOOR)
        self.mean = self.mean + self.var * phi * (y - p)
        self.n_obs += 1

    def uncertainty(self, x):
        """Predictive variance term s -- the active-selection signal."""
        phi = self.space.phi(x)
        return float(phi @ (self.var * phi))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def global_prob(space, x):
    """The frozen crowd model's P(pick A) -- the baseline curve on the UI."""
    return _sigmoid(space.scale * float(space.w @ np.asarray(x)))


def select_pair(posterior, cand_feats, top_frac=0.2, rng=None):
    """Active selection: among candidate pair features, pick one of the most
    informative (highest predictive uncertainty), randomized within the top
    fraction so sessions do not all see the same pairs. Returns the index."""
    rng = rng or np.random.default_rng()
    s = np.array([posterior.uncertainty(x) for x in cand_feats])
    k = max(1, int(len(s) * top_frac))
    top = np.argsort(-s)[:k]
    return int(rng.choice(top))


def replay_session(space, feats, choices, measure_mask, **kw):
    """Offline replay of one logged session -> per-swipe hit/miss for the
    personalized and global models, on measure pairs only. Train pairs update
    the posterior but never score."""
    post = UserPosterior(space, **kw)
    hits_p, hits_g, idx = [], [], []
    for i, (x, y) in enumerate(zip(feats, choices)):
        if measure_mask[i]:
            hits_p.append((post.predict(x) > 0.5) == bool(y))
            hits_g.append((global_prob(space, x) > 0.5) == bool(y))
            idx.append(i)
        post.update(x, y)
    return {"swipe_idx": np.array(idx), "hit_personal": np.array(hits_p),
            "hit_global": np.array(hits_g)}
