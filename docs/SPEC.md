# how-predictable -- ML system spec

Hot-or-not for pets with a live accuracy line. Two pet photos side by side, the
user clicks the one they like. Before the click, the model secretly predicts
which one. A rolling accuracy line at the top of the UI climbs as the model
learns the user: ~50-60% cold, ~70% after 20-30 swipes. The line is the product.

## AI-system card

| | |
|---|---|
| Prediction problem | Pairwise preference: P(user picks left of {A,B}). Ranking/binary hybrid. |
| KPI | The visible one: personalized accuracy gap over the global model, per session. Proxy for "the system learns an individual user online". |
| ML proxy metric | Pairwise accuracy (and log-loss) on held-out preference pairs; cold-start curve: accuracy vs number of swipes, via offline replay of logged sessions. |
| Data sources | 1. Kaggle `petfinder-pawpularity-score`: 9,912 shelter-pet photos with real engagement scores (cold-start prior). 2. HF `drzraf/petfinder-dogs`: 700k images / 150k dogs, 4-12 photos each (the pair pool; no labels). 3. `swipe_events` FG: the game's own clicks (flywheel). |
| ML-system type | Real-time. Online feature store lookup (precomputed embeddings) + per-session online learner in the app. |
| Consumed via | Streamlit app (the game). Deployment API underneath. |
| Monitoring | Inference logging of every pair served + prediction + outcome to `swipe_events`; rolling accuracy is itself the monitor. Retrain job reads it. |

## Model design (decided, see OPERATING-NOTES NEXT UP)

- Global prior: Bradley-Terry pairwise head trained on Pawpularity scores
  converted to preference pairs. Trains on the task the game plays.
- Images -> vectors at the door: one frozen encoder (benchmark DINOv2 vs SigLIP
  vs CLIP B/32), embeddings in an online FG, never pixels.
- Per-user layer: Bayesian logistic in a LOW-DIM taste subspace (ADF, diagonal
  Gaussian), in the app session. Raw 768-d is unlearnable in 30 swipes (~1
  effective dim of info per swipe); phi(x) = [crowd logit, top-24 pool PCs],
  prior mean [1, 0...] = "start as the crowd, learn your delta". Simulation
  (2026-07-02): global flat 61%, personal 61% -> 71% @ swipe 20-30 -> 77% @
  40-60; active selection +2.2pt held-out over random at swipe 20. The
  TasteSpace (PCA + logit scale) ships inside the pet_taste model artifact.
- Active pair selection trains, randomly-interleaved pairs measure. The accuracy
  line is computed ONLY on the random measure pairs. Non-negotiable.
- The UI line shows global-model accuracy AND personalized accuracy; the gap is
  the proof of personalization.
- Supervised online preference learning. NOT RL, docs say so. Bandit (which
  pets/photos to show) is explicitly v2.

## Pipelines (FTI, joined only through the feature store)

### F1. collect -- pool images (no Hopsworks deps)
`collect/pool.py`: stream a capped subset (~30-50k dogs, lead photo each) from
HF `drzraf/petfinder-dogs` to `data/pool/`, resumable, per-shard parquet
manifest. `collect/pawpularity.py`: kaggle download (needs kaggle.json) to
`data/pawpularity/`.
Blocked by: kaggle creds (pawpularity only). Skill: hops-features.

### F2. embed -- images -> vectors (Hopsworks JOBS)
`tools/benchmark_encoders.py` (job): DINOv2 / SigLIP / CLIP B/32 embeddings of
pawpularity images, small pairwise CV, pick the encoder by held-out pairwise
accuracy. `pipelines/embed.py` (shard-parallel job fleet, 2 cores each): embed
pool + pawpularity with the winner, per-shard parquet in `data/emb/`.
Blocked by: F1. Skill: hops-features / hops-job.

### F3. feature pipeline -- parquet -> FGs -> FV (Hopsworks JOB)
FGs: `pet_embeddings` v1 (online; pet_id PK, embedding, source, url/path meta),
`pawpularity_labels` v1 (pet_id, score), `swipe_events` v1 (event_time =
last-activity; session_id, pair ids, choice, model_pick, pair_kind
train|measure). FV `taste_fv` = pawpularity_labels join pet_embeddings.
Insert via SDK (timestamp quirk). Blocked by: F2. Skill: hops-fg / hops-fv.

### T1. train -- global prior (Hopsworks JOB)
`pipelines/train.py`: pawpularity scores -> preference pairs (score-gap
threshold; pairs GROUPED by pet across folds, no pet in both sides of a split).
Bradley-Terry logistic head on embedding differences. Baselines that must be
beaten: coin flip, zero-shot text-prompt score (CLIP "an appealing photo of a
pet"), score-regression-then-rank. k-fold CV mean. Register `pet_taste` with
card + images (pairwise ROC, calibration, accuracy-vs-score-gap,
run-progression). Blocked by: F3. Skill: hops-train.

### I1. app -- the game, and the online inference pipeline (Streamlit on Hopsworks)
No KServe deployment in v1: every pool pet is pre-embedded, so pair scoring is
one dot product on 768 floats -- an endpoint would be a fake predictor wrapping
a lookup. The app IS the inference pipeline: loads `w_global.npy` from the model
registry at session start; pair display; secret model pick; per-user Bayesian
logistic in session state updated per swipe; two rolling accuracy curves (global
vs personalized) computed on measure pairs only; swipe logging to
`swipe_events`. `.streamlit/config.toml` fileWatcherType=none.
KServe returns in v2 with "upload YOUR pet" (frozen encoder at request time =
genuinely heavy = real predictor). Blocked by: T1. Skill: hops-app.

### I2. flywheel -- retrain from swipes (scheduled Hopsworks JOB)
`pipelines/retrain.py`: swipe_events (train pairs only) + pawpularity pairs ->
retrain global prior -> register challenger; fixed eval pairs never move.
Blocked by: I1. Skill: hops-job / hops-monitoring.

## Order

F1 -> F2 -> F3 -> T1 -> I1 -> I2. Pawpularity (kaggle creds) gates F2's
benchmark and T1; the pool download and scaffolding are not gated.

## Honesty rules (README-bound)

- Pawpularity engagement is population-level "photo appeal", not any single
  user's taste: report the global model as the baseline the personal layer must
  beat, per session.
- Measure-pair accuracy only; train pairs are actively selected and would
  inflate/deflate the line.
- Cold-start curves reported per swipe count, not one number.
