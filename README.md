# how predictable.

![how predictable.](assets/banner.svg)

Two pets. Click the one you like. The machine has already guessed which one
you'll pick, and the line at the top shows how often it reads you right. It
starts near a coin flip and climbs as it learns you: that line is the product.

Live demo: Hopsworks app `howpredictable` (project createnew).

![model comparison](assets/model_comparison.png)

## What it shows

Online per-user preference learning on top of a crowd prior, as a game:

- A **crowd model** (Bradley-Terry logistic head on frozen image embeddings)
  trained on 9,912 PetFinder photos with real engagement scores (Kaggle
  Pawpularity). It knows what everyone likes.
- A **per-user Bayesian layer** that starts as the crowd and learns your delta,
  one swipe at a time, in the app session. No account, no history: 25
  parameters, updated in closed form per click.
- The UI plots both: the frozen crowd model and your personalized model, on the
  same swipes. The gap between the curves is what the machine learned about you.

Supervised online preference learning. Not RL: nothing optimizes which pets you
see for engagement; active selection maximizes information about your taste.

## The two design decisions that matter

**Personalization happens in 25 dimensions, not 768.** A swipe carries roughly
one dimension of information, so 30 swipes cannot pin a 768-weight model.
The per-user layer works on phi(x) = [crowd logit, top-24 pool PCs] with prior
mean [1, 0, ..., 0]: "start as the crowd, learn where you disagree".
Simulation before any deployment: crowd model flat at 61%, personalized 71%
by swipe 20-30, 77% by 50.

**The accuracy line only counts random pairs.** Two of three pairs are chosen
actively (highest posterior uncertainty: they teach the model most). Actively
chosen pairs are harder than average, so scoring them would corrupt the metric.
Every third pair is uniform random and is the only one the accuracy line sees.
Train pairs train, measure pairs measure.

## FTI pipelines

```mermaid
flowchart LR
    A[PetFinder dogs\n700k photos] -->|embed fleet| F1[pet_embeddings FG\nonline]
    B[Pawpularity\n9,912 scored photos] -->|embed| F2[pawpularity_photos FG]
    F2 --> FV[taste_fv]
    FV -->|preference pairs\ngrouped CV| T[train job\nBradley-Terry prior]
    T --> MR[(model registry\npet_taste + TasteSpace)]
    MR --> APP[app: the game\nper-user Bayesian layer]
    F1 --> APP
    APP -->|swipes jsonl| FW[retrain job\npromotion gate]
    FW --> MR
```

Feature, training and inference pipelines are independent programs joined
through the Hopsworks feature store. The v1 app serves inference in-process
(the model is a weight vector; a KServe endpoint would wrap a dot product in
an HTTP hop). The day "upload your own pet" ships, the frozen encoder moves
behind a real KServe deployment.

## Honest numbers

| model | pairwise accuracy (5-fold CV) |
|---|---|
| coin flip | 0.500 |
| zero-shot appeal prompts | TBD |
| ridge score-then-rank | TBD |
| Bradley-Terry head (champion) | TBD |

Caveats, loud:

- Pawpularity scores are population-level photo appeal, not any individual's
  taste. The crowd model is the floor the personal layer must beat, per session.
- Preference pairs with score gap < 10 are noise and are excluded from
  training; accuracy vs gap is in the model card.
- The pool photos are 2023 PetFinder shelter listings (dataset license
  unknown, dataset card on HF: `drzraf/petfinder-dogs`). Non-commercial demo.

## Rebuild

```
make pawpularity      # needs kaggle token + accepted competition rules
make benchmark-job    # pins the encoder in taste_features.py
make embed-fleet      # petfinder zips -> embeddings + lead photos
make insert           # parquets -> FGs + FV
make train-job        # prior + baselines -> model registry
make app              # the game
make retrain-job      # flywheel (schedule daily)
```
