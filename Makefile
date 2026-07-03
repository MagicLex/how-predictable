# how-predictable -- FTI on Hopsworks: the machine learns your taste, live
# Feature (embeddings FGs) -> Training (Bradley-Terry prior vs zero-shot) -> Inference (KServe + game app)

pawpularity:         ## download the scored cold-start set (needs ~/.kaggle/kaggle.json)
	python3 collect/pawpularity.py

benchmark-job:       ## encoder shoot-out on pawpularity pairwise CV (pins taste_features.ENCODER)
	hops job deploy predictable-benchmark tools/benchmark_encoders.py \
		--env torch-training-pipeline --run --wait --overwrite

embed-fleet:         ## shard-parallel embed jobs: petfinder zips + pawpularity -> data/emb/
	python3 tools/launch_fleet.py

insert:              ## embedded parquets -> FGs + FV
	python3 pipelines/insert_fg.py

train-job:           ## Bradley-Terry prior vs baselines, register champion
	hops job deploy predictable-train pipelines/train.py \
		--env pandas-training-pipeline --run --wait --overwrite

app:                 ## deploy the Streamlit game (v1 inference lives in-app)
	python3 app/deploy_app.py

retrain-job:         ## flywheel: swipe train-pairs + pawpularity -> challenger (schedule daily)
	hops job deploy predictable-retrain pipelines/retrain.py \
		--env pandas-training-pipeline --run --wait --overwrite

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/  --/'
.PHONY: pawpularity benchmark-job embed-fleet insert train-job app retrain-job help
