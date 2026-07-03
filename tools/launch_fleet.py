"""Launch the embed fleet: one job per petfinder zip (disjoint, resumable) plus
the pawpularity embed on the base job.

Requires the shell job to exist first (gives us an appPath to clone):

    hops job deploy predictable-embed pipelines/embed.py --env torch-training-pipeline

Then:  python tools/launch_fleet.py
"""
import hopsworks

ZIPS = {"predictable-embed-a": "587", "predictable-embed-b": "588",
        "predictable-embed-c": "1xx"}
CAP = 15000
RES = {"cores": 2.0, "memory": 8192, "gpus": 0, "shmSize": 128}
# 2 cores packs better than 4 when the cluster is tight (where-on-earth)


def main():
    proj = hopsworks.login()
    ja = proj.get_job_api()
    base = ja.get_job("predictable-embed")
    for name, zip_name in ZIPS.items():
        args = f"pool --zip {zip_name} --cap {CAP}"
        cfg = dict(base.config)
        cfg["appName"] = name
        cfg["defaultArgs"] = args
        cfg["resourceConfig"] = dict(RES)
        job = ja.get_job(name) or ja.create_job(name, cfg)
        job.config.update(cfg)
        job.save()
        ex = job.run(args=args, await_termination=False)
        print(f"{name}: zip {zip_name}, execution {ex.id}", flush=True)

    base.config["resourceConfig"] = dict(RES)
    base.save()
    ex = base.run(args="pawpularity", await_termination=False)
    print(f"predictable-embed: pawpularity, execution {ex.id}", flush=True)


if __name__ == "__main__":
    main()
