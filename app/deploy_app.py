"""Deploy the game as a Hopsworks Streamlit app.

Stock python-app-pipeline base: the app imports numpy/pandas/streamlit/hopsworks
only (taste_online is pure numpy) -- no pickle, no pinned ML stack, no cloned
env. The thin-client rule from the playbook applies even though inference is
in-process, because the "model" is a weight vector, not a pickle.

Redeploys use the full recovery sequence (stop, purge lingering k8s deployment,
drain, stop zombie executions, settle, run) -- app.stop() returns before the
execution dies and the naive stop-then-run desyncs the platform state machine.
"""
import subprocess
import time
from pathlib import Path

import hopsworks

APP_NAME = "howpredictable"
ENV_NAME = "python-app-pipeline"

rel = str(Path(__file__).resolve()).split("/hopsfs/", 1)[1]
APP_PATH = str(Path(rel).parent / "app.py")


def _pods():
    out = subprocess.run(["kubectl", "get", "pods"], capture_output=True, text=True).stdout
    return [l.split()[0] for l in out.splitlines() if APP_NAME in l]


def _purge_k8s():
    out = subprocess.run(["kubectl", "get", "deployment"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if APP_NAME in line:
            name = line.split()[0]
            subprocess.run(["kubectl", "delete", "deployment", name], capture_output=True)
            print(f"purged k8s deployment {name}", flush=True)
    for _ in range(60):                     # bounded wait for pods to drain
        if not _pods():
            return
        time.sleep(5)
    raise RuntimeError("app pods refused to drain")


def _stop_zombies(project):
    job = project.get_job_api().get_job(APP_NAME)
    if job is None:
        return
    for ex in job.get_executions() or []:
        if ex.final_status in ("UNDEFINED", None):
            try:
                ex.stop()
                print(f"stopped zombie execution {ex.id}", flush=True)
            except Exception:
                pass


def main():
    project = hopsworks.login()
    apps = project.get_app_api()
    print(f"app_path={APP_PATH} env={ENV_NAME}", flush=True)
    app = apps.get_app(APP_NAME)
    if app is None:
        app = apps.create_app(name=APP_NAME, app_path=APP_PATH,
                              environment=ENV_NAME, memory=2048, cores=1.0)
    else:
        try:
            app.stop()
        except Exception:
            pass
        _purge_k8s()
        _stop_zombies(project)
        time.sleep(10)                      # let the platform settle before run
    app.run(await_serving=True)
    print("serving:", app.serving)
    print("URL:", app.get_url())


if __name__ == "__main__":
    main()
