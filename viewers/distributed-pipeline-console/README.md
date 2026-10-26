# Distributed Pipeline Console

The console prepares runs in an external PostgreSQL database and displays their
stages, queue backlog, worker heartbeats and Parquet production. Sample previews
include metadata and audio playback.

## Start and use

Use the same database connection and artifact root as the workers:

```bash
export DIST_PIPELINE_DATABASE_URL='postgresql://user:password@host:5432/pipeline'
export ARTIFACT_DIR=/path/to/artifacts/edit-data-pipeline
envs/pipe-cpu/.venv/bin/python viewers/distributed-pipeline-console/app.py \
  --host 127.0.0.1 --port 20110
```

Open `http://127.0.0.1:20110`. Use a secure tunnel for a remote console; the
application is intended for trusted operators, not anonymous Internet access.
`--data-root` selects another directory of runtime outputs.

Choose a pipeline config to create a run. This seeds its input queue; launch
workers on your allocated machines as described in the
[runtime guide](../../speech_edit/runtime/README.md). The console does not start
remote processes. Deployment configs may also be supplied to the create-run API
or CLI by absolute path outside the repository.

### Status and controls

- **Preparing**: input seeding has not finished.
- **Waiting for workers**: work remains but no worker has a recent active heartbeat.
- **Running**: workers are active.
- **Stopping / stopped**: processing is disabled; active workers finish their current tasks.
- **Completed**: queues drained successfully.
- **Failed**: seeding failed or a queue item exhausted its attempts.
- **Unavailable**: runtime information cannot be read; it is not treated as a successful run.

Stop disables new claims. Resume enables claims but does not start processes.
Restart workers using the same effective config. Workers with stale heartbeats
are marked unresponsive; inspect their hosts to determine the cause.

### Inspect output

The dashboard shows registered row counts, audio duration, file sizes and
throughput. Open a Parquet preview, then a sample to inspect its transcript,
edit metadata and source/target audio. Preview paths are restricted to the run
directory.
