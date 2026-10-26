# Pipeline runtime

The runtime executes the stages and queues declared in `topology.bindings`.
Pipeline adapters own business logic; workers call tool services over HTTP.
PostgreSQL stores durable tasks, worker heartbeats and run control state, while
shared storage holds intermediate payloads and final Parquet files.

![Distributed pipeline runtime overview](../../docs/images/distributed-pipeline-overview.png)

## Run a pipeline

### Prepare the deployment

Provide an existing PostgreSQL database accessible to every worker and the
console. Use a dedicated database and a user allowed to create the runtime tables.
Runs share these tables and are isolated by `run_id`.

```bash
export DIST_PIPELINE_DATABASE_URL='postgresql://user:password@host:5432/pipeline'
export ARTIFACT_DIR=/path/to/artifacts/edit-data-pipeline
```

Keep credentials outside the repository. Configure source inputs, model files,
service Python environments and worker counts in your deployment YAML using the
[configuration guide](../../configs/README.md). Every worker needs access to the
same effective config, input files and artifact paths.

### Create and execute

Create a run and start its workers on the current machine:

```bash
envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.submit_run \
  --run-config /path/to/deployment.yaml --run-id example01 --local
```

Omit `--local` to prepare the run for workers launched elsewhere. The command
prints the effective `run_config.yaml` path. Run identities and output directories
must be new; use the existing effective config when resuming work.

The local runner starts `stages.<name>.max_workers` processes per stage using
`worker_python_bin`. Configure `local_workers.<stage>.gpu_ids` for local GPU
placement. The runner assigns distinct service-port offsets to its processes.
It waits for completion or a stop request and reports failed work or child exits.
There is no default run time limit; `--timeout-sec` is an explicit operator option.

On each allocated machine, start one stage worker using the effective config:

```bash
envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.entry \
  --run-config /path/to/run/run_config.yaml \
  --pipeline text_edit --stage eval --worker-id eval-machine1
```

Use distinct worker IDs. Set `CUDA_VISIBLE_DEVICES` for each process and
`DIST_PIPELINE_WORKER_INDEX` to distinct nonnegative values when workers share a
host, so service ports do not collide. Configure each service's GPU IDs within
its visible devices. The runtime does not allocate machines or launch remote
processes.

## Monitor and control

The [console](../../viewers/distributed-pipeline-console/README.md) creates runs,
shows queue and worker state, and previews written samples with audio playback.
Stop requests prevent workers from claiming new tasks; workers finish their
current work and flush Parquet before exiting. Resume enables processing again;
start replacement workers with the same effective configuration:

```bash
envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.local_runner \
  --run-config /path/to/run/run_config.yaml
```

An unresponsive heartbeat is not proof that a process has exited. Inspect the
worker host when diagnosing failures. Expired task leases return to the queue;
processing failures obey the task attempt limit and become dead items when it
is exhausted. Workers do not silently change algorithms after failures.

A run completes when all its queue work is done and no dead item remains.
Stopping a run does not shut down PostgreSQL or release externally allocated
machines. Database connectivity is required for monitoring and control.

### Output and cleanup

Save stages write Parquet parts before acknowledging their input tasks. They use
one queue-processing client per worker. `parquet_output` controls batching and
`pipeline_config.output_dataset_name` supplies the dataset name.

Workers remove consumed intermediate payloads only after a successful queue
acknowledgement. Set `runtime.artifact_cleanup.enabled: false` to retain them.
The existing `speech_edit.runtime.artifact_gc` command can inspect and remove
unreferenced artifacts. Outputs, logs and effective configs live under
`ARTIFACT_DIR`; source indexes can be shared across runs.
