from __future__ import annotations
from speech_edit.pipelines import load_parameters
PARAMETERS = load_parameters("pipeline/text_edit/defaults")


import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from speech_edit.runtime.models import DEFAULT_STAGE_WORKER_PYTHON_BIN
from speech_edit.runtime.local_runner import run_local
from speech_edit.runtime.queue_store import QueueStore
from speech_edit.runtime.source_seed import seed_parquet_queue


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_text_edit_dist_pipeline_local_e2e():
    if os.environ.get("RUN_TEXT_EDIT_DIST_PIPELINE_E2E") != "1":
        pytest.skip("set RUN_TEXT_EDIT_DIST_PIPELINE_E2E=1 to run real local E2E")

    source = os.environ.get("TEXT_EDIT_E2E_SOURCE")
    if not source:
        pytest.skip("TEXT_EDIT_E2E_SOURCE must point to a parquet/list source")

    repo_root = _repo_root()
    date = datetime.utcnow().strftime("%Y-%m-%d")
    run_dir = Path(
        os.environ.get(
            "TEXT_EDIT_E2E_RUN_DIR",
            str(
                repo_root
                / "outputs"
                / date
                / f"text_edit_dist_pipeline_e2e_{uuid.uuid4().hex[:8]}"
            ),
        )
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = run_dir / "run_config.json"

    run_config = {
        "run_id": run_dir.name,
        "pipeline": "text_edit",
        "adapter_class": (
            "speech_edit.pipelines.text.stage_adapter:"
            "TextEditStageAdapter"
        ),
        "project_root": str(repo_root),
        "run_dir": str(run_dir),
        "artifact_root": str(run_dir / "artifacts"),
        "pipeline_config": {**load_parameters("pipeline/text_edit/defaults"),
            "seed": int(os.environ.get("TEXT_EDIT_E2E_SEED", "42")),
            "num_generations": int(os.environ.get("TEXT_EDIT_E2E_NUM_GENERATIONS", "1")),
            "aligner_backend": "qwen3",
            "output_dataset_name": "TextEdit-DistPipeline-E2E",
        },
        "queues": {
            "source_items": {"max_size": 8},
            "text.sample_to_plan": {"max_size": 16},
            "text.plan_to_synth": {"max_size": 8},
            "text.synth_to_eval": {"max_size": 8},
            "text.eval_to_save": {"max_size": 8},
        },
        "stages": {
            "sample": {
                "worker_python_bin": DEFAULT_STAGE_WORKER_PYTHON_BIN,
                "max_workers": 1,
            },
            "plan": {
                "worker_python_bin": DEFAULT_STAGE_WORKER_PYTHON_BIN,
                "max_workers": int(os.environ.get("TEXT_EDIT_E2E_PLAN_WORKERS", "2")),
                "services": {
                    "llm": {
                        "managed": False,
                        "url": os.environ.get("TEXT_EDIT_E2E_LLM_URL", "http://127.0.0.1:8012"),
                    }
                },
            },
            "synth": {
                "worker_python_bin": DEFAULT_STAGE_WORKER_PYTHON_BIN,
                "max_workers": 1,
                "services": {
                    "f5": {
                        "managed": False,
                        "url": os.environ.get("TEXT_EDIT_E2E_F5_URL", "http://127.0.0.1:8001"),
                    },
                    "aligner": {
                        "managed": False,
                        "url": os.environ.get("TEXT_EDIT_E2E_ALIGNER_URL", "http://127.0.0.1:8029"),
                    },
                },
            },
            "eval": {
                "worker_python_bin": DEFAULT_STAGE_WORKER_PYTHON_BIN,
                "max_workers": 1,
                "services": {
                    "asr": {
                        "managed": False,
                        "url": os.environ.get("TEXT_EDIT_E2E_ASR_URL", "http://127.0.0.1:8010"),
                    },
                    "wer": {
                        "managed": False,
                        "url": os.environ.get("TEXT_EDIT_E2E_WER_URL", "http://127.0.0.1:8011"),
                    },
                    "utmos": {
                        "managed": False,
                        "url": os.environ.get("TEXT_EDIT_E2E_UTMOS_URL", "http://127.0.0.1:8017"),
                    },
                    "spk_sim": {
                        "managed": False,
                        "url": os.environ.get("TEXT_EDIT_E2E_SPK_SIM_URL", "http://127.0.0.1:8019"),
                    },
                    "wdtw": {
                        "managed": False,
                        "url": os.environ.get("TEXT_EDIT_E2E_WDTW_URL", "http://127.0.0.1:8032"),
                    },
                },
            },
            "save": {
                "worker_python_bin": DEFAULT_STAGE_WORKER_PYTHON_BIN,
                "max_workers": 1,
                "parquet_output": {
                    "enabled": True,
                    "output_dir": str(run_dir / "parquet"),
                    "flush_rows": 1,
                    "flush_bytes_mb": 256,
                    "flush_interval_sec": 1,
                },
            },
        },
        "topology": {
            "bindings": {
                "source_items": {"consumers": ["sample"]},
                "text.sample_to_plan": {"producer": "sample", "consumers": ["plan"]},
                "text.plan_to_synth": {"producer": "plan", "consumers": ["synth"]},
                "text.synth_to_eval": {"producer": "synth", "consumers": ["eval"]},
                "text.eval_to_save": {"producer": "eval", "consumers": ["save"]},
            }
        },
        "local_workers": {
            "sample": {"count": 1},
            "plan": {"count": int(os.environ.get("TEXT_EDIT_E2E_PLAN_WORKERS", "2"))},
            "synth": {"count": 1, "gpu_ids": [os.environ.get("TEXT_EDIT_E2E_SYNTH_GPU", "0")]},
            "eval": {"count": 1, "gpu_ids": [os.environ.get("TEXT_EDIT_E2E_EVAL_GPU", "1")]},
            "save": {"count": 1},
        },
    }
    run_config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    inserted = seed_parquet_queue(
        run_id=str(run_config["run_id"]),
        queue_name="source_items",
        sources=[source],
        rows_per_item=int(os.environ.get("TEXT_EDIT_E2E_ROWS_PER_ITEM", "1")),
        queue_max_size=8,
    )
    assert inserted > 0

    stats = run_local(
        run_config_path=str(run_config_path),
        timeout_sec=float(os.environ.get("TEXT_EDIT_E2E_TIMEOUT_SEC", "7200")),
        poll_sec=5,
    )

    assert stats["run_complete"] == 1
    assert QueueStore(run_id=str(run_config["run_id"])).completion_stats()["run_complete"] == 1
    parts = sorted((run_dir / "parquet").glob("worker_*/part-*.parquet"))
    assert parts
    table = pq.read_table(parts[0])
    assert table.num_rows > 0
    assert {"audio_id", "audios", "segments", "instructions", "eval_results"}.issubset(
        set(table.column_names)
    )
    row = table.slice(0, 1).to_pylist()[0]
    segments = json.loads(row["segments"])
    assert "alignment" in segments["src"]
    assert "alignment" in segments["tgt"]
