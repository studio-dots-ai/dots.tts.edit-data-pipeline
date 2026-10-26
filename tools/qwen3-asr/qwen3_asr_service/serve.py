"""Launch the official vLLM entrypoint after verifying its source checkout."""
import runpy


if __name__ == "__main__":
    runpy.run_module("qwen_asr.cli.serve", run_name="__main__")
