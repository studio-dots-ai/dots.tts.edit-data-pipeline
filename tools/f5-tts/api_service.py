"""Launch the locally maintained F5-TTS synthesis service."""
import runpy

if __name__ == "__main__":
    runpy.run_module("f5_service.synthesis_service", run_name="__main__")
