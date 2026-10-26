"""
Shared utility functions for tool services.

Provides audio encoding/decoding, numpy serialization, temp file cleanup,
and logging setup used across multiple services.
"""

import base64
import logging
import os
import uuid
from typing import List, Optional, Union

import numpy as np

from .models import EncodedAudio, EncodedTensor


def decode_audio_input(
    audio_input: Union["EncodedAudio", str],
    temp_dir: str,
    prefix: str = "audio",
) -> str:
    """Decode audio input to a file path.

    If *audio_input* is already a path string the path is returned unchanged
    (after verifying it exists).  If it is an ``EncodedAudio`` object the
    base64 payload is written to a temporary file in *temp_dir*.

    Args:
        audio_input: Either a file-path string or an ``EncodedAudio`` object.
        temp_dir: Directory for writing temporary decoded files.
        prefix: Filename prefix for the temp file.

    Returns:
        Absolute path to the audio file (either the original or the temp file).

    Raises:
        ValueError: If a path string does not exist or base64 decoding fails.
    """
    if isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise ValueError(f"Audio file not found: {audio_input}")
        return audio_input

    try:
        audio_bytes = base64.b64decode(audio_input.data)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 audio: {e}")

    file_ext = audio_input.format.lower()
    if not file_ext.startswith("."):
        file_ext = f".{file_ext}"
    temp_path = os.path.join(temp_dir, f"{prefix}_{uuid.uuid4().hex}{file_ext}")

    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    return temp_path


def encode_numpy(arr: np.ndarray) -> EncodedTensor:
    """Encode a numpy array to a base64-encoded ``EncodedTensor``."""
    tensor_bytes = arr.tobytes()
    data_b64 = base64.b64encode(tensor_bytes).decode("utf-8")
    return EncodedTensor(
        data=data_b64,
        shape=list(arr.shape),
        dtype=str(arr.dtype),
    )


def decode_numpy(
    data: str,
    shape: List[int],
    dtype: str = "float32",
) -> np.ndarray:
    """Decode a base64-encoded tensor back to a numpy array.

    Args:
        data: Base64-encoded byte string.
        shape: Target shape for the array.
        dtype: Numpy dtype string.

    Returns:
        Reconstructed numpy array.
    """
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.dtype(dtype))
    return arr.reshape(shape)


def encode_audio_to_base64(path: str) -> str:
    """Read an audio file and return its contents as a base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def cleanup_temp_files(
    paths: List[str],
    temp_dir: Optional[str] = None,
) -> None:
    """Remove temporary files created during request handling.

    Only files whose path starts with *temp_dir* (when provided) are deleted.
    Errors are logged but never raised.

    Args:
        paths: List of file paths to consider for deletion.
        temp_dir: If given, only delete files under this directory.
    """
    logger = logging.getLogger(__name__)
    for path in paths:
        try:
            if not path or not os.path.exists(path):
                continue
            if temp_dir and not path.startswith(temp_dir):
                continue
            os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {path}: {e}")


def setup_logging(
    service_name: str = __name__,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure basic logging and return a named logger.

    Uses the standard format:
        ``%(asctime)s - %(name)s - %(levelname)s - %(message)s``

    Args:
        service_name: Logger name (typically ``__name__`` of the calling module).
        level: Logging level.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(service_name)
