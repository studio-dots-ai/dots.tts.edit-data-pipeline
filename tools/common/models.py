"""
Shared Pydantic models for tool services.

These models are used across multiple services for encoding/decoding audio
and tensor data over HTTP APIs.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class EncodedAudio(BaseModel):
    """Base64-encoded audio bytes with format metadata."""

    data: str = Field(..., description="Base64-encoded audio bytes")
    format: str = Field("wav", description="Audio format: 'wav', 'mp3', 'flac', etc.")


class EncodedTensor(BaseModel):
    """Base64-encoded numpy array with metadata."""

    data: str = Field(..., description="Base64-encoded tensor bytes")
    shape: List[int] = Field(..., description="Tensor shape")
    dtype: str = Field("float32", description="Numpy dtype string")


class HealthResponse(BaseModel):
    """Standard health check response."""

    status: str = Field(..., description="Service health status")
    service: str = Field("", description="Service name")
    version: str = Field("1.0.0", description="Service version")
