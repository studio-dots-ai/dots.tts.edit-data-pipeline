"""Explicit-unit alignment adapters over official Qwen3-ASR."""
from typing import Any, Dict, List
import numpy as np
import torch
from qwen_asr.inference.qwen3_forced_aligner import (
    Qwen3ForceAlignProcessor as OfficialProcessor,
    Qwen3ForcedAligner as OfficialAligner,
    ForcedAlignResult,
)
from qwen_asr.inference.utils import AudioLike, normalize_audios


class Qwen3ForceAlignProcessor(OfficialProcessor):
    def tokenize_text(self, text: str, language: str) -> List[str]:
        language = language.lower()

        if language.lower() == "japanese":
            word_list = self.tokenize_japanese(text)
        elif language.lower() == "korean":
            if self.ko_tokenizer is None:
                from soynlp.tokenizer import LTokenizer

                self.ko_tokenizer = LTokenizer(scores=self.ko_score)
            word_list = self.tokenize_korean(self.ko_tokenizer, text)
        else:
            word_list = self.tokenize_space_lang(text)

        return word_list


    def build_aligner_input_text(self, word_list: List[str]) -> str:
        input_text = "<timestamp><timestamp>".join(word_list) + "<timestamp><timestamp>"
        input_text = "<|audio_start|><|audio_pad|><|audio_end|>" + input_text

        return input_text


    def encode_timestamp(self, text: str, language: str) -> List[str]:
        word_list = self.tokenize_text(text, language)
        input_text = self.build_aligner_input_text(word_list)

        return word_list, input_text



class Qwen3ForcedAligner(OfficialAligner):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        instance = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        instance.aligner_processor = Qwen3ForceAlignProcessor()
        return instance

    def _align_prepared(
        self,
        audios: List[np.ndarray],
        word_lists: List[List[str]],
        aligner_input_texts: List[str],
    ) -> List[ForcedAlignResult]:
        if not (len(audios) == len(word_lists) == len(aligner_input_texts)):
            raise ValueError(
                "Batch size mismatch: "
                f"audio={len(audios)}, word_lists={len(word_lists)}, "
                f"aligner_input_texts={len(aligner_input_texts)}"
            )

        inputs = self.processor(
            text=aligner_input_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        logits = self.model.thinker(**inputs).logits
        output_ids = logits.argmax(dim=-1)

        results: List[ForcedAlignResult] = []
        for input_id, output_id, word_list in zip(
            inputs["input_ids"], output_ids, word_lists
        ):
            masked_output_id = output_id[input_id == self.timestamp_token_id]
            timestamp_ms = (
                (masked_output_id * self.timestamp_segment_time).to("cpu").numpy()
            )
            timestamp_output = self.aligner_processor.parse_timestamp(
                word_list, timestamp_ms
            )
            for it in timestamp_output:
                it["start_time"] = round(it["start_time"] / 1000.0, 3)
                it["end_time"] = round(it["end_time"] / 1000.0, 3)
            results.append(self._to_structured_items(timestamp_output))

        return results


    @torch.inference_mode()
    def align_with_units(
        self,
        audio: AudioLike,
        units: List[str],
    ) -> ForcedAlignResult:
        """Run forced alignment against an explicit list of alignment units."""

        results = self._align_prepared(
            audios=normalize_audios([audio]),
            word_lists=[list(units)],
            aligner_input_texts=[
                self.aligner_processor.build_aligner_input_text(list(units))
            ],
        )
        return results[0]
