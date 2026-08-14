"""ONNX export wrapper for the maintained ECal+TriggerPad Transformer."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


CANONICAL_FEATURES = (
    "is_ecal",
    "is_tpad",
    "ecal_x",
    "ecal_y",
    "ecal_z",
    "ecal_energy",
    "tpad_centroid",
    "tpad_pe",
)


class ECalTpadTransformerONNXWrapper(nn.Module):
    """Apply training preprocessing and return per-token class logits.

    Input
    -----
    raw_tokens:
        Float tensor shaped ``[num_tokens, 8]`` in the canonical feature order.

    Output
    ------
    logits:
        Float tensor shaped ``[num_tokens, num_classes]``. The caller should
        consume only rows whose input ``is_ecal`` feature is one.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        ecal_energy_transform: str,
        tpad_pe_transform: str,
        feature_norm: Mapping[str, Any] | None,
    ) -> None:
        super().__init__()
        if ecal_energy_transform not in {"raw", "log1p"}:
            raise ValueError(f"Unsupported ECal energy transform: {ecal_energy_transform}")
        if tpad_pe_transform not in {"raw", "log1p"}:
            raise ValueError(f"Unsupported TriggerPad PE transform: {tpad_pe_transform}")

        self.model = model
        self.apply_ecal_log1p = ecal_energy_transform == "log1p"
        self.apply_tpad_log1p = tpad_pe_transform == "log1p"

        if feature_norm is None:
            first_col = len(CANONICAL_FEATURES)
            mean = torch.empty(0, dtype=torch.float32)
            std = torch.empty(0, dtype=torch.float32)
        else:
            first_col = int(feature_norm["first_continuous_col"])
            mean = torch.as_tensor(feature_norm["mean"], dtype=torch.float32)
            std = torch.as_tensor(feature_norm["std"], dtype=torch.float32)
            expected = len(CANONICAL_FEATURES) - first_col
            if mean.numel() != expected or std.numel() != expected:
                raise ValueError(
                    "Checkpoint normalization has incompatible dimensions: "
                    f"first_continuous_col={first_col}, mean={mean.numel()}, "
                    f"std={std.numel()}, expected={expected}."
                )
            if torch.any(std <= 0):
                raise ValueError("Checkpoint normalization standard deviations must be positive.")

        self.first_continuous_col = first_col
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)

    def preprocess(self, raw_tokens: torch.Tensor) -> torch.Tensor:
        # Clone because deployment callers may reuse their input buffer.
        x = raw_tokens.to(dtype=torch.float32).clone()

        # Masking prevents a context token's placeholder ECal energy (or an ECal
        # token's placeholder TPad PE) from acquiring unintended values.
        if self.apply_ecal_log1p:
            ecal_mask = x[:, 0:1]
            transformed = torch.log1p(torch.clamp_min(x[:, 5:6], 0.0))
            x[:, 5:6] = transformed * ecal_mask
        if self.apply_tpad_log1p:
            tpad_mask = x[:, 1:2]
            transformed = torch.log1p(torch.clamp_min(x[:, 7:8], 0.0))
            x[:, 7:8] = transformed * tpad_mask

        if self.feature_mean.numel() != 0:
            continuous = x[:, self.first_continuous_col :]
            x[:, self.first_continuous_col :] = (
                continuous - self.feature_mean
            ) / self.feature_std
        return x

    def forward(self, raw_tokens: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(raw_tokens)

        # The maintained model accepts either one event [T,F] or a padded batch.
        # Calling its layers directly avoids exporting Python rank/shape checks
        # and fixes the deployment graph to the one-event contract.
        batched = x.unsqueeze(0)
        hidden = self.model.input_proj(batched)
        hidden = self.model.encoder(hidden)
        return self.model.head(hidden).squeeze(0)
