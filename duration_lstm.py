from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from cde import UNet1D


class DurationLSTM(nn.Module):
    """Duration-aware recurrent baseline for text encoder features."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        projection_num_layers: int = 2,
        time_norm_mode: str = "utterance",
        time_norm_value: float = 1024.0,
        min_duration: float = 0.001,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"Expected num_layers >= 1, got {num_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"Expected dropout in [0, 1), got {dropout}")
        if time_norm_mode not in {"utterance", "global"}:
            raise ValueError(f"Unknown time_norm_mode '{time_norm_mode}'")
        if time_norm_mode == "global" and time_norm_value <= 0.0:
            raise ValueError(
                f"Expected time_norm_value > 0 for global mode, got {time_norm_value}"
            )
        if min_duration <= 0.0:
            raise ValueError(f"Expected min_duration > 0, got {min_duration}")

        self.channels = int(channels)
        self.time_norm_mode = str(time_norm_mode)
        self.time_norm_value = float(time_norm_value)
        self.min_duration = float(min_duration)

        self.lstm = nn.LSTM(
            input_size=self.channels + 1,
            hidden_size=int(hidden_channels),
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=bool(bidirectional),
            dropout=float(dropout) if num_layers > 1 else 0.0,
        )
        output_channels = int(hidden_channels) * (2 if bidirectional else 1)
        self.readout = UNet1D(
            in_channels=output_channels,
            mid_channels=int(hidden_channels),
            out_channels=self.channels,
            num_layers=int(projection_num_layers),
            dropout=float(dropout),
        )

        gate_init = 3.0
        self.gate_logit = nn.Parameter(torch.tensor(gate_init))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        durations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected x to have shape (batch, channels, length), got {tuple(x.shape)}"
            )
        if mask.ndim != 3:
            raise ValueError(
                f"Expected mask to have shape (batch, 1, length), got {tuple(mask.shape)}"
            )

        batch, channels, length = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected x with {self.channels} channels, got {channels}")
        if mask.shape != (batch, 1, length):
            raise ValueError(
                f"Expected mask with shape {(batch, 1, length)}, got {tuple(mask.shape)}"
            )

        out_dtype = x.dtype
        compute_dtype = torch.float32
        valid_mask = mask[:, 0, :].bool()
        lengths = valid_mask.long().sum(dim=1).clamp_min(1)

        if durations is None:
            durations = torch.ones(
                (batch, length), device=x.device, dtype=compute_dtype
            )
        else:
            if durations.ndim == 3:
                durations = durations[:, 0, :]
            if durations.shape != (batch, length):
                raise ValueError(
                    "Expected durations to have shape "
                    f"{(batch, length)} or {(batch, 1, length)}"
                )
            durations = durations.to(device=x.device, dtype=compute_dtype)

        autocast_ctx = (
            torch.autocast(device_type="cuda", enabled=False)
            if x.is_cuda
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with autocast_ctx:
            starts = self._phone_start_times(durations, valid_mask)
            path = torch.cat(
                [x.transpose(1, 2).to(dtype=compute_dtype), starts.unsqueeze(-1)],
                dim=-1,
            )
            packed = pack_padded_sequence(
                path,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.lstm(packed)
            output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=length,
            )
            residual = self.readout(
                output.transpose(1, 2),
                mask.to(dtype=compute_dtype),
            )

        residual = residual.to(dtype=out_dtype)
        gate = torch.sigmoid(self.gate_logit)
        return (x + gate * residual) * mask.to(dtype=out_dtype)

    def _phone_start_times(
        self,
        durations: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        durations = durations.clamp_min(self.min_duration)
        durations = durations * valid_mask.to(dtype=durations.dtype)
        starts = torch.cat(
            [
                durations.new_zeros(durations.shape[0], 1),
                torch.cumsum(durations[:, :-1], dim=1),
            ],
            dim=1,
        )
        if self.time_norm_mode == "utterance":
            denominator = durations.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            denominator = durations.new_tensor(self.time_norm_value)
        return starts / denominator
