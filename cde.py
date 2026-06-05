from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcde

# by matchatts https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/decoder.py
class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Block1D(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8, dropout: float = 0.0):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask

class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=True, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs):
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs

class UNet1D(nn.Module):
    """Small 1D UNet using existing decoder.py blocks."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mid_channels = int(mid_channels)
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError(f"Expected num_layers >= 1, got {self.num_layers}")
        groups = self._groups_for(self.mid_channels)
        self.in_block = Block1D(in_channels, self.mid_channels, groups=groups, dropout=dropout)
        self.down = Downsample1D(self.mid_channels)
        self.mid_blocks = nn.ModuleList(
            [
                Block1D(self.mid_channels, self.mid_channels, groups=groups, dropout=dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.up = Upsample1D(self.mid_channels, use_conv_transpose=True)
        self.out_block = Block1D(
            2 * self.mid_channels,
            self.mid_channels,
            groups=groups,
            dropout=dropout,
        )
        self.proj = nn.Conv1d(self.mid_channels, out_channels, 1)

    @staticmethod
    def _groups_for(channels: int) -> int:
        for g in (8, 4, 2, 1):
            if channels % g == 0:
                return g
        return 1

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x0 = self.in_block(x, mask)
        d = self.down(x0)
        d_mask = torch.nn.functional.interpolate(mask, size=d.shape[-1], mode="nearest")
        m = d
        for mid_block in self.mid_blocks:
            m = mid_block(m, d_mask)
        u = self.up(m)
        if u.shape[-1] != x0.shape[-1]:
            u = torch.nn.functional.interpolate(u, size=x0.shape[-1], mode="nearest")
        h = torch.cat([x0, u], dim=1)
        h = self.out_block(h, mask)
        return self.proj(h) * mask


class CDEFunc(torch.nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        width=None,
        num_layers: int = 2,
        output_activation: str = "none",
        dropout: float = 0.0,
    ):
        super(CDEFunc, self).__init__()
        del width
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        if output_activation not in {"none", "tanh"}:
            raise ValueError(f"Unknown output_activation '{output_activation}'")
        self.output_activation = output_activation
        self.unet = UNet1D(
            in_channels=1,
            mid_channels=hidden_channels,
            out_channels=input_channels,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        del t
        input_dtype = z.dtype
        z_in = z.unsqueeze(1).to(dtype=torch.float32)  # (b, 1, hidden)
        z_mask = torch.ones((z_in.shape[0], 1, z_in.shape[-1]), device=z.device, dtype=z_in.dtype)
        vf = self.unet(z_in, z_mask)  # (b, input, hidden)
        vf = vf.transpose(1, 2).contiguous()  # (b, hidden, input)
        if self.output_activation == "tanh":
            vf = vf.tanh()
        return vf.to(input_dtype)


class NeuralCDE(nn.Module):
    """Neural CDE block for token sequences.

    This module is shaped to match the rest of Matcha's text-side components:
    it consumes an encoder sequence `(B, C, L)` along with a `(B, 1, L)` mask and
    per-token durations `(B, L)`/`(B, 1, L)`, and returns `(B, C, L)`.

    Phone timing is represented as an extra control-path channel. The CDE
    itself is solved on a shared index-space grid so the whole batch can be
    integrated in one call.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        *,
        interpolation: str = "linear",
        solver: str = "rk4",
        num_layers: int = 2,
        vf_output_activation: str = "none",
        readout_type: str = "unet",
        time_norm_mode: str = "utterance",
        time_norm_value: float = 1024.0,
        min_duration: float = 1e-3,
        adjoint: bool = True,
        dt: float = 0.01,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        dropout: float = 0.0,
    ):
        super().__init__()
        if interpolation not in {"linear", "cubic"}:
            raise ValueError(f"Unknown interpolation '{interpolation}'")
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)

        self.interpolation = interpolation
        self.solver = solver
        if readout_type not in {"unet", "linear"}:
            raise ValueError(f"Unknown readout_type '{readout_type}'")
        self.readout_type = str(readout_type)
        if time_norm_mode not in {"utterance", "global"}:
            raise ValueError(f"Unknown time_norm_mode '{time_norm_mode}'")
        self.time_norm_mode = str(time_norm_mode)
        self.time_norm_value = float(time_norm_value)
        if self.time_norm_mode == "global" and self.time_norm_value <= 0.0:
            raise ValueError(f"Expected time_norm_value > 0 for global mode, got {self.time_norm_value}")
        self.min_duration = float(min_duration)
        if self.min_duration <= 0.0:
            raise ValueError(f"Expected min_duration > 0, got {self.min_duration}")
        self.adjoint = bool(adjoint)
        self.dt = float(dt)
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.dropout = float(dropout)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"Expected dropout in [0, 1), got {self.dropout}")

        self.input_channels = self.channels + 1
        self.func = CDEFunc(
            self.input_channels,
            self.hidden_channels,
            num_layers=num_layers,
            output_activation=vf_output_activation,
            dropout=self.dropout,
        )
        self.init_rf = 8
        self.initial_unet = UNet1D(
            in_channels=self.input_channels,
            mid_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            num_layers=num_layers,
            dropout=self.dropout,
        )
        if self.readout_type == "unet":
            self.readout_unet = UNet1D(
                in_channels=self.hidden_channels,
                mid_channels=self.hidden_channels,
                out_channels=self.channels,
                num_layers=num_layers,
                dropout=self.dropout,
            )
            self.readout_linear = None
        else:
            self.readout_unet = None
            self.readout_linear = nn.Linear(self.hidden_channels, self.channels)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, durations: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected x to have shape (batch, channels, length), got {tuple(x.shape)}"
            )
        if mask.ndim != 3:
            raise ValueError(
                f"Expected mask to have shape (batch, 1, length), got {tuple(mask.shape)}"
            )

        b, c, t = x.shape
        if c != self.channels:
            raise ValueError(f"Expected x with {self.channels} channels, got {c}")

        out_dtype = x.dtype
        compute_dtype = torch.float32
        mask_bool = mask[:, 0, :].bool()
        x_t = x.transpose(1, 2).to(dtype=compute_dtype)  # (b, t, c)

        if durations is None:
            durations = torch.ones((b, t), device=x.device, dtype=compute_dtype)
        else:
            if durations.ndim == 3:
                durations = durations[:, 0, :]
            if durations.ndim != 2:
                raise ValueError(
                    f"Expected durations to have shape (batch, length) or (batch, 1, length)"
                )
            durations = durations.to(device=x.device, dtype=compute_dtype)

        autocast_ctx = (
            torch.autocast(device_type="cuda", enabled=False)
            if x.is_cuda
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with autocast_ctx:
            starts = self._phone_start_times(durations, mask_bool)
            path = torch.cat([x_t, starts.unsqueeze(-1)], dim=-1)
            path = self._fill_forward(path, mask_bool)

            if t == 1:
                z_t = self._initial_state(path).unsqueeze(1)
            else:
                X, coeffs = self._make_interpolation(path)
                z0 = self._initial_state(path)
                t_grid = torch.arange(t, device=x.device, dtype=compute_dtype)
                cdeint_kwargs = self._cdeint_kwargs(X, z0, t_grid, coeffs)
                z_t = torchcde.cdeint(**cdeint_kwargs)  # (b, t, hidden)

            y = self._readout(z_t, mask_bool)
        y = y.to(dtype=out_dtype)
        return y * mask.to(dtype=out_dtype)

    def _phone_start_times(self, durations: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        durations = durations.clamp_min(self.min_duration) * valid_mask.to(dtype=durations.dtype)
        starts = torch.cat(
            [durations.new_zeros(durations.shape[0], 1), torch.cumsum(durations[:, :-1], dim=1)],
            dim=1,
        )
        if self.time_norm_mode == "utterance":
            denom = durations.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            denom = durations.new_tensor(self.time_norm_value)
        return starts / denom

    def _fill_forward(self, path: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        lengths = valid_mask.long().sum(dim=1).clamp_min(1)
        final_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, path.shape[-1])
        final_values = path.gather(dim=1, index=final_idx)
        return torch.where(valid_mask.unsqueeze(-1), path, final_values.expand_as(path))

    def _make_interpolation(self, path: torch.Tensor):
        if self.interpolation == "linear":
            coeffs = torchcde.linear_interpolation_coeffs(path)
            return torchcde.LinearInterpolation(coeffs), coeffs
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(path)
        return torchcde.CubicSpline(coeffs), coeffs

    def _initial_state(self, path: torch.Tensor) -> torch.Tensor:
        rf = min(self.init_rf, path.shape[1])
        init_x = path[:, :rf, :].transpose(1, 2)  # (b, input, rf)
        init_mask = torch.ones((path.shape[0], 1, rf), device=path.device, dtype=path.dtype)
        init_feats = self.initial_unet(init_x, init_mask)  # (b, hidden, rf)
        return init_feats[:, :, -1]  # (b, hidden)

    def _cdeint_kwargs(self, X, z0: torch.Tensor, t_grid: torch.Tensor, coeffs: torch.Tensor):
        cdeint_kwargs = dict(
            X=X,
            z0=z0,
            func=self.func,
            t=t_grid,
            adjoint=self.adjoint,
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol,
        )
        if self.adjoint:
            cdeint_kwargs["adjoint_params"] = tuple(self.func.parameters()) + (coeffs,)
        if self.solver == "reversible_heun":
            cdeint_kwargs["backend"] = "torchsde"
            cdeint_kwargs["dt"] = self.dt
        else:
            cdeint_kwargs["options"] = {"step_size": self.dt}
        return cdeint_kwargs

    def _readout(self, z_t: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if self.readout_type == "unet":
            z_seq = z_t.transpose(1, 2)  # (b, hidden, length)
            mask = valid_mask.unsqueeze(1).to(device=z_seq.device, dtype=z_seq.dtype)
            return self.readout_unet(z_seq, mask)  # (b, channels, length)
        y_t = self.readout_linear(z_t)  # (b, length, channels)
        return y_t.transpose(1, 2)  # (b, channels, length)
