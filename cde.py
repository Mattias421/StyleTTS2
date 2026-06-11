from __future__ import annotations

import torch
from torch._functorch.autograd_function import generate_single_level_function
import torch.nn as nn
import torch.nn.functional as F
import torchcde
from torch.nn.utils import weight_norm


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)

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

class SingleNeuralCDE(nn.Module):
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
        init_type: str = "reverse_lstm",
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
        if init_type not in {"unet", "reverse_lstm"}:
            raise ValueError(f"Unknown init_type '{init_type}'")
        self.init_type = str(init_type)
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
        if self.init_type == "unet":
            self.initial_unet = UNet1D(
                in_channels=self.input_channels,
                mid_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                num_layers=num_layers,
                dropout=self.dropout,
            )
            self.initial_lstm = None
        else:
            self.initial_unet = None
            self.initial_lstm = nn.LSTM(
                input_size=self.input_channels,
                hidden_size=self.hidden_channels,
                num_layers=1,
                batch_first=True,
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

        gate_init = 3.0
        self.gate_logit = nn.Parameter(torch.tensor(gate_init))

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
                z_t = self._initial_state(path, mask_bool).unsqueeze(1)
            else:
                X, coeffs = self._make_interpolation(path)
                z0 = self._initial_state(path, mask_bool)
                t_grid = torch.arange(t, device=x.device, dtype=compute_dtype)
                cdeint_kwargs = self._cdeint_kwargs(X, z0, t_grid, coeffs)
                z_t = torchcde.cdeint(**cdeint_kwargs)  # (b, t, hidden)

            y = self._readout(z_t, mask_bool)
        y = y.to(dtype=out_dtype)
        gate = torch.sigmoid(self.gate_logit)
        y = x + gate * y
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

    def _initial_state(
        self, path: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.init_type == "reverse_lstm":
            if valid_mask is None:
                valid_mask = torch.ones(
                    path.shape[:2], device=path.device, dtype=torch.bool
                )
            lengths = valid_mask.long().sum(dim=1).clamp_min(1)
            positions = torch.arange(path.shape[1], device=path.device).unsqueeze(0)
            reverse_idx = (lengths.unsqueeze(1) - 1 - positions).clamp_min(0)
            reverse_idx = reverse_idx.unsqueeze(-1).expand_as(path)
            reversed_path = path.gather(dim=1, index=reverse_idx)
            packed = nn.utils.rnn.pack_padded_sequence(
                reversed_path,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            self.initial_lstm.flatten_parameters()
            _, (hidden, _) = self.initial_lstm(packed)
            return hidden[-1]

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

class StackedCDEFunc(nn.Module):
    """Vector field for a stack of CDE layers solved in one cdeint call.

    Layer 0:
        dz_0 = f_0(z_0) dX

    Layer 1:
        dz_1 = f_1(z_1) dz_0

    Layer 2:
        dz_2 = f_2(z_2) dz_1

    etc.

    During integration we rewrite every layer as being controlled by dX:

        dz_1 = f_1(z_1) f_0(z_0) dX
        dz_2 = f_2(z_2) f_1(z_1) f_0(z_0) dX

    so torchcde only sees one control path X.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_cde_layers: int,
        *,
        num_layers: int = 2,
        output_activation: str = "none",
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_cde_layers < 1:
            raise ValueError(f"Expected num_cde_layers >= 1, got {num_cde_layers}")

        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_cde_layers = int(num_cde_layers)

        funcs = []

        # First CDE is controlled by the external path X.
        funcs.append(
            CDEFunc(
                input_channels=self.input_channels,
                hidden_channels=self.hidden_channels,
                num_layers=num_layers,
                output_activation=output_activation,
                dropout=dropout,
            )
        )

        # Higher CDEs are controlled by the previous hidden path.
        for _ in range(1, self.num_cde_layers):
            funcs.append(
                CDEFunc(
                    input_channels=self.hidden_channels,
                    hidden_channels=self.hidden_channels,
                    num_layers=num_layers,
                    output_activation=output_activation,
                    dropout=dropout,
                )
            )

        self.funcs = nn.ModuleList(funcs)

    @property
    def total_hidden_channels(self) -> int:
        return self.num_cde_layers * self.hidden_channels

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # z: (B, num_cde_layers * hidden)
        zs = z.split(self.hidden_channels, dim=-1)

        effective_vfs = []
        prev_effective_vf = None

        for layer_idx, func in enumerate(self.funcs):
            local_vf = func(t, zs[layer_idx])

            # local_vf shapes:
            #   layer 0: (B, hidden, input_channels)
            #   layer k: (B, hidden, hidden)
            if layer_idx == 0:
                effective_vf = local_vf
            else:
                # Chain rule:
                # local_vf:          (B, hidden, hidden)
                # prev_effective_vf: (B, hidden, input_channels)
                # effective_vf:      (B, hidden, input_channels)
                effective_vf = torch.bmm(local_vf, prev_effective_vf)

            effective_vfs.append(effective_vf)
            prev_effective_vf = effective_vf

        # torchcde expects vector field shape:
        #   (B, total_hidden, input_channels)
        return torch.cat(effective_vfs, dim=-2)

class NeuralCDE(nn.Module):
    """Multilayer Neural CDE block for token sequences.

    Consumes:
        x:         (B, C, L)
        mask:      (B, 1, L)
        durations: (B, L) or (B, 1, L)

    Returns:
        y:         (B, C, L)

    The first CDE layer is controlled by the text/timing path.
    Each higher CDE layer is controlled by the previous CDE hidden path.
    All layers are solved together in one torchcde.cdeint call.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        *,
        num_cde_layers: int = 2,
        interpolation: str = "linear",
        solver: str = "rk4",
        num_layers: int = 2,
        vf_output_activation: str = "none",
        readout_type: str = "unet",
        time_norm_mode: str = "utterance",
        time_norm_value: float = 1024.0,
        min_duration: float = 1e-3,
        init_type: str = "reverse_lstm",
        adjoint: bool = True,
        dt: float = 0.01,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        dropout: float = 0.0,
    ):
        super().__init__()
        if interpolation not in {"linear", "cubic"}:
            raise ValueError(f"Unknown interpolation '{interpolation}'")
        if readout_type not in {"unet", "linear"}:
            raise ValueError(f"Unknown readout_type '{readout_type}'")
        if time_norm_mode not in {"utterance", "global"}:
            raise ValueError(f"Unknown time_norm_mode '{time_norm_mode}'")
        if init_type not in {"unet", "reverse_lstm"}:
            raise ValueError(f"Unknown init_type '{init_type}'")
        if num_cde_layers < 1:
            raise ValueError(f"Expected num_cde_layers >= 1, got {num_cde_layers}")

        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.num_cde_layers = int(num_cde_layers)

        self.interpolation = interpolation
        self.solver = solver
        self.readout_type = str(readout_type)
        self.time_norm_mode = str(time_norm_mode)
        self.time_norm_value = float(time_norm_value)
        self.min_duration = float(min_duration)
        self.init_type = str(init_type)
        self.adjoint = bool(adjoint)
        self.dt = float(dt)
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.dropout = float(dropout)

        if self.time_norm_mode == "global" and self.time_norm_value <= 0.0:
            raise ValueError(
                f"Expected time_norm_value > 0 for global mode, got {self.time_norm_value}"
            )
        if self.min_duration <= 0.0:
            raise ValueError(f"Expected min_duration > 0, got {self.min_duration}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"Expected dropout in [0, 1), got {self.dropout}")

        self.input_channels = self.channels + 1

        self.func = CDEFunc(
            input_channels=self.input_channels,
            hidden_channels=self.hidden_channels,
            num_layers=num_layers,
            output_activation=vf_output_activation,
            dropout=self.dropout,
        )

        self.init_rf = 8

        if self.init_type == "unet":
            self.initial_unet = UNet1D(
                in_channels=self.input_channels,
                mid_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                num_layers=num_layers,
                dropout=self.dropout,
            )
            self.initial_lstm = None
        else:
            self.initial_unet = None
            self.initial_lstm = nn.LSTM(
                input_size=self.input_channels,
                hidden_size=self.hidden_channels,
                num_layers=1,
                batch_first=True,
            )

        # Initialisers for higher CDE layers.
        # First layer z0 comes from the path; higher z0s come from previous z0.
        self.higher_initializers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_channels, self.hidden_channels),
                    nn.SiLU(),
                    nn.Linear(self.hidden_channels, self.hidden_channels),
                )
                for _ in range(self.num_cde_layers - 1)
            ]
        )

        # Read out only the top CDE layer.
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

        b, c, t = x.shape
        if c != self.channels:
            raise ValueError(f"Expected x with {self.channels} channels, got {c}")

        out_dtype = x.dtype
        compute_dtype = torch.float32

        mask_bool = mask[:, 0, :].bool()
        x_t = x.transpose(1, 2).to(dtype=compute_dtype)  # (B, L, C)

        if durations is None:
            durations = torch.ones((b, t), device=x.device, dtype=compute_dtype)
        else:
            if durations.ndim == 3:
                durations = durations[:, 0, :]
            if durations.ndim != 2:
                raise ValueError(
                    "Expected durations to have shape "
                    "(batch, length) or (batch, 1, length)"
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
                z0 = self._initial_state(path, mask_bool)
                z_t = z0.unsqueeze(1)
            else:
                X, coeffs = self._make_interpolation(path)
                z0 = self._initial_state(path, mask_bool)

                t_grid = torch.arange(t, device=x.device, dtype=compute_dtype)

                cdeint_kwargs = self._cdeint_kwargs(X, z0, t_grid, coeffs)
                z_t = torchcde.cdeint(**cdeint_kwargs)

            # z_t: (B, L, num_cde_layers * hidden)
            z_top = self._top_hidden(z_t)

            y = self._readout(z_top, mask_bool)

        y = y.to(dtype=out_dtype)
        gate = torch.sigmoid(self.gate_logit)
        y = x + gate * y
        return y * mask.to(dtype=out_dtype)

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
            denom = durations.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            denom = durations.new_tensor(self.time_norm_value)

        return starts / denom

    def _fill_forward(
        self,
        path: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        lengths = valid_mask.long().sum(dim=1).clamp_min(1)
        final_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, path.shape[-1])
        final_values = path.gather(dim=1, index=final_idx)
        return torch.where(
            valid_mask.unsqueeze(-1),
            path,
            final_values.expand_as(path),
        )

    def _make_interpolation(self, path: torch.Tensor):
        if self.interpolation == "linear":
            coeffs = torchcde.linear_interpolation_coeffs(path)
            return torchcde.LinearInterpolation(coeffs), coeffs

        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(path)
        return torchcde.CubicSpline(coeffs), coeffs

    def _initial_state(
        self,
        path: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # First-layer initial state.
        if self.init_type == "reverse_lstm":
            if valid_mask is None:
                valid_mask = torch.ones(
                    path.shape[:2],
                    device=path.device,
                    dtype=torch.bool,
                )

            lengths = valid_mask.long().sum(dim=1).clamp_min(1)

            positions = torch.arange(path.shape[1], device=path.device).unsqueeze(0)
            reverse_idx = (lengths.unsqueeze(1) - 1 - positions).clamp_min(0)
            reverse_idx = reverse_idx.unsqueeze(-1).expand_as(path)

            reversed_path = path.gather(dim=1, index=reverse_idx)

            packed = nn.utils.rnn.pack_padded_sequence(
                reversed_path,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

            self.initial_lstm.flatten_parameters()
            _, (hidden, _) = self.initial_lstm(packed)

            z0_first = hidden[-1]
        else:
            rf = min(self.init_rf, path.shape[1])
            init_x = path[:, :rf, :].transpose(1, 2)  # (B, input, rf)

            init_mask = torch.ones(
                (path.shape[0], 1, rf),
                device=path.device,
                dtype=path.dtype,
            )

            init_feats = self.initial_unet(init_x, init_mask)  # (B, hidden, rf)
            z0_first = init_feats[:, :, -1]  # (B, hidden)

        z0s = [z0_first]

        # Higher-layer initial states.
        prev = z0_first
        for init in self.higher_initializers:
            prev = init(prev)
            z0s.append(prev)

        # Combined initial state for all CDE layers:
        #   (B, num_cde_layers * hidden)
        return torch.cat(z0s, dim=-1)

    def _top_hidden(self, z_t: torch.Tensor) -> torch.Tensor:
        # z_t: (B, L, num_cde_layers * hidden)
        return z_t[..., -self.hidden_channels:]

    def _cdeint_kwargs(
        self,
        X,
        z0: torch.Tensor,
        t_grid: torch.Tensor,
        coeffs: torch.Tensor,
    ):
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

    def _readout(
        self,
        z_t: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        # z_t is only the top layer: (B, L, hidden)
        if self.readout_type == "unet":
            z_seq = z_t.transpose(1, 2)  # (B, hidden, L)

            mask = valid_mask.unsqueeze(1).to(
                device=z_seq.device,
                dtype=z_seq.dtype,
            )

            return self.readout_unet(z_seq, mask)  # (B, C, L)

        y_t = self.readout_linear(z_t)  # (B, L, C)
        return y_t.transpose(1, 2)  # (B, C, L)

class TextEncoderCDE(nn.Module):
    """Text encoder whose convolutional features control stacked Neural CDEs."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        depth: int,
        n_symbols: int,
        *,
        cde_depth: int = 2,
        hidden_channels: int | None = None,
        actv: nn.Module | None = None,
        **cde_kwargs,
    ):
        super().__init__()
        if cde_depth < 1:
            raise ValueError(f"Expected cde_depth >= 1, got {cde_depth}")
        if actv is None:
            actv = nn.LeakyReLU(0.2)

        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                ChannelLayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))

        hidden_channels = hidden_channels or channels // 2
        cde_kwargs = dict(cde_kwargs)
        cde_kwargs.setdefault("init_type", "reverse_lstm")
        self.cde_layer = NeuralCDE(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    **cde_kwargs,
                )

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        m: torch.Tensor,
        attn: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(device=x.device, dtype=torch.bool)
        if m.ndim == 2:
            m = m.unsqueeze(1)
        if m.ndim != 3 or m.shape[1] != 1:
            raise ValueError(
                f"Expected m to have shape (batch, length) or (batch, 1, length), got {tuple(m.shape)}"
            )
        x.masked_fill_(m, 0.0)

        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)

        durations = self._durations_from_attention(attn, x.shape[-1])
        valid_mask = (~m).to(dtype=x.dtype)
        x = cde_layer(x, valid_mask, durations)
        return x.masked_fill(m, 0.0)

    @staticmethod
    def _durations_from_attention(
        attn: torch.Tensor | None, text_length: int
    ) -> torch.Tensor | None:
        if attn is None:
            return None
        if attn.ndim == 4 and attn.shape[1] == 1:
            attn = attn[:, 0]
        if attn.ndim != 3:
            raise ValueError(
                f"Expected attn to have shape (batch, text, frames), got {tuple(attn.shape)}"
            )
        if attn.shape[1] != text_length:
            raise ValueError(
                f"Attention text length {attn.shape[1]} does not match encoder length {text_length}"
            )
        return attn.sum(dim=-1).detach()

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        lengths = torch.full(
            (x.shape[0],), x.shape[1], device=x.device, dtype=torch.long
        )
        mask = torch.zeros_like(x, dtype=torch.bool)
        return self.forward(x, lengths, mask).transpose(1, 2)
