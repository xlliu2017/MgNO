"""
MgNO: Multigrid Neural Operator models.

This module implements the MgNO architecture, a family of neural operators
inspired by multigrid methods for solving PDEs. The key components follow the
classical multigrid V-cycle:

  - MgIte / MgIte_init : learnable smoothing iterations (analogous to Gauss-Seidel)
  - Restrict            : coarsening (restriction + prolongation projections)
  - MgConv variants     : full V-cycle convolution blocks
  - MgNO variants       : top-level models for specific PDE benchmarks

Reference:
  Liu, X. et al. "MgNO: Efficient Parameterization of Linear Operators via
  Multigrid." ICLR 2024.
  https://openreview.net/forum?id=eb3c8135137c8a60425a0320869ad87e
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from utilities3 import count_params


# ---------------------------------------------------------------------------
# Core multigrid building blocks
# ---------------------------------------------------------------------------

class MgIte(nn.Module):
    """One multigrid smoothing iteration: u <- u + S(f - A(u)).

    Args:
        A: Learnable operator approximating the PDE operator (u -> f space).
        S: Learnable smoother (f -> u space).
    """

    def __init__(self, A, S):
        super().__init__()
        self.A = A
        self.S = S

    def forward(self, out):
        """Apply one smoothing step.

        Args:
            out: Either a tuple ``(u, f)`` for subsequent iterations, or just
                 ``f`` for the very first iteration (handled by MgIte_init).

        Returns:
            tuple: Updated ``(u, f)``.
        """
        if isinstance(out, tuple):
            u, f = out
            u = u + self.S(f - self.A(u))
        else:
            f = out
            u = self.S(f)
        return (u, f)


class MgIte_init(nn.Module):
    """Initial smoothing step when no previous approximation is available.

    Computes ``u = S(f)`` and packages the result as ``(u, f)``.

    Args:
        S: Learnable smoother (f -> u space).
    """

    def __init__(self, S):
        super().__init__()
        self.S = S

    def forward(self, f):
        u = self.S(f)
        return (u, f)


class Restrict(nn.Module):
    """Multigrid restriction operator: projects ``(u, f)`` to a coarser grid.

    Optionally computes the residual ``f - A(u)`` before restricting, which
    corresponds to the *full approximation scheme* (FAS) when ``A`` is given.

    Args:
        Pi: Prolongation/restriction operator for *u* (fine -> coarse).
        R:  Restriction operator for *f* (fine -> coarse).
        A:  Optional operator for residual computation (u -> f space).
    """

    def __init__(self, Pi=None, R=None, A=None):
        super().__init__()
        self.Pi = Pi
        self.R = R
        self.A = A

    def forward(self, out):
        u, f = out
        if self.A is not None:
            f = self.R(f - self.A(u))
        else:
            f = self.R(f)
        u = self.Pi(u)
        return (u, f)

# ---------------------------------------------------------------------------
# Multigrid convolution blocks (V-cycle)
# ---------------------------------------------------------------------------

class MgConv(nn.Module):
    """Multigrid V-cycle convolutional block with layer-norm on the upsampling path.

    Builds a hierarchy of *num_iteration* levels.  Each level applies pre-smoothing
    iterations (``MgIte``), restricts to the next coarser grid, and then on the
    upward pass applies post-smoothing after adding the prolongated coarse correction.

    Args:
        num_iteration: List of ``[num_pre, num_post]`` iteration counts per level.
        num_channel_u: Number of channels for the solution field *u*.
        num_channel_f: Number of channels for the right-hand-side field *f*.
        padding_mode: Convolution padding mode (default: ``'zeros'``).
        bias: Whether to use bias in smoothing convolutions.
        use_res: If True, use residual-based restriction (FAS-style).
        elementwise_affine: Whether LayerNorm has learnable affine parameters.
    """

    def __init__(self, num_iteration, num_channel_u, num_channel_f,
                 padding_mode='zeros', bias=False, use_res=False,
                 elementwise_affine=True):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        # Layer norms applied after prolongation at each level (coarsest excluded)
        self.norm_layer_list = nn.ModuleList([
            nn.LayerNorm(
                [num_channel_u, 64 // 2**j, 64 // 2**j],
                elementwise_affine=elementwise_affine,
            )
            for j in range(len(num_iteration) - 1)
        ])

        # Transposed convolutions for prolongation (coarse -> fine)
        self.RTlayers = nn.ModuleList([
            nn.ConvTranspose2d(num_channel_u, num_channel_u,
                               kernel_size=4, stride=2, padding=1, bias=False)
            for _ in range(len(num_iteration) - 1)
        ])

        # Build per-level pre-smooth and post-smooth sequential modules
        layers = []
        for l, num_iteration_l in enumerate(num_iteration):
            post_smooth_layers = []

            # Pre-smoothing iterations
            for i in range(num_iteration_l[0]):
                S = nn.Conv2d(num_channel_f, num_channel_u,
                              kernel_size=3, stride=1, padding=1,
                              bias=bias, padding_mode=padding_mode)
                if l == 0 and i == 0:
                    layers.append(MgIte_init(S))
                else:
                    A = nn.Conv2d(num_channel_u, num_channel_f,
                                  kernel_size=3, stride=1, padding=1,
                                  bias=bias, padding_mode=padding_mode)
                    layers.append(MgIte(A, S))

            # Post-smoothing iterations (applied after prolongation)
            if num_iteration_l[1] != 0:
                for _ in range(num_iteration_l[1]):
                    S = nn.Conv2d(num_channel_f, num_channel_u,
                                  kernel_size=3, stride=1, padding=1,
                                  bias=bias, padding_mode=padding_mode)
                    A = nn.Conv2d(num_channel_u, num_channel_f,
                                  kernel_size=3, stride=1, padding=1,
                                  bias=bias, padding_mode=padding_mode)
                    post_smooth_layers.append(MgIte(A, S))
            else:
                post_smooth_layers.append(nn.Identity())

            setattr(self, 'layer' + str(l), nn.Sequential(*layers))
            setattr(self, 'post_smooth_layer' + str(l),
                    nn.Sequential(*post_smooth_layers))

            # Restriction operators to transition to coarser grid
            if l < len(num_iteration) - 1:
                A = nn.Conv2d(num_channel_u, num_channel_f,
                              kernel_size=3, stride=1, padding=1,
                              bias=bias, padding_mode=padding_mode)
                Pi = nn.Conv2d(num_channel_u, num_channel_u,
                               kernel_size=3, stride=2, padding=1,
                               bias=False, padding_mode=padding_mode)
                R = nn.Conv2d(num_channel_f, num_channel_f,
                              kernel_size=3, stride=2, padding=1,
                              bias=False, padding_mode=padding_mode)
                layers = [Restrict(Pi, R, A) if use_res else Restrict(Pi=Pi, R=R)]

    def forward(self, f):
        """V-cycle forward pass: downward (restriction) then upward (prolongation)."""
        out_list = [0] * len(self.num_iteration)
        out = f

        # Downward pass: apply pre-smooth + restrict at each level
        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer' + str(l))(out)
            out_list[l] = out

        # Upward pass: prolongate coarse correction, apply layer norm, post-smooth
        for j in range(len(self.num_iteration) - 2, -1, -1):
            u, f = out_list[j][0], out_list[j][1]
            u_post = self.norm_layer_list[j](
                u + self.RTlayers[j](out_list[j + 1][0])
            )
            out_list[j] = getattr(self, 'post_smooth_layer' + str(j))((u_post, f))

        return out_list[0][0]


# ---------------------------------------------------------------------------
# MgNO for Navier-Stokes (time-stepping, periodic boundaries)
# ---------------------------------------------------------------------------

class MgNO_NS(nn.Module):
    """MgNO model for Navier-Stokes time-stepping.

    Uses circular padding to respect periodic boundary conditions on the 2-D
    spatial grid.  A residual skip connection is added from input to output.

    Args:
        num_layer: Number of stacked MgConv blocks.
        num_channel_u: Feature channels for the solution field.
        num_channel_f: Feature channels for the right-hand side (input).
        num_classes: Number of output classes (kept for API consistency).
        num_iteration: Per-level ``[pre, post]`` iteration counts for MgConv.
        in_chans: Number of input time-channels.
        normalizer: Optional output normalizer (not used in forward; kept for
            API compatibility).
        mlp_hidden_dim: If > 0, the final 1x1 projection uses a two-layer MLP
            with this hidden dimension.
        output_dim: Number of output channels.
        activation: Nonlinearity – one of ``'relu'``, ``'gelu'``, ``'tanh'``,
            ``'silu'``.
        padding_mode: Convolution padding mode (default: ``'circular'``).
        bias: Whether smoothing convolutions have a bias term.
        use_res: Residual-based restriction (FAS-style).
        elementwise_affine: Learnable affine in LayerNorm.
    """

    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes,
                 num_iteration, in_chans=3, normalizer=None, mlp_hidden_dim=0,
                 output_dim=1, activation='gelu', padding_mode='circular',
                 bias=False, use_res=False, elementwise_affine=True):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration

        self.conv_list = nn.ModuleList([
            MgConv(num_iteration, num_channel_u, num_channel_f,
                   padding_mode='circular', bias=bias, use_res=use_res,
                   elementwise_affine=elementwise_affine)
        ])
        for _ in range(num_layer - 1):
            self.conv_list.append(
                MgConv(num_iteration, num_channel_u, num_channel_u,
                       padding_mode='circular', bias=bias, use_res=use_res,
                       elementwise_affine=elementwise_affine)
            )

        if mlp_hidden_dim:
            self.last_layer = nn.Sequential(
                nn.Conv2d(num_channel_u, mlp_hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(mlp_hidden_dim, output_dim, kernel_size=1, bias=False),
            )
        else:
            self.last_layer = nn.Conv2d(
                num_channel_u, 1, kernel_size=1, stride=1, padding=0, bias=False
            )

        activation_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'silu': nn.SiLU(),
        }
        if activation not in activation_map:
            raise NameError(f"Invalid activation '{activation}'. "
                            f"Choose from {list(activation_map)}")
        self.act = activation_map[activation]

    def forward(self, u):
        u_0 = u
        for i in range(self.num_layer):
            u = self.act(self.conv_list[i](u))
        u = self.last_layer(u)
        return u + u_0  # residual skip connection


# ---------------------------------------------------------------------------
# MgConv_DC: DC (Darcy / general) variant – no LayerNorm on upsampling path
# ---------------------------------------------------------------------------

class MgConv_DC(nn.Module):
    """V-cycle MgConv block for Darcy-flow and pipe-flow benchmarks.

    Similar to :class:`MgConv` but omits layer normalization after
    prolongation, which works better for non-periodic, Dirichlet-type problems.

    Args:
        num_iteration: List of ``[num_pre, num_post]`` smoothing counts per level.
        num_channel_u: Channels for the solution field.
        num_channel_f: Channels for the forcing/rhs field.
        padding_mode: Convolution padding (default: ``'zeros'``).
        bias: Bias in smoothing convolutions.
        use_res: Use residual-based restriction.
    """

    def __init__(self, num_iteration, num_channel_u, num_channel_f,
                 padding_mode='zeros', bias=False, use_res=False):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        # Prolongation (transposed conv) for each level transition
        self.RTlayers = nn.ModuleList([
            nn.ConvTranspose2d(num_channel_u, num_channel_u,
                               kernel_size=4, stride=2, padding=1, bias=False)
            for _ in range(len(num_iteration) - 1)
        ])

        layers = []
        for l, num_iteration_l in enumerate(num_iteration):
            post_smooth_layers = []

            for i in range(num_iteration_l[0]):
                S = nn.Conv2d(num_channel_f, num_channel_u,
                              kernel_size=3, stride=1, padding=1,
                              bias=bias, padding_mode=padding_mode)
                if l == 0 and i == 0:
                    layers.append(MgIte_init(S))
                else:
                    A = nn.Conv2d(num_channel_u, num_channel_f,
                                  kernel_size=3, stride=1, padding=1,
                                  bias=bias, padding_mode=padding_mode)
                    layers.append(MgIte(A, S))

            if num_iteration_l[1] != 0:
                for _ in range(num_iteration_l[1]):
                    S = nn.Conv2d(num_channel_f, num_channel_u,
                                  kernel_size=3, stride=1, padding=1,
                                  bias=bias, padding_mode=padding_mode)
                    A = nn.Conv2d(num_channel_u, num_channel_f,
                                  kernel_size=3, stride=1, padding=1,
                                  bias=bias, padding_mode=padding_mode)
                    post_smooth_layers.append(MgIte(A, S))
            else:
                post_smooth_layers.append(nn.Identity())

            setattr(self, 'layer' + str(l), nn.Sequential(*layers))
            setattr(self, 'post_smooth_layer' + str(l),
                    nn.Sequential(*post_smooth_layers))

            if l < len(num_iteration) - 1:
                A = nn.Conv2d(num_channel_u, num_channel_f,
                              kernel_size=3, stride=1, padding=1,
                              bias=bias, padding_mode=padding_mode)
                Pi = nn.Conv2d(num_channel_u, num_channel_u,
                               kernel_size=3, stride=2, padding=1,
                               bias=False, padding_mode=padding_mode)
                R = nn.Conv2d(num_channel_f, num_channel_f,
                              kernel_size=3, stride=2, padding=1,
                              bias=False, padding_mode=padding_mode)
                layers = [Restrict(Pi, R, A) if use_res else Restrict(Pi=Pi, R=R)]

    def forward(self, f):
        """V-cycle forward pass without LayerNorm on prolongation."""
        out_list = [0] * len(self.num_iteration)
        out = f

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer' + str(l))(out)
            out_list[l] = out

        for j in range(len(self.num_iteration) - 2, -1, -1):
            u, f = out_list[j][0], out_list[j][1]
            u_post = u + self.RTlayers[j](out_list[j + 1][0])
            out_list[j] = getattr(self, 'post_smooth_layer' + str(j))((u_post, f))

        return out_list[0][0]


def _build_activation(activation: str) -> nn.Module:
    """Return a PyTorch activation module by name."""
    activation_map = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'silu': nn.SiLU(),
    }
    if activation not in activation_map:
        raise NameError(f"Invalid activation '{activation}'. "
                        f"Choose from {list(activation_map)}")
    return activation_map[activation]


class MgNO_DC(nn.Module):
    """MgNO model for Darcy-flow and pipe-flow benchmarks.

    Each layer consists of a parallel MgConv_DC branch (multi-scale nonlocal)
    and a pointwise 1×1 convolution (local), whose outputs are summed before
    the nonlinearity.

    Args:
        num_layer: Number of stacked MgConv_DC + 1×1 blocks.
        num_channel_u: Feature channels.
        num_channel_f: Input channels.
        num_classes: Unused; kept for API consistency.
        num_iteration: Per-level smoothing counts for MgConv_DC.
        in_chans: Number of input channels (default: 1).
        normalizer: Optional output normalizer with a ``decode`` method.
        output_dim: Number of output channels.
        activation: Nonlinearity name.
        padding_mode: Convolution padding mode.
    """

    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes,
                 num_iteration, in_chans=1, normalizer=None, output_dim=1,
                 activation='gelu', padding_mode='zeros'):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration

        self.conv_list = nn.ModuleList([
            MgConv_DC(num_iteration, num_channel_u, num_channel_f,
                      padding_mode=padding_mode)
        ])
        self.linear_list = nn.ModuleList([
            nn.Conv2d(num_channel_f, num_channel_u,
                      kernel_size=1, stride=1, padding=0, bias=True)
        ])
        for _ in range(num_layer - 1):
            self.conv_list.append(
                MgConv_DC(num_iteration, num_channel_u, num_channel_u,
                          padding_mode=padding_mode)
            )
            self.linear_list.append(
                nn.Conv2d(num_channel_u, num_channel_u,
                          kernel_size=1, stride=1, padding=0, bias=True)
            )

        self.linear = nn.Conv2d(num_channel_u, 1, kernel_size=1, bias=False)
        self.normalizer = normalizer
        self.act = _build_activation(activation)

    def forward(self, u):
        for i in range(self.num_layer):
            u = self.act(self.conv_list[i](u) + self.linear_list[i](u))
        u = self.linear(u)
        return self.normalizer.decode(u) if self.normalizer else u


class MgNO_DC_smooth(nn.Module):
    """MgNO-DC variant tuned for smooth Darcy-flow data.

    Identical structure to :class:`MgNO_DC` but uses
    :class:`MgConv_DC_smooth` internally, which has level-specific
    prolongation kernels suited for smooth solutions.

    Args:
        See :class:`MgNO_DC` for full argument descriptions.
    """

    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes,
                 num_iteration, in_chans=1, normalizer=None, output_dim=1,
                 activation='gelu', padding_mode='zeros'):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration

        self.conv_list = nn.ModuleList([
            MgConv_DC_smooth(num_iteration, num_channel_u, num_channel_f,
                             padding_mode=padding_mode)
        ])
        self.linear_list = nn.ModuleList([
            nn.Conv2d(num_channel_f, num_channel_u,
                      kernel_size=1, stride=1, padding=0, bias=True)
        ])
        for _ in range(num_layer - 1):
            self.conv_list.append(
                MgConv_DC_smooth(num_iteration, num_channel_u, num_channel_u,
                                 padding_mode=padding_mode)
            )
            self.linear_list.append(
                nn.Conv2d(num_channel_u, num_channel_u,
                          kernel_size=1, stride=1, padding=0, bias=True)
            )

        self.linear = nn.Conv2d(num_channel_u, 1, kernel_size=1, bias=False)
        self.normalizer = normalizer
        self.act = _build_activation(activation)


    def forward(self, u):
        for i in range(self.num_layer):
            u = self.act(self.conv_list[i](u) + self.linear_list[i](u))
        u = self.linear(u)
        return self.normalizer.decode(u) if self.normalizer else u


# ---------------------------------------------------------------------------
# Helmholtz-specific MgConv variants (non-square spatial grids, reflect padding)
# ---------------------------------------------------------------------------

class MgConv_helm(nn.Module):
    """MgConv variant for the Helmholtz benchmark (101×101 spatial grid).

    Channel counts grow with depth (``num_channel * (l+1)`` at level *l*),
    which increases expressiveness for the high-frequency Helmholtz solutions.
    Uses ``reflect`` padding throughout.

    Args:
        num_iteration: List of pre-smoothing iteration counts per level
            (integer per level, not ``[pre, post]`` pairs).
        num_channel_u: Base channel count for the solution field.
        num_channel_f: Base channel count for the rhs field.
        init: If True, apply Xavier initialisation to restriction operators.
    """

    def __init__(self, num_iteration, num_channel_u, num_channel_f, init=False):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.init = init

        # Prolongation layers with growing/shrinking channel counts
        self.RTlayers = nn.ModuleList([
            nn.ConvTranspose2d(num_channel_u * 2, num_channel_u,
                               kernel_size=3, stride=2, padding=0, bias=False)
        ])
        for j in range(len(num_iteration) - 3):
            self.RTlayers.append(
                nn.ConvTranspose2d(num_channel_u * (j + 3), num_channel_u * (j + 2),
                                   kernel_size=4, stride=2, padding=0, bias=False)
            )
        self.RTlayers.append(
            nn.ConvTranspose2d(num_channel_u * 5, num_channel_u * 4,
                               kernel_size=3, stride=2, padding=0, bias=False)
        )

        layers = []
        for l, num_iteration_l in enumerate(num_iteration):
            for _ in range(num_iteration_l):
                A = nn.Conv2d(num_channel_u * (l + 1), num_channel_f * (l + 1),
                              kernel_size=3, stride=1, padding=1,
                              bias=True, padding_mode='reflect')
                S = nn.Conv2d(num_channel_f * (l + 1), num_channel_u * (l + 1),
                              kernel_size=3, stride=1, padding=1,
                              bias=True, padding_mode='reflect')
                layers.append(MgIte(A, S))

            setattr(self, 'layer' + str(l), nn.Sequential(*layers))

            if l < len(num_iteration) - 1:
                Pi = nn.Conv2d(num_channel_u * (l + 1), num_channel_u * (l + 2),
                               kernel_size=3, stride=2, padding=0,
                               bias=False, padding_mode='zeros')
                R = nn.Conv2d(num_channel_f * (l + 1), num_channel_f * (l + 2),
                              kernel_size=3, stride=2, padding=0,
                              bias=False, padding_mode='zeros')
                if self.init:
                    nn.init.xavier_uniform_(Pi.weight, gain=1 / num_channel_u**2)
                    nn.init.xavier_uniform_(R.weight, gain=1 / num_channel_u**2)
                layers = [Restrict(Pi, R)]

    def forward(self, f):
        u_list = []
        out = f

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer' + str(l))(out)
            u, f = out
            u_list.append(u)

        for j in range(len(self.num_iteration) - 2, -1, -1):
            u_list[j] = u_list[j] + self.RTlayers[j](u_list[j + 1])

        return u_list[0]


class MgConv_helm2(nn.Module):
    """MgConv variant for Helmholtz with growing channels and LayerNorm.

    Similar to :class:`MgConv_helm` but adds layer normalisation on the
    upsampling path (as in :class:`MgConv`).

    Args:
        num_iteration: List of ``[num_pre, num_post]`` counts per level.
        num_channel_u: Base channel count.
        num_channel_f: Base rhs channel count.
        padding_mode: Padding mode (default ``'zeros'``).
        bias: Bias in smoothing convolutions.
        elementwise_affine: Learnable affine in LayerNorm.
        init: Xavier init for restriction ops.
    """

    def __init__(self, num_iteration, num_channel_u, num_channel_f,
                 padding_mode='zeros', bias=False, elementwise_affine=True,
                 init=False):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        self.norm_layer_list = nn.ModuleList([
            nn.LayerNorm([num_channel_u, 101, 101]),
            nn.LayerNorm([num_channel_u * 2, 50, 50], elementwise_affine=False),
            nn.LayerNorm([num_channel_u * 3, 24, 24], elementwise_affine=False),
            nn.LayerNorm([num_channel_u * 4, 11, 11], elementwise_affine=False),
        ])

        self.RTlayers = nn.ModuleList([
            nn.ConvTranspose2d(num_channel_u * 2, num_channel_u,
                               kernel_size=3, stride=2, padding=0, bias=False)
        ])
        for j in range(len(num_iteration) - 3):
            self.RTlayers.append(
                nn.ConvTranspose2d(num_channel_u * (j + 3), num_channel_u * (j + 2),
                                   kernel_size=4, stride=2, padding=0, bias=False)
            )
        self.RTlayers.append(
            nn.ConvTranspose2d(num_channel_u * 5, num_channel_u * 4,
                               kernel_size=3, stride=2, padding=0, bias=False)
        )

        layers = []
        for l, num_iteration_l in enumerate(num_iteration):
            post_smooth_layers = []

            for i in range(num_iteration_l[0]):
                S = nn.Conv2d(num_channel_f * (l + 1), num_channel_u * (l + 1),
                              kernel_size=3, stride=1, padding=1,
                              bias=bias, padding_mode='reflect')
                if l == 0 and i == 0:
                    layers.append(MgIte_init(S))
                else:
                    A = nn.Conv2d(num_channel_u * (l + 1), num_channel_f * (l + 1),
                                  kernel_size=3, stride=1, padding=1,
                                  bias=bias, padding_mode='reflect')
                    layers.append(MgIte(A, S))

            if num_iteration_l[1] != 0:
                for _ in range(num_iteration_l[1]):
                    S = nn.Conv2d(num_channel_f * (l + 1), num_channel_u * (l + 1),
                                  kernel_size=3, stride=1, padding=1,
                                  bias=bias, padding_mode='reflect')
                    A = nn.Conv2d(num_channel_u * (l + 1), num_channel_f * (l + 1),
                                  kernel_size=3, stride=1, padding=1,
                                  bias=bias, padding_mode='reflect')
                    post_smooth_layers.append(MgIte(A, S))
            else:
                post_smooth_layers.append(nn.Identity())

            setattr(self, 'layer' + str(l), nn.Sequential(*layers))
            setattr(self, 'post_smooth_layer' + str(l),
                    nn.Sequential(*post_smooth_layers))

            if l < len(num_iteration) - 1:
                Pi = nn.Conv2d(num_channel_u * (l + 1), num_channel_u * (l + 2),
                               kernel_size=3, stride=2, padding=0,
                               bias=False, padding_mode='zeros')
                R = nn.Conv2d(num_channel_f * (l + 1), num_channel_f * (l + 2),
                              kernel_size=3, stride=2, padding=0,
                              bias=False, padding_mode='zeros')
                layers = [Restrict(Pi=Pi, R=R)]

    def forward(self, f):
        out_list = [0] * len(self.num_iteration)
        out = f

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer' + str(l))(out)
            out_list[l] = out

        for j in range(len(self.num_iteration) - 2, -1, -1):
            u, f = out_list[j][0], out_list[j][1]
            u_post = self.norm_layer_list[j](
                u + self.RTlayers[j](out_list[j + 1][0])
            )
            out_list[j] = getattr(self, 'post_smooth_layer' + str(j))((u_post, f))

        return out_list[0][0]


class MgNO_helm(nn.Module):
    """MgNO model for the Helmholtz equation (growing-channel variant).

    Uses :class:`MgConv_helm` with channel growth across levels and an
    additional MgConv_helm block as the final projection layer.

    Args:
        num_layer: Number of intermediate MgConv_helm blocks.
        num_channel_u: Base channel count.
        num_channel_f: Input channel count.
        num_classes: Unused; kept for API consistency.
        num_iteration: Per-level smoothing counts (integers, not pairs).
        in_chans: Input spatial channels.
        normalizer: Optional output normalizer.
        output_dim: Number of output channels.
        activation: Nonlinearity name.
        init: Xavier init for restriction operators.
    """

    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes,
                 num_iteration, in_chans=3, normalizer=None, output_dim=1,
                 activation='gelu', init=False):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration

        self.norm_layer_list = nn.ModuleList([
            nn.LayerNorm([num_channel_u, 101, 101])
            for _ in range(num_layer)
        ])

        self.conv_list = nn.ModuleList([
            MgConv_helm(num_iteration, num_channel_u, num_channel_f, init=init)
        ])
        for _ in range(num_layer - 1):
            self.conv_list.append(
                MgConv_helm(num_iteration, num_channel_u, num_channel_u, init=init)
            )

        self.last_layer = MgConv_helm(num_iteration, 1, num_channel_u, init=init)
        self.normalizer = normalizer
        self.act = _build_activation(activation)

    def forward(self, u):
        for i in range(self.num_layer):
            u = self.act(self.norm_layer_list[i](self.conv_list[i](u)))
        out = self.last_layer(u)
        return self.normalizer.decode(torch.squeeze(out)) if self.normalizer else out


class MgNO_helm2(nn.Module):
    """MgNO model for Helmholtz with uniform channels (no channel growth).

    Uses :class:`MgConv_helm3` instead of :class:`MgConv_helm`, keeping the
    channel count constant across levels.  This is the default model used in
    the paper's Helmholtz experiment.

    Args:
        num_layer: Number of intermediate MgConv_helm3 blocks.
        num_channel_u: Channel count (uniform across levels).
        num_channel_f: Input channel count.
        num_classes: Unused; kept for API consistency.
        num_iteration: Per-level ``[pre, post]`` smoothing counts.
        in_chans: Input spatial channels.
        normalizer: Optional output normalizer.
        mlp_hidden_dim: Hidden dimension for optional MLP projection (unused).
        output_dim: Number of output channels.
        activation: Nonlinearity name.
        init: Xavier init for restriction operators.
        if_mlp: If True, use MgConv_helm3 for the output projection; otherwise
            use a simple 1×1 convolution.
    """

    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes,
                 num_iteration, in_chans=3, normalizer=None, mlp_hidden_dim=128,
                 output_dim=1, activation='gelu', init=False, if_mlp=False):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration

        self.conv_list = nn.ModuleList([
            MgConv_helm3(num_iteration, num_channel_u, num_channel_f, init=init)
        ])
        for _ in range(num_layer - 1):
            self.conv_list.append(
                MgConv_helm3(num_iteration, num_channel_u, num_channel_u, init=init)
            )

        self.mlp = (
            MgConv_helm3(num_iteration, 1, num_channel_u, init=init)
            if if_mlp
            else nn.Conv2d(num_channel_u, 1, kernel_size=1)
        )
        self.normalizer = normalizer
        self.act = _build_activation(activation)

    def forward(self, u):
        for i in range(self.num_layer):
            u = self.act(self.conv_list[i](u))
        out = self.mlp(u)
        return self.normalizer.decode(torch.squeeze(out)) if self.normalizer else out


class MgConv_helm3(nn.Module):
    """MgConv for Helmholtz with uniform channels and LayerNorm on upsampling.

    Keeps channels constant (``num_channel_u``) across all levels (unlike
    :class:`MgConv_helm`).  Uses ``reflect`` padding for smoothing and
    ``zeros`` padding for restriction/prolongation.

    Args:
        num_iteration: List of ``[num_pre, num_post]`` counts per level.
        num_channel_u: Uniform channel count.
        num_channel_f: Input channel count.
        padding_mode: Smoothing convolution padding (default ``'reflect'``).
        bias: Bias in smoothing convolutions.
        elementwise_affine: Learnable affine in LayerNorm.
        init: Xavier init for restriction operators.
    """

    def __init__(self, num_iteration, num_channel_u, num_channel_f,
                 padding_mode='reflect', bias=False, elementwise_affine=True,
                 init=False):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        self.norm_layer_list = nn.ModuleList([
            nn.LayerNorm([num_channel_u, 101, 101]),
            nn.LayerNorm([num_channel_u, 50, 50], elementwise_affine=False),
            nn.LayerNorm([num_channel_u, 24, 24], elementwise_affine=False),
            nn.LayerNorm([num_channel_u, 11, 11], elementwise_affine=False),
        ])

        self.RTlayers = nn.ModuleList([
            nn.ConvTranspose2d(num_channel_u, num_channel_u,
                               kernel_size=3, stride=2, padding=0, bias=False)
        ])
        for _ in range(len(num_iteration) - 3):
            self.RTlayers.append(
                nn.ConvTranspose2d(num_channel_u, num_channel_u,
                                   kernel_size=4, stride=2, padding=0, bias=False)
            )
        self.RTlayers.append(
            nn.ConvTranspose2d(num_channel_u, num_channel_u,
                               kernel_size=3, stride=2, padding=0, bias=False)
        )

        layers = []
        for l, num_iteration_l in enumerate(num_iteration):
            post_smooth_layers = []

            for i in range(num_iteration_l[0]):
                S = nn.Conv2d(num_channel_f, num_channel_u,
                              kernel_size=3, stride=1, padding=1,
                              bias=bias, padding_mode='reflect')
                if l == 0 and i == 0:
                    layers.append(MgIte_init(S))
                else:
                    A = nn.Conv2d(num_channel_u, num_channel_f,
                                  kernel_size=3, stride=1, padding=1,
                                  bias=bias, padding_mode='reflect')
                    layers.append(MgIte(A, S))

            if num_iteration_l[1] != 0:
                for _ in range(num_iteration_l[1]):
                    S = nn.Conv2d(num_channel_f, num_channel_u,
                                  kernel_size=3, stride=1, padding=1,
                                  bias=bias, padding_mode='reflect')
                    A = nn.Conv2d(num_channel_u, num_channel_f,
                                  kernel_size=3, stride=1, padding=1,
                                  bias=bias, padding_mode='reflect')
                    post_smooth_layers.append(MgIte(A, S))
            else:
                post_smooth_layers.append(nn.Identity())

            setattr(self, 'layer' + str(l), nn.Sequential(*layers))
            setattr(self, 'post_smooth_layer' + str(l),
                    nn.Sequential(*post_smooth_layers))

            if l < len(num_iteration) - 1:
                Pi = nn.Conv2d(num_channel_u, num_channel_u,
                               kernel_size=3, stride=2, padding=0,
                               bias=False, padding_mode='zeros')
                R = nn.Conv2d(num_channel_f, num_channel_f,
                              kernel_size=3, stride=2, padding=0,
                              bias=False, padding_mode='zeros')
                layers = [Restrict(Pi=Pi, R=R)]

    def forward(self, f):
        out_list = [0] * len(self.num_iteration)
        out = f

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer' + str(l))(out)
            out_list[l] = out

        for j in range(len(self.num_iteration) - 2, -1, -1):
            u, f = out_list[j][0], out_list[j][1]
            u_post = self.norm_layer_list[j](
                u + self.RTlayers[j](out_list[j + 1][0])
            )
            out_list[j] = getattr(self, 'post_smooth_layer' + str(j))((u_post, f))

        return out_list[0][0]


# ---------------------------------------------------------------------------
# MgConv_DC_smooth: smooth Darcy variant with non-uniform prolongation kernels
# ---------------------------------------------------------------------------

class MgConv_DC_smooth(nn.Module):
    """V-cycle MgConv for smooth Darcy data with alternating prolongation kernels.

    Uses kernel sizes 3 and 4 alternately for the transposed convolutions to
    handle the specific grid sizes arising in the smooth Darcy benchmark.

    Args:
        num_iteration: List of ``[num_pre, num_post]`` smoothing counts per level.
        num_channel_u: Channel count for the solution field.
        num_channel_f: Channel count for the rhs field.
        padding_mode: Convolution padding (default: ``'zeros'``).
        bias: Bias in smoothing convolutions.
        use_res: Residual-based restriction.
    """

    def __init__(self, num_iteration, num_channel_u, num_channel_f,
                 padding_mode='zeros', bias=False, use_res=False):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        # Alternating kernel sizes (3, 4, 3, 3, 4) to match grid halving behaviour
        self.RTlayers = nn.ModuleList([
            nn.ConvTranspose2d(num_channel_u, num_channel_u, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ConvTranspose2d(num_channel_u, num_channel_u, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ConvTranspose2d(num_channel_u, num_channel_u, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ConvTranspose2d(num_channel_u, num_channel_u, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ConvTranspose2d(num_channel_u, num_channel_u, kernel_size=4, stride=2, padding=1, bias=False),
        ])

        layers = []
        for l, num_iteration_l in enumerate(num_iteration):
            post_smooth_layers = []

            for i in range(num_iteration_l[0]):
                S = nn.Conv2d(num_channel_f, num_channel_u,
                              kernel_size=3, stride=1, padding=1,
                              bias=bias, padding_mode=padding_mode)
                if l == 0 and i == 0:
                    layers.append(MgIte_init(S))
                else:
                    A = nn.Conv2d(num_channel_u, num_channel_f,
                                  kernel_size=3, stride=1, padding=1,
                                  bias=bias, padding_mode=padding_mode)
                    layers.append(MgIte(A, S))

            if num_iteration_l[1] != 0:
                for _ in range(num_iteration_l[1]):
                    S = nn.Conv2d(num_channel_f, num_channel_u,
                                  kernel_size=3, stride=1, padding=1,
                                  bias=bias, padding_mode=padding_mode)
                    A = nn.Conv2d(num_channel_u, num_channel_f,
                                  kernel_size=3, stride=1, padding=1,
                                  bias=bias, padding_mode=padding_mode)
                    post_smooth_layers.append(MgIte(A, S))
            else:
                post_smooth_layers.append(nn.Identity())

            setattr(self, 'layer' + str(l), nn.Sequential(*layers))
            setattr(self, 'post_smooth_layer' + str(l),
                    nn.Sequential(*post_smooth_layers))

            if l < len(num_iteration) - 1:
                A = nn.Conv2d(num_channel_u, num_channel_f,
                              kernel_size=3, stride=1, padding=1,
                              bias=bias, padding_mode=padding_mode)
                Pi = nn.Conv2d(num_channel_u, num_channel_u,
                               kernel_size=3, stride=2, padding=1,
                               bias=False, padding_mode=padding_mode)
                R = nn.Conv2d(num_channel_f, num_channel_f,
                              kernel_size=3, stride=2, padding=1,
                              bias=False, padding_mode=padding_mode)
                layers = [Restrict(Pi, R, A) if use_res else Restrict(Pi=Pi, R=R)]

    def forward(self, f):
        out_list = [0] * len(self.num_iteration)
        out = f

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer' + str(l))(out)
            out_list[l] = out

        for j in range(len(self.num_iteration) - 2, -1, -1):
            u, f = out_list[j][0], out_list[j][1]
            u_post = u + self.RTlayers[j](out_list[j + 1][0])
            out_list[j] = getattr(self, 'post_smooth_layer' + str(j))((u_post, f))

        return out_list[0][0]









if __name__ == "__main__":
    """Quick sanity check: instantiate MgNO_DC and run a forward + backward pass."""
    torch.autograd.set_detect_anomaly(True)
    model = MgNO_DC(
        num_layer=5, num_channel_u=32, num_channel_f=1,
        num_classes=1, num_iteration=[[1, 0], [1, 0], [1, 0], [1, 1], [2, 0]],
    ).cuda()

    print(model)
    print('Parameter count:', count_params(model))
    inp = torch.randn(10, 1, 129, 129).cuda()
    out = model(inp)
    print('Output shape:', out.shape)
    summary(model, input_size=(10, 1, 129, 129))
    out.sum().backward()
    print('Forward + backward pass successful!')

