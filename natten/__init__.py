"""
Neighborhood Attention

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from .nattencuda1d import NeighborhoodAttention1D
from .nattencuda2d import NeighborhoodAttention2D
from .nattentorch2d import LegacyNeighborhoodAttention2D

from .nattencuda import NeighborhoodAttention
from .nattentorch import LegacyNeighborhoodAttention
