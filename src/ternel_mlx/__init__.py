"""MLX/Metal execution path for the TQ1_G128 lossless ternary format.

The historical Python package remains ``bonsai_tq1``; this package holds the
Apple Silicon distribution path built on top of the same frozen format.
"""

from __future__ import annotations

LAYOUT_VERSION = 1
LAYOUT_NAME = "TQ1_G128_MLX_TILED"
