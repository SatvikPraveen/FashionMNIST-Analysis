"""
Shared pytest setup.

Disables oneDNN CPU convolutions on hosts where they cannot be created
(virtualised login nodes without AVX), so the suite is portable.
"""

from src.training.utils import ensure_cpu_conv_backend

ensure_cpu_conv_backend()
