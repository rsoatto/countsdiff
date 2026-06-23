"""
Collection of third‑party baseline models used for comparison.
Convenience imports live here for easier access, but the subpackages can
still be imported directly.
"""

try:
    from .MAGIC.magic_wrapper import MAGICWrapper  # type: ignore
except Exception:  # pragma: no cover
    MAGICWrapper = None

try:
    from .scIDPMs.scidpm_wrapper import scIDPMWrapper  # type: ignore
except Exception:  # pragma: no cover
    scIDPMWrapper = None

try:
    from .scGPT.scgpt_wrapper import SCGPTWrapper  # type: ignore
except Exception:  # pragma: no cover
    SCGPTWrapper = None

try:
    from .xTrimoGene.xtrimogene_wrapper import XTrimoGeneWrapper  # type: ignore
except Exception:  # pragma: no cover
    XTrimoGeneWrapper = None

try:
    from .ReMDM.remdm import ReMDM  # type: ignore
except Exception:  # pragma: no cover
    ReMDM = None

__all__ = [
    "MAGICWrapper",
    "scIDPMWrapper",
    "SCGPTWrapper",
    "XTrimoGeneWrapper",
    "ReMDM",
]
