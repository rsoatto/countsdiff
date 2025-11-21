"""
Collection of third‑party baseline models used for comparison.
Convenience imports live here for easier access, but the subpackages can
still be imported directly.
"""

from .MAGIC.magic_wrapper import MAGICWrapper  
from .scIDPMs.scidpm_wrapper import scIDPMWrapper  
# Add other common entry points as you need them, e.g.
# from .ForestDiff.forestdiff import ForestDiffModel
from .ReMDM.remdm import ReMDM  