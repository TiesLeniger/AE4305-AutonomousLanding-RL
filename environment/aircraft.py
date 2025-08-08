from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np

@dataclass
class AircraftParams:
    # Mass an
    mass: float = 660.0                 # [kg], aircraft empty mass
    S: float = 16.17                    # [m^2], aircraft reference wing area
    wing_span: float = 10.97            # [m], wing span
    cog_loc: tuple = (1.041, 0.927)     # [m], empty COG location (x, z)
    mac = 1.494                         # [m], mean aerodynamic chord

    # Aerodynamic forces
    CL0: float = 0.25                   # [-], zero aoa lift coefficient
    CL_alpha: float = 4.66              # [1/rad], lift slope in linear part of lift curve
    CL_de: float = 0.347                # [1/rad], elevator lift slope
    CD0: float = 0.032                  # [-], clean zero lift drag
    oswald_e: float = 0.82              # [-], oswald span efficiency factor

    # Aerodynamic moments
    CM0: float = 0.1                    # [-], zero lift moment coefficient
    CM_alpha: float = -1.8              # [1/rad], moment slope
    CM_de: float = -1.28                # [1/rad], elevator effectiveness
    CM_q: float = -12.4                 #

    def __post_init__(self):
        self.AR = (self.wing_span**2)/self.S
        self.drag_k = 1/(np.pi * self.AR * self.oswald_e)

