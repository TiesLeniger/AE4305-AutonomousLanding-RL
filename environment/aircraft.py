import numpy as np
import tomllib

G0 = 9.80665            # [m/s^2], gravitational constant
HP2KW = 0.745699872     # [KW/HP], conversion factor HP to KW
FUEL_DENSITY = 720.0    # [kg/m^3], density of 100LL avgas
# -----------------------------------------------------
class Location(np.ndarray):
    def __new__(cls, x: float, y: float, z: float):
        obj = np.asarray([x, y, z], dtype = float).view(cls)
        return obj
    
    def __array_finalize__(self, obj):
        pass

    @property
    def x(self) -> float:
        return float(self[0])
    
    @property
    def y(self) -> float:
        return float(self[1])

    @property
    def z(self) -> float:
        return float(self[2])
    
class PointMass:
    """Represents a discrete mass with a 3D location."""
    __slots__ = ("name", "mass", "location")

    def __init__(self, name: str, mass: float, location: Location):
        self.name = name
        self.mass = float(mass)
        self.location = location

    def __repr__(self):
        return f"PointMass(name='{self.name}', mass={self.mass} [kg], location={self.location.tolist()} [m])"

    @property
    def weight(self) -> float:
        """Returns weight in newtons (assuming g = 9.80665 m/s²)."""
        return self.mass * G0

    def moment_about(self, point: Location) -> np.ndarray:
        """Returns the moment vector r x F about a reference point."""
        r = self.location - point
        F = np.array([0.0, 0.0, -self.weight])  # weight acts in -Z
        return np.cross(r, F)

class FuelTank:
    """"Represents a fuel tank in the aircraft with a 3D location."""
    __slots__ = ("capacity", "fill_level", "location")

    def __init__(self, capacity: float, fill_level: float, location: Location):
        self.capacity = capacity
        if fill_level <= 0.0 or fill_level > 1.0:
            raise ValueError(f"Fuel tank fill level must be larger than 0.0 and smaller than or equal to 1.0, got {fill_level}")
        else:
            self.fill_level = fill_level
        self.location = location

    def __repr__(self):
        return f"FuelTank(capacity= {self.capacity} [kg], fill level = {self.fill_level}, location = {self.location.tolist()} [m])"
    
    @property
    def mass(self) -> float:
        return self.capacity * self.fill_level

    @property
    def weight(self) -> float:
        return self.mass * G0
    
    @property
    def fuel_volume(self) -> float:
        return (self.mass*1000)/FUEL_DENSITY
    
    def moment_about(self, point: Location) -> np.ndarray:
        r = self.location - point
        F = np.array([0.0, 0.0, -self.weight])
        return np.cross(r, F)
class Aircraft:

    def __init__(self, config_file: str):
        with open(config_file, "rb") as f:
            raw = tomllib.load(f)
        
        # Metrics
        m = raw["metrics"]
        self.S = m["wing_area"]         # [m^2], wing reference area
        self.b = m["wing_span"]         # [m], wing span
        self.c = m["c_bar"]             # [m], mean aerodynamic chord

        # Mass and balance
        mb = raw["mass_balance"]
        self.EOM = mb["empty_mass"]                     # [kg], empty operating mass
        self.Iyy = mb["Iyy"]                            # [kg m^2], pitch axis moment of inertia of the aircraft
        self.empty_cg = Location(mb["empty_cg"]["x"],
                                 mb["empty_cg"]["y"],
                                 mb["empty_cg"]["z"])   # [m], x, y and z location of the empty cg
        pm_list = mb["pointmasses"]
        self.point_masses = [PointMass(pm["name"], 
                                       pm["mass"], 
                                       Location(pm["location"]["x"],
                                                pm["location"]["y"],
                                                pm["location"]["z"])) for pm in pm_list]
        
        # Propulsion
        pr = raw["propulsion"]
        self.engine_max_power = pr["max_power"] * HP2KW
        self.engine_bsfc = pr["bsfc"]
        self.engine_rpm_range = (550.0, 2700.0)
        self.engine_throttle_range = (0.1, 1.0)
        self.engine_mech_drag = pr["propeller"]["k_mech"]
        self.prop_ixx = pr["propeller"]["ixx"]
        self.prop_diameter = pr["propeller"]["diameter"]
        self.prop_cT_curve = np.asarray([pr["propeller"]["ct_curve"]["J"],
                                         pr["propeller"]["ct_curve"]["Ct"]], dtype = float)
        self.prop_cP_curve = np.asarray([pr["propeller"]["cp_curve"]["J"],
                                         pr["propeller"]["cp_curve"]["Cp"]], dtype = float)
        self.fuel_tanks = [FuelTank(ft["capacity"],
                                    ft["fill_level"],
                                    Location(ft["location"]["x"],
                                             ft["location"]["y"],
                                             ft["location"]["z"])) for ft in pr["fuel_tank"]]
        
        ## Aerodynamics
        a = raw["aerodynamics"]
        # Lift
        self.CL_alpha_curve = np.asarray([a["CL_curve"]["alpha"],
                                          a["CL_curve"]["CL"]], dtype = float)
        self.dCL_flaps = np.asarray([a["dCL_flaps"]["flap_deg"],
                                     a["dCL_flaps"]["dCL_flap"]], dtype = float)
        self.CLde = a["CLde"]
        self.CLadot = a["CLadot"]
        self.CLq = a["CLq"]
        self.ground_effect_lift_curve = np.asarray([a["ground_effect_lift"]["h_b"],
                                                    a["ground_effect_lift"]["lift_multiplier"]], dtype = float)
        # Drag
        self.CD0 = a["CD0"]
        self.CDde = a["CDde"]
        self.CD_alpha_curve = np.asarray([a["CD_curve"]["alpha"],
                                          a["CD_curve"]["CD_flap0"],
                                          a["CD_curve"]["CD_flap10"],
                                          a["CD_curve"]["CD_flap20"],
                                          a["CD_curve"]["CD_flap30"]], dtype = float)
        self.ground_effect_drag_curve = np.asarray([a["ground_effect_drag"]["h_b"],
                                                    a["ground_effect_drag"]["lift_multiplier"]], dtype = float)
        # Pitching moment
        self.CM0 = a["CM0"]
        self.CMa = a["CMa"]
        self.CMq = a["CMq"]
        self.CMadot = a["CMadot"]
        CMde = a["CMde"]
        self.dCM_flaps = np.asarray([a["dCM_flaps"]["flap_deg"],
                                     a["dCM_flaps"]["dCM_flap"]], dtype = float)