import numpy as np
import tomllib

G0 = 9.80665                # [m/s^2], gravitational constant
HP2KW = 0.745699872         # [kW/HP], conversion factor HP to kW
FUEL_DENSITY = 720.0        # [kg/m^3], density of 100LL avgas
RPM_TO_OMEGA = 2*np.pi/60   # Conversion from rpm to angular velocity omega [rad/s]
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
    def fuel_volume_litres(self) -> float:
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
        self.MTOM = mb["mtom"]                          # [kg], max take-off mass
        self.EOM = mb["empty_mass"]                     # [kg], empty operating mass
        self.Iyy = mb["iyy"]                            # [kg m^2], pitch axis moment of inertia of the aircraft
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
        eng = raw["propulsion"]["engine"]
        self.engine_max_power = eng["max_power"] * HP2KW
        self.engine_bsfc = eng["bsfc"]
        self.engine_rpm_range = (eng["idle_rpm"], eng["max_rpm"])
        self.engine_throttle_range = (eng["min_throttle"], eng["max_throttle"])
        self.engine_mech_drag = eng["k_mech"]
        prop = raw["propulsion"]["propeller"]
        self.prop_ixx = prop["ixx"]
        self.prop_diameter = prop["diameter"]
        self.prop_cT_curve = np.asarray([prop["ct_curve"]["J"],
                                         prop["ct_curve"]["Ct"]], dtype = float)
        self.prop_cP_curve = np.asarray([prop["cp_curve"]["J"],
                                         prop["cp_curve"]["Cp"]], dtype = float)
        self.fuel_tanks = [FuelTank(ft["capacity"],
                                    ft["fill_level"],
                                    Location(ft["location"]["x"],
                                             ft["location"]["y"],
                                             ft["location"]["z"])) for ft in raw["propulsion"]["fuel_tank"]]
        
        ## Aerodynamics
        a = raw["aerodynamics"]
        self.flap_settings = np.asarray(a["dCL_flaps"]["flap_deg"], dtype = float)
        # Lift
        self.CL_alpha_curve = np.asarray([a["CL_curve"]["alpha"],
                                          a["CL_curve"]["CL"]], dtype = float)
        self.dCL_flaps = np.asarray([self.flap_settings,
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
                                                    a["ground_effect_drag"]["drag_multiplier"]], dtype = float)
        # Pitching moment
        self.CM0 = a["CM0"]
        self.CMa = a["CMa"]
        self.CMq = a["CMq"]
        self.CMadot = a["CMadot"]
        self.CMde = a["CMde"]
        self.dCM_flaps = np.asarray([self.flap_settings,
                                     a["dCM_flaps"]["dCM_flap"]], dtype = float)
        
        # post config loading attributes
        self.total_mass = self._evaluate_total_mass()
        if self.total_mass > self.MTOM:
            print(f"Warning: maximum take-off mass exceeded (mass: {self.total_mass:.1f} [kg], max allowed: {self.MTOM:.1f} [kg]). Reducing pax/luggage weight")
            order = ["luggage", "passenger 2", "passenger 1", "co-pilot"]
            i = 0
            while self.total_mass > self.MTOM:
                self.point_masses = [pm for pm in self.point_masses if pm.name != order[i]]
                self.total_mass = self._evaluate_total_mass()
                i += 1

    def _evaluate_total_mass(self) -> float:
        return self.EOM + sum([pm.mass for pm in self.point_masses]) + sum([ft.mass for ft in self.fuel_tanks])
    
    def update_total_mass(self, d_mass: float):
        self.total_mass -= d_mass
    
    def calculate_current_cg(self) -> Location:
        cg_mass_product = self.empty_cg * self.EOM
        for pm in self.point_masses:
            cg_mass_product += pm.location * pm.mass
        for ft in self.fuel_tanks:
            cg_mass_product += ft.location * ft.mass
        cg = cg_mass_product / self.total_mass
        self.cg = Location(cg[0], cg[1], cg[2])
        return self.cg

    def get_current_engine_power(self, throttle_setting: float) -> float:
        if not self.engine_throttle_range[0] <= throttle_setting <= self.engine_throttle_range[1]:
            raise ValueError("Throttle setting is outside of the permitted range")
        P_eng = throttle_setting * self.engine_max_power
        return P_eng
    
    def calculate_thrust_coefficient(self, J: float, clamp: bool = True) -> float:
        J_grid, cT = self.prop_cT_curve
        if clamp:
            J = np.clip(J, J_grid[0], J_grid[-1])
        return np.interp(J, J_grid, cT)
    
    def calculate_power_coefficient(self, J: float, clamp: bool = True) -> float:
        J_grid, Cp = self.prop_cP_curve
        if clamp:
            J = np.clip(J, J_grid[0], J_grid[-1])
        return np.interp(J, J_grid, Cp)
    
    def calculate_CL_due_to_alpha(self, alpha: float, clamp: bool = True) -> float:
        alpha_grid, CL = self.CL_alpha_curve
        if clamp:
            alpha = np.clip(alpha, alpha_grid[0], alpha_grid[-1])
        return np.interp(alpha, alpha_grid, CL)
    
    def get_CL_due_to_flaps(self, flap_setting_deg: float) -> float:
        flap_idx = np.nonzero(np.isclose(self.flap_settings, flap_setting_deg))[0]
        if flap_idx.size == 0:
            raise ValueError(f"Flap {flap_setting_deg}° not in {self.flap_settings.tolist()}")
        flap_idx = int(flap_idx[0])
        return float(self.dCL_flaps[1, flap_idx])
    
    def calculate_CD_due_to_alpha(self, alpha: float, flap_setting_deg: float, clamp: bool = True) -> float:
        flap_idx = np.nonzero(np.isclose(self.flap_settings, flap_setting_deg))[0]
        if flap_idx.size == 0:
            raise ValueError(f"Flap {flap_setting_deg}° not in {self.flap_settings.tolist()}")
        flap_idx = int(flap_idx[0])
        alpha_grid, CD = self.CD_alpha_curve[0], self.CD_alpha_curve[flap_idx + 1]
        if clamp:
            alpha = np.clip(alpha, alpha_grid[0], alpha_grid[-1])
        return np.interp(alpha, alpha_grid, CD)
    
    def calculate_ground_effect_lift_factor(self, h: float, num_tol: float = 1e-6) -> float:
        h_b = h / self.b
        if h_b < 0.0:
            raise ValueError(f"Negative altitude encountered: {h:.2f} [m]")
        elif h_b > self.ground_effect_lift_curve[0, -1]:
            return 1.0
        elif h_b < num_tol:
            return self.ground_effect_lift_curve[1, 0]
        else:
            return np.interp(h_b, self.ground_effect_lift_curve[0], self.ground_effect_lift_curve[1])

    def calculate_ground_effect_drag_factor(self, h: float, num_tol: float = 1e-6) -> float:
        h_b = h / self.b
        if h_b < 0.0:
            raise ValueError(f"Negative altitude encountered: {h:.2f} [m]")
        elif h_b > self.ground_effect_drag_curve[0, -1]:
            return 1.0
        elif h_b < num_tol:
            return self.ground_effect_drag_curve[1, 0]
        else:
            return np.interp(h_b, self.ground_effect_drag_curve[0], self.ground_effect_drag_curve[1])
        
    def get_CM_due_to_flaps(self, flap_setting_deg: float) -> float:
        flap_idx = np.nonzero(np.isclose(self.flap_settings, flap_setting_deg))[0]
        if flap_idx.size == 0:
            raise ValueError(f"Flap {flap_setting_deg}° not in {self.flap_settings.tolist()}")
        flap_idx = int(flap_idx[0])
        return float(self.dCM_flaps[1, flap_idx])