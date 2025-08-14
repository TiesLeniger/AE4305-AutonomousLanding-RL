import numpy as np

class IntStdAtmosphere:

    def __init__(self):
        self.p0 = 101325.0                      # [Pa], mean sea level pressure
        self.rho0 = 1.225                       # [kg/m^3], mean sea level density
        self.T0 = 288.15                        # [K], mean sea level temperature
        self.gamma = 1.4                        # [-], ratio of specific heats for air
        self.R = 287.05                         # [J/kg K], gas constant of air
        self.g0 = 9.80665                       # [m/s^2], gravitational acceleration
        self.lapse_rate = -0.0065               # [m^-1], lapse rate in the troposphere
        self.mu_air = 1.789e-5                  # [Pa s], absolute viscosity of air
        self.altitude_range = [0.0, 11000.0]    # [m], lower and upper limit of the troposphere (only layer modelled)

    def calculate_temperature(self, h: float) -> float:
        if not self.altitude_range[0] <= h <= self.altitude_range[-1]:
            raise ValueError(f"Only the troposphere is modelled. {h:.1f} [m] is outside of the troposphere")
        return self.T0 + self.lapse_rate*h
    
    def calculate_pressure(self, h: float) -> float:
        if not self.altitude_range[0] <= h <= self.altitude_range[-1]:
            raise ValueError(f"Only the troposphere is modelled. {h:.1f} [m] is outside of the troposphere")
        return self.p0*(1+(self.lapse_rate*h/self.T0))**(-self.g0/(self.R*self.lapse_rate))
    
    def calculate_density(self, p: float, T: float) -> float:
        return p/(self.R * T)
    
    def calculate_atmospheric_properties(self, h: float) -> tuple[float]:
        if not self.altitude_range[0] <= h <= self.altitude_range[-1]:
            raise ValueError(f"Only the troposphere is modelled. {h:.1f} [m] is outside of the troposphere")
        T = self.calculate_temperature(h)
        p = self.calculate_pressure(h)
        rho = self.calculate_density(p, T)

        return T, p, rho