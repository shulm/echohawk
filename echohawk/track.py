"""DOA Tracking for moving sources.

Implements a simple 1-D alpha-beta (or Kalman) tracker over azimuth to
smooth noisy MUSIC/SRP-PHAT bearings over time.
"""
from __future__ import annotations

import numpy as np

class AzimuthTracker:
    """A simple 1-D alpha-beta tracker for azimuth.
    
    Parameters
    ----------
    alpha : float
        Position update gain (0 < alpha < 1).
    beta : float
        Velocity update gain (0 < beta < 1).
    dt : float
        Time step between updates.
    """
    def __init__(self, alpha: float = 0.2, beta: float = 0.05, dt: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.azimuth = None
        self.velocity = 0.0
        
    def _circular_diff(self, a, b):
        """Difference a - b wrapped to [-180, 180)."""
        diff = (a - b) % 360
        if diff >= 180:
            diff -= 360
        return diff

    def update(self, measurement: float) -> float:
        """Update tracker with a new azimuth measurement (degrees).
        
        Returns
        -------
        estimated_azimuth : float
        """
        if self.azimuth is None:
            self.azimuth = measurement
            self.velocity = 0.0
            return self.azimuth
            
        # Predict
        pred_az = self.azimuth + self.velocity * self.dt
        
        # Calculate residual (wrapping angle)
        residual = self._circular_diff(measurement, pred_az)
        
        # Update
        self.azimuth = (pred_az + self.alpha * residual) % 360
        self.velocity = self.velocity + (self.beta / self.dt) * residual
        
        return self.azimuth
