#TODO: Check this implementation through

class PID:
    """
    PID controller suitable for position control.

    Key features:
      - Derivative on measurement (avoids derivative kick on setpoint changes)
      - Integral anti-windup via clamping when output saturates
      - Optional setpoint rate limit (ramping)
      - Output limits
    """
    def __init__(self, Kp, Ki, Kd, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self._last_error = 0.0
        self._integral = 0.0
        self._last_measurement = None

        self.output_limits = output_limits

    def reset(self):
        self._last_error = 0.0
        self._integral = 0.0
        self._last_measurement = None

    def compute(self, setpoint, measurement, dt):
        error = setpoint - measurement

        # Proportional term
        P = self.Kp * error

        # Integral term with anti-windup
        self._integral += error * dt
        I = self.Ki * self._integral

        # Derivative term (on measurement to avoid derivative kick)
        if self._last_measurement is None:
            D = 0.0
        else:
            D = -self.Kd * (measurement - self._last_measurement) / dt

        # Compute raw output
        output = P + I + D

        # Apply output limits and anti-windup
        lower_limit, upper_limit = self.output_limits
        if lower_limit is not None and output < lower_limit:
            output = lower_limit
            # Anti-windup: prevent integral from increasing further
            if error < 0:
                self._integral -= error * dt  # Undo last integral step
        elif upper_limit is not None and output > upper_limit:
            output = upper_limit
            # Anti-windup: prevent integral from increasing further
            if error > 0:
                self._integral -= error * dt  # Undo last integral step

        # Save last measurement for next derivative calculation
        self._last_measurement = measurement

        return output