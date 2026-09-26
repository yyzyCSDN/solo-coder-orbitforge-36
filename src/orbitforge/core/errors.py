class OrbitForgeError(Exception):
    pass

class ValidationError(OrbitForgeError):
    pass

class ConvergenceError(OrbitForgeError):
    pass

class GeometryError(OrbitForgeError):
    pass

class NoSolutionError(OrbitForgeError):
    pass

class TimelineError(OrbitForgeError):
    pass

class ReproducibilityError(OrbitForgeError):
    pass
