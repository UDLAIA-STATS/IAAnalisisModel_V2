from enum import StrEnum


class DetectorTypes(StrEnum):
    """Enum for the types of detectors."""
    TRACKING = "tracker"
    DETECTION = "detector"