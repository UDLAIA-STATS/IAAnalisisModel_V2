from enum import StrEnum


class BucketTypes(StrEnum):
    ARTEFACTS = "artifacts"
    PLAYERS = "players"

class FilePurposeTypes(StrEnum):
    PLAYER_IMAGE = "player_image"
    HEATMAP = "heatmap"
    REPORTS = "reports"
