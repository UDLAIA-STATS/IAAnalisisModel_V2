PITCH_WIDTH = 105.0
FIELD_HEIGHT = 68.0

FIELD_KEYPOINTS: dict[str, tuple[float, float]] = {
    # Four corners
    "tl_corner": (0.0, 0.0),
    "tr_corner": (PITCH_WIDTH, 0.0),
    "bl_corner": (0.0, FIELD_HEIGHT),
    "br_corner": (PITCH_WIDTH, FIELD_HEIGHT),
    # Halfway line × touchlines
    "mid_top": (PITCH_WIDTH / 2, 0.0),
    "mid_bottom": (PITCH_WIDTH / 2, FIELD_HEIGHT),
    # Centre spot
    "centre_spot": (PITCH_WIDTH / 2, FIELD_HEIGHT / 2),
    # Left penalty box corners (goal line side)
    "lpen_tl": (0.0, 13.84),
    "lpen_tr": (16.5, 13.84),
    "lpen_bl": (0.0, 54.16),
    "lpen_br": (16.5, 54.16),
    # Right penalty box corners
    "rpen_tl": (PITCH_WIDTH - 16.5, 13.84),
    "rpen_tr": (PITCH_WIDTH, 13.84),
    "rpen_bl": (PITCH_WIDTH - 16.5, 54.16),
    "rpen_br": (PITCH_WIDTH, 54.16),
    # Left six-yard box corners
    "lsix_tl": (0.0, 24.84),
    "lsix_tr": (5.5, 24.84),
    "lsix_bl": (0.0, 43.16),
    "lsix_br": (5.5, 43.16),
    # Right six-yard box corners
    "rsix_tl": (PITCH_WIDTH - 5.5, 24.84),
    "rsix_tr": (PITCH_WIDTH, 24.84),
    "rsix_bl": (PITCH_WIDTH - 5.5, 43.16),
    "rsix_br": (PITCH_WIDTH, 43.16),
    # Penalty spots
    "lpen_spot": (11.0, FIELD_HEIGHT / 2),
    "rpen_spot": (PITCH_WIDTH - 11.0, FIELD_HEIGHT / 2),
    # Halfway line × penalty box lines
    "lpen_mid": (16.5, FIELD_HEIGHT / 2),
    "rpen_mid": (PITCH_WIDTH - 16.5, FIELD_HEIGHT / 2),
}