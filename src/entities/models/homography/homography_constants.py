from src.config.constants import PITCH_WIDTH, PITCH_LENGTH

FIELD_KEYPOINTS = {
    # Esquinas
    "tl_corner": (0.0, 0.0),
    "tr_corner": (PITCH_LENGTH, 0.0),
    "bl_corner": (0.0, PITCH_WIDTH),
    "br_corner": (PITCH_LENGTH, PITCH_WIDTH),

    # Línea central
    "mid_top": (PITCH_LENGTH / 2, 0.0),
    "mid_bottom": (PITCH_LENGTH / 2, PITCH_WIDTH),

    # Centro
    "centre_spot": (PITCH_LENGTH / 2, PITCH_WIDTH / 2),

    # Área grande izquierda
    "lpen_tl": (0.0, 13.84),
    "lpen_tr": (16.5, 13.84),
    "lpen_bl": (0.0, PITCH_WIDTH - 13.84),
    "lpen_br": (16.5, PITCH_WIDTH - 13.84),

    # Área grande derecha
    "rpen_tl": (PITCH_LENGTH - 16.5, 13.84),
    "rpen_tr": (PITCH_LENGTH, 13.84),
    "rpen_bl": (PITCH_LENGTH - 16.5, PITCH_WIDTH - 13.84),
    "rpen_br": (PITCH_LENGTH, PITCH_WIDTH - 13.84),

    # Área pequeña izquierda
    "lsix_tl": (0.0, 24.84),
    "lsix_tr": (5.5, 24.84),
    "lsix_bl": (0.0, PITCH_WIDTH - 24.84),
    "lsix_br": (5.5, PITCH_WIDTH - 24.84),

    # Área pequeña derecha
    "rsix_tl": (PITCH_LENGTH - 5.5, 24.84),
    "rsix_tr": (PITCH_LENGTH, 24.84),
    "rsix_bl": (PITCH_LENGTH - 5.5, PITCH_WIDTH - 24.84),
    "rsix_br": (PITCH_LENGTH, PITCH_WIDTH - 24.84),

    # Puntos de penal
    "lpen_spot": (11.0, PITCH_WIDTH / 2),
    "rpen_spot": (PITCH_LENGTH - 11.0, PITCH_WIDTH / 2),

    # Centro del área
    "lpen_mid": (16.5, PITCH_WIDTH / 2),
    "rpen_mid": (PITCH_LENGTH - 16.5, PITCH_WIDTH / 2),
}