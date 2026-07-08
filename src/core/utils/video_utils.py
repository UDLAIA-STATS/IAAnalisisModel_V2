import numpy as np

def extract_player_torso(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(int)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return crop

    h, w = crop.shape[:2]

    if h < 40 or w < 20:
        return crop

    x_margin = max(2, int(w * 0.12))
    top_margin = max(2, int(h * 0.12))
    bottom_cut = max(2, int(h * 0.40))

    torso = crop[
        top_margin : h - bottom_cut,
        x_margin : w - x_margin,
    ]

    if torso.shape[0] < 10 or torso.shape[1] < 10:
        return crop

    return torso