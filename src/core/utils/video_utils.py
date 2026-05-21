import numpy as np

def extract_player_torso(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    new_y1 = int(y1 + h * 0.15)
    new_y2 = int(y1 + h * 0.50)
    new_x1 = int(x1 + w * 0.25)
    new_x2 = int(x2 - w * 0.25)

    return frame[new_y1:new_y2, new_x1:new_x2]