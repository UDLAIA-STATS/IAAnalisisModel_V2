from dataclasses import dataclass
from pathlib import Path

from src.config.routes import POST_PROCESSING_MODELS

@dataclass
class PostProcessingConfig:
    # Generales
    use_ml: bool = True
    train_sample_fraction: float = 0.8          # Fracción de datos etiquetados para entrenar
    model_dir: Path = POST_PROCESSING_MODELS             # Directorio para guardar/cargar modelos
    force_retrain: bool = True                 # Si True, ignora modelos guardados
    fallback_to_heuristic: bool = True          # Si falla ML, usar heurística
    cache_dataframes: bool = True               # Cachear DataFrames intermedios

    # Umbrales para detección
    possession_threshold: float = 0.5
    shot_threshold: float = 0.4
    goal_association_window: float = 5.0        # segundos
    min_possession_seconds: float = 0.09

    # Limpieza de balón
    min_ball_confidence: float = 0.25
    min_bbox_area: float = 15.0
    max_position_jump_px: float = 180.0
    max_ball_speed_kmh: float = 130.0
    max_ball_accel_mss: float = 150.0
    static_speed_threshold: float = 0.5         # km/h
    static_frame_threshold: int = 8

    # Interpolación
    max_interpolation_gap: int = 10
    frame_rate: float = 30.0

    # Limpieza del arco
    min_post_confidence: float = 0.20
    min_post_area: float = 500.0
    min_consecutive_frames: int = 1
    cluster_distance_threshold: float = 150.0   # px

    # Modelos de trayectoria (regresión)
    trajectory_window: int = 5

    # Otros
    possession_lookback_frames: int = 30
    possession_lookback_seconds: float = 2.0
    shot_lookahead_frames: int = 15
    shot_lookahead_seconds: float = 1.0
    min_shot_speed_kmh: float = 20.0
    max_shot_speed_kmh: float = 120.0
    near_goal_cm_threshold: float = 200.0       # cm
    goal_area_reduction_factor: float = 0.75    # 75% del área original