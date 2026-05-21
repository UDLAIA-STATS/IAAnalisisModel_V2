from pathlib import Path
from src.config.configuration import settings

# BASE
BASE_RES_DIR = Path("./res")
DATABASE_DIR = BASE_RES_DIR / "database"
INPUT_VIDEOS_DIR = BASE_RES_DIR / "inputs"


# OUTPUTS
OUTPUTS_DIR = BASE_RES_DIR / "outputs"
OUTPUT_VIDEOS_DIR = OUTPUTS_DIR / "videos"
OUTPUT_IMAGES_DIR = OUTPUTS_DIR / "images"
OUTPUT_REPORTS_DIR = OUTPUTS_DIR / "reports"

# ANNOTATIONS
ANNOTATED_FILES_DIR = OUTPUTS_DIR / "annotated"
ANOTATED_VIDEOS_DIR = ANNOTATED_FILES_DIR / "videos"
ANOTATED_OUTPUT_IMAGES = ANNOTATED_FILES_DIR / "images"

# METRICS
METRICS_DIR = OUTPUT_REPORTS_DIR / "metrics"
DETECTED_OBJECTS_METRICS_DIR = OUTPUT_REPORTS_DIR / "detected_objects_metrics"
MEMORY_TRACKER_DIR = OUTPUT_REPORTS_DIR / "memory_tracker"
TIME_REPORTS_DIR = OUTPUT_REPORTS_DIR / "time_reports"

# MODELS
MODELS_DIR = BASE_RES_DIR / "models"
MODELS_BACKUP_DIR = MODELS_DIR / "backup"
YOLO_MODELS_DIR = MODELS_DIR / "yolo"
TROCR_PATH = MODELS_DIR / "trocr"
CONFIG_MODELS_DIR = MODELS_DIR / "config"
BALL_MODEL_PATH = YOLO_MODELS_DIR / str(settings.BALL_MODEL_NAME)
PLAYER_MODEL_PATH = YOLO_MODELS_DIR / str(settings.PLAYER_MODEL_NAME)
MODEL_GOALS_PATH = YOLO_MODELS_DIR / str(settings.GOAL_MODEL_NAME)
DEPTH_MODEL_PATH = MODELS_DIR / str(settings.DEPTH_MODEL_NAME)
BYTETRACK_CONFIG_PATH = CONFIG_MODELS_DIR / "bytetrack.yaml"

# RETRAINING
DATASETS_DIR = Path("./retraining", "data")
PLAYER_YOLO_DATA = MODELS_DIR / "YOLO_pickles"
RETRAINED_MODELS = MODELS_DIR / "retrained_models"
PLAYER_CUSTOM_DATASET = DATASETS_DIR / "custom_player_dataset"
BALL_CUSTOM_DATASET = DATASETS_DIR / "custom_ball_dataset"
CUSTOM_MODELS = DATASETS_DIR / "custom_models"


def ensure_directories():
    """
    Asegura que las carpetas necesarias existan.
    """
    for directory in [
        OUTPUT_VIDEOS_DIR,
        OUTPUT_IMAGES_DIR,
        MODELS_DIR,
        DATASETS_DIR,
        TROCR_PATH,
        INPUT_VIDEOS_DIR,
        OUTPUT_REPORTS_DIR,
        DATABASE_DIR,
        ANOTATED_VIDEOS_DIR,
        ANOTATED_OUTPUT_IMAGES,
        METRICS_DIR,
        MEMORY_TRACKER_DIR,
        DETECTED_OBJECTS_METRICS_DIR,
        PLAYER_YOLO_DATA,
        RETRAINED_MODELS,
        PLAYER_CUSTOM_DATASET,
        BALL_CUSTOM_DATASET,
        MODELS_BACKUP_DIR,
        CUSTOM_MODELS,
        TIME_REPORTS_DIR
    ]:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)


def validate_model():
    for file in [
        BALL_MODEL_PATH,
        PLAYER_MODEL_PATH,
        MODEL_GOALS_PATH,
        DEPTH_MODEL_PATH,
        BYTETRACK_CONFIG_PATH,
    ]:
        if not file.exists() or not file.is_file():
            raise FileNotFoundError(f"Modelo no encontrado: {file.name}")
