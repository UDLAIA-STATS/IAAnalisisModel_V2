from pydantic import BaseModel


class PlayerResume(BaseModel):
    player_id: int
    track_id: int
    match_id: int

    team_color: str
    shirt_number: int
    team_goals: int
    goals: int
    passes: int

    avg_speed_kmh: float
    distance_km: float
    avg_acceleration: float
    avg_possession_time_s: float

    player_crop_path: str
    heatmap_image_path: str
    team_heatmap_path: str
    movement_trajectories_path: str
    player_movement_trajectories_path: str
    team_color_time_kde_path: str
    voronoi_territories_path: str
