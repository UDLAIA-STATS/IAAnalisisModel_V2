from pydantic import BaseModel


class PlayerResume(BaseModel):
    player_id: int
    track_id: int
    match_id: int

    team_color: str
    shirt_number: int
    goals: int

    avg_speed: float
    avg_distance: float
    avg_acceleration: float
    ball_possession_time: float

    player_crop_path: str
    player_heatmap_path: str
    team_heatmap_path: str
    movement_trajectories_path: str
