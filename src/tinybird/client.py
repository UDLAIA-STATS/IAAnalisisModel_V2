import os

from tinybird_sdk import Tinybird
from src.tinybird.tinybird_resources import page_views, top_pages

tinybird = Tinybird(
    {
        "datasources": {"page_views": page_views},
        "pipes": {"top_pages": top_pages},
        "base_url": os.getenv("TINYBIRD_API_URL", "https://api.tinybird.co"),
        "token": os.getenv("TINYBIRD_TOKEN"),
    }
)
