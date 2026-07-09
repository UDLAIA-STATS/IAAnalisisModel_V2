import os

from dynaconf import Dynaconf


settings = Dynaconf(
    settings_file="settings.toml",
    environments=True,
    envvar_prefix="APP",
)

os.environ["USE_CUDA"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
