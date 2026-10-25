"""SkyRL import path for the env class (examples.train.rpg_v10.env:RPGv10Env); the code is rl_v10/skyrl_env.py."""
import os
import sys

_SRC = os.environ.get("RLV10_SRC", "/work/ADS_shared/dataset_generation_code/rl_v10")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from skyrl_env import RPGv10Env  # noqa: E402,F401
