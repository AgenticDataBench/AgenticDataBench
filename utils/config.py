"""
Configuration
"""

import os
import random

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly


DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_API_BASE = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
QWEN_MODEL = "qwen3-max-2026-01-23"

# Agent LLM settings ---------------------------------------------------------
# These back the coding-agent wrappers under testbed/da_agent/agent/. Each
# agent reads its own provider's settings from here so credentials live in one
# place (with environment variables as override).

# Claude Code agent (Anthropic API). Passed into the agent's Docker container
# as environment variables and consumed by the `claude` CLI.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", DASHSCOPE_API_KEY)
ANTHROPIC_BASE_URL = 'https://dashscope.aliyuncs.com/apps/anthropic'

# Codex agent (OpenAI-compatible API, e.g. DashScope's compatible mode).
# The `codex` CLI reads the key via the `env_key` set in ~/.codex/config.toml.
# Defaults fall back to the DashScope endpoint/key since that is the intended
# provider; override with the dedicated environment variables if needed.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", DASHSCOPE_API_KEY)
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", DASHSCOPE_API_BASE)

DATASETS_DIR = "./datasets"
OUTPUT_DIR = "./output"

# Eval case file constraints
MAX_FILES_PER_CASE = 12  # One eval case involves at most this many (physical) files
MIN_STEPS_PER_CASE = 3   # Cases with fewer steps are considered generation failures

# Per-domain min files range (min, max) inclusive. Each eval case samples one value in [min, max].
# Ranges are capped by min(range_value, len(available_files), MAX_FILES_PER_CASE) at runtime.
MIN_FILES_PER_DOMAIN = {
    "agriculture": (3, 5),       # 7 files total (4 CropYield CSVs + shapefile + CropRec + FoodPrices)
    "e-commerce": (6, 8),        # 12 files (Brazilian 9-file star schema + Amazon pair + eBay)
    "energy": (5, 7),            # 9 files (Sunroof×4 + US Prices + US Gen + Global + OWID + Steel)
    "entertainment": (6, 8),     # 14 files (Movies×7 + Spotify + top×3 + vgsales + netflix + books)
    "healthcare": (6, 8),        # ~11 files (5 standalone CSVs + thrombosis.sqlite + CKD + TCGA×3)
    "real_estate": (6, 10),      # 48 files (Zillow×40 + Census×2 + Crime×4 + Education + Unemployment)
    "social_network": (3, 4),    # 8 files total (6 Twitter + YouTube + Instagram)
    "sports": (6, 8),            # 19 files (NBA×16 CSVs + Soccer.sqlite + Olympic×2)
    "tourism": (6, 8),           # ~17 main files (TripAdvisor×9 + FISETIO×3 + Airbnb + 4 standalone)
    "transportation": (5, 7),    # 12 files (yellow_tripdata×9 + taxi_zones + ncr + NTAD)
}
DEFAULT_MIN_FILES_RANGE = (6, 8)


def get_min_files_for_domain(domain: str) -> int:
    """Return a random min_files value for this domain (for one eval case)."""
    low, high = MIN_FILES_PER_DOMAIN.get(domain, DEFAULT_MIN_FILES_RANGE)
    return random.randint(low, high)
