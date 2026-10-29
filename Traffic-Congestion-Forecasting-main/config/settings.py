# Data source configuration
# DATA_SOURCE = "static"  # Switch between "static" or "realtime"

# # Static data paths
# STATIC_DATA_PATH = "data/raw/Banglore_traffic_Dataset.csv"

# # Real-time API settings
# REALTIME_CONFIG = {
#     "provider": "mapmyindia",  # "tomtom", "here"
#     "update_interval": 300,  # Seconds between updates
#     "location": "Bengaluru",
#     "cache_enabled": True
# }


# Static mode configuration
# from config import settings
# from pathlib import Path
# import sys
# STATIC_MODE = {
#     "DATA_SOURCE": "static",
#     "STATIC_DATA_PATH":  "data/raw/Banglore_traffic_Dataset.csv"
# }

# # Real-time mode configuration
# REALTIME_MODE = {
#     "DATA_SOURCE": "realtime",
#     "REALTIME_CONFIG": {
#         "provider": "mapmyindia",
#         "update_interval": 300,
#         "location": "Bengaluru",
#         "cache_enabled": True
#     }
# }

# # Active configuration (change this to switch modes)
# ACTIVE_CONFIG = STATIC_MODE  # Or REALTIME_MODE

# Add project root to sys.path first
# sys.path.append(str(Path(__file__).parent.parent))

# settings.py
# Data source configuration
DATA_SOURCES = {
    "static": {
        "active": True,
        "path": "data/raw/Banglore_traffic_Dataset.csv"
    },
    "realtime": {
        "active": True,
        "provider": "mapmyindia",  # "tomtom", "here"
        "update_interval": 300,  # Seconds between updates
        "location": "Bengaluru",
        "cache_enabled": True
    }
}