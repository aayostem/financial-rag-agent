"""Configuration factory - exports the right config based on environment"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Determine environment
ENV = os.getenv("FINRAG_ENV", "development").lower()

# Import the appropriate config
if ENV == "production":
    from .environments.production import ProductionConfig as Config
elif ENV == "testing":
    from .environments.testing import TestingConfig as Config
else:  # development is default
    from .environments.development import DevelopmentConfig as Config

# Create config instance
config = Config()

# Export
__all__ = ["config"]
