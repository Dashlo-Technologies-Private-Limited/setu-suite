import sys
import os

# Add parent directory to path so it can import app.py and database
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app