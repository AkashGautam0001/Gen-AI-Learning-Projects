import logging
import os

# Log file path (same folder)
log_file = os.path.join(os.path.dirname(__file__), "app.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)