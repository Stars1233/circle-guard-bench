import logging
from pathlib import Path
from datetime import datetime
import uuid

from dotenv import load_dotenv

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _setup_logging():
    """
    Settings up logger and logs file handler
    """
    unique_id = uuid.uuid4().hex[:8]
    log_filename = (
        LOG_DIR
        / f"guardbench_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),
        ],
    )

    # Suppress logs from specific modules
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


# Setup logging
_setup_logging()

# Load env values from local .env
load_dotenv()
