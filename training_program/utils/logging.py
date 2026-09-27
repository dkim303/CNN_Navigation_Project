import time
from pathlib import Path
import logging
from datetime import datetime, timezone

def setup_logging(program_type: str, log_dir: Path) -> None:         
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_name = f"{program_type}_{timestamp}.log"

    logging.basicConfig(
        filename=log_dir / log_name,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )