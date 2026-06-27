import logging
import sys

# Third-party libraries that log chatty per-request INFO lines (e.g. httpx
# emitting a line for every HuggingFace Hub file probe, including expected 404s).
# Capped at WARNING so they don't drown out our own logs.
# _NOISY_LOGGERS = ("httpx", "httpcore", "huggingface_hub", "sentence_transformers")

def setup_logging(level="INFO"):
    """Configure basic logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # for noisy in _NOISY_LOGGERS:
        # logging.getLogger(noisy).setLevel(logging.WARNING)

def get_logger(name: str):
    """Return a logger with the given name."""
    return logging.getLogger(name)
