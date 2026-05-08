"""Central configuration for CRYPTO AI."""

DBSCAN_EPS = 1.15
DBSCAN_MIN_SAMPLES = 5

FCM_CLUSTERS = 3
FCM_M = 2.0
FCM_ERROR = 0.005
FCM_MAX_ITER = 300

SVM_KERNEL = "rbf"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Main autonomous action threshold.
ACTION_THRESHOLD = 0.72

FORECAST_STEPS = 7
DRIFT_THRESHOLD = 0.30

# Supported cipher families in the final CRYPTO AI pipeline.
# Important:
# - Base64, Base32, Morse are the original project cipher types.
# - Hex and ASCII were added as extra deterministic decoding families.
SUPPORTED_CIPHER_TYPES = [
    "Base64",
    "Base32",
    "Morse",
    "Hex",
    "ASCII",
]

# Optional aliases used by validation/normalization layers.
# These help the system understand different names for the same cipher type.
CIPHER_TYPE_ALIASES = {
    "base64": "Base64",
    "b64": "Base64",
    "base_64": "Base64",

    "base32": "Base32",
    "b32": "Base32",
    "base_32": "Base32",

    "morse": "Morse",
    "morsecode": "Morse",
    "morse_code": "Morse",

    "hex": "Hex",
    "hexadecimal": "Hex",
    "hexadecimalcode": "Hex",
    "hex_code": "Hex",

    "ascii": "ASCII",
    "ascii_decimal": "ASCII",
    "decimalascii": "ASCII",
    "decimal_ascii": "ASCII",
    "ascii_codes": "ASCII",
}

DEFAULT_CSV_PATH = "data/raw/dataset.csv"
DEFAULT_EXCEL_PATH = "data/raw/dataset.xlsx"
DEFAULT_KNOWLEDGE_PATH = "data/knowledge/cipher_knowledge.csv"
ACTION_LOG_PATH = "data/outputs/action_log.csv"