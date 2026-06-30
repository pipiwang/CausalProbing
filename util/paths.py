from pathlib import Path


PROJECT_ROOT = Path(".")
DATA_ROOT = PROJECT_ROOT / "data"
RAW_IMAGE_ROOT = Path("data/raw_mri")
PROFOUND_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PROFOUND_CONV_CHECKPOINT = PROFOUND_CHECKPOINT_DIR / "profound_conv_checkpoint-799.pth"
PROFOUND_VIT_CHECKPOINT = PROFOUND_CHECKPOINT_DIR / "profound_vit_checkpoint-799.pth"

META_INFO_PROCESSED_CSV = DATA_ROOT / "meta_info_processed.csv"
META_INFO_CLEANED_CSV = DATA_ROOT / "meta_info_cleaned.csv"
META_INFO_CLEANED_DICTIONARY_CSV = DATA_ROOT / "meta_info_cleaned_dictionary.csv"
GLEASON_CLASSIFICATION_CSV = DATA_ROOT / "gleason_classification.csv"
GLEASON_PREPROCESS_FAILURES_CSV = DATA_ROOT / "gleason_preprocess_failures.csv"
GLEASON_SPLIT_SUMMARY_CSV = DATA_ROOT / "gleason_split_summary.csv"
IMAGE_CACHE_DIR = DATA_ROOT / "img"
QC_DIR = DATA_ROOT / "qc"

CLASSIFICATION_OUTPUT_DIR = PROJECT_ROOT / "output_cls"
CLASSIFICATION_LOG_DIR = PROJECT_ROOT / "output_cls"
GEOMETRY_REPORT_CSV = DATA_ROOT / "geometry_report.csv"
