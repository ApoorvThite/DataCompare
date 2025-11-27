import os
from pathlib import Path

# --------- PROJECT ROOT ----------
PROJECT_ROOT = Path("data-formats-ml-tableau")

# --------- FOLDERS TO CREATE ----------
DIRS = [
    "data_raw",
    "data_canonical",
    "data_formatted",
    "data_for_tableau",
    "src",
    "src/schemas",
    "src/formats",
    "src/ml",
    "src/metrics",
    "src/utils",
    "tableau",
    "docs",
]

# --------- FILES TO CREATE (with default content) ----------
FILES = {
    "README.md": "# Data Formats vs ML vs Tableau Project\n\nProject initialized.\n",
    "requirements.txt": "pandas\npyarrow\nscikit-learn\nnumpy\n",
    "src/config.py": (
        "from pathlib import Path\n\n"
        "PROJECT_ROOT = Path(__file__).resolve().parent.parent\n\n"
        "DATA_RAW = PROJECT_ROOT / 'data_raw'\n"
        "DATA_CANONICAL = PROJECT_ROOT / 'data_canonical'\n"
        "DATA_FORMATTED = PROJECT_ROOT / 'data_formatted'\n"
        "DATA_FOR_TABLEAU = PROJECT_ROOT / 'data_for_tableau'\n\n"
        "TARGET_COL = 'job_category'  # change if needed\n"
        "RANDOM_STATE = 42\n"
    ),
    "src/schemas/canonical_schema.py": (
        "# Canonical schema definition\n\n"
        "CANONICAL_SCHEMA = {\n"
        "    'job_id': 'string',\n"
        "    'title': 'string',\n"
        "    'company_location': 'string',\n"
        "    'experience_level': 'string',\n"
        "    'employment_type': 'string',\n"
        "    'salary_min': 'float',\n"
        "    'salary_max': 'float',\n"
        "    'currency': 'string',\n"
        "    'skills': 'list<string>',\n"
        "    'job_category': 'string',\n"
        "}\n"
    ),
    "src/schemas/toon_schema.toon": (
        "entity Job {\n"
        "  field job_id: string [primary_key]\n"
        "  field title: string\n"
        "  field company_location: string\n"
        "  field experience_level: string\n"
        "  field employment_type: string\n"
        "  field salary_min: float\n"
        "  field salary_max: float\n"
        "  field currency: string\n"
        "  field skills: list<string> [required]\n"
        "  field job_category: string\n"
        "}\n"
    ),
    "src/schemas/toon_parser.py": (
        "# Minimal TOON parser (placeholder)\n\n"
        "def parse_toon_schema(path):\n"
        "    pass\n\n"
        "def parse_toon_records(path):\n"
        "    pass\n\n"
        "def toon_to_canonical(path):\n"
        "    pass\n"
    ),
    "src/formats/csv_loader.py": "# CSV loader placeholder\n",
    "src/formats/json_loader.py": "# JSON loader placeholder\n",
    "src/formats/toon_loader.py": "# TOON loader placeholder\n",
    "src/formats/canonical_to_csv.py": "# canonical to CSV converter placeholder\n",
    "src/formats/canonical_to_json.py": "# canonical to JSON converter placeholder\n",
    "src/formats/canonical_to_toon.py": "# canonical to TOON converter placeholder\n",
    "src/ml/preprocessing_common.py": "# shared preprocessing placeholder\n",
    "src/ml/features_from_canonical.py": "# feature extraction placeholder\n",
    "src/ml/pipeline_csv.py": "# ML pipeline for CSV placeholder\n",
    "src/ml/pipeline_json.py": "# ML pipeline for JSON placeholder\n",
    "src/ml/pipeline_toon.py": "# ML pipeline for TOON placeholder\n",
    "src/ml/evaluate_pipeline.py": "# evaluate all pipelines placeholder\n",
    "src/metrics/benchmark_parsing.py": "# parsing benchmark placeholder\n",
    "src/metrics/benchmark_pipeline.py": "# pipeline benchmark placeholder\n",
    "src/metrics/schema_richness.py": "# schema richness calculator placeholder\n",
    "src/utils/io_helpers.py": "# IO helpers placeholder\n",
    "src/utils/timing.py": (
        "import time\n"
        "from contextlib import contextmanager\n\n"
        "@contextmanager\n"
        "def time_block(label: str):\n"
        "    start = time.perf_counter()\n"
        "    yield\n"
        "    end = time.perf_counter()\n"
        "    print(f'{label}: {(end - start) * 1000:.2f} ms')\n"
    ),
    "docs/project_overview.md": "# Project Overview\n",
    "docs/toon_format_spec.md": "# TOON Format Specification\n",
}

# --------- SCRIPT EXECUTION ----------
def create_project():
    print(f"Creating project at: {PROJECT_ROOT.resolve()}\n")

    # Create directories
    for d in DIRS:
        path = PROJECT_ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")

    # Create files
    for file_path, content in FILES.items():
        full_path = PROJECT_ROOT / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        if not full_path.exists():
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Created file: {full_path}")
        else:
            print(f"File already exists (skipped): {full_path}")

    print("\nProject structure created successfully!")


if __name__ == "__main__":
    create_project()
