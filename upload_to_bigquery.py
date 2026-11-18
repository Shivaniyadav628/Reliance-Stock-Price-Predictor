
from pathlib import Path
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError

# ==========================================
# 1. GOOGLE CLOUD CONFIGURATION
# ==========================================

PROJECT_ID = "stock-news-analysis-514715"
DATASET_ID = "stock_predictor"
TABLE_ID = "fusion_dataset"

LOCATION = "US"

# ==========================================
# 2. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

CSV_FILE = BASE_DIR / "data" / "fusion_dataset.csv"

# ==========================================
# 3. CHECK INPUT FILE
# ==========================================

print("\nStarting BigQuery upload...")

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Fusion dataset not found: {CSV_FILE}"
    )

if CSV_FILE.stat().st_size == 0:
    raise ValueError("Fusion dataset is empty!")

print(f"Input file: {CSV_FILE}")
print(f"File size: {CSV_FILE.stat().st_size / (1024 * 1024):.2f} MB")

# ==========================================
# 4. CONNECT TO BIGQUERY
# ==========================================

try:
    client = bigquery.Client(project=PROJECT_ID)

    print("\nConnected to Google BigQuery!")

    # Verify destination dataset
    dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"
    dataset = client.get_dataset(dataset_ref)

    if dataset.location != LOCATION:
        raise ValueError(
            f"Dataset location is {dataset.location}, "
            f"expected {LOCATION}"
        )

    # ==========================================
    # 5. CONFIGURE DESTINATION TABLE
    # ==========================================

    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition=(
            bigquery.WriteDisposition.WRITE_TRUNCATE
        ),
        encoding="UTF-8",
        allow_quoted_newlines=True,
    )

    # ==========================================
    # 6. UPLOAD CSV TO BIGQUERY
    # ==========================================

    print("\nUploading fusion_dataset.csv...")
    print(f"Destination: {table_ref}")

    with CSV_FILE.open("rb") as file:
        job = client.load_table_from_file(
            file,
            table_ref,
            job_config=job_config,
            location=LOCATION,
        )

    # Wait for upload to finish
    job.result()

    # ==========================================
    # 7. DISPLAY UPLOAD RESULTS
    # ==========================================

    table = client.get_table(table_ref)

    print("\n" + "=" * 45)
    print("BIGQUERY UPLOAD COMPLETED SUCCESSFULLY!")
    print("=" * 45)

    print(f"Project ID : {PROJECT_ID}")
    print(f"Dataset    : {DATASET_ID}")
    print(f"Table      : {TABLE_ID}")
    print(f"Rows       : {table.num_rows}")
    print(f"Columns    : {len(table.schema)}")
    print(f"Location   : {table.location}")

    print("\nTable schema:")

    for field in table.schema:
        print(f"  {field.name}: {field.field_type}")

    print("\nYour fusion dataset is now in BigQuery!")

except GoogleAPIError as error:
    print("\nBIGQUERY API ERROR:")
    print(error)

except Exception as error:
    print("\nUPLOAD FAILED:")
    print(error)

    raise

finally:
    if "client" in locals():
        client.close()
