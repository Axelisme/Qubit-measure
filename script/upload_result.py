import argparse
import os
import sys
from dotenv import load_dotenv
from google.cloud import storage
from google.api_core import exceptions

def upload_to_gcs(bucket_name, source_folder, destination_blob_folder):
    """Uploads a folder to the GCS bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
    except Exception as e:
        print(f"Error connecting to Google Cloud Storage: {e}")
        print("Please check your authentication credentials (GOOGLE_APPLICATION_CREDENTIALS).")
        return

    if not bucket.exists():
        print(f"Error: Bucket '{bucket_name}' does not exist.")
        return

    # Check if the source folder exists
    if not os.path.isdir(source_folder):
        print(f"Error: Source folder '{source_folder}' not found.")
        sys.exit(1)

    print(f"Uploading contents of '{source_folder}' to GCS bucket '{bucket_name}' under '{destination_blob_folder}'...")

    # Walk through the source folder
    for dirpath, _, filenames in os.walk(source_folder):
        for filename in filenames:
            local_path = os.path.join(dirpath, filename)
            # Create a relative path to maintain folder structure in the bucket
            relative_path = os.path.relpath(local_path, source_folder)
            blob_path = os.path.join(destination_blob_folder, relative_path)

            try:
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(local_path)
                print(f"  - Uploaded {local_path} to {blob_path}")
            except exceptions.GoogleAPICallError as e:
                print(f"Error uploading file {local_path}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while uploading {local_path}: {e}")
    
    print("\nUpload complete.")


def main():
    """Main function to parse arguments and trigger the upload."""
    parser = argparse.ArgumentParser(description="Upload qubit measurement results to Google Cloud Storage.")
    parser.add_argument("qubit_name", help="The name of the qubit (e.g., Si001). This corresponds to a folder inside the 'results/' directory.")
    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    bucket_name = os.getenv("GCP_STORAGE_BUCKET")
    if not bucket_name:
        print("Error: The 'GCP_STORAGE_BUCKET' environment variable is not set.")
        print("Please create a '.env' file based on the '.env.example' and provide your bucket name.")
        sys.exit(1)

    # Assuming '@result/Si001' means a 'results' directory in the project root.
    # The script is in 'script/', so we go up one level to the project root.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_folder = os.path.join(project_root, "results", args.qubit_name)
    
    # The destination folder in the bucket will be the qubit name itself.
    destination_blob_folder = args.qubit_name

    upload_to_gcs(bucket_name, source_folder, destination_blob_folder)


if __name__ == "__main__":
    main()
