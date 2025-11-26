import argparse
import os
import sys

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_FILE = "token.json"


def authenticate_drive():
    """Authenticates with the Google Drive API using OAuth 2.0."""
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired credentials...")
            creds.refresh(Request())
        else:
            print("No valid credentials found. Starting authentication flow...")
            client_secret_file = os.getenv("GOOGLE_CLIENT_SECRET_FILE")
            if not client_secret_file or not os.path.exists(client_secret_file):
                print(
                    "Error: The 'GOOGLE_CLIENT_SECRET_FILE' environment variable is not set or the file does not exist."
                )
                print(
                    f"Please point it to your client_secret.json file. Searched at: {client_secret_file}"
                )
                sys.exit(1)

            flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
            creds = flow.run_local_server(port=0)
            print("Authentication successful.")

        # Save the credentials for the next run
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
            print(f"Credentials saved to '{TOKEN_FILE}'.")

    try:
        service = build("drive", "v3", credentials=creds)
        print("Successfully built Google Drive API service.")
        return service
    except Exception as e:
        print(f"Failed to build Google Drive API service: {e}")
        sys.exit(1)


def find_or_create_folder(service, name, parent_id):
    """Find a folder by name within a parent folder, or create it if it doesn't exist."""
    query = f"name='{name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    try:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="files(id, name)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files = response.get("files", [])

        if files:
            print(f"Found existing folder '{name}' (ID: {files[0].get('id')}).")
            return files[0].get("id")
        else:
            print(f"Folder '{name}' not found. Creating it...")
            file_metadata = {
                "name": name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_id],
            }
            folder = (
                service.files()
                .create(body=file_metadata, fields="id", supportsAllDrives=True)
                .execute()
            )
            print(f"Created new folder '{name}' (ID: {folder.get('id')}).")
            return folder.get("id")
    except HttpError as e:
        print(f"An error occurred while finding/creating folder '{name}': {e}")
        return None


def list_files_in_folder(service, folder_id):
    """Lists all files in a specific Google Drive folder to avoid duplicates."""
    files_set = set()
    page_token = None
    try:
        while True:
            response = (
                service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed=false and mimeType!='application/vnd.google-apps.folder'",
                    spaces="drive",
                    fields="nextPageToken, files(name)",
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            for file in response.get("files", []):
                files_set.add(file.get("name"))

            page_token = response.get("nextPageToken", None)
            if page_token is None:
                break
    except HttpError as e:
        print(f"Warning: Could not list files in folder {folder_id}: {e}")
    return files_set


def upload_folder_to_drive(service, local_folder_path, parent_folder_id):
    """Recursively uploads a local folder to a Google Drive folder, skipping existing files."""
    if not os.path.isdir(local_folder_path):
        print(f"Error: Source folder '{local_folder_path}' not found.")
        return

    print(
        f"Starting upload of '{local_folder_path}' to Drive folder ID '{parent_folder_id}'..."
    )

    # Cache for folder IDs to avoid redundant API calls
    # Key: relative path from local_folder_path (e.g., "." or "subdir/subsubdir")
    # Value: Google Drive Folder ID
    folder_id_cache = {".": parent_folder_id}

    for dirpath, _, filenames in os.walk(local_folder_path):
        relative_dir = os.path.relpath(dirpath, local_folder_path)

        # Determine the Drive folder ID for the current local directory
        if relative_dir == ".":
            current_parent_id = parent_folder_id
        else:
            # os.walk is top-down, so the parent folder should already be in cache
            parent_rel = os.path.dirname(relative_dir)
            if parent_rel == "":
                parent_rel = "."

            parent_id = folder_id_cache.get(parent_rel)

            if not parent_id:
                print(
                    f"Warning: Parent folder for '{relative_dir}' not found in cache. Skipping."
                )
                continue

            folder_name = os.path.basename(relative_dir)
            current_parent_id = find_or_create_folder(service, folder_name, parent_id)

            if current_parent_id:
                folder_id_cache[relative_dir] = current_parent_id
            else:
                print(
                    f"Could not create/find folder '{folder_name}'. Skipping files in it."
                )
                continue

        # Get existing files in the target Drive folder to skip duplicates
        existing_files = list_files_in_folder(service, current_parent_id)

        for filename in filenames:
            if filename in existing_files:
                print(f"  - Skipping '{filename}' (already exists).")
                continue

            local_path = os.path.join(dirpath, filename)
            print(f"  - Uploading file '{local_path}'...")

            file_metadata = {"name": filename, "parents": [current_parent_id]}
            media = MediaFileUpload(local_path, resumable=True)

            try:
                service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields="id",
                    supportsAllDrives=True,
                ).execute()
            except HttpError as e:
                print(f"    Error uploading file '{filename}': {e}")

    print("\nUpload complete.")


def main():
    """Main function to parse arguments and trigger the upload."""
    parser = argparse.ArgumentParser(
        description="Upload qubit measurement results to a specified Google Drive folder."
    )
    parser.add_argument(
        "qubit_names",
        nargs="+",
        help="The names of the qubits (e.g., Si001 Si002). These correspond to folders inside the 'result/' directory, which will be uploaded.",
    )
    args = parser.parse_args()

    load_dotenv()

    parent_folder_id = os.getenv("GOOGLE_DRIVE_PARENT_FOLDER_ID")
    if not parent_folder_id:
        print(
            "Error: The 'GOOGLE_DRIVE_PARENT_FOLDER_ID' environment variable is not set."
        )
        sys.exit(1)

    drive_service = authenticate_drive()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for qubit_name in args.qubit_names:
        print(f"\nProcessing qubit: {qubit_name}")
        source_folder = os.path.join(project_root, "result", qubit_name)

        if not os.path.isdir(source_folder):
            print(f"Error: Source folder '{source_folder}' not found. Skipping.")
            continue

        qubit_folder_id = find_or_create_folder(
            drive_service, qubit_name, parent_folder_id
        )

        if qubit_folder_id:
            upload_folder_to_drive(drive_service, source_folder, qubit_folder_id)


if __name__ == "__main__":
    main()
