import argparse
import fnmatch
import os
import sys
import threading
from datetime import datetime, timezone
from typing import List

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from joblib import Parallel, delayed

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_FILE = "token.json"
IGNORE_FILE: List[str] = ["*.lock"]

# Thread-local storage for Google Drive service instances
_thread_local = threading.local()


def get_thread_local_service(creds):
    """Get or create a thread-local Google Drive service instance.

    Args:
        creds: Google OAuth2 credentials.

    Returns:
        A Google Drive API service instance unique to this thread.
    """
    if not hasattr(_thread_local, "service"):
        _thread_local.service = build("drive", "v3", credentials=creds)
    return _thread_local.service


def authenticate_drive():
    """Authenticates with the Google Drive API using OAuth 2.0.

    Returns:
        tuple: (service, creds) - The Drive API service and credentials.
               Credentials are returned for creating thread-local services.
    """
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
        return service, creds
    except Exception as e:
        print(f"Failed to build Google Drive API service: {e}")
        sys.exit(1)


def parse_drive_timestamp(timestamp_str):
    """Parses Google Drive's RFC 3339 timestamp into a datetime object."""
    # Replace 'Z' with '+00:00' for compatibility with datetime.fromisoformat
    if timestamp_str.endswith("Z"):
        timestamp_str = timestamp_str[:-1] + "+00:00"
    return datetime.fromisoformat(timestamp_str)


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
    files_data = {}  # Change to dictionary to store more file info
    page_token = None
    try:
        while True:
            response = (
                service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed=false and mimeType!='application/vnd.google-apps.folder'",
                    spaces="drive",
                    fields="nextPageToken, files(id, name, modifiedTime)",  # Request id and modifiedTime
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            for file in response.get("files", []):
                file_name = file.get("name")
                if file_name:
                    files_data[file_name] = {
                        "id": file.get("id"),
                        "modifiedTime": file.get("modifiedTime"),
                    }

            page_token = response.get("nextPageToken", None)
            if page_token is None:
                break
    except HttpError as e:
        print(f"Warning: Could not list files in folder {folder_id}: {e}")
    return files_data


def collect_upload_tasks(
    service, local_folder_path, parent_folder_id, tasks_list, prune=False
):
    """Recursively traverses a local folder and collects upload/update/delete tasks.

    Args:
        service: Google Drive API service instance.
        local_folder_path: The local folder path to traverse.
        parent_folder_id: The Google Drive parent folder ID.
        tasks_list: A list to append tasks to.
        prune: If True, delete remote files that do not exist locally.
    """
    if not os.path.isdir(local_folder_path):
        print(f"Error: Source folder '{local_folder_path}' not found.")
        return

    print(
        f"Collecting tasks from '{local_folder_path}' to Drive folder ID '{parent_folder_id}'..."
    )

    # Cache for folder IDs to avoid redundant API calls
    folder_id_cache = {".": parent_folder_id}

    for dirpath, dirnames, filenames in os.walk(local_folder_path):
        # Prune ignored directories and files
        dirnames[:] = [
            d for d in dirnames if not any(fnmatch.fnmatch(d, p) for p in IGNORE_FILE)
        ]
        filenames[:] = [
            f for f in filenames if not any(fnmatch.fnmatch(f, p) for p in IGNORE_FILE)
        ]

        relative_dir = os.path.relpath(dirpath, local_folder_path)

        # Determine the Drive folder ID for the current local directory
        if relative_dir == ".":
            current_parent_id = parent_folder_id
        else:
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

        # Get existing files in the target Drive folder
        existing_files = list_files_in_folder(service, current_parent_id)
        processed_remote_files = set()

        for filename in filenames:
            local_path = os.path.join(dirpath, filename)
            local_mtime = os.path.getmtime(local_path)
            local_datetime = datetime.fromtimestamp(local_mtime, tz=timezone.utc)

            if filename in existing_files:
                processed_remote_files.add(filename)
                remote_file_info = existing_files[filename]
                remote_file_id = remote_file_info["id"]
                remote_modified_time_str = remote_file_info["modifiedTime"]
                remote_datetime = parse_drive_timestamp(remote_modified_time_str)

                if local_datetime > remote_datetime:
                    print(f"  - Queuing update for '{filename}' (local is newer).")
                    tasks_list.append(
                        (
                            "update",
                            local_path,
                            filename,
                            current_parent_id,
                            remote_file_id,
                        )
                    )
                else:
                    print(
                        f"  - Skipping '{filename}' (remote is up-to-date: {remote_datetime} vs local: {local_datetime})."
                    )
                continue

            print(f"  - Queuing new file '{filename}' for upload...")
            tasks_list.append(("create", local_path, filename, current_parent_id, None))

        # Handle pruning of extraneous remote files
        if prune:
            for remote_filename, remote_info in existing_files.items():
                if remote_filename not in processed_remote_files:
                    print(
                        f"  - Queuing deletion for '{remote_filename}' (not found locally)."
                    )
                    tasks_list.append(
                        (
                            "delete",
                            None,
                            remote_filename,
                            current_parent_id,
                            remote_info["id"],
                        )
                    )


def process_upload_task(creds, task):
    """Process a single task using a thread-local service.

    Args:
        creds: Google OAuth2 credentials for creating thread-local services.
        task: A tuple of (task_type, local_path, filename, folder_id, remote_file_id).
    """
    task_type, local_path, filename, folder_id, remote_file_id = task

    # Get or create thread-local service
    service = get_thread_local_service(creds)

    try:
        if task_type == "delete":
            service.files().delete(fileId=remote_file_id, supportsAllDrives=True).execute()
            print(f"    Deleted remote file: {filename}")
            return

        media = MediaFileUpload(local_path, resumable=True)

        if task_type == "update":
            service.files().update(
                fileId=remote_file_id,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            ).execute()
            print(f"    Updated: {local_path}")
        else:  # task_type == "create"
            file_metadata = {"name": filename, "parents": [folder_id]}
            service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            ).execute()
            print(f"    Uploaded: {local_path}")
    except HttpError as e:
        action = "deleting" if task_type == "delete" else "uploading"
        print(f"    Error {action} file '{filename}': {e}")


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
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=4,
        help="Number of parallel upload threads (default: 4).",
    )
    parser.add_argument(
        "--prune-remote",
        "--delete-extraneous",
        action="store_true",
        dest="prune",
        help="Delete files on the remote that do not exist locally.",
    )
    args = parser.parse_args()

    load_dotenv()

    parent_folder_id = os.getenv("GOOGLE_DRIVE_PARENT_FOLDER_ID")
    if not parent_folder_id:
        print(
            "Error: The 'GOOGLE_DRIVE_PARENT_FOLDER_ID' environment variable is not set."
        )
        sys.exit(1)

    drive_service, creds = authenticate_drive()
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
            # Collect all upload tasks
            tasks = []
            collect_upload_tasks(
                drive_service,
                source_folder,
                qubit_folder_id,
                tasks,
                prune=args.prune,
            )

            if tasks:
                print(
                    f"\nProcessing {len(tasks)} file(s) (upload/update/delete) using {args.jobs} threads..."
                )
                # Execute uploads in parallel using joblib
                Parallel(n_jobs=args.jobs, backend="threading")(
                    delayed(process_upload_task)(creds, task) for task in tasks
                )
                print(f"Operation complete for {qubit_name}.")
            else:
                print(f"No files to upload for {qubit_name}.")


if __name__ == "__main__":
    main()
