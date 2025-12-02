import argparse
import io
import os
import sys
import threading
from datetime import datetime, timezone

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from joblib import Parallel, delayed

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_FILE = "token.json"

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


def find_folder_id(service, name, parent_id):
    """Find a folder by name within a parent folder."""
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
            print(f"Found remote folder '{name}' (ID: {files[0].get('id')}).")
            return files[0].get("id")
        else:
            print(f"Remote folder '{name}' not found.")
            return None
    except HttpError as e:
        print(f"An error occurred while finding folder '{name}': {e}")
        return None


def list_items_in_folder(service, folder_id):
    """Lists all files and folders in a specific Google Drive folder."""
    items = []
    page_token = None
    try:
        while True:
            response = (
                service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed=false",
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            items.extend(response.get("files", []))
            page_token = response.get("nextPageToken", None)
            if page_token is None:
                break
    except HttpError as e:
        print(f"Warning: Could not list items in folder {folder_id}: {e}")
    return items


def download_file(service, file_id, local_path):
    """Downloads a file from Google Drive."""
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            # print(f"Download {int(status.progress() * 100)}%.")

        with open(local_path, "wb") as f:
            f.write(fh.getbuffer())
        print(f"    Downloaded: {local_path}")
    except HttpError as e:
        print(f"    Error downloading file ID '{file_id}': {e}")


def collect_download_tasks(service, remote_folder_id, local_folder_path, tasks_list):
    """Recursively traverses a Google Drive folder and collects download tasks.

    Args:
        service: Google Drive API service instance.
        remote_folder_id: The ID of the remote folder to traverse.
        local_folder_path: The local path to download files to.
        tasks_list: A list to append download tasks to. Each task is a tuple of
                    (file_id, local_path, remote_timestamp, name).
    """
    if not os.path.exists(local_folder_path):
        os.makedirs(local_folder_path)
        print(f"Created local directory: {local_folder_path}")

    items = list_items_in_folder(service, remote_folder_id)

    for item in items:
        name = item.get("name")
        item_id = item.get("id")
        mime_type = item.get("mimeType")
        modified_time_str = item.get("modifiedTime")
        remote_datetime = parse_drive_timestamp(modified_time_str)

        local_item_path = os.path.join(local_folder_path, name)

        if mime_type == "application/vnd.google-apps.folder":
            # Recursive call for subfolders
            print(f"  Entering subfolder: {name}")
            collect_download_tasks(service, item_id, local_item_path, tasks_list)
        else:
            # It's a file - check if we need to download it
            if os.path.exists(local_item_path):
                local_mtime = os.path.getmtime(local_item_path)
                local_datetime = datetime.fromtimestamp(local_mtime, tz=timezone.utc)

                # Check if local file is newer or same as remote file
                if local_datetime >= remote_datetime:
                    print(
                        f"  - Skipping '{name}' (local is up-to-date: {local_datetime} vs remote: {remote_datetime})."
                    )
                    continue
                else:
                    print(f"  - Queuing update for '{name}' (remote is newer).")
            else:
                print(f"  - Queuing new file '{name}' for download...")

            # Add task to the list
            remote_timestamp = remote_datetime.timestamp()
            tasks_list.append((item_id, local_item_path, remote_timestamp, name))


def process_download_task(creds, task):
    """Process a single download task using a thread-local service.

    Args:
        creds: Google OAuth2 credentials for creating thread-local services.
        task: A tuple of (file_id, local_path, remote_timestamp, name).
    """
    file_id, local_path, remote_timestamp, name = task

    # Get or create thread-local service
    service = get_thread_local_service(creds)

    # Download the file
    download_file(service, file_id, local_path)

    # Set local file modification time to match remote
    # Note: os.utime expects timestamp in seconds
    try:
        os.utime(local_path, (remote_timestamp, remote_timestamp))
    except OSError as e:
        print(f"    Warning: Could not set modification time for '{name}': {e}")


def main():
    """Main function to parse arguments and trigger the download."""
    parser = argparse.ArgumentParser(
        description="Download qubit measurement results from a specified Google Drive folder."
    )
    parser.add_argument(
        "qubit_names",
        nargs="+",
        help="The names of the qubits (e.g., Si001 Si002). These correspond to folders inside the 'result/' directory.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=4,
        help="Number of parallel download threads (default: 4).",
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

        # Find the remote folder for this qubit
        qubit_folder_id = find_folder_id(drive_service, qubit_name, parent_folder_id)

        if qubit_folder_id:
            local_target_folder = os.path.join(project_root, "result", qubit_name)
            print(
                f"Collecting download tasks from Drive folder '{qubit_name}' to '{local_target_folder}'..."
            )

            # Collect all download tasks
            tasks = []
            collect_download_tasks(
                drive_service, qubit_folder_id, local_target_folder, tasks
            )

            if tasks:
                print(
                    f"\nDownloading {len(tasks)} file(s) using {args.jobs} threads..."
                )
                # Execute downloads in parallel using joblib
                Parallel(n_jobs=args.jobs, backend="threading")(
                    delayed(process_download_task)(creds, task) for task in tasks
                )
                print(f"Download complete for {qubit_name}.")
            else:
                print(f"No files to download for {qubit_name}.")
        else:
            print(f"Skipping download for {qubit_name} as remote folder was not found.")


if __name__ == "__main__":
    main()
