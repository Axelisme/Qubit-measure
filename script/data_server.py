import argparse
from pathlib import Path

from flask import Flask, request, send_file
from zcu_tools.utils.datasaver import safe_labber_filepath

KEYWORD = "Database"
ROOT_DIR = Path(__file__).resolve().parents[1] / KEYWORD
DEFAULT_IP = "0.0.0.0"
DEFAULT_PORT = 4999


def is_allowed_file(filename: str):
    allowed_ls = ["hdf5", "h5"]
    return "." in filename and filename.split(".")[-1].lower() in allowed_ls


def get_relpath(path_str: str) -> Path:
    # normalize separators to POSIX style, handling both Windows and Unix inputs
    normalized = path_str.replace("\\", "/")
    segments = [seg for seg in normalized.split("/") if seg]
    if KEYWORD in segments:
        idx = segments.index(KEYWORD)
        segments = segments[idx + 1 :]
    return Path(*segments)


def save_file(file):
    # determine destination relative path and full filepath
    rel = get_relpath(file.filename)
    dest = ROOT_DIR / rel
    dest_str = str(dest)
    filepath = safe_labber_filepath(dest_str)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    file.save(filepath)

    return f"{filepath} uploaded successfully", 200


def load_file(remote_path):
    # determine source relative path and full filepath
    rel = get_relpath(remote_path)
    filepath = ROOT_DIR / rel
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return "File not found", 404
    return send_file(str(filepath), as_attachment=True)


app = Flask(__name__)


@app.route("/upload", methods=["POST"])
def remote2server():
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "" or file.filename is None:
        return "No selected file", 400
    if not is_allowed_file(file.filename):
        return "Invalid file format", 400

    if file:
        return save_file(file)
    else:
        return "No file part", 400


@app.route("/download", methods=["POST"])
def server2remote():
    filepath = request.json.get("path")
    if not filepath:
        return "No file path", 400

    return load_file(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR)
    parser.add_argument("--ip", type=str, default=DEFAULT_IP)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    ROOT_DIR = args.root_dir

    # localPC_ip check by cmd > ipconfig
    print(f"Save data to {ROOT_DIR}")
    app.run(host=args.ip, port=args.port)
