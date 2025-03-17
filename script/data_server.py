import os
from os.path import abspath, dirname, join

from flask import Flask, request, send_file

ROOT_DIR = abspath(join(dirname(dirname(__file__)), "Database"))


def is_allowed_file(filename: str):
    allowed_ls = ["hdf5", "h5"]
    return "." in filename and filename.split(".")[-1].lower() in allowed_ls


def safe_filepath(filepath: str):
    filepath = os.path.abspath(filepath)

    def parse_filepath(filepath):
        filename, ext = os.path.splitext(filepath)
        count = filename.split("_")[-1]
        if count.isdigit():
            filename = "_".join(filename.split("_")[:-1])  # remove number
            return filename, int(count), ext
        else:
            return filename, 1, ext

    filename, count, ext = parse_filepath(filepath)

    filepath = filename + f"_{count}" + ext
    while os.path.exists(filepath):
        count += 1
        filepath = filename + f"_{count}" + ext

    return filepath


def save_file(file):
    remote_path = file.filename

    # convert to windows path
    remote_path = remote_path.replace("/", "\\")

    # check root directory
    if "Database\\" not in remote_path:
        return "Cannot find root directory in given path", 400

    # remove path before root directory
    relpath = remote_path.split("Database\\")[1]

    # save file
    filepath = safe_filepath(os.path.join(ROOT_DIR, relpath))
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file.save(filepath)

    return f"{filepath} uploaded successfully", 200


def load_file(remote_path):
    # convert to windows path
    remote_path = remote_path.replace("/", "\\")

    # check root directory
    if "Database\\" not in remote_path:
        return "Cannot find root directory in given path", 400

    # remove path before root directory
    relpath = remote_path.split("Database\\")[1]

    # load file
    filepath = os.path.join(ROOT_DIR, relpath)
    if not os.path.exists(filepath):
        return "File not found", 404

    # send file to client
    return send_file(filepath, as_attachment=True)


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


@app.route("/download", methods=["POST"])
def server2remote():
    filepath = request.json.get("path")
    if not filepath:
        return "No file path", 400

    return load_file(filepath)


if __name__ == "__main__":
    # localPC_ip check by cmd > ipconfig
    localPC_ip = "0.0.0.0"
    port = 4999
    print(f"Save data to {ROOT_DIR}")
    app.run(host=localPC_ip, port=port)
