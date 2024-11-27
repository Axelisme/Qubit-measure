import os

from flask import Flask, request

ROOT_DIR = r"C:\Users\SQC\Desktop\QICK\Database"


def is_allowed_file(filename: str):
    allowed_ls = ["hdf5", "h5"]
    return "." in filename and filename.split(".")[-1].lower() in allowed_ls


def save_file(file):
    filepath = file.filename

    # convert to windows path
    filepath = filepath.replace("/", "\\")

    # check root directory
    if "Database\\" not in filepath:
        return "Cannot find root directory in given path", 400

    # remove path before root directory
    filepath = filepath.split("Database\\")[1]

    # save file
    filepath = os.path.join(ROOT_DIR, filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file.save(filepath)

    return f"{filepath} uploaded successfully", 200


app = Flask(__name__)


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400
    if not is_allowed_file(file.filename):
        return "Invalid file format", 400
    if file:
        return save_file(file)


if __name__ == "__main__":
    # localPC_ip check by cmd > ipconfig
    # localPC_ip = "192.168.10.252"
    localPC_ip = "100.76.229.37"
    port = 4999
    app.run(host=localPC_ip, port=port)
