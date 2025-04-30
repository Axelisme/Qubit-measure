import io
import shutil
import tempfile
import unittest
from pathlib import Path

import script.data_server as data_server


class FakeFile:
    def __init__(self, filename, data):
        self.filename = filename
        self._data = data

    def save(self, dst):
        with open(dst, "wb") as f:
            f.write(self._data)


class TestDataServer(unittest.TestCase):
    def setUp(self):
        # prepare temporary ROOT_DIR and override safe_labber_filepath
        self.test_dir = tempfile.mkdtemp()
        data_server.ROOT_DIR = Path(self.test_dir)
        data_server.safe_labber_filepath = lambda p: p
        data_server.app.testing = True
        self.app = data_server.app
        # push app context for send_file
        self.ctx = self.app.test_request_context()
        self.ctx.push()

    def tearDown(self):
        # pop app context
        self.ctx.pop()
        shutil.rmtree(self.test_dir)

    def test_is_allowed_file(self):
        self.assertTrue(data_server.is_allowed_file("file.hdf5"))
        self.assertTrue(data_server.is_allowed_file("file.H5"))
        self.assertFalse(data_server.is_allowed_file("file.txt"))
        self.assertFalse(data_server.is_allowed_file("no_extension"))

    def test_get_relpath_unix(self):
        self.assertEqual(
            data_server.get_relpath("Database/dir/file.h5"), Path("dir") / "file.h5"
        )
        self.assertEqual(
            data_server.get_relpath("/prefix/Database/dir/sub/file.hdf5"),
            Path("dir") / "sub" / "file.hdf5",
        )

    def test_get_relpath_windows(self):
        path = r"\\server\\Database\\dir\\file.HDF5"
        self.assertEqual(data_server.get_relpath(path), Path("dir") / "file.HDF5")

    def test_save_and_load_file(self):
        # test save_file
        fake = FakeFile("Database/test.hdf5", b"hello")
        msg, code = data_server.save_file(fake)
        self.assertEqual(code, 200)
        saved_path = msg.split()[0]
        self.assertTrue(Path(saved_path).exists())
        # test load_file for missing
        not_msg, not_code = data_server.load_file("Database/not_exist.hdf5")
        self.assertEqual(not_code, 404)
        # test load_file for existing
        fake2 = FakeFile("Database/hey.h5", b"world")
        data_server.save_file(fake2)
        resp = data_server.load_file("Database/hey.h5")
        from flask.wrappers import Response

        self.assertIsInstance(resp, Response)
        self.assertEqual(resp.status_code, 200)

    def test_flask_routes(self):
        client = self.app.test_client()
        # upload route
        data = {"file": (io.BytesIO(b"data123"), "Database/foo.h5")}
        resp = client.post("/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        # download route
        resp2 = client.post("/download", json={"path": "Database/foo.h5"})
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(resp2.data, b"data123")


if __name__ == "__main__":
    unittest.main()
