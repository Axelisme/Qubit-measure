import unittest

import numpy as np
from single_shot import singleshot_analysis


class TestSingleShot(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from tkinter.ttk import Style

        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("Accept the plot?")
        self.root.geometry("800x600")
        self.style = Style(self.root)
        self.style.configure("TButton", font=("Arial", 25))

    def tearDown(self):
        self.root.destroy()

    def generate_data(self, seed=0, n=5000):
        # Generate two dimensional binary classification gaussian data
        np.random.seed(seed)
        x1 = np.random.randn(n)
        y1 = np.random.randn(n)

        x2 = np.random.randn(n) + 0.3
        y2 = np.random.randn(n) + 0.3

        return x1, y1, x2, y2

    def check_result(self, fid, threshold, angle):
        self.assertTrue(isinstance(fid, float))
        self.assertTrue(isinstance(threshold, float))
        self.assertTrue(isinstance(angle, float))

    def ask_user(self, name: str):
        # use tkinter to ask user
        from tkinter import messagebox

        # show accept/reject botton
        if not messagebox.askyesno(name, "Accept the plot?"):
            raise AssertionError("User rejected the plot")

    def test_fit_by_center(self):
        x1, y1, x2, y2 = self.generate_data()

        fid, threshold, angle = singleshot_analysis(
            (x1, x2), (y1, y2), plot=True, backend="center"
        )

        self.check_result(fid, threshold, angle)
        self.ask_user("center")

    def test_fit_by_regression(self):
        x1, y1, x2, y2 = self.generate_data()

        fid, threshold, angle = singleshot_analysis(
            (x1, x2), (y1, y2), plot=True, backend="regression"
        )

        self.check_result(fid, threshold, angle)
        self.ask_user("regression")


if __name__ == "__main__":
    unittest.main()
