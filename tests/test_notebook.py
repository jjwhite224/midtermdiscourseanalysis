"""Check that the class walkthrough notebook is valid and safe to commit."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kmeans_midterm_walkthrough.ipynb"


class NotebookStructureTest(unittest.TestCase):
    def test_valid_json_and_python_cells(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        n = 0
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            n += 1
            source = "".join(cell.get("source", []))
            compile(source, f"notebook_cell_{i}", "exec")
            self.assertEqual(cell.get("outputs"), [], "Do not publish raw comment outputs")
            self.assertIsNone(cell.get("execution_count"))
        self.assertGreaterEqual(n, 8)


if __name__ == "__main__":
    unittest.main()
