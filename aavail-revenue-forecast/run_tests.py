import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py",
                                                top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.testsRun == 0:
        print("no tests were collected")
        return 1
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
