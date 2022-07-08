import unittest
from typing import List


def is_increasing(s: List[float]):
    return all(x < y for x, y in zip(s, s[1:]))


class SimpleTest(unittest.TestCase):
    def setUp(self):
        self.dataset = [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_dummy(self):
        self.assertEqual(is_increasing(self.dataset), True)


if __name__ == "__main__":
    unittest.main()
