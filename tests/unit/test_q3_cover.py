import unittest

from dproblem.q3.cover import minimum_candidate_cover


class CoverTests(unittest.TestCase):
    def test_minimum_cover(self):
        masks = {"a": 0b011, "b": 0b110, "c": 0b001}
        covers = minimum_candidate_cover(masks, 0b111, maximum_candidates=2)
        self.assertIn(("a", "b"), covers)
        self.assertTrue(all(len(row) == 2 for row in covers))

    def test_no_two_candidate_cover(self):
        masks = {"a": 0b001, "b": 0b010, "c": 0b100}
        self.assertEqual(minimum_candidate_cover(masks, 0b111, 2), [])


if __name__ == "__main__":
    unittest.main()
