import unittest

from pages import parse_pages


class PageTests(unittest.TestCase):
    def test_inclusive_sorted_unique(self):
        self.assertEqual(parse_pages("4, 2-4, 1, 2"), [1, 2, 3, 4])

    def test_singleton_range_and_whitespace(self):
        self.assertEqual(parse_pages(" 3 - 3 , 1 "), [1, 3])

    def test_reject_invalid_items(self):
        for value in ("", "1,", ",1", "1,,2", "0", "-1", "3-2", "1-0", "1.5", "x", "1-2-3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_pages(value)

    def test_cardinality_limit(self):
        self.assertEqual(len(parse_pages("1-10000")), 10000)
        with self.assertRaises(ValueError):
            parse_pages("1-10001")

    def test_duplicates_do_not_count_against_limit(self):
        self.assertEqual(len(parse_pages("1-10000,1-10000")), 10000)


if __name__ == "__main__":
    unittest.main()
