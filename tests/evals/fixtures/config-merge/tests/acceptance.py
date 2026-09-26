import copy
import unittest

from config_merge import merge_config


class MergeTests(unittest.TestCase):
    def test_recursive_merge(self):
        self.assertEqual(
            merge_config({"db": {"host": "local", "port": 123}}, {"db": {"port": 456}}),
            {"db": {"host": "local", "port": 456}},
        )

    def test_falsey_values_and_type_changes(self):
        base = {"x": 1, "y": True, "z": {"nested": 2}, "a": [1, 2], "b": 2}
        override = {"x": 0, "y": False, "z": None, "a": [3], "b": {"c": 4}}
        self.assertEqual(merge_config(base, override), override)

    def test_no_input_mutation_or_shared_descendants(self):
        base = {"base": {"values": [1]}, "common": {"keep": [2]}}
        override = {"override": {"values": [3]}, "common": {"new": [4]}}
        originals = copy.deepcopy((base, override))
        result = merge_config(base, override)
        self.assertEqual((base, override), originals)
        result["base"]["values"].append(9)
        result["override"]["values"].append(9)
        result["common"]["keep"].append(9)
        result["common"]["new"].append(9)
        self.assertEqual((base, override), originals)


if __name__ == "__main__":
    unittest.main()
