import unittest

from extend_train_plan import OWNER_RANGES, chunked_allocations


class ExtendTrainPlanTests(unittest.TestCase):
    def test_twenty_family_chunks_cover_each_owner_range_once(self):
        allocations = chunked_allocations(20)
        actual = [i for item in allocations for i in range(item['from'], item['to'] + 1)]
        self.assertEqual(actual, list(range(1, 1751)))
        self.assertEqual(len(actual), len(set(actual)))
        for producer, start, end in OWNER_RANGES:
            owned = [item for item in allocations if item['producer'] == producer]
            self.assertEqual(owned[0]['from'], start)
            self.assertEqual(owned[-1]['to'], end)
            self.assertTrue(all(item['to'] - item['from'] + 1 <= 20 for item in owned))


if __name__ == '__main__':
    unittest.main()
