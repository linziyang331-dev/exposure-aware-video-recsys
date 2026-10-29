import unittest


class DataProtocolTests(unittest.TestCase):
    def protocol(self, length=2):
        from src.data.protocol import DataProtocol
        return DataProtocol(train_start_ms=100, train_end_ms=200, validation_end_ms=300,
                            test_end_ms=400, max_history=length)

    def event(self, number, user, item, time, label=1, source='standard'):
        return {'event_id': number, 'user_id': user, 'video_id': item, 'time_ms': time,
                'is_click': label, 'long_view': label, 'tab': 1, 'source': source}

    def test_calendar_boundary_belongs_to_later_split(self):
        from src.data.protocol import split_timestamp
        config = self.protocol()
        self.assertEqual([split_timestamp(t, config) for t in [100, 199, 200, 299, 300, 399]],
                         ['train', 'train', 'validation', 'validation', 'test', 'test'])
        for t in [99, 400]:
            with self.assertRaises(ValueError):split_timestamp(t, config)

    def test_vocab_and_popularity_use_only_standard_train_exposures(self):
        from src.data.protocol import learn_train_vocabulary
        events = [self.event(1, 0, 0, 110), self.event(2, 9, 7, 120, label=0),
                  self.event(3, 88, 99, 220), self.event(4, 77, 66, 250, source='random')]
        users, items, popularity = learn_train_vocabulary(events, self.protocol())
        self.assertEqual(users, {0: 1, 9: 2})
        self.assertEqual(items, {0: 1, 7: 2})
        self.assertEqual(dict(popularity), {0: 1})

    def test_history_excludes_self_tied_feedback_and_random_outcomes(self):
        from src.data.protocol import iter_examples
        events = [self.event(1, 1, 10, 110), self.event(2, 1, 20, 110),
                  self.event(3, 1, 30, 150, label=0),
                  self.event(4, 1, 40, 210, source='random'),
                  self.event(5, 1, 10, 220), self.event(6, 1, 20, 310)]
        examples = list(iter_examples(events, self.protocol(), {1: 1}, {10: 1, 20: 2, 30: 3, 40: 4}))
        self.assertEqual([row['history_item_indices'] for row in examples],
                         [[], [], [1, 2], [1, 2], [1, 2], [2, 1]])
        self.assertEqual(examples[-1]['history_time_ms'], [110, 220])
        self.assertEqual(examples[-1]['split'], 'test')

    def test_cold_entities_remain_and_map_to_zero_without_new_history(self):
        from src.data.protocol import iter_examples
        events = [self.event(1, 1, 10, 110), self.event(2, 1, 99, 220),
                  self.event(3, 1, 10, 310), self.event(4, 2, 10, 320)]
        rows = list(iter_examples(events, self.protocol(), {1: 1}, {10: 1}))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[1]['item_index'], 0)
        self.assertFalse(rows[1]['item_seen_in_train'])
        self.assertEqual(rows[2]['history_item_indices'], [1])
        self.assertEqual(rows[3]['user_index'], 0)
        self.assertEqual(rows[3]['history_item_indices'], [])

    def test_unsorted_input_is_rejected(self):
        from src.data.protocol import iter_examples
        with self.assertRaisesRegex(ValueError, 'sorted'):
            list(iter_examples([self.event(1, 1, 10, 150), self.event(2, 1, 10, 110)],
                               self.protocol(), {1: 1}, {10: 1}))

    def test_random_train_rows_are_rejected(self):
        from src.data.protocol import iter_examples
        with self.assertRaisesRegex(ValueError, 'random.*train'):
            list(iter_examples([self.event(1, 1, 10, 150, source='random')],
                               self.protocol(), {1: 1}, {10: 1}))

    def test_nonpositive_history_limit_is_rejected(self):
        with self.assertRaises(ValueError):self.protocol(length=0)
