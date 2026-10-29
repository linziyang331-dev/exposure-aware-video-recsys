"""Time and history rules shared by preparation and leakage regressions."""
from collections import Counter, deque
from dataclasses import dataclass
from itertools import groupby


@dataclass(frozen=True)
class DataProtocol:
    train_start_ms: int
    train_end_ms: int
    validation_end_ms: int
    test_end_ms: int
    max_history: int = 50

    def __post_init__(self):
        if not self.train_start_ms < self.train_end_ms < self.validation_end_ms < self.test_end_ms:
            raise ValueError('Split boundaries must be strictly increasing')
        if self.max_history <= 0:
            raise ValueError('max_history must be positive')


def split_timestamp(time_ms, protocol):
    if not protocol.train_start_ms <= time_ms < protocol.test_end_ms:
        raise ValueError(f'Timestamp outside configured calendar window: {time_ms}')
    if time_ms < protocol.train_end_ms:
        return 'train'
    return 'validation' if time_ms < protocol.validation_end_ms else 'test'


def learn_train_vocabulary(events, protocol):
    """Use standard TRAIN exposures only; raw ID 0 remains a legitimate entity."""
    users, items, popularity = set(), set(), Counter()
    for row in events:
        if row['source'] != 'standard' or not protocol.train_start_ms <= row['time_ms'] < protocol.train_end_ms:
            continue
        users.add(int(row['user_id'])); items.add(int(row['video_id']))
        if row['is_click'] == 1:
            popularity[int(row['video_id'])] += 1
    return ({value: index+1 for index, value in enumerate(sorted(users))},
            {value: index+1 for index, value in enumerate(sorted(items))}, popularity)


def iter_examples(events, protocol, user_map, item_map):
    """Yield bounded-history rows from events sorted by (user_id, time_ms, event_id).

    Only standard, positive, train-vocabulary items update history. Prior held-out
    standard feedback is allowed in rolling evaluation; random outcomes never
    update history. All events sharing a timestamp see the same earlier history.
    """
    history = deque(maxlen=protocol.max_history)
    previous_key, previous_user = None, None
    for (user, time_ms), group in groupby(events, key=lambda row: (row['user_id'], row['time_ms'])):
        if user != previous_user:
            history.clear()
        indices, times = [pair[0] for pair in history], [pair[1] for pair in history]
        pending = deque(maxlen=protocol.max_history)
        for row in group:
            key = (user, time_ms, row['event_id'])
            if previous_key is not None and key < previous_key:
                raise ValueError('Events must be sorted by user, time and event ID')
            previous_key = key
            split = split_timestamp(time_ms, protocol)
            if row['source'] not in {'standard', 'random'}:
                raise ValueError(f'Unknown source: {row["source"]}')
            if row['source'] == 'random' and split == 'train':
                raise ValueError('random exposure rows cannot enter train')
            item = item_map.get(int(row['video_id']), 0)
            user_index = user_map.get(int(user), 0)
            yield {'event_id': int(row['event_id']), 'user_id': int(user),
                   'video_id': int(row['video_id']), 'user_index': user_index, 'item_index': item,
                   'time_ms': int(time_ms), 'label': int(row['is_click']),
                   'long_view': int(row['long_view']), 'tab': int(row['tab']),
                   'source': row['source'], 'split': split,
                   'user_seen_in_train': bool(user_index), 'item_seen_in_train': bool(item),
                   'history_item_indices': indices, 'history_time_ms': times,
                   'history_length': len(indices)}
            if row['source'] == 'standard' and row['is_click'] == 1 and item:
                pending.append((item, int(time_ms)))
        history.extend(pending)
        previous_user = user
