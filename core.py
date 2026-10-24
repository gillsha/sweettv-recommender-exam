"""Auditable pure-Python metric, interval join and submission contract.

No competition data or credentials are embedded. Times must be seconds on a
single, verified axis. All intervals use half-open [start, end) semantics.
"""
from bisect import bisect_left, bisect_right
from collections import defaultdict
import csv
import math


def ap_at_k(actual, predicted, k=5):
    """Kaggle-style AP@k; empty truth is 0, duplicates get no second credit."""
    if k < 1:
        raise ValueError('k must be positive')
    actual = set(actual)
    if not actual:
        return 0.0
    seen, hits, total = set(), 0, 0.0
    for rank, item in enumerate(predicted[:k], 1):
        if item in actual and item not in seen:
            hits += 1
            total += hits / rank
        seen.add(item)
    return total / min(k, len(actual))


def map_at_k(truth, predictions, users, k=5):
    """Fixed evaluation cohort, INCLUDING users without relevant validation items."""
    users = list(users)
    if not users or len(users) != len(set(users)):
        raise ValueError('Evaluation users must be unique and nonempty')
    if set(predictions) != set(users):
        raise ValueError('Predictions must cover the exact evaluation cohort')
    return sum(ap_at_k(truth.get(u, ()), predictions[u], k) for u in users) / len(users)


def union_length(intervals):
    """Seconds watched without double counting overlapping view-log intervals."""
    end, total = None, 0
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if end is None or left > end:
            total += right-left
        elif right > end:
            total += right-end
        end = right if end is None else max(end, right)
    return total


class ScheduleIndex:
    """Supports overlapping broadcasts, not just a single as-of match.

    rows: (channel, start_seconds, end_seconds, nonzero_item_id, airing_key).
    Prefix maximum ends allow efficient removal of all broadcasts ending before
    a viewing starts, even when the schedule itself contains overlapping slots.
    """
    def __init__(self, rows):
        by_channel = defaultdict(list)
        for channel, start, end, item, key in rows:
            if end <= start:
                raise ValueError('Broadcast length must be positive')
            if item != 0:
                by_channel[channel].append((start, end, item, key))
        self.channels = {}
        for channel, values in by_channel.items():
            values.sort()
            starts, prefix, largest = [], [], -math.inf
            for start, end, _, _ in values:
                starts.append(start)
                largest=max(largest, end)
                prefix.append(largest)
            self.channels[channel]=(values, starts, prefix)

    def overlaps(self, channel, start, end):
        if end <= start:
            return []
        if channel not in self.channels:
            return []
        values, starts, prefix = self.channels[channel]
        lo=bisect_right(prefix, start)
        hi=bisect_left(starts, end)
        result=[]
        for bstart, bend, item, key in values[lo:hi]:
            left, right=max(start,bstart), min(end,bend)
            if right > left:
                result.append((item,key,left,right,bend-bstart))
        return result


def aggregate_watches(index, views, window_start, window_end, min_seconds):
    """Clip at split BEFORE matching and merge duplicate watches per airing.

    views: (user,channel,start,end). Returns qualified per-airing interactions.
    The duration threshold is a parameter; its final value requires data audit.
    """
    if window_end <= window_start or min_seconds <= 0:
        raise ValueError('Invalid window or threshold')
    segments=defaultdict(list)
    for user,channel,start,end in views:
        left,right=max(start,window_start),min(end,window_end)
        for item,key,a,b,length in index.overlaps(channel,left,right):
            segments[(user,item,key,length)].append((a,b))
    results=[]
    for (user,item,key,length),intervals in segments.items():
        seconds=union_length(intervals)
        if seconds >= min_seconds:
            results.append({'user_id':user,'tv_show_id':item,'airing_key':key,
                            'watch_seconds':seconds,'programme_seconds':length,
                            'watch_fraction':seconds/length})
    return results


def validate_submission(template_path, submission_path, candidates):
    """Use the ACTUAL template's header/order, never an assumed column name."""
    def read(path):
        with open(path,newline='',encoding='utf-8-sig') as stream:
            rows=list(csv.reader(stream))
        return rows[0],rows[1:]
    header,template=read(template_path)
    out_header,rows=read(submission_path)
    if out_header != header or len(header)!=2 or len(rows)!=len(template):
        raise ValueError('Header or row count differs from real template')
    if any(len(r)!=2 for r in rows):
        raise ValueError('Unexpected CSV field count')
    if [r[0] for r in rows] != [r[0] for r in template]:
        raise ValueError('User identifiers/order differ from real template')
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError('Duplicate users')
    allowed=set(map(int,candidates))-{0}
    for user,ranking in rows:
        ids=[int(x) for x in ranking.split()]
        if len(ids)!=5 or len(set(ids))!=5 or not set(ids)<=allowed:
            raise ValueError(f'Invalid ranking for user {user}')
    return {'users':len(rows),'header':header,'five_unique_valid_ids':True}
