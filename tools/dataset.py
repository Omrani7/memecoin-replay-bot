"""Load a kit's frozen replay window (dataset/manifest.json + dataset/events/*.ndjson.gz) for tools/replay.py.

The events are read in file order (the collector sorted them by slot, mint, chain_seq), reduced to the
universe the way the harness does (sim/universe.ts, grader/harnesses.ts universeDataset), and grouped
into one batch per slot. A pickle cache next to the output keeps reloads fast.
"""

import gzip
import json
import os
import pickle

from replay import universe_ok


class Dataset:
    def __init__(self, batches, end_slot, slot_time, manifest):
        #: [(slot, [event dicts])] for every slot carrying universe records, in order.
        self.batches = batches
        self.end_slot = end_slot
        #: slot -> block_time of its first record that carries one.
        self.slot_time = slot_time
        self.manifest = manifest

    @property
    def n_events(self):
        return sum(len(ev) for _, ev in self.batches)


def load_events(dataset_dir):
    manifest = json.load(open(os.path.join(dataset_dir, "manifest.json"), encoding="utf-8"))
    events_dir = os.path.join(dataset_dir, "events")
    events = []
    for name in sorted(os.listdir(events_dir)):
        if not name.endswith(".ndjson.gz"):
            continue
        with gzip.open(os.path.join(events_dir, name), "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
    return manifest, events


def build(manifest, events, end_slot=None):
    sample = manifest["universe"]["sample"]
    excluded = {x["mint"] if isinstance(x, dict) else x for x in manifest.get("excluded_mints", [])}
    members = set()
    batches = []
    slot_time = {}
    for e in events:
        bt = e.get("block_time")
        if bt is not None and e["slot"] not in slot_time:
            slot_time[e["slot"]] = bt
        if e["event"] == "CreateEvent" and universe_ok(e, sample, excluded):
            members.add(e["mint"])
        if e["mint"] not in members:
            continue
        if batches and batches[-1][0] == e["slot"]:
            batches[-1][1].append(e)
        else:
            batches.append((e["slot"], [e]))
    last = events[-1]["slot"] if events else 0
    if end_slot is None:
        end_slot = manifest.get("window", {}).get("end_slot", last)
    return Dataset(batches, end_slot, slot_time, manifest)


def load_dataset(dataset_dir, cache=None):
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    manifest, events = load_events(dataset_dir)
    ds = build(manifest, events)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)
    return ds


def slice_dataset(ds, start_slot, end_slot):
    """The batches with start_slot <= slot <= end_slot, as a window of its own (end_slot its end).

    Coins created before start_slot are left out entirely (their CreateEvent is not in the slice), so a
    slice is a fresh replay of the coins launched inside it, as a real window would be.
    """
    created = set()
    out = []
    for slot, evs in ds.batches:
        if slot < start_slot or slot > end_slot:
            continue
        keep = []
        for e in evs:
            if e["event"] == "CreateEvent":
                created.add(e["mint"])
            if e["mint"] in created:
                keep.append(e)
        if keep:
            out.append((slot, keep))
    return Dataset(out, end_slot, ds.slot_time, ds.manifest)
