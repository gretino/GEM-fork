# filip/data/sampler.py

import random
from collections import defaultdict
from torch.utils.data import Sampler


class UniqueReportBatchSampler(Sampler):
    """Batch sampler that ensures every mini-batch contains unique report texts.

    Prevents in-batch duplicate report strings (e.g. machine-generated 'sinus rhythm')
    from creating false negatives in contrastive loss calculations.
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group sample indices by normalized report text
        self.report_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            record = dataset.records[idx]
            r_text = record.get("report_text", "")
            norm_r = " ".join(r_text.strip().lower().split())
            if not norm_r:
                norm_r = f"__empty_{idx}__"
            self.report_to_indices[norm_r].append(idx)

        # Precompute total valid batches per epoch
        self.total_samples = len(dataset)
        print(
            f"[UniqueReportBatchSampler] Loaded {len(self.report_to_indices)} unique report groups "
            f"across {self.total_samples} total samples (Batch size = {batch_size})."
        )

    def __iter__(self):
        # Shallow copy index lists for iteration
        available_reports = {
            r: list(indices) for r, indices in self.report_to_indices.items()
        }
        if self.shuffle:
            for r in available_reports:
                random.shuffle(available_reports[r])

        active_report_keys = list(available_reports.keys())
        if self.shuffle:
            random.shuffle(active_report_keys)

        while len(active_report_keys) >= self.batch_size:
            if self.shuffle:
                selected_keys = random.sample(active_report_keys, self.batch_size)
            else:
                selected_keys = active_report_keys[:self.batch_size]

            batch = []
            keys_to_remove = []

            for key in selected_keys:
                idx = available_reports[key].pop()
                batch.append(idx)
                if len(available_reports[key]) == 0:
                    keys_to_remove.append(key)

            yield batch

            for key in keys_to_remove:
                active_report_keys.remove(key)

    def __len__(self):
        # Estimate number of full unique batches per epoch
        return self.total_samples // self.batch_size
