from .data_generator import DataGenerator
import numpy as np
from typing import List


class Reconstructor:

    def __init__(self, data_gen: DataGenerator):
        PG = data_gen.struct['PG']
        BG = data_gen.struct['BG']

        # ensure the order
        if PG.n_workers > 1:
            assert PG.ordered
        assert BG.n_workers == 1
        if 'AG' in data_gen.struct:
            assert data_gen.struct['AG'].n_workers == 1

        self.data_list = PG.data_list
        self.partition = PG.partition
        self.restore = PG.restore

    def on_batches(self, batch_list: List[dict]):

        assert len(batch_list) > 0
        len_queue = 0
        batch_size = None
        for key in batch_list[0]:
            assert key in ['match', 'total', 'prediction']
            if batch_size is None:
                batch_size = len(batch_list[0][key])
            else:
                assert batch_size == len(batch_list[0][key])
        assert batch_size > 0

        for (data_idx, partition_per_data) in zip(
            self.data_list,
            self.partition
        ):
            while len_queue < partition_per_data:
                batch = batch_list.pop(0)
                if len_queue == 0:
                    queue: dict = batch
                else:
                    for key in batch:
                        queue[key] = np.concatenate(
                            (queue[key], batch[key]),
                            axis=0
                        )
                len_queue += batch_size

            output = {'idx': data_idx}
            if 'match' in queue or 'total' in queue:
                assert 'match' in queue
                assert 'total' in queue
                match = queue['match'][:partition_per_data]
                total = queue['total'][:partition_per_data]
                output['score'] = np.sum(
                    2 * match / total,
                    axis=1
                )

            if 'prediction' in queue:
                output['prediction'] = self.restore(
                    data_idx,
                    queue['prediction'][:partition_per_data]
                )

            for key in queue:
                queue[key] = queue[key][partition_per_data:]
            len_queue -= partition_per_data

            yield output
