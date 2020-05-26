from .data_generator import DataGenerator
import numpy as np
from typing import List


class Reverter:

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
        self.revert = PG.revert
        self.batch_size = BG.batch_size
        self.revertible = ['match', 'total', 'prediction']
        self.ROIs = data_gen.struct['DL'].ROIs

    def on_batches(self, batch_list: List[dict], output_threshold=0.35):

        assert len(batch_list) * self.batch_size >= sum(self.partition), \
            (len(batch_list) * self.batch_size, sum(self.partition))

        len_queue = 0
        batch_idx = 0
        for (data_idx, partition_per_data) in zip(
            self.data_list,
            self.partition
        ):
            while len_queue < partition_per_data:
                if len_queue == 0:
                    queue = {
                        key: batch_list[batch_idx][key] for key
                        in batch_list[batch_idx] if key in self.revertible
                    }
                else:
                    for key in queue:
                        queue[key] = np.concatenate(
                            (queue[key], batch_list[batch_idx][key]),
                            axis=0
                        )
                batch_idx += 1
                len_queue += self.batch_size

            output = {'idx': data_idx}
            for key in queue:
                if key == 'match':
                    assert 'total' in queue
                    match = np.sum(queue['match'][:partition_per_data], axis=0)[1:]
                    total = np.sum(queue['total'][:partition_per_data], axis=0)[1:]
                    dice_score = 2 * match / total
                    dice_score = dice_score.astype(float)
                    output['score'] = {
                        roi: score for roi, score
                        in zip(self.ROIs, dice_score)
                    }

                elif key == 'prediction':
                    output['prediction'] = self.revert(
                        data_idx,
                        queue['prediction'][:partition_per_data],
                        output_threshold=output_threshold
                    )
                else:
                    assert key in self.revertible
                queue[key] = queue[key][partition_per_data:]
            len_queue -= partition_per_data

            yield output
