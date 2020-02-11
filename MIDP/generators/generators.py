# Generator template
from .multi_thread_queue_generator import MultiThreadQueueGenerator # NOQA

# Partition generator
from .block_generator import BlockGenerator                         # NOQA
from .block_sampler import BlockSampler                             # NOQA
from .patch_generator import PatchGenerator                         # NOQA
PG = [
    'PatchGenerator',
    'BlockGenerator',
    'BlockSampler',
]

# Augmentation generator
from .augmentor import Augmentor                                    # NOQA
AG = [
    'Augmentor',
]

# Batch generator
from .batch_generator import BatchGenerator                         # NOQA
BG = [
    'BatchGenerator',
]
