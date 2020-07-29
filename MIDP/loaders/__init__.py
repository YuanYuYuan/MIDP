from .parsing_loader import ParsingLoader
from .nifti_loader import NIfTILoader
from .nrrd_loader import NRRDLoader
from .cyb_loader import CybLoader
from .msd_loader import MSDLoader
from .abcs_loader import ABCSLoader

LOADERS = {
    'NIfTILoader': NIfTILoader,
    'ParsingLoader': ParsingLoader,
    'NRRDLoader': NRRDLoader,
    'CybLoader': CybLoader,
    'MSDLoader': MSDLoader,
    'ABCSLoader': ABCSLoader,
}


class DataLoader:

    def __new__(self, name, **kwargs):
        return LOADERS[name](**kwargs)
