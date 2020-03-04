from .parsing_loader import ParsingLoader
from .nifti_loader import NIfTILoader
from .nrrd_loader import NRRDLoader

LOADERS = {
    'NIfTILoader': NIfTILoader,
    'ParsingLoader': ParsingLoader,
    'NRRDLoader': NRRDLoader,
}


class DataLoader:

    def __new__(self, name, **kwargs):
        return LOADERS[name](**kwargs)
