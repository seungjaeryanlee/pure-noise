import inspect
from torchvision import transforms
    

TRASNFORMS_NAME_TO_CLASS = {k: v for k, v in inspect.getmembers(transforms) if inspect.isclass(v)}


def initialize_transforms(transform_repr):
    return eval(transform_repr, TRASNFORMS_NAME_TO_CLASS)
