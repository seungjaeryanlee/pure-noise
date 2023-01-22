import inspect
from torchvision import transforms
from .custom_transforms import Cutout

TRASNFORMS_NAME_TO_CLASS = {k: v for k, v in inspect.getmembers(transforms) if inspect.isclass(v)}
CUSTOM_TRASNFORMS_NAME_TO_CLASS = {'Cutout': Cutout}


def initialize_transforms(transform_reprs: list):
    transform_list = [eval(transform_repr, TRASNFORMS_NAME_TO_CLASS | CUSTOM_TRASNFORMS_NAME_TO_CLASS) for transform_repr in transform_reprs]

    return transforms.Compose(transform_list)
