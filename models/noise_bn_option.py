from enum import Enum
from .dar_bn import dar_bn
from .aux_bn import aux_bn

from enum import Enum

class NoiseBnOption(Enum):
    STANDARD = 1;
    DARBN = 2;
    AUXBN = 3;
    
def run_noise_bn(x, noise_bn_option, natural_bn, noise_bn=None, noise_mask=None):
    if noise_bn_option == NoiseBnOption.STANDARD:
        return natural_bn(x)
    if noise_bn_option == NoiseBnOption.DARBN:
        return dar_bn(natural_bn, x, noise_mask)
    if noise_bn_option == NoiseBnOption.AUXBN:
        return aux_bn(natural_bn, noise_bn, x, noise_mask)