from enum import Enum
from .dar_bn import dar_bn
from .aux_bn import aux_bn

class NoiseBnRunner(Enum):
    STANDARD = 1;
    DARBN = 2;
    AUXBN = 3;
    
    def run(self, x, noise_mask=None, natural_bn=None, noise_bn=None):
        if isinstance(self, NoiseBnRunner.STANDARD):
            return natural_bn(x)
        if isinstance(self, NoiseBnRunner.DARBN):
            return dar_bn(natural_bn, x, noise_mask)
        if isinstance(self, NoiseBnRunner.AUXBN):
            return aux_bn(natural_bn, noise_bn, x, noise_mask)