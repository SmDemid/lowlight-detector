from .clahe import CLAHEEnhancer
from .gamma import GammaEnhancer
from .msrcr import MSRCREnhancer
from .bilateral import BilateralEnhancer
from .zero_dce import ZeroDCEEnhancer

def get_enhancer(name: str):
    enhancers = {
        'clahe': CLAHEEnhancer,
        'gamma': GammaEnhancer,
        'msrcr': MSRCREnhancer,
        'bilateral': BilateralEnhancer,
        'zero_dce': ZeroDCEEnhancer
    }
    if name not in enhancers:
        raise ValueError(f"Unknown enhancer: {name}")
    return enhancers[name]()