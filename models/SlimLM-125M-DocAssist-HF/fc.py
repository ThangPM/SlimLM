from torch import nn
from .layers_registry import fcs
fcs.register('torch', func=nn.Linear)
try:
    import transformer_engine.pytorch as te
    fcs.register('te', func=te.Linear)
except:
    pass