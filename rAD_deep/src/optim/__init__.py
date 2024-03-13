#----------------------------------------------------------------------------
# © 2024 – UMONS
#
# Created By  : Sukanya Patra
# Created Date: 11-Mar-2024
# version ='1.0'
# ---------------------------------------------------------------------------

from .rAD_trainer import rADTrainer

def get_method(method_name: str):

    available_methods = {
        'rad': rADTrainer,
    }

    return available_methods[method_name]