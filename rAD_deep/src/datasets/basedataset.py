#----------------------------------------------------------------------------
# © 2024 – UMONS
#
# Created By  : Sukanya Patra
# Created Date: 11-Mar-2024
# version ='1.0'
# ---------------------------------------------------------------------------

class BaseDataset():

    def __init__(self):
        pass

    def get_prior(self):
        raise NotImplementedError


    def loaders(self, batch_size: int, shuffle=True, num_workers: int = 0):
        raise NotImplementedError