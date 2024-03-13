#----------------------------------------------------------------------------
# © 2024 – UMONS
#
# Created By  : Sukanya Patra
# Created Date: 11-Mar-2024
# version ='1.0'
# ---------------------------------------------------------------------------

import torch
import click
import logging
import random
import json
import numpy as np

from pathlib import Path
from utils.config import Config
from optim import get_method
from datasets import get_dataset

@click.command()
@click.option('--exp_config', type = click.Path(exists=True), default=None, help='configuration file')
def main(exp_config):
    """
    r-AD, a fully deep method for anomaly detection using PU learning.
    """

    # Get configuration
    config = Config(locals().copy())
    config.load_config(Path(exp_config))
    config.settings['seed'] = 1234 #default seeds for all

    config.settings['log_path'] = config.settings['log_path'] + '/' + config.settings['experiment']
    config.settings['model_path'] = config.settings['model_path'] + '/'

    # Set up logger
    if not Path.exists(Path(config.settings['log_path'])):
        Path.mkdir(Path(config.settings['log_path']), parents=True, exist_ok=True)

    if config.settings['train']:
        log_path = Path(config.settings['log_path']).joinpath('log.txt')
    else:
        log_path = Path(config.settings['log_path']).joinpath('log_test.txt')

    logging.basicConfig(level = logging.INFO,
                        filemode = 'w',
                        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename = log_path)
    logger = logging.getLogger()

    logger.info('Log file is %s.' % (log_path))

    # Set seed
    if config.settings['seed'] != -1:

        # if -1 then keep randomised
        random.seed(config.settings['seed'])
        np.random.seed(config.settings['seed'])
        torch.manual_seed(config.settings['seed'])
        torch.cuda.manual_seed(config.settings['seed'])
        torch.cuda.manual_seed_all(config.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if torch.cuda.is_available() and config.settings['device'] == 'cuda':
        config.settings['device'] = 'cuda'
    else:
        config.settings['device'] = 'cpu'

    logger.info(json.dumps(config.formatted_config(), indent=2))

    dataset = get_dataset(config.settings['dataset'])(config)
    train_set=dataset.train_set
    val_set=dataset.val_set
    test_set=dataset.test_set

    run_method=get_method(config.settings['method'])(config)
    if config.settings["early_stopping"]:
        run_method.train(train_set=train_set,val_set=val_set)
    else:
        run_method.train(train_set)

    run_method.test(test_set=test_set)

if __name__ == '__main__':
    main()
