import torch

import colossalai
from colossalai.core import global_context as gpc
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer
from colossalai.logging import disable_existing_loggers, get_dist_logger

def Trainer():
    assert torch.cuda.is_available()
    disable_existing_loggers()

    parser = colossalai.get_default_parser()

    parser.add_argument(
        '--use_trainer',
        action='store_true',
        help='whether to use trainer'
    )

    args = parser.parse_args()

    colossalai.launch_from_torch(
        config='',
        seed=0
    )