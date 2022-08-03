import torch

import colossalai
from colossalai.core import global_context as gpc
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer
from colossalai.logging import disable_existing_loggers, get_dist_logger

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from baobei.autoregressive import AutoregressiveWrapper
from baobei.baobei import baobei_model  

def Trainer():
    assert torch.cuda.is_available()
    assert hasattr(gpc.config, "epochs"), "Please provide epochs in your configuration"
    assert hasattr(gpc.config, "lr"), "Please provide LEARNING_RATE in your configuration"
    assert hasattr(gpc.config, "gradient_accumulation"), "Please provide gradient_accumulation in your configuration"
    assert hasattr(gpc.config, "clip_grad_norm"), "Please provide clip_grad_norm in your configuration"
    assert hasattr(gpc.config, "seed"), "Please provide seed in your configuration"

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

    # Colossal logger
    logger = get_dist_logger()
    logger.info("Initialized environment", ranks=[0])

    model = baobei_model()
    model = AutoregressiveWrapper(model)

    # build dataloaders
    train_dataloader, eval_dataloader = build_dataloaders()

    # Loss Function

    class loss_function(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x_inp, x_labels):
            x_inp, x_labels = x_inp[:, :-1], x_labels[:, 1:]
            loss = F.cross_entropy(rearrange(x_inp, "b c n -> b n c"), x_labels)
            return loss

    loss_fn = loss_function()

    # optimizer function

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr = gpc.config.lr,
        weight_decay=gpc.config.WEIGHT_DECAY
    )

    # initialze model, optimizer, criterion, and data loaders

    engine, train_dataloader, _, _ = colossalai.initialize(
        model,
        optimizer,
        loss_fn,
        train_dataloader = train_dataloader
    )

    def batch_data_process_func(batch_data):
        data = batch_data["input_ids"]
        labels = batch_data["labels"]
        return data, labels

    # Time session with ColossalAI
    timer = MultiTimer()

    # trainer
    trainer = Trainer(
        engine = engine,
        timer =  timer,
        logger = logger
    )

    hook_list = [
        hooks.LogMetricByStepHook(),
        hooks.LossHook(),
        hooks.LogMetricByEpochHook(logger)
    ]

    trainer.fit(
        train_dataloader = train_dataloader,
        epochs = gpc.config.epochs,
        hooks = hook_list,
        display_progress = True
    )