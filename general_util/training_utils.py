import random
import os
from typing import Dict, Union
from fairscale.nn.data_parallel.fully_sharded_data_parallel import FullyShardedDataParallel as FullyShardedDP
from omegaconf import DictConfig, OmegaConf

from general_util.logger import get_child_logger
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

logger = get_child_logger("Training utils")


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).
    Args:
        model (:obj:`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def save_model(model: Union[torch.nn.Module, FullyShardedDP], cfg: DictConfig, output_dir: str):
    # Save model checkpoint.
    if cfg.local_rank != -1:
        state_dict = model.state_dict()
        if cfg.local_rank == 0:
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    # Save tokenizer and training args.
    if cfg.local_rank in [-1, 0]:
        OmegaConf.save(cfg, os.path.join(output_dir, "training_config.yaml"))
        logger.info("Saving model checkpoint to %s", output_dir)


def batch_to_device(batch: Dict[str, torch.Tensor], device):
    batch_on_device = {}
    for k, v in batch.items():
        batch_on_device[k] = v.to(device)
    return batch_on_device


def initialize_optimizer(cfg: DictConfig, model: torch.nn.Module):
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': cfg.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if "optimizer" in cfg and cfg.optimizer == 'lamb':
        if "bit_training" in cfg and cfg.bit_training:
            from bitsandbytes.optim import LAMB8bit

            optimizer = LAMB8bit(optimizer_grouped_parameters,
                                 lr=cfg.learning_rate,
                                 betas=eval(cfg.adam_betas),
                                 eps=cfg.adam_epsilon,
                                 max_unorm=cfg.max_grad_norm)
        else:
            from apex.optimizers.fused_lamb import FusedLAMB

            optimizer = FusedLAMB(optimizer_grouped_parameters,
                                  lr=cfg.learning_rate,
                                  betas=eval(cfg.adam_betas),
                                  eps=cfg.adam_epsilon,
                                  use_nvlamb=(cfg.use_nvlamb if "use_nvlamb" in cfg else False),
                                  max_grad_norm=cfg.max_grad_norm)
    else:
        if "bit_training" in cfg and cfg.bit_training:
            from bitsandbytes.optim import AdamW8bit

            optimizer = AdamW8bit(optimizer_grouped_parameters, lr=cfg.learning_rate, eps=cfg.adam_epsilon, betas=(eval(cfg.adam_betas)))
        else:
            if hasattr(cfg, "multi_tensor") and cfg.multi_tensor:
                from torch.optim._multi_tensor import AdamW
            else:
                from transformers import AdamW

            optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate, eps=cfg.adam_epsilon, betas=(eval(cfg.adam_betas)))

    return optimizer


def note_best_checkpoint(cfg: DictConfig, results: Dict[str, float], sub_path: str):
    metric = results[cfg.prediction_cfg.metric]
    if (not cfg.prediction_cfg.best_result) or (cfg.prediction_cfg.measure > 0 and metric > cfg.prediction_cfg.best_result) or (
            cfg.prediction_cfg.measure < 0 and metric < cfg.prediction_cfg.best_result):
        cfg.prediction_cfg.best_result = metric
        cfg.prediction_cfg.best_checkpoint = sub_path
        return True
    return False


class SummaryWriterHelper:
    def __init__(self, writer: SummaryWriter):
        self.writer = writer

    def __call__(self, batch, step):
        self.writer.add_scalar('node_num', batch['input_emb_index'].size(0), global_step=step)
