#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
#                                                   Haowen Qiu
#                                                   Fangjun Kuang)
#                2021  University of Chinese Academy of Sciences (author: Han Zhu)
# Apache 2.0

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pytorch_lightning as pl
import torch
from torch import nn

from snowfall.common import describe, str2bool
from snowfall.common import setup_logger
from snowfall.data.librispeech import LibriSpeechAsrDataModule
from snowfall.lexicon import Lexicon
from snowfall.models import AcousticModel
from snowfall.models.conformer import Conformer
from snowfall.models.transformer import Noam, Transformer
from snowfall.objectives import LFMMILoss, encode_supervisions
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler
from snowfall.training.mmi_graph import create_bigram_phone_lm


class LightningWrapper(pl.LightningModule):
    def __init__(self, model: AcousticModel, loss_fn: nn.Module, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss_fn = loss_fn
        self.args = args

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(
            self,
            batch: Dict,
            batch_idx: int,
            is_training: bool = True
    ):
        feature = batch['inputs']
        # at entry, feature is [N, T, C]
        feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]
        assert feature.ndim == 3
        feature = feature.to(self.device)

        supervisions = batch['supervisions']
        supervision_segments, texts = encode_supervisions(supervisions)

        nnet_output, encoder_memory, memory_mask = self.model(feature, supervisions)
        if is_training:
            if self.args.att_rate != 0.0:
                att_loss = self.model.decoder_forward(
                    encoder_memory,
                    memory_mask,
                    supervisions,
                    self.loss_fn.graph_compiler
                )

        # nnet_output is [N, C, T]
        nnet_output = nnet_output.permute(0, 2, 1)  # now nnet_output is [N, T, C]

        mmi_loss, tot_frames, all_frames = self.loss_fn(nnet_output, texts, supervision_segments)

        if is_training:
            # def maybe_log_gradients(tag: str):
            #     if tb_writer is not None and global_batch_idx_train is not None and global_batch_idx_train % 200 == 0:
            #         tb_writer.add_scalars(
            #             tag,
            #             measure_gradient_norms(model, norm='l1'),
            #             global_step=global_batch_idx_train
            #         )

            if self.args.att_rate != 0.0:
                loss = (- (1.0 - self.args.att_rate) * mmi_loss + self.args.att_rate * att_loss) / len(texts)
            else:
                loss = (-mmi_loss) / len(texts)

            self.loss_fn.P.set_scores_stochastic_(self.model.P_scores)
            # loss.backward()
            # if is_update:
            #     maybe_log_gradients('train/grad_norms')
            #     clip_grad_value_(model.parameters(), 5.0)
            #     maybe_log_gradients('train/clipped_grad_norms')
            #     if tb_writer is not None and (global_batch_idx_train // accum_grad) % 200 == 0:
            #         # Once in a time we will perform a more costly diagnostic
            #         # to check the relative parameter change per minibatch.
            #         deltas = optim_step_and_measure_param_change(model, optimizer)
            #         tb_writer.add_scalars(
            #             'train/relative_param_change_per_minibatch',
            #             deltas,
            #             global_step=global_batch_idx_train
            #         )
            #     else:
            #         optimizer.step()
            #     optimizer.zero_grad()

        report_keys = {
            'mmi_loss': mmi_loss
        }
        if is_training:
            report_keys['loss'] = loss
            if self.args.att_rate != 0:
                report_keys['att_loss'] = self.args.att_rate * att_loss
        report_keys.update({
            'tot_frames': tot_frames,
            'all_frames': all_frames
        })

        return report_keys

    def validation_step(self, batch: Dict, batch_idx: int):
        return self.training_step(batch, batch_idx, is_training=False)

    def validation_epoch_end(self, outputs: List[Dict]):
        agg = defaultdict(float)
        for batch in outputs:
            for key, val in batch.items():
                agg[key] += val
        return {
            # For losses, report their average
            **{f'avg_{k}': v / agg['tot_frames'] for k, v in agg.items() if 'loss' in k},
            # For everything else, report as it is
            **{k: v for k, v in agg.items() if 'loss' not in k}
        }

    def test_step(self, batch: Dict, batch_idx: int):
        return self.training_step(batch, batch_idx, is_training=False)

    def configure_optimizers(self):
        optimizer = Noam(self.model.parameters(),
                         model_size=self.args.attention_dim,
                         factor=1.0,
                         warm_step=self.args.warm_step)
        return optimizer


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-type',
        type=str,
        default="conformer",
        choices=["transformer", "conformer"],
        help="Model type.")
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help="Number of traning epochs.")
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help="Number of start epoch.")
    parser.add_argument(
        '--warm-step',
        type=int,
        default=5000,
        help='The number of warm-up steps for Noam optimizer.'
    )
    parser.add_argument(
        '--accum-grad',
        type=int,
        default=1,
        help="Number of gradient accumulation.")
    parser.add_argument(
        '--den-scale',
        type=float,
        default=1.0,
        help="denominator scale in mmi loss.")
    parser.add_argument(
        '--att-rate',
        type=float,
        default=0.0,
        help="Attention loss rate.")
    parser.add_argument(
        '--nhead',
        type=int,
        default=4,
        help="Number of attention heads in transformer.")
    parser.add_argument(
        '--attention-dim',
        type=int,
        default=256,
        help="Number of units in transformer attention layers.")
    parser.add_argument(
        '--tensorboard',
        type=str2bool,
        default=True,
        help='Should various information be logged in tensorboard.'
    )
    return parser


class SetSamplerEpoch(pl.Callback):
    def on_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        trainer.train_dataloader.sampler.set_epoch(trainer.current_epoch)


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    model_type = args.model_type
    start_epoch = args.start_epoch
    att_rate = args.att_rate

    pl.seed_everything(42)
    # fix_random_seed(42)

    exp_dir = Path('exp-' + model_type + '-noam-mmi-att-musan-sa')
    exp_dir.mkdir(exist_ok=True, parents=True)
    setup_logger('{}/log/log-train'.format(exp_dir))
    # tb_writer = SummaryWriter(log_dir=f'{exp_dir}/tensorboard') if args.tensorboard else None

    logging.info("Loading lexicon and symbol tables")
    lang_dir = Path('data/lang_nosp')
    lexicon = Lexicon(lang_dir)

    device_id = 0
    device = torch.device('cuda', device_id)

    graph_compiler = MmiTrainingGraphCompiler(
        lexicon=lexicon,
        device=device,
    )
    phone_ids = lexicon.phone_symbols()
    P = create_bigram_phone_lm(phone_ids)
    P.scores = torch.zeros_like(P.scores)
    P = P.to(device)

    librispeech = LibriSpeechAsrDataModule(args)
    train_dl = librispeech.train_dataloaders()
    valid_dl = librispeech.valid_dataloaders()

    if not torch.cuda.is_available():
        logging.error('No GPU detected!')
        sys.exit(-1)

    logging.info("About to create model")

    if att_rate != 0.0:
        num_decoder_layers = 6
    else:
        num_decoder_layers = 0

    if model_type == "transformer":
        model = Transformer(
            num_features=80,
            nhead=args.nhead,
            d_model=args.attention_dim,
            num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
            subsampling_factor=4,
            num_decoder_layers=num_decoder_layers)
    else:
        model = Conformer(
            num_features=80,
            nhead=args.nhead,
            d_model=args.attention_dim,
            num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
            subsampling_factor=4,
            num_decoder_layers=num_decoder_layers)

    model.P_scores = nn.Parameter(P.scores.clone(), requires_grad=True)

    model.to(device)
    describe(model)

    model_pl = LightningWrapper(
        model,
        loss_fn=LFMMILoss(
            graph_compiler=graph_compiler,
            P=P,
            den_scale=args.den_scale
        ),
        args=args
    )

    trainer = pl.Trainer(
        default_root_dir=exp_dir,
        gradient_clip_val=5.0,
        gpus=[device_id],
        accumulate_grad_batches=args.accum_grad,
        max_epochs=args.num_epochs,
        resume_from_checkpoint=exp_dir / 'last.pt',
        replace_sampler_ddp=False,
        callbacks=[
            SetSamplerEpoch()
        ]
    )
    trainer.fit(
        model_pl,
        train_dataloader=train_dl,
        val_dataloaders=valid_dl,
    )

    logging.warning('Done')


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()
