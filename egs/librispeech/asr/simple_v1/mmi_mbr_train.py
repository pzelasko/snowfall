#!/usr/bin/env python3
# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
#                                                   Haowen Qiu
#                                                   Fangjun Kuang)
# Apache 2.0

import k2
import k2.sparse
import logging
import numpy as np
import os
import torch
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from torch import nn
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple

from lhotse import CutSet
from lhotse.dataset import CutConcatenate, CutMix, K2SpeechRecognitionDataset, SingleCutSampler
from lhotse.utils import fix_random_seed
from snowfall.common import describe
from snowfall.common import load_checkpoint, save_checkpoint
from snowfall.common import save_training_info
from snowfall.common import setup_logger
from snowfall.models import AcousticModel
from snowfall.models.tdnn_lstm import TdnnLstm1b
from snowfall.training.mmi_graph import create_bigram_phone_lm
from snowfall.training.mmi_graph import get_phone_symbols
from snowfall.training.mmi_mbr_graph import MmiMbrTrainingGraphCompiler

den_scale = 1.0


def get_tot_objf_and_num_frames(tot_scores: torch.Tensor,
                                frames_per_seq: torch.Tensor,
                                finite_indexes: Optional[torch.Tensor]=None
                                ) -> Tuple[float, int, int]:
    ''' Figures out the total score(log-prob) over all successful supervision segments
    (i.e. those for which the total score wasn't -infinity), and the corresponding
    number of frames of neural net output
         Args:
            tot_scores: a Torch tensor of shape (num_segments,) containing total scores
                       from forward-backward. If `finite_indexes` is not None, `-inf` entries
                       from tot_scores should have already been removed. If `finite_indexes`
                       is None, then `tot_scores` may contain `-inf` elements.
        frames_per_seq: a Torch tensor of shape (num_segments,) containing the number of
                       frames for each segment
        finite_indexes: a tensor containing successful segment indexes, e.g. [0 1 3 4 5]
        Returns:
             Returns a tuple of 3 scalar tensors:  (tot_score, ok_frames, all_frames)
        where ok_frames is the frames for successful (finite) segments, and
       all_frames is the frames for all segments (finite or not).
    '''
    if finite_indexes is None:
        finite_indexes = torch.isfinite(tot_scores)
        tot_scores = tot_scores[finite_indexes]
    if False:
        bad_indexes = ~finite_indexes
        if bad_indexes.shape[0] > 0:
            print("Bad indexes: ", bad_indexes, ", bad lengths: ",
                  frames_per_seq[bad_indexes], " vs. max length ",
                  torch.max(frames_per_seq), ", avg ",
                  (torch.sum(frames_per_seq) / frames_per_seq.numel()))
    # print("finite_indexes = ", finite_indexes, ", tot_scores = ", tot_scores)
    ok_frames = frames_per_seq[finite_indexes].sum()
    all_frames = frames_per_seq.sum()
    return (tot_scores.sum(), ok_frames, all_frames)


def get_loss(batch: Dict,
             model: AcousticModel,
             P: k2.Fsa,
             device: torch.device,
             graph_compiler: MmiMbrTrainingGraphCompiler,
             is_training: bool,
             optimizer: Optional[torch.optim.Optimizer] = None):
    assert P.device == device
    feature = batch['inputs']
    supervisions = batch['supervisions']
    supervision_segments = torch.stack(
        (supervisions['sequence_idx'],
         torch.floor_divide(supervisions['start_frame'],
                            model.subsampling_factor),
         torch.floor_divide(supervisions['num_frames'],
                            model.subsampling_factor)), 1).to(torch.int32)
    indices = torch.argsort(supervision_segments[:, 2], descending=True)
    supervision_segments = supervision_segments[indices]

    texts = supervisions['text']
    texts = [texts[idx] for idx in indices]
    assert feature.ndim == 3
    # print(supervision_segments[:, 1] + supervision_segments[:, 2])

    feature = feature.to(device)
    # at entry, feature is [N, T, C]
    feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]
    if is_training:
        nnet_output = model(feature)
    else:
        with torch.no_grad():
            nnet_output = model(feature)

    # nnet_output is [N, C, T]
    nnet_output = nnet_output.permute(0, 2, 1)  # now nnet_output is [N, T, C]

    if is_training:
        num_graph, den_graph, decoding_graph = graph_compiler.compile(texts, P)
    else:
        with torch.no_grad():
            num_graph, den_graph, decoding_graph = graph_compiler.compile(
                texts, P)

    assert num_graph.requires_grad == is_training
    assert den_graph.requires_grad is False
    assert decoding_graph.requires_grad is False
    assert len(decoding_graph.shape) == 2 or decoding_graph.shape == (1, None, None)

    num_graph = num_graph.to(device)
    den_graph = den_graph.to(device)

    decoding_graph = decoding_graph.to(device)

    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)
    assert nnet_output.device == device

    num_lats = k2.intersect_dense(num_graph,
                                  dense_fsa_vec,
                                  10.0,
                                  seqframe_idx_name='seqframe_idx')


    mbr_lats = k2.intersect_dense_pruned(decoding_graph,
                                         dense_fsa_vec,
                                         20.0,
                                         7.0,
                                         30,
                                         10000,
                                         seqframe_idx_name='seqframe_idx')

    if True:
        # WARNING: the else branch is not working at present (the total loss is not stable)
        den_lats = k2.intersect_dense(den_graph, dense_fsa_vec, 10.0)
    else:
        # in this case, we can remove den_graph
        den_lats = mbr_lats

    num_tot_scores = num_lats.get_tot_scores(log_semiring=True,
                                             use_double_scores=True)

    den_tot_scores = den_lats.get_tot_scores(log_semiring=True,
                                             use_double_scores=True)

    if id(den_lats) == id(mbr_lats):
        # Some entries in den_tot_scores may be -inf.
        # The corresponding sequences are discarded/ignored.
        finite_indexes = torch.isfinite(den_tot_scores)
        den_tot_scores = den_tot_scores[finite_indexes]
        num_tot_scores = num_tot_scores[finite_indexes]
    else:
        finite_indexes =  None

    tot_scores = num_tot_scores - den_scale * den_tot_scores

    (tot_score, tot_frames,
     all_frames) = get_tot_objf_and_num_frames(tot_scores,
                                               supervision_segments[:, 2],
                                               finite_indexes)


    num_rows = dense_fsa_vec.scores.shape[0]
    num_cols = dense_fsa_vec.scores.shape[1] - 1
    mbr_num_sparse = k2.create_sparse(rows=num_lats.seqframe_idx,
                                      cols=num_lats.phones,
                                      values=num_lats.get_arc_post(True,
                                                                   True).exp(),
                                      size=(num_rows, num_cols),
                                      min_col_index=0)

    mbr_den_sparse = k2.create_sparse(rows=mbr_lats.seqframe_idx,
                                      cols=mbr_lats.phones,
                                      values=mbr_lats.get_arc_post(True,
                                                                   True).exp(),
                                      size=(num_rows, num_cols),
                                      min_col_index=0)
    # NOTE: Due to limited support of PyTorch's autograd for sparse tensors,
    # we cannot use (mbr_num_sparse - mbr_den_sparse) here
    #
    # The following works only for torch >= 1.7.0
    mbr_loss = torch.sparse.sum(
        k2.sparse.abs((mbr_num_sparse + (-mbr_den_sparse)).coalesce()))

    mmi_loss = -tot_score

    total_loss = mmi_loss + mbr_loss

    if is_training:
        optimizer.zero_grad()
        total_loss.backward()
        clip_grad_value_(model.parameters(), 5.0)
        optimizer.step()

    ans = (
        mmi_loss.detach().cpu().item(),
        mbr_loss.detach().cpu().item(),
        tot_frames.cpu().item(),
        all_frames.cpu().item(),
    )
    return ans


def get_validation_loss(dataloader: torch.utils.data.DataLoader,
                        model: AcousticModel,
                        P: k2.Fsa,
                        device: torch.device,
                        graph_compiler: MmiMbrTrainingGraphCompiler):
    total_loss = 0.
    total_mmi_loss = 0.
    total_mbr_loss = 0.
    total_frames = 0.  # for display only
    total_all_frames = 0.  # all frames including those seqs that failed.

    model.eval()

    for batch_idx, batch in enumerate(dataloader):
        mmi_loss, mbr_loss, frames, all_frames = get_loss(
            batch=batch,
            model=model,
            P=P,
            device=device,
            graph_compiler=graph_compiler,
            is_training=False)
        cur_loss = mmi_loss + mbr_loss
        total_loss += cur_loss
        total_mmi_loss += mmi_loss
        total_mbr_loss += mbr_loss
        total_frames += frames
        total_all_frames += all_frames

    return total_loss, total_mmi_loss, total_mbr_loss, total_frames, total_all_frames


def train_one_epoch(dataloader: torch.utils.data.DataLoader,
                    valid_dataloader: torch.utils.data.DataLoader,
                    model: AcousticModel, P: k2.Fsa,
                    device: torch.device,
                    graph_compiler: MmiMbrTrainingGraphCompiler,
                    optimizer: torch.optim.Optimizer,
                    current_epoch: int,
                    tb_writer: SummaryWriter,
                    num_epochs: int,
                    global_batch_idx_train: int):
    total_loss, total_mmi_loss, total_mbr_loss, total_frames, total_all_frames = 0., 0., 0., 0., 0.
    valid_average_loss = float('inf')
    time_waiting_for_batch = 0
    prev_timestamp = datetime.now()

    model.train()
    for batch_idx, batch in enumerate(dataloader):
        global_batch_idx_train += 1
        timestamp = datetime.now()
        time_waiting_for_batch += (timestamp - prev_timestamp).total_seconds()

        P.set_scores_stochastic_(model.P_scores)
        assert P.requires_grad is True

        curr_batch_mmi_loss, curr_batch_mbr_loss, curr_batch_frames, curr_batch_all_frames = get_loss(
            batch=batch,
            model=model,
            P=P,
            device=device,
            graph_compiler=graph_compiler,
            is_training=True,
            optimizer=optimizer)

        total_mmi_loss += curr_batch_mmi_loss
        total_mbr_loss += curr_batch_mbr_loss
        curr_batch_loss = curr_batch_mmi_loss + curr_batch_mbr_loss
        total_loss += curr_batch_loss
        total_frames += curr_batch_frames
        total_all_frames += curr_batch_all_frames

        if batch_idx % 10 == 0:
            logging.info(
                'batch {}, epoch {}/{} '
                'global average loss: {:.6f}, '
                'global average mmi loss: {:.6f}, '
                'global average mbr loss: {:.6f} over {} '
                'frames ({:.1f}% kept), '
                'current batch average loss: {:.6f}, '
                'current batch average mmi loss: {:.6f}, '
                'current batch average mbr loss: {:.6f} '
                'over {} frames ({:.1f}% kept) '
                'avg time waiting for batch {:.3f}s'.format(
                    batch_idx, current_epoch, num_epochs,
                    total_loss / total_frames,
                    total_mmi_loss / total_frames,
                    total_mbr_loss / total_frames, total_frames,
                    100.0 * total_frames / total_all_frames,
                    curr_batch_loss / (curr_batch_frames + 0.001),
                    curr_batch_mmi_loss / (curr_batch_frames + 0.001),
                    curr_batch_mbr_loss / (curr_batch_frames + 0.001),
                    curr_batch_frames,
                    100.0 * curr_batch_frames / curr_batch_all_frames,
                    time_waiting_for_batch / max(1, batch_idx)))

            tb_writer.add_scalar('train/global_average_loss',
                                 total_loss / total_frames, global_batch_idx_train)

            tb_writer.add_scalar('train/global_average_mmi_loss',
                                 total_mmi_loss / total_frames, global_batch_idx_train)

            tb_writer.add_scalar('train/global_average_mbr_loss',
                                 total_mbr_loss / total_frames, global_batch_idx_train)

            tb_writer.add_scalar('train/current_batch_average_loss',
                                 curr_batch_loss / (curr_batch_frames + 0.001),
                                 global_batch_idx_train)

            tb_writer.add_scalar('train/current_batch_average_mmi_loss',
                                 curr_batch_mmi_loss / (curr_batch_frames + 0.001),
                                 global_batch_idx_train)

            tb_writer.add_scalar('train/current_batch_average_mbr_loss',
                                 curr_batch_mbr_loss / (curr_batch_frames + 0.001),
                                 global_batch_idx_train)
            # if batch_idx >= 10:
            #    print("Exiting early to get profile info")
            #    sys.exit(0)

        if batch_idx > 0 and batch_idx % 3000 == 0:
            total_valid_loss, total_valid_mmi_loss, total_valid_mbr_loss, \
                    total_valid_frames, total_valid_all_frames = get_validation_loss(
                dataloader=valid_dataloader,
                model=model,
                P=P,
                device=device,
                graph_compiler=graph_compiler)
            valid_average_loss = total_valid_loss / total_valid_frames
            model.train()
            logging.info(
                'Validation average loss: {:.6f}, '
                'Validation average mmi loss: {:.6f}, '
                'Validation average mbr loss: {:.6f} '
                'over {} frames ({:.1f}% kept)'
                    .format(total_valid_loss / total_valid_frames,
                            total_valid_mmi_loss / total_valid_frames,
                            total_valid_mbr_loss / total_valid_frames,
                            total_valid_frames,
                            100.0 * total_valid_frames / total_valid_all_frames))

            tb_writer.add_scalar('train/global_valid_average_loss',
                             total_valid_loss / total_valid_frames,
                             global_batch_idx_train)

            tb_writer.add_scalar('train/global_valid_average_mmi_loss',
                             total_valid_mmi_loss / total_valid_frames,
                             global_batch_idx_train)

            tb_writer.add_scalar('train/global_valid_average_mbr_loss',
                             total_valid_mbr_loss / total_valid_frames,
                             global_batch_idx_train)

        prev_timestamp = datetime.now()
    return total_loss / total_frames, valid_average_loss, global_batch_idx_train


def main():
    fix_random_seed(42)

    start_epoch = 0
    num_epochs = 10
    use_adam = True

    exp_dir = f'exp-lstm-adam-mmi-mbr-musan'
    setup_logger('{}/log/log-train'.format(exp_dir))
    tb_writer = SummaryWriter(log_dir=f'{exp_dir}/tensorboard')

    if not torch.cuda.is_available():
        logging.warn('No GPU detected!')
        logging.warn('USE CPU (very slow)!')
        device = torch.device('cpu')
    else:
        logging.info('Use GPU')
        device_id = 0
        device = torch.device('cuda', device_id)

    # load L, G, symbol_table
    lang_dir = Path('data/lang_nosp')
    phone_symbol_table = k2.SymbolTable.from_file(lang_dir / 'phones.txt')
    word_symbol_table = k2.SymbolTable.from_file(lang_dir / 'words.txt')

    logging.info("Loading L.fst")
    if (lang_dir / 'Linv.pt').exists():
        logging.info('Loading precompiled L')
        L_inv = k2.Fsa.from_dict(torch.load(lang_dir / 'Linv.pt'))
    else:
        logging.info('Compiling L')
        with open(lang_dir / 'L.fst.txt') as f:
            L = k2.Fsa.from_openfst(f.read(), acceptor=False)
            L_inv = k2.arc_sort(L.invert_())
            torch.save(L_inv.as_dict(), lang_dir / 'Linv.pt')

    logging.info("Loading L_disambig.fst")
    if (lang_dir / 'L_disambig.pt').exists():
        logging.info('Loading precompiled L_disambig')
        L_disambig = k2.Fsa.from_dict(torch.load(lang_dir / 'L_disambig.pt'))
    else:
        logging.info('Compiling L_disambig')
        with open(lang_dir / 'L_disambig.fst.txt') as f:
            L_disambig = k2.Fsa.from_openfst(f.read(), acceptor=False)
            L_disambig = k2.arc_sort(L_disambig)
            torch.save(L_disambig.as_dict(), lang_dir / 'L_disambig.pt')

    logging.info("Loading G.fst")
    if (lang_dir / 'G_uni.pt').exists():
        logging.info('Loading precompiled G')
        G = k2.Fsa.from_dict(torch.load(lang_dir / 'G_uni.pt'))
    else:
        logging.info('Compiling G')
        with open(lang_dir / 'G_uni.fst.txt') as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
            G = k2.arc_sort(G)
            torch.save(G.as_dict(), lang_dir / 'G_uni.pt')

    graph_compiler = MmiMbrTrainingGraphCompiler(
        L_inv=L_inv,
        L_disambig=L_disambig,
        G=G,
        device=device,
        phones=phone_symbol_table,
        words=word_symbol_table
    )
    phone_ids = get_phone_symbols(phone_symbol_table)
    P = create_bigram_phone_lm(phone_ids)
    P.scores = torch.zeros_like(P.scores)

    # load dataset
    feature_dir = Path('exp/data')
    logging.info("About to get train cuts")
    cuts_train = CutSet.from_json(feature_dir /
                                  'cuts_train-clean-100.json.gz')
    logging.info("About to get dev cuts")
    cuts_dev = CutSet.from_json(feature_dir / 'cuts_dev-clean.json.gz')
    logging.info("About to get Musan cuts")
    cuts_musan = CutSet.from_json(feature_dir / 'cuts_musan.json.gz')

    logging.info("About to create train dataset")
    train = K2SpeechRecognitionDataset(
        cuts_train,
        cut_transforms=[
            CutConcatenate(),
            CutMix(
                cuts=cuts_musan,
                prob=0.5,
                snr=(10, 20)
            )
        ]
    )
    train_sampler = SingleCutSampler(
        cuts_train,
        max_frames=30000,
        shuffle=True,
    )
    logging.info("About to create train dataloader")
    train_dl = torch.utils.data.DataLoader(
        train,
        sampler=train_sampler,
        batch_size=None,
        num_workers=4
    )
    logging.info("About to create dev dataset")
    validate = K2SpeechRecognitionDataset(cuts_dev)
    valid_sampler = SingleCutSampler(cuts_dev, max_frames=60000)
    logging.info("About to create dev dataloader")
    valid_dl = torch.utils.data.DataLoader(
        validate,
        sampler=valid_sampler,
        batch_size=None,
        num_workers=1
    )

    logging.info("About to create model")
    model = TdnnLstm1b(num_features=40,
                       num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
                       subsampling_factor=3)
    model.P_scores = nn.Parameter(P.scores.clone(), requires_grad=True)

    model.to(device)
    describe(model)

    P = P.to(device)

    if use_adam:
        learning_rate = 1e-3
        weight_decay = 5e-4
        optimizer = optim.AdamW(model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay)
        # Equivalent to the following in the epoch loop:
        #  if epoch > 6:
        #      curr_learning_rate *= 0.8
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda ep: 1.0 if ep < 7 else 0.8 ** (ep - 6)
        )
    else:
        learning_rate = 5e-5
        weight_decay = 1e-5
        momentum = 0.9
        lr_schedule_gamma = 0.7
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=lr_schedule_gamma
        )

    best_objf = np.inf
    best_valid_objf = np.inf
    best_epoch = start_epoch
    best_model_path = os.path.join(exp_dir, 'best_model.pt')
    best_epoch_info_filename = os.path.join(exp_dir, 'best-epoch-info')
    global_batch_idx_train = 0  # for logging only

    if start_epoch > 0:
        model_path = os.path.join(exp_dir, 'epoch-{}.pt'.format(start_epoch - 1))
        ckpt = load_checkpoint(filename=model_path, model=model, optimizer=optimizer, scheduler=lr_scheduler)
        best_objf = ckpt['objf']
        best_valid_objf = ckpt['valid_objf']
        global_batch_idx_train = ckpt['global_batch_idx_train']
        logging.info(f"epoch = {ckpt['epoch']}, objf = {best_objf}, valid_objf = {best_valid_objf}")

    for epoch in range(start_epoch, num_epochs):
        # LR scheduler can hold multiple learning rates for multiple parameter groups;
        # For now we report just the first LR which we assume concerns most of the parameters.
        train_sampler.set_epoch(epoch)
        curr_learning_rate = lr_scheduler.get_last_lr()[0]
        tb_writer.add_scalar('train/learning_rate', curr_learning_rate, global_batch_idx_train)
        tb_writer.add_scalar('train/epoch', epoch, global_batch_idx_train)

        logging.info('epoch {}, learning rate {}'.format(epoch, curr_learning_rate))
        objf, valid_objf, global_batch_idx_train = train_one_epoch(
            dataloader=train_dl,
            valid_dataloader=valid_dl,
            model=model,
            P=P,
            device=device,
            graph_compiler=graph_compiler,
            optimizer=optimizer,
            current_epoch=epoch,
            tb_writer=tb_writer,
            num_epochs=num_epochs,
            global_batch_idx_train=global_batch_idx_train,
        )

        lr_scheduler.step()

        # the lower, the better
        if valid_objf < best_valid_objf:
            best_valid_objf = valid_objf
            best_objf = objf
            best_epoch = epoch
            save_checkpoint(filename=best_model_path,
                            model=model,
                            optimizer=None,
                            scheduler=None,
                            epoch=epoch,
                            learning_rate=curr_learning_rate,
                            objf=objf,
                            valid_objf=valid_objf,
                            global_batch_idx_train=global_batch_idx_train)
            save_training_info(filename=best_epoch_info_filename,
                               model_path=best_model_path,
                               current_epoch=epoch,
                               learning_rate=curr_learning_rate,
                               objf=objf,
                               best_objf=best_objf,
                               valid_objf=valid_objf,
                               best_valid_objf=best_valid_objf,
                               best_epoch=best_epoch)

        # we always save the model for every epoch
        model_path = os.path.join(exp_dir, 'epoch-{}.pt'.format(epoch))
        save_checkpoint(filename=model_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        epoch=epoch,
                        learning_rate=curr_learning_rate,
                        objf=objf,
                        valid_objf=valid_objf,
                        global_batch_idx_train=global_batch_idx_train)
        epoch_info_filename = os.path.join(exp_dir, 'epoch-{}-info'.format(epoch))
        save_training_info(filename=epoch_info_filename,
                           model_path=model_path,
                           current_epoch=epoch,
                           learning_rate=curr_learning_rate,
                           objf=objf,
                           best_objf=best_objf,
                           valid_objf=valid_objf,
                           best_valid_objf=best_valid_objf,
                           best_epoch=best_epoch)

    logging.warning('Done')

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()
