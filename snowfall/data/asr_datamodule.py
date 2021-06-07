import argparse
import logging
from pathlib import Path
from typing import List, Union

from torch.utils.data import DataLoader

from lhotse import Fbank, FbankConfig, load_manifest
from lhotse import LibrosaFbank, LibrosaFbankConfig
from lhotse.dataset import AudioSamples, BucketingSampler, CutConcatenate, CutMix, K2SpeechRecognitionDataset, \
    RandomizedSmoothing, SingleCutSampler, \
    SpecAugment
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from snowfall.common import str2bool
from snowfall.data.datamodule import DataModule


class AsrDataModule(DataModule):
    """
    DataModule for K2 ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean and test-other).

    It contains all the common data pipeline modules used in ASR experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,
    - on-the-fly feature extraction

    This class should be derived for specific corpora used in ASR tasks.
    """

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        group = parser.add_argument_group(
            title='ASR data related options',
            description='These options are used for the preparation of PyTorch DataLoaders '
                        'from Lhotse CutSet\'s -- they control the effective batch sizes, '
                        'sampling strategies, applied data augmentations, etc.'
        )
        group.add_argument(
            '--feature-dir',
            type=Path,
            default=Path('exp/data'),
            help='Path to directory with train/valid/test cuts.'
        )
        group.add_argument(
            '--max-duration',
            type=int,
            default=500.0,
            help="Maximum pooled recordings duration (seconds) in a single batch.")
        group.add_argument(
            '--bucketing-sampler',
            type=str2bool,
            default=False,
            help='When enabled, the batches will come from buckets of '
                 'similar duration (saves padding frames).')
        group.add_argument(
            '--num-buckets',
            type=int,
            default=30,
            help='The number of buckets for the BucketingSampler'
                 '(you might want to increase it for larger datasets).')
        group.add_argument(
            '--concatenate-cuts',
            type=str2bool,
            default=True,
            help='When enabled, utterances (cuts) will be concatenated '
                 'to minimize the amount of padding.')
        group.add_argument(
            '--duration-factor',
            type=float,
            default=1.0,
            help='Determines the maximum duration of a concatenated cut '
                 'relative to the duration of the longest cut in a batch.')
        group.add_argument(
            '--gap',
            type=float,
            default=1.0,
            help='The amount of padding (in seconds) inserted between concatenated cuts. '
                 'This padding is filled with noise when noise augmentation is used.')
        group.add_argument(
            '--on-the-fly-feats',
            type=str2bool,
            default=False,
            help='When enabled, use on-the-fly cut mixing and feature extraction. '
                 'Will drop existing precomputed feature manifests if available.'
        )
        group.add_argument(
            '--raw-audio',
            type=str2bool,
            default=False,
            help='When enabled, return audio waveforms from the DataLoader.'
        )
        group.add_argument(
            '--use-rand-smooth',
            type=str2bool,
            default=False,
            help='When enabled, use randomized smoothing (additive gaussian noise to the waveform).'
        )
        group.add_argument(
            '--librosa',
            type=str2bool,
            default=False,
            help='When enabled, use LibrosaFbank with sampling rate 22050Hz.'
        )

    def train_dataloaders(self) -> DataLoader:
        logging.info("About to get train cuts")
        cuts_train = self.train_cuts()

        logging.info("About to get Musan cuts")
        cuts_musan = load_manifest(self.args.feature_dir / 'cuts_musan.json.gz')
        if self.args.librosa:
            cuts_train = cuts_train.filter(lambda c: '_sp' not in c.id)
            cuts_train = cuts_train.resample(22050)
            cuts_musan = cuts_musan.resample(22050)

        logging.info("About to create train dataset")
        transforms = [CutMix(cuts=cuts_musan, prob=0.5, snr=(10, 20))]
        if self.args.concatenate_cuts:
            logging.info(f'Using cut concatenation with duration factor '
                         f'{self.args.duration_factor} and gap {self.args.gap}.')
            # Cut concatenation should be the first transform in the list,
            # so that if we e.g. mix noise in, it will fill the gaps between different utterances.
            transforms = [
                             CutConcatenate(
                                 duration_factor=self.args.duration_factor,
                                 gap=self.args.gap
                             )
                         ] + transforms

        smoothing = RandomizedSmoothing(
            sigma=[
                (0, 0.0),
                (1000, 0.001),
                (2000, 0.01),
                (5000, 0.1),
                (10000, 0.2),
                (20000, 0.3),
            ],
            sample_sigma=True,
            p=0.66
        )

        if self.args.raw_audio:
            input_transforms = []
            if self.args.use_rand_smooth:
                input_transforms.append(smoothing)

            train = K2SpeechRecognitionDataset(
                cuts=cuts_train,
                cut_transforms=transforms,
                input_strategy=AudioSamples(),
                input_transforms=input_transforms
            )
        else:
            input_transforms = [
                SpecAugment(num_frame_masks=2, features_mask_size=27, num_feature_masks=2, frames_mask_size=100)
            ]

            if self.args.on_the_fly_feats:
                # NOTE: the PerturbSpeed transform should be added only if we remove it from data prep stage.
                # # Add on-the-fly speed perturbation; since originally it would have increased epoch
                # # size by 3, we will apply prob 2/3 and use 3x more epochs.
                # # Speed perturbation probably should come first before concatenation,
                # # but in principle the transforms order doesn't have to be strict (e.g. could be randomized)
                # Drop feats to be on the safe side.
                cuts_train = cuts_train.drop_features()
                if self.args.librosa:
                    logging.info('Using LibrosaFbank')
                    extractor = LibrosaFbank()
                    #from lhotse.dataset import PerturbSpeed
                    #transforms = [PerturbSpeed(factors=[0.9, 1.1], p=2 / 3)] + transforms
                else:
                    extractor=Fbank(FbankConfig(num_mel_bins=80))
                train = K2SpeechRecognitionDataset(
                    cuts=cuts_train,
                    cut_transforms=transforms,
                    input_strategy=OnTheFlyFeatures(
                        extractor=extractor,
                        wave_transforms=[smoothing] if self.args.use_rand_smooth else []
                    ),
                    input_transforms=input_transforms
                )
            else:
                train = K2SpeechRecognitionDataset(
                    cuts_train,
                    cut_transforms=transforms,
                    input_transforms=input_transforms
                )

        if self.args.bucketing_sampler:
            logging.info('Using BucketingSampler.')
            train_sampler = BucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=True,
                num_buckets=self.args.num_buckets
            )
        else:
            logging.info('Using SingleCutSampler.')
            train_sampler = SingleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=True,
            )
        logging.info("About to create train dataloader")
        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=4,
            persistent_workers=True,
        )
        return train_dl

    def valid_dataloaders(self) -> DataLoader:
        logging.info("About to get dev cuts")
        cuts_valid = self.valid_cuts()
        if self.args.librosa:
            cuts_valid = cuts_valid.resample(22050)

        transforms = []
        if self.args.concatenate_cuts:
            transforms = [CutConcatenate(
                duration_factor=self.args.duration_factor,
                gap=self.args.gap)
                         ] + transforms

        logging.info("About to create dev dataset")
        if self.args.raw_audio:
            validate = K2SpeechRecognitionDataset(
                cuts_valid,
                cut_transforms=transforms,
                input_strategy=AudioSamples(),
            )
        else:
            if self.args.on_the_fly_feats:
                if self.args.librosa:
                    logging.info('Using LibrosaFbank')
                    extractor = LibrosaFbank()
                else:
                    extractor=Fbank(FbankConfig(num_mel_bins=80))
                validate = K2SpeechRecognitionDataset(
                    cuts_valid.drop_features(),
                    cut_transforms=transforms,
                    input_strategy=OnTheFlyFeatures(extractor=extractor)
                )
            else:
                validate = K2SpeechRecognitionDataset(cuts_valid,
                                                      cut_transforms=transforms)
        valid_sampler = SingleCutSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=True,
        )
        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=2,
            persistent_workers=True,
        )
        return valid_dl

    def test_dataloaders(self) -> Union[DataLoader, List[DataLoader]]:
        cuts = self.test_cuts()
        is_list = isinstance(cuts, list)
        test_loaders = []
        if not is_list:
            cuts = [cuts]

        for cuts_test in cuts:
            logging.debug("About to create test dataset")
            if self.args.librosa:
                cuts_test = cuts_test.resample(22050)
            if self.args.raw_audio:
                test = K2SpeechRecognitionDataset(
                    cuts_test,
                    #input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
                    input_strategy=AudioSamples()
                )
            else:
                if self.args.librosa:
                    logging.info('Using LibrosaFbank')
                    extractor = LibrosaFbank()
                else:
                    extractor=Fbank(FbankConfig(num_mel_bins=80))
                test = K2SpeechRecognitionDataset(
                    cuts_test,
                    input_strategy=OnTheFlyFeatures(extractor=extractor)
                )
            sampler = SingleCutSampler(cuts_test, max_duration=self.args.max_duration)
            logging.debug("About to create test dataloader")
            test_dl = DataLoader(test, batch_size=None, sampler=sampler, num_workers=1)
            test_loaders.append(test_dl)

        if is_list:
            return test_loaders
        else:
            return test_loaders[0]
