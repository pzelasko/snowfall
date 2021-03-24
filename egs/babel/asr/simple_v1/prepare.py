#!/usr/bin/env python3
# Copyright (c)  2021  John's Hopkins University (authors: Piotr Żelasko)
# Apache 2.0

import logging
import os
import re
import sys
import torch
from pathlib import Path

from lhotse import CutSet, combine, load_manifest
from lhotse.recipes import prepare_musan, prepare_single_babel_language
from lhotse.recipes.babel import BABELCODE2LANG
from snowfall.common import setup_logger

# Mapping from directory names to the language names that we will
# encounter in the manifests.
BABEL_MAP = {
    name.lower().replace('-', ''): name
    for name in BABELCODE2LANG.values()
}

# Torch's multithreaded behavior needs to be disabled or it wastes a lot of CPU and
# slow things down.  Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def locate_corpus(*corpus_dirs) -> Path:
    for d in corpus_dirs:
        if os.path.exists(d):
            return Path(d)
    logging.error("Please add the BABEL corpus location on your system to `corpus_dirs`.")
    sys.exit(1)


def main():
    output_dir = Path('exp/data')
    setup_logger(f'{output_dir}/log/log-prepare', log_level='info')

    corpus_dir = locate_corpus(
        Path('/export/common/data/corpora/babel_CLSP'),
    )
    logging.info(f'Found Babel at {corpus_dir}')
    musan_dir = locate_corpus(
        Path('/export/corpora5/JHU/musan'),
        Path('/export/common/data/corpora/MUSAN/musan'),
        Path('/root/fangjun/data/musan'),
    )
    logging.info(f'Found MUSAN at {musan_dir}')

    # For other user's reference:
    # There is no standard directory layout for BABEL, and there's a good chance this recipe
    # will not work out-of-the-box for you.
    # For your convenience in adapting it, I am providing the value of "babel_dirs"
    # variable, so that you can set it yourself to match the distribution on your system.
    # >>> print('babel_dirs =', [str(d) for d in babel_dirs])
    # ... babel_dirs = ['/export/common/data/corpora/babel_CLSP/101-cantonese', '/export/common/data/corpora/babel_CLSP/102-assamese', '/export/common/data/corpora/babel_CLSP/103-bengali', '/export/common/data/corpora/babel_CLSP/104-pashto', '/export/common/data/corpora/babel_CLSP/105-turkish', '/export/common/data/corpora/babel_CLSP/106-tagalog', '/export/common/data/corpora/babel_CLSP/107-vietnamese', '/export/common/data/corpora/babel_CLSP/201-haitian', '/export/common/data/corpora/babel_CLSP/202-swahili', '/export/common/data/corpora/babel_CLSP/203-lao', '/export/common/data/corpora/babel_CLSP/204-tamil', '/export/common/data/corpora/babel_CLSP/205-kurmanji', '/export/common/data/corpora/babel_CLSP/206-zulu', '/export/common/data/corpora/babel_CLSP/207-tokpisin', '/export/common/data/corpora/babel_CLSP/301-cebuano', '/export/common/data/corpora/babel_CLSP/302-kazakh', '/export/common/data/corpora/babel_CLSP/303-telugu', '/export/common/data/corpora/babel_CLSP/304-lithuanian', '/export/common/data/corpora/babel_CLSP/305-guarani', '/export/common/data/corpora/babel_CLSP/306-igbo', '/export/common/data/corpora/babel_CLSP/307-amharic', '/export/common/data/corpora/babel_CLSP/401-mongolian', '/export/common/data/corpora/babel_CLSP/402-javanese', '/export/common/data/corpora/babel_CLSP/403-dholuo', '/export/common/data/corpora/babel_CLSP/404-georgian']
    babel_pattern = re.compile(r'\d\d\d-.+')
    babel_dirs = [d for d in corpus_dir.glob('*') if babel_pattern.match(d.name) and d.is_dir()]
    babel_langs = [BABEL_MAP[''.join(d.name.split('-')[1:])] for d in babel_dirs]
    logging.info(f'Found the following BABEL languages: {", ".join(babel_langs)}')

    babel_manifests = {}
    for lang, lang_dir in zip(babel_langs, babel_dirs):
        if (output_dir / lang / f'recordings_{lang}_train.json').exists():
            babel_manifests[lang] = {
                split: {
                    'recordings': load_manifest(output_dir / lang / f'recordings_{lang}_{split}.json'),
                    'supervisions': load_manifest(output_dir / lang / f'supervisions_{lang}_{split}.json')
                }
                for split in ('train', 'dev')  # we will skip eval for now, 'eval')
            }
            logging.info(f'Skipping {lang} - already prepared!')
            continue
        if lang == 'Swahili':
            lang_dir = Path('/export/common/data/corpora/babel_CLSP/202-swahili/IARPA-babel202b-v1.0d-build')
        if lang == 'Cebuano':
            lang_dir = Path('/export/common/data/corpora/babel_CLSP/301-cebuano/IARPA-babel301b-v2.0b-build')
        if lang == 'Kazakh':
            lang_dir = Path('/export/common/data/corpora/babel_CLSP/302-kazakh/IARPA-babel302b-v1.0a-build')
        if lang == 'Lithuanian':
            lang_dir = Path('/export/common/data/corpora/babel_CLSP/304-lithuanian/IARPA-babel304b-v1.0b-build')
        if lang == 'Amharic':
            lang_dir = Path('/export/common/data/corpora/babel_CLSP/307-amharic/IARPA-babel307b-v1.0b-build')
        if lang == 'Javanese':
            lang_dir = Path('/export/common/data/corpora/babel_CLSP/402-javanese/IARPA-babel402b-v1.0b-build')
        if lang == 'Dholuo':
            lang_dir = Path('/export/common/data/corpora/babel_CLSP/403-dholuo/IARPA-babel403b-v1.0b-build')
        logging.info(f'Preparing {lang} from "{lang_dir}"')
        babel_manifests[lang] = prepare_single_babel_language(
            corpus_dir=lang_dir,
            output_dir=output_dir / lang,
            no_eval_ok=True,
        )

    logging.info('Musan manifest preparation')
    musan_manifests = prepare_musan(
        corpus_dir=musan_dir,
        output_dir=output_dir,
        parts=('music', 'speech', 'noise')
    )

    logging.info('Creating Babel monolingual cuts')
    babel_cuts = {}
    for lang, lang_manifests in babel_manifests.items():
        babel_cuts[lang] = {}
        for split, manifests in lang_manifests.items():
            cuts_path = output_dir / lang / f'cuts_{split}.json'
            if not cuts_path.is_file():
                logging.info(f'\t- {lang} : {split}')
                babel_cuts[lang][split] = CutSet.from_manifests(**manifests)
                babel_cuts[lang][split].to_json(cuts_path)
            else:
                babel_cuts[lang][split] = load_manifest(cuts_path)

    logging.info('Creating Babel multilingual cuts')
    (output_dir / 'multilingual').mkdir(parents=True, exist_ok=True)
    for split in ('train', 'dev', 'eval'):
        cuts_path = output_dir / 'multilingual' / f'cuts_{split}.json.gz'
        if not cuts_path.is_file():
            # Below is equivalent to cuts1 + cuts2 + cuts3 + ... for each split
            multilingual_cuts = sum((
                cuts
                for all_splits in babel_cuts.values()
                for name, cuts in all_splits.items()
                if name == split
            ), CutSet({}))
            multilingual_cuts.to_json(cuts_path)

    musan_cuts_path = output_dir / 'cuts_musan.json'
    if not musan_cuts_path.is_file():
        # create chunks of Musan with duration 5 - 10 seconds
        musan_cuts = CutSet.from_manifests(
            recordings=combine(
                part['recordings'] for part in musan_manifests.values()
            ).resample(8000)
        ).cut_into_windows(10.0).filter(lambda c: c.duration > 5)
        musan_cuts.to_json(musan_cuts_path)


if __name__ == '__main__':
    main()
