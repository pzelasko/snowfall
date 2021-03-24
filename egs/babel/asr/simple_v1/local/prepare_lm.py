#!/usr/bin/env python
import logging
import subprocess
from pathlib import Path

from lhotse import load_manifest
from snowfall.common import setup_logger

omit_words = frozenset((
    '<laugh>'
    '<noise>',
    '<silence>',
    '<v-noise>'
))

LM_CMD = """lmplz -o 3 --text exp/data/{lang}/lm_train_text --arpa exp/data/{lang}/arpa.3g.txt --skip_symbols"""

GFST_CMD = """python3 -m kaldilm \
    --read-symbol-table="data/lang_mono/{lang}/words.txt" \
    --disambig-symbol='#0' \
    --max-order=3 \
    exp/data/{lang}/arpa.3g.txt >exp/data/{lang}/G.fst.txt
"""

setup_logger('log/prepare_lm')

root = Path('exp/data/')
lang_dirs = [p for p in root.glob('*') if p.is_dir() and p.name != 'log' and p.name != 'multilingual']

for d in lang_dirs:
    lang = d.name
    if Path(f'exp/data/{lang}/G.fst.txt').is_file():
        logging.info(f'Skipping {lang} - G.fst.txt already exists.')
        continue
    logging.info(f'Processing language {lang} (directory: {d})')
    # Read Lhotse supervisions, filter out silence regions, remove special non-lexical tokens,
    # and write the sentences to a text file for LM training.
    logging.info(f'Preparing LM training text.')
    sups = load_manifest(d / f'supervisions_{lang}_train.json').filter(lambda s: s.text != '<silence>')
    with (d / 'lm_train_text').open('w') as f:
        for s in sups:
            text = ' '.join(w for w in s.text.split() if w not in omit_words)
            if text:
                print(s.text, file=f)
    # Run KenLM n-gram training
    logging.info(f'Training KenLM n-gram model.')
    subprocess.run(LM_CMD.format(lang=lang), text=True, shell=True, check=True)
    # Create G.fst.txt for K2 decoding
    logging.info(f'Compiling G.fst.txt with kaldilm.')
    subprocess.run(GFST_CMD.format(lang=lang), text=True, shell=True, check=True)
