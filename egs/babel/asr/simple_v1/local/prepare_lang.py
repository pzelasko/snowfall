#!/usr/bin/env python
import subprocess

from pathlib import Path

cmd_mono = """
local/prepare_lang.sh \
  --position-dependent-phones false \
  data/dict_mono/{lang} \
  "<unk>" \
  data/local/lang_tmp/{lang} \
  data/lang_mono/{lang}
"""

for lang in Path('data/dict_mono').glob('*'):
    print(' '.join(cmd_mono.format(lang=lang.name).replace('\\', '').split()))
    subprocess.run(cmd_mono.format(lang=lang.name), shell=True, text=True)

cmd_multi = """
local/prepare_lang.sh \
  --position-dependent-phones false \
  data/dict_multi \
  "<unk>" \
  data/local/lang_tmp/{lang} \
  data/lang_multi
"""

subprocess.run(cmd_multi, shell=True, text=True)
