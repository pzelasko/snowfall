#!/usr/bin/env python
import subprocess

from pathlib import Path

from snowfall.bash import ParallelBash

CMD_MONO = """
local/prepare_lang.sh \
  --position-dependent-phones false \
  data/dict_mono/{lang} \
  "<unk>" \
  data/local/lang_tmp/{lang} \
  data/lang_mono/{lang}
"""

bash = ParallelBash()
dict_root = Path('data/dict_mono')
lang_root = Path('data/lang_mono')
for lang in dict_root.glob('*'):
    print(' '.join(CMD_MONO.format(lang=lang.name).replace('\\', '').split()))
    lang_dir = lang_root / lang.name
    lang_dir.mkdir(exist_ok=True, parents=True)
    bash.run(
        cmd=CMD_MONO.format(lang=lang.name),
        log_path=lang_dir / 'prepare_lang.log'
    )
bash.join()

CMD_MULTI = """
local/prepare_lang.sh \
  --position-dependent-phones false \
  data/dict_multi \
  "<unk>" \
  data/local/lang_multi_tmp/ \
  data/lang_multi
"""

bash.run(CMD_MULTI)
bash.join()
