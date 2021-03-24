#!/usr/bin/env python
import subprocess
from itertools import chain
from pathlib import Path

BABEL_ROOT = Path("/export/common/data/corpora/babel_CLSP")

BABEL_LEXICON_PATHS = {
    # "101-cantonese/release-babel101b-v0.4c-rc1/conversational/reference_materials/lexicon.txt",
    "Cantonese": "101-cantonese/release-babel101-v0.1d/conversational/reference_materials/lexicon.txt",
    "Assamese": "102-assamese/release-babel102b-v0.4/conversational/reference_materials/lexicon.txt",
    "Bengali": "103-bengali/release-babel103b-v0.3/conversational/reference_materials/lexicon.txt",
    # "104-pashto/release-babel104-0.91-final/conversational/reference_materials/lexicon.txt",
    # "104-pashto/release-babel104b-v04.aY/conversational/reference_materials/lexicon.txt",
    "Pashto": "104-pashto/release-babel104b-v04.bY/conversational/reference_materials/lexicon.txt",
    # "105-turkish/release-babel105-v0.6/conversational/reference_materials/lexicon.txt",
    "Turkish": "105-turkish/release-babel105b-v0.4-rc1/conversational/reference_materials/lexicon.txt",
    # "106-tagalog/release-babel106b-v0.2f/conversational/reference_materials/lexicon.txt",
    # "106-tagalog/release-babel106-v0.5/conversational/reference_materials/lexicon.txt",
    # "106-tagalog/release-babel106b-v0.2g/conversational/reference_materials/lexicon.txt",
    "Tagalog": "106-tagalog/release-current/conversational/reference_materials/lexicon.txt",
    "Vietnamese": "107-vietnamese/release-babel107b-v0.7/conversational/reference_materials/lexicon.txt",
    "Haitian": "201-haitian/release-babel201b-v0.2b/conversational/reference_materials/lexicon.txt",
    "Swahili": "202-swahili/IARPA-babel202b-v1.0d-build/BABEL_OP2_202/conversational/reference_materials/lexicon.txt",
    # "203-lao/release-babel203b-v2.1a/conversational/reference_materials/lexicon.txt",
    "Lao": "203-lao/release-IARPA-babel203b-v3.1a/conversational/reference_materials/lexicon.txt",
    # "204-tamil/release-IARPA-babel204b-v1.1b/conversational/reference_materials/lexicon.txt",
    # "204-tamil/release-IARPA-babel204b-v1.1b/conversational/reference_materials.orig/lexicon.txt",
    "Tamil": "204-tamil/BABEL_OP1_204/conversational/reference_materials/lexicon.txt",
    "Kurmanji": "205-kurmanji/IARPA-babel205b-v1.0a-build/BABEL_OP2_205/conversational/reference_materials/lexicon.txt",
    # "206-zulu/release-babel206b-v0.1d/conversational/reference_materials/lexicon.txt",
    "Zulu": "206-zulu/release-babel206b-v0.1e/conversational/reference_materials/lexicon.txt",
    "Tok-Pisin": "207-tokpisin/IARPA-babel207b-v1.0e-build/BABEL_OP2_207/conversational/reference_materials/lexicon.txt",
    "Cebuano": "301-cebuano/IARPA-babel301b-v2.0b-build/BABEL_OP2_301/conversational/reference_materials/lexicon.txt",
    "Kazakh": "302-kazakh/IARPA-babel302b-v1.0a-build/BABEL_OP2_302/conversational/reference_materials/lexicon.txt",
    "Teluhu": "303-telugu/IARPA-babel303b-v1.0a-build/BABEL_OP2_303/conversational/reference_materials/lexicon.txt",
    "Lithuanian": "304-lithuanian/IARPA-babel304b-v1.0b-build/BABEL_OP2_304/conversational/reference_materials/lexicon.txt",
    "Guarani": "305-guarani/IARPA-babel305b-v1.0b-build/BABEL_OP3_305/conversational/reference_materials/lexicon.txt",
    "Igbo": "306-igbo/IARPA-babel306b-v2.0c-build/BABEL_OP3_306/conversational/reference_materials/lexicon.txt",
    "Amharic": "307-amharic/IARPA-babel307b-v1.0b-build/BABEL_OP3_307/conversational/reference_materials/lexicon.txt",
    "Mongolian": "401-mongolian/IARPA-babel401b-v2.0b-build/BABEL_OP3_401/conversational/reference_materials/lexicon.txt",
    "Javanese": "402-javanese/IARPA-babel402b-v1.0b-build/BABEL_OP3_402/conversational/reference_materials/lexicon.txt",
    "Dholuo": "403-dholuo/IARPA-babel403b-v1.0b-build/BABEL_OP3_403/conversational/reference_materials/lexicon.txt",
}
for key in BABEL_LEXICON_PATHS:
    BABEL_LEXICON_PATHS[key] = BABEL_ROOT / BABEL_LEXICON_PATHS[key]


def needs_romanized_flag(lang: str) -> bool:
    return lang in [
        'Amharic', 'Assamese', 'Bengali', 'Cantonese', 'Kazakh', 'Lao', 'Mongolian', 'Pashto', 'Tamil', 'Teluhu'
    ]


# Create monolingual lexicons
Path('data/dict_mono').mkdir(parents=True, exist_ok=True)
for lang, path in BABEL_LEXICON_PATHS.items():
    print(f'Creating dict for {lang}')
    romanized = '--romanized' if needs_romanized_flag(lang) else ''
    subprocess.run(f'local/convert_babel_lexicon.pl {romanized} {path} data/dict_mono/{lang}', shell=True, text=True)

# Create multilingual lexicon
lexicons = {}
for lex_path in Path('data/dict_mono').rglob('lexicon.txt'):
    lang = lex_path.parent.name
    lexicons[lang] = lex_path.read_text().splitlines()
    print(lang, len(lexicons[lang]))
Path('data/dict_multi').mkdir(parents=True, exist_ok=True)
lexicon_multi = sorted(set(chain.from_iterable(lexicons.values())))
with open('data/dict_multi/lexicon.txt', 'w') as f:
    for line in lexicon_multi:
        print(line, file=f)
