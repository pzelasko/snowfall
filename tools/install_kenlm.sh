#!/usr/bin/env bash

# The script downloads and installs KenLM

GIT=${GIT:-git}

set -e

# Make sure we are in the tools/ directory.
if [ `basename $PWD` == extras ]; then
  cd ..
fi

! [ `basename $PWD` == tools ] && \
   echo "You must call this script from the tools/ directory" && exit 1;

echo "Installing KenLM"

if [ ! -d "kenlm" ]; then
  $GIT clone https://github.com/kpu/kenlm || exit 1
fi

cd kenlm
mkdir build
cd build
cmake ..
make -j8 || exit 1;
cd bin

(
  set +u

  wd=`pwd`
  echo "export PATH=\$PATH:$wd/kaldi_lm"
) >> env.sh

echo >&2 "Installation of KenLM finished successfully"
echo >&2 "Please source tools/env.sh in your path.sh to enable it"
