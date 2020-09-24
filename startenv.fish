#!/usr/bin/fish

set -l basename ""
set -l hostname (hostname)

if [ $hostname = "mukunar1" ];
  set basename "/home/mukunar1/hub/work";
else
  set basename "/mnt/lascar/qqmukund";
  set -l cudadir "/mnt/lascar/qqmukund/cuda/cuda-11.0";
  eval (eval $LMOD_CMD fish "load Python/3.7.4-GCCcore-8.3.0 PyTorch/1.5.0-fosscuda-2019b-Python-3.7.4 torchvision/0.6.0-fosscuda-2019b-Python-3.7.4-PyTorch-1.5.0");
  set -gx LD_LIBRARY_PATH (string join ":" $LD_LIBRARY_PATH);
end

set -gx PYTHONPATH $basename/endzone/logpolar/
set -gx PYTHONPATH (string join : $basename/endzone/MKDNet/ $PYTHONPATH)
set -gx PYTHONPATH (string join : $basename/endzone/utils/ $PYTHONPATH)

source "$basename/endzone/venv/p37/bin/activate.fish"

