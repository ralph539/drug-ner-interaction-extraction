#! /bin/bash
#SBATCH -p cuda
#SBATCH -A cudabig
#SBATCH --qos=cudabig3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH -c 4
#SBATCH --mem=64Gb 


## Usage: 
##    sbatch FT-train.sh llama32B3 prompt01 train devel [-quant]

# [MOD-1.3] venv updated per professor's 2026-04-10 notice: mml0/MML.venv -> mgl0/AHLT.venv
source /scratch/nas/1/PDI/mgl0/AHLT.venv/bin/activate

MODEL=$1
PROMPTS=$2
TRAIN=$3
TEST=$4
QUANT=$5

python3 finetune-train.py $MODEL $PROMPTS $TRAIN $TEST $QUANT
if (test $? != 0); then exit; fi

deactivate

# [MOD-1.3] Auto-submit the matching inference job now that training is done.
# Without this, Phase D inference had to wait for the FT-inference sbatch to be
# resubmitted by hand because cudabig3080's QOS caps us at 10 queued jobs and
# we were already at the cap while Phase C was running.
# [MOD-1.3] Phase G: honour FT_TAG so the auto-chained inference loads the
# tagged weights dir (e.g. FT-llama32B3-quant-r32.weights) produced above.
WEIGHTS_SUFFIX=${QUANT:+$QUANT}
TAG_SUFFIX=""
if [ -n "$FT_TAG" ]; then
  case "$FT_TAG" in
    -*) TAG_SUFFIX="$FT_TAG" ;;
    *)  TAG_SUFFIX="-$FT_TAG" ;;
  esac
fi
sbatch --export=ALL FT-inference.sh $MODEL $PROMPTS $TEST FT-${MODEL}${WEIGHTS_SUFFIX}${TAG_SUFFIX}.weights $QUANT || true
