#! /bin/bash
#SBATCH -p cuda
#SBATCH -A cudabig
#SBATCH --qos=cudabig3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH -c 2
#SBATCH --mem=48Gb 


## Usage: 
##    sbatch fewshot.sh llama32B3 prompt01 15 train devel [-quant]

source /scratch/nas/1/PDI/mml0/MML.venv/bin/activate

MODEL=$1
PROMPTS=$2
SHOTS=$3
TRAIN=$4
TEST=$5
QUANT=$6

python3 fewshot.py $MODEL $PROMPTS $SHOTS $TRAIN $TEST $QUANT
if (test $? != 0); then exit; fi

# [MOD-2.3] Stamp the prompt-variant tag into the output filename so
# multiple prompts (prompts01/02/03) at the same shots don't clobber
# each other (the same fix we applied in 1.3 after the same collision).
TAG=$(basename "$PROMPTS" .json)
BASE=../results/FS-$MODEL-$SHOTS-${TEST}${QUANT}
TAGGED=../results/FS-$MODEL-$TAG-$SHOTS-${TEST}${QUANT}
for ext in out json; do
  if [ -f $BASE.$ext ]; then mv $BASE.$ext $TAGGED.$ext; fi
done

python3 ../../../util/evaluator.py DDI ../../../data/$TEST.xml $TAGGED.out $TAGGED.stats


deactivate
