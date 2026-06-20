#! /bin/bash
#SBATCH -p cuda
#SBATCH -A cudabig
#SBATCH --qos=cudabig3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH -c 2
#SBATCH --mem=48Gb 


## Usage: 
##    sbatch fewshot.sh llama32B3 prompt01 15 train devel [-quant]

# [MOD-1.3] venv updated per professor's 2026-04-10 notice: mml0/MML.venv -> mgl0/AHLT.venv
source /scratch/nas/1/PDI/mgl0/AHLT.venv/bin/activate

MODEL=$1
PROMPTS=$2
SHOTS=$3
TRAIN=$4
TEST=$5
QUANT=$6
# [MOD-1.3] Phase G: optional 7th argument `-balanced` forwarded to
# fewshot.py and appended to the output filename tag.
BAL=$7

python3 fewshot.py $MODEL $PROMPTS $SHOTS $TRAIN $TEST $QUANT $BAL
if (test $? != 0); then exit; fi

# [MOD-1.3] Rename outputs to include prompt variant so the phase-C grid
# (prompts01/prompts02/prompts03 with the same model+shots) does not overwrite
# itself. fewshot.py names outputs FS-<model>-<shots>-<test><quant>, which is
# ambiguous when varying the prompt file. We strip the ".json" extension from
# $PROMPTS before stamping it into the filename for cleaner names.
TAG=$(basename "$PROMPTS" .json)
# [MOD-1.3] also append balanced tag so Phase G runs don't overwrite Phase C/F.
BALTAG=""
if [ "$BAL" = "-balanced" ]; then BALTAG="-balanced"; fi
BASE=../results/FS-$MODEL-$SHOTS-${TEST}${QUANT}
TAGGED=../results/FS-$MODEL-$TAG-$SHOTS-${TEST}${QUANT}${BALTAG}
for ext in out json; do
  if [ -f $BASE.$ext ]; then mv $BASE.$ext $TAGGED.$ext; fi
done

python3 ../../../util/evaluator.py NER ../../../data/$TEST.xml  $TAGGED.out $TAGGED.stats

deactivate
