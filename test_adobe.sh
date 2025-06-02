PROMPT=$1
SEED=$2
STEP=$3

python retouch_adobe.py --promptdir $PROMPT --seed $SEED --ddim_steps $STEP
