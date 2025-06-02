PROMPT=$1
SEED=$2
STEP=$3

python retouch_512_batch_affine_prompt_batch_cons_ddpm.py --promptdir $PROMPT --seed $SEED --ddim_steps $STEP
