#! /bin/bash

random_seed=41
gpunum=0
model=vbert
#model=twinrbert
fold=f1
exp=""
MAX_EPOCH=40
freeze_bert=0   ## 1 : freeze

if [ ! -z "$1" ]; then
    fold=$1
fi

# 1. make ./models/$model/weights.p (weights file) in ./models.
echo "training"
outdir="$model"_"$fold""$exp"
echo $outdir

   echo $model
   python train.py \
      --model $model \
      --datafiles ../data/robust/queries.tsv ../data/robust/documents.tsv \
      --qrels ../data/robust/qrels \
      --train_pairs ../data/robust/$fold.train.pairs \
      --valid_run ../data/robust/$fold.valid.run \
      --model_out_dir models/$outdir \
      --max_epoch $MAX_EPOCH \
      --gpunum $gpunum \
      --random_seed $random_seed  \
      --freeze_bert $freeze_bert

# 2. load model weights from ./models/$model/weights.p, run tests, and ./models/$model/test.run
echo "testing"
    python rerank.py \
      --model $model \
      --datafiles ../data/robust/queries.tsv ../data/robust/documents.tsv \
      --run ../data/robust/$fold.test.run \
      --model_weights models/$outdir/weights.p \
      --out_path models/$outdir/test.run \
      --gpunum $gpunum 


#3. read ./models/$model/test.run, calculate scores using various metrics and save the result to ./models/$model/eval.result
echo "evaluating"
../bin/trec_eval -m all_trec ../data/robust/qrels models/$outdir/test.run > models/$outdir/eval.result




