import os
import argparse
import subprocess
import random
from tqdm import tqdm
import torch
import modeling
import data
import torch.nn as nn

trec_eval_f = '../bin/trec_eval'

def setRandomSeed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)

def main(model, dataset, train_pairs, qrels, valid_run, qrelf, model_out_dir, max_epoch, warmup_epoch, freeze_bert):
    LR = 0.001
    BERT_LR = 2e-5
    MAX_EPOCH = max_epoch
    _verbose = False
    _logf = os.path.join(model_out_dir, 'train.log')

    ## freeze_bert
    model_name = type(model).__name__
    if(freeze_bert == 1):
        model.freeze_bert()

    ## parameter update setting
    nonbert_params, bert_params = model.get_params()
    optim_nonbert_params = {'params': nonbert_params}
    optim_bert_params = {'params': bert_params, 'lr':BERT_LR}
    optimizer = torch.optim.Adam([optim_nonbert_params, optim_bert_params], lr=LR)

    ## training & validation
    logf = open(_logf, "w")
    print(f'max_epoch={max_epoch}', file=logf)
    if(warmup_epoch > 0):
        print(f'warmup_epoch={warmup_epoch}', file=logf)
    epoch = 0
    top_valid_score = None
    for epoch in range(MAX_EPOCH):
        warmup=""
        if(epoch < warmup_epoch): warmup="warmup"
        if(model_name.startswith('Distil')):
            loss, vloss, closs = distil_train_iteration(model, optimizer, dataset, train_pairs, qrels, warmup_epoch, epoch)
            print(f'train epoch={epoch} loss={loss} vloss={vloss} closs={closs} {warmup}')
            print(f'train epoch={epoch} loss={loss} vloss={vloss} closs={closs} {warmup}', file=logf)
        else:
            loss = train_iteration(model, optimizer, dataset, train_pairs, qrels)
            print(f'train epoch={epoch} loss={loss}')
            print(f'train epoch={epoch} loss={loss}', file=logf)

        valid_score = validate(model, dataset, valid_run, qrelf, epoch, model_out_dir)
        print(f'validation epoch={epoch} score={valid_score}')
        print(f'validation epoch={epoch} score={valid_score}', file=logf)
        if(epoch >= warmup_epoch and (top_valid_score is None or valid_score > top_valid_score)):
            top_valid_score = valid_score
            print('new top validation score, saving weights')
            print(f'newtopsaving epoch={epoch} score={top_valid_score}', file=logf)
            model.save(os.path.join(model_out_dir, 'weights.p'))

        logf.flush()

    print(f'topsaving score={top_valid_score}', file=logf)

def train_iteration(model, optimizer, dataset, train_pairs, qrels):
    BATCH_SIZE = 16
    BATCHES_PER_EPOCH = 64
    GRAD_ACC_SIZE = 2
    total = 0
    model.train()
    total_loss = 0.
    cq_sum = 0.
    cd_sum = 0.
    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_train_pairs(model, dataset, train_pairs, qrels, GRAD_ACC_SIZE): 
            scores = model(record['query_tok'],
                           record['query_mask'],
                           record['doc_tok'],
                           record['doc_mask'])
            count = len(record['query_id']) // 2
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pariwse softmax
            loss.backward()
            total_loss += loss.item()
            total += count
            if total % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.update(count)
            if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
                return total_loss

def distil_train_iteration(model, optimizer, dataset, train_pairs, qrels, warmup_epoch, epoch):
    BATCH_SIZE = 16
    BATCHES_PER_EPOCH = 64
    GRAD_ACC_SIZE = 2
    total = 0
    model.train()
    total_loss = 0.
    total_vloss = 0.
    total_closs = 0.
    cq_sum = 0.
    cd_sum = 0.
    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_train_pairs(model, dataset, train_pairs, qrels, GRAD_ACC_SIZE): 
            scores, cls_loss, simmat_loss = model.train_forward(record['query_tok'],
                           record['query_mask'],
                           record['doc_tok'],
                           record['doc_mask'])
            count = len(record['query_id']) // 2
            scores = scores.reshape(count, 2)
            #print(cls_loss)
            #print(simmat_loss)
            #print(scores)

            ## 
            score_loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pariwse softmax

            loss = score_loss + cls_loss + simmat_loss 
            #loss = score_loss  ## without distillation

            loss.backward()

            total_loss += loss.item()
            total_vloss += cls_loss.item()
            if(isinstance(simmat_loss, torch.Tensor)): total_closs += simmat_loss.item()
            total += count
            if total % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.update(count)
            if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
                return total_loss, total_vloss, total_closs

def validate(model, dataset, run, qrelf, epoch, model_out_dir):
    VALIDATION_METRIC = 'P.20'
    runf = os.path.join(model_out_dir, f'{epoch}.run')
    run_model(model, dataset, run, runf)
    return trec_eval(qrelf, runf, VALIDATION_METRIC)


def run_model(model, dataset, run, runf, desc='valid'):
    BATCH_SIZE = 16
    rerank_run = {}
    model_name = type(model).__name__
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        for records in data.iter_valid_records(model, dataset, run, BATCH_SIZE):
            scores = model(records['query_tok'],
                           records['query_mask'],
                           records['doc_tok'],
                           records['doc_mask'])
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run.setdefault(qid, {})[did] = score.item()
            pbar.update(len(records['query_id']))
    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')

def trec_eval(qrelf, runf, metric):
    output = subprocess.check_output([trec_eval_f, '-m', metric, qrelf, runf]).decode().rstrip()
    output = output.replace('\t', ' ').split('\n')
    assert len(output) == 1
    return float(output[0].split()[2])

def main_cli():
    MODEL_MAP = modeling.MODEL_MAP
    parser = argparse.ArgumentParser('TRMD model training and validation')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vbert')
    parser.add_argument('--submodel1', choices=MODEL_MAP.keys(), default=None)
    parser.add_argument('--submodel2', choices=MODEL_MAP.keys(), default=None)
    parser.add_argument('--submodel3', choices=MODEL_MAP.keys(), default=None)
    parser.add_argument('--submodel4', choices=MODEL_MAP.keys(), default=None)
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--qrels', type=argparse.FileType('rt'))
    parser.add_argument('--train_pairs', type=argparse.FileType('rt'))
    parser.add_argument('--valid_run', type=argparse.FileType('rt'))
    parser.add_argument('--initial_bert_weights', type=str, default=None)
    parser.add_argument('--model_out_dir')
    parser.add_argument('--max_epoch', type=int, default=100, help='max epoch')
    parser.add_argument('--warmup_epoch', type=int, default=0, help='warmup epoch')
    parser.add_argument('--gpunum', type=str, default="0", help='gpu number')
    parser.add_argument('--random_seed', type=int, default=42, help='ranodm seed number')
    parser.add_argument('--freeze_bert', type=int, default=0, help='freezing bert')
    args = parser.parse_args()

    setRandomSeed(args.random_seed)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpunum

    print("GPU count=", torch.cuda.device_count())
   
    if(args.model == 'distilbert'):
        has_colbert=False
        if("colbert" in args.submodel1 or "colbert" in args.submodel2): has_colbert=True
        model = MODEL_MAP[args.model](args.submodel1, args.submodel2, trainable=True, late=True, colbert=has_colbert)
    else:
        model = MODEL_MAP[args.model]().cuda()

    dataset = data.read_datafiles(args.datafiles)
    qrels = data.read_qrels_dict(args.qrels)
    train_pairs = data.read_pairs_dict(args.train_pairs)
    valid_run = data.read_run_dict(args.valid_run)

    ## initial
    if(args.initial_bert_weights is not None):
        wts = args.initial_bert_weights.split(',')
        if(len(wts) == 1):
            model.load(wts[0])
        elif(len(wts) == 2):
            model.load_duet(wts[0], wts[1])

    os.makedirs(args.model_out_dir, exist_ok=True)
    main(model, dataset, train_pairs, qrels, valid_run, args.qrels.name, args.model_out_dir, args.max_epoch, args.warmup_epoch, args.freeze_bert)


if __name__ == '__main__':
    main_cli()
