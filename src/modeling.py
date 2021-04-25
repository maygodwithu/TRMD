from pytools import memoize_method
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import modeling_util
import string

class BertRanker(torch.nn.Module):
    def __init__(self, without_bert=False):
        super().__init__()
        self.BERT_MODEL = 'bert-base-uncased'
        self.CHANNELS = 12 + 1 # from bert-base-uncased
        self.BERT_SIZE = 768 # from bert-base-uncased
        if(without_bert):
            self.bert = None
        else:
            self.bert = BertModel.from_pretrained(self.BERT_MODEL, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_MODEL)

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            state[key] = state[key].data
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
        print("load model : ", path)

    def load_cuda(self, path, device):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)), strict=False)
        print("load model set device : ", path, device)

    def freeze_bert(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def get_params(self):
        params = [(k, v) for k, v in self.named_parameters() if v.requires_grad]
        non_bert_params = [v for k, v in params if not k.startswith('bert')]
        bert_params = [v for k, v in params if k.startswith('bert')]
        return non_bert_params, bert_params 

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask):
        BATCH, QLEN = query_tok.shape
        DIFF = 3 # = [CLS] and 2x[SEP]
        maxlen = self.bert.config.max_position_embeddings

        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (2 + QLEN) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0 # remove padding (will be masked anyway)

        # execute BERT model
        #result = self.bert(toks, segment_ids.long(), mask)
        result_tuple = self.bert(toks, mask, segment_ids.long())
        result = result_tuple[2] ## all hidden_states

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1:QLEN+1] for r in result]
        doc_results = [r[:, QLEN+2:-1] for r in result]
        doc_results = [modeling_util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i*BATCH:(i+1)*BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)

        return cls_results, query_results, doc_results

    def encode_colbert(self, query_tok, query_mask, doc_tok, doc_mask, device='cuda:0'):
        # encode without subbatching
        BATCH, QLEN = query_tok.shape
        DIFF = 5 # = [CLS], 2x[SEP], [Q], [D]
        maxlen = self.bert.config.max_position_embeddings

        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        query_toks = query_tok
        # query_mask = query_mask
        doc_toks = doc_tok[:, :MAX_DOC_TOK_LEN]
        doc_mask = doc_mask[:, :MAX_DOC_TOK_LEN]
        
        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        Q_tok = torch.full(
            size=(BATCH, 1), fill_value=1, dtype=torch.long
        ).cuda(device)  # [unused0] = 1
        D_tok = torch.full(
            size=(BATCH, 1), fill_value=2, dtype=torch.long
        ).cuda(device)  # [unused1] = 2

        # Query augmentation with [MASK] tokens ([MASK] = 103)
        query_toks[query_toks == -1] = torch.tensor(103).cuda(device)
        query_mask = torch.ones_like(query_mask)

        # build BERT input sequences
        toks = torch.cat([CLSS, Q_tok, query_toks, SEPS, D_tok, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, ONES, query_mask, ONES, ONES, doc_mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (3+QLEN) + [ONES] * (2+doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0 # remove padding (will be masked anyway)
        
        # modifiy doc_mask
        doc_mask = torch.cat([ONES, doc_mask, ONES], dim=1)

        # execute BERT model
        result_tuple = self.bert(toks, mask, segment_ids.long())
        result = result_tuple[2] ## all hidden_states

        # extract relevant subsequences for query and doc
        query_results = [r[:, :QLEN+3] for r in result]
        doc_results = [r[:, QLEN+3:] for r in result]

        cls_results = [r[:, 0] for r in result]

        return cls_results, query_results, query_mask, doc_results, doc_mask

class TwoBertRanker(torch.nn.Module):
    def __init__(self, without_bert=False, asym=False):
        super().__init__()
        self.BERT_MODEL = 'bert-base-uncased'
        self.CHANNELS = 12 + 1 # from bert-base-uncased
        self.BERT_SIZE = 768 # from bert-base-uncased
        if(without_bert): 
            self.bert = None
        else:
            self.bert = BertModel.from_pretrained(self.BERT_MODEL, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_MODEL)

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            state[key] = state[key].data
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
        print("load model : ", path)

    def freeze_bert(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def get_params(self):
        params = [(k, v) for k, v in self.named_parameters() if v.requires_grad]
        non_bert_params = [v for k, v in params if not k.startswith('bert')]
        bert_params = [v for k, v in params if k.startswith('bert')]
        return non_bert_params, bert_params 

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask):
        BATCH, QLEN = query_tok.shape
        DIFF = 3 # = [CLS] and 2x[SEP]
        maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences query & doc
        q_toks = torch.cat([CLSS, query_toks, SEPS], dim=1)
        q_mask = torch.cat([ONES, query_mask, ONES], dim=1)
        q_segid = torch.cat([NILS] * (2+QLEN), dim=1)
        q_toks[q_toks == -1] = 0

        d_toks = torch.cat([CLSS, doc_toks, SEPS], dim=1)
        d_mask = torch.cat([ONES, doc_mask, ONES], dim=1)
        d_segid = torch.cat([NILS] * (2+doc_toks.shape[1]), dim=1)
        d_toks[d_toks == -1] = 0

        # execute BERT model
        q_result_tuple = self.bert(q_toks, q_mask, q_segid.long())
        d_result_tuple = self.bert(d_toks, d_mask, d_segid.long())
        q_result = q_result_tuple[2]
        d_result = d_result_tuple[2]

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1:-1] for r in q_result]
        doc_results = [r[:, 1:-1] for r in d_result]
        doc_results = [modeling_util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        # build CLS representation
        q_cls_results = []
        for layer in q_result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i*BATCH:(i+1)*BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            q_cls_results.append(cls_result)

        d_cls_results = []
        for layer in d_result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i*BATCH:(i+1)*BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            d_cls_results.append(cls_result)

        return q_cls_results, d_cls_results, query_results, doc_results

    def encode_colbert(self, query_tok, query_mask, doc_tok, doc_mask, device='cuda:0'):
        # encode without subbatching
        query_lengths = (query_mask > 0).sum(1)
        doc_lengths = (doc_mask > 0).sum(1)
        BATCH, QLEN = query_tok.shape
        # QLEN : 20
        # DIFF = 2  # = [CLS] and [SEP]
        maxlen = self.bert.config.max_position_embeddings
        # MAX_DOC_TOK_LEN = maxlen - DIFF  # doc maxlen: 510

        doc_toks = F.pad(doc_tok[:, : maxlen - 2], pad=(0, 1, 0, 0), value=-1)
        doc_mask = F.pad(doc_mask[:, : maxlen - 2], pad=(0, 1, 0, 0), value=0)
        query_toks = query_tok

        query_lengths = torch.where(query_lengths > 19, torch.tensor(19).cuda(device), query_lengths)
        query_toks[torch.arange(BATCH), query_lengths] = self.tokenizer.vocab["[SEP]"]
        query_mask[torch.arange(BATCH), query_lengths] = 1
        doc_lengths = torch.where(doc_lengths > 510, torch.tensor(510).cuda(device), doc_lengths)
        doc_toks[torch.arange(BATCH), doc_lengths] = self.tokenizer.vocab["[SEP]"]
        doc_mask[torch.arange(BATCH), doc_lengths] = 1

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab["[CLS]"])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab["[SEP]"])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences query & doc
        q_toks = torch.cat([CLSS, query_toks], dim=1)
        q_mask = torch.cat([ONES, query_mask], dim=1)
        q_segid = torch.cat([NILS] * (1 + QLEN), dim=1)
        # 2) Query augmentation with [MASK] tokens ([MASK] = 103)
        q_toks[q_toks == -1] = torch.tensor(103).cuda(device)

        d_toks = torch.cat([CLSS, doc_toks], dim=1)
        d_mask = torch.cat([ONES, doc_mask], dim=1)
        d_segid = torch.cat([NILS] * (1 + doc_toks.shape[1]), dim=1)
        d_toks[d_toks == -1] = 0

        # execute BERT model
        q_result_tuple = self.bert(q_toks, q_mask, q_segid.long())
        d_result_tuple = self.bert(d_toks, d_mask, d_segid.long())
        q_result = q_result_tuple[2]
        d_result = d_result_tuple[2]

        # extract relevant subsequences for query and doc
        query_results = [r[:, :] for r in q_result]  # missing representation for cls and sep?
        doc_results = [r[:, :] for r in d_result]

        q_cls_result = [r[:, 0] for r in q_result]
        d_cls_result = [r[:, 0] for r in d_result]

        return q_cls_result, d_cls_result, query_results, q_mask, doc_results, d_mask

class VanillaBertRanker(BertRanker):
    def __init__(self, without_bert=False):
        super().__init__(without_bert)
        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, value_return=False):
        cls_reps, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        if(value_return):
            return self.cls(self.dropout(cls_reps[-1])), cls_reps, None
        else:
            return self.cls(self.dropout(cls_reps[-1]))
    
    def forward_without_bert(self, cls_reps, q_reps, d_reps, query_tok, doc_tok):
        return self.cls(self.dropout(cls_reps[-1])), None
    

class TwinBertRanker(TwoBertRanker):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        q_cls_reps, d_cls_reps, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        q_cls_rep = F.normalize(q_cls_reps[-1], p=2, dim=0)
        d_cls_rep = F.normalize(d_cls_reps[-1], p=2, dim=0)
        score = F.cosine_similarity(q_cls_rep, d_cls_rep)
        #print(score)
        return score

class TwinBertResRanker(TwoBertRanker):
    def __init__(self, without_bert=False, qd=True, asym=False):
        super().__init__(without_bert, asym)
        self.qd = qd
        self.dropout = torch.nn.Dropout(0.1)
        self.wpool = torch.nn.AdaptiveAvgPool2d((1,self.BERT_SIZE))
        self.res = torch.nn.Linear(self.BERT_SIZE, self.BERT_SIZE)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, value_return=False):
        q_cls_reps, d_cls_reps, q_reps, d_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)

        if(self.qd):
            x1 = self.wpool(q_reps[-1]).squeeze(dim=1)
            x2 = self.wpool(d_reps[-1]).squeeze(dim=1)
        else:
            x1 = q_cls_reps[-1]
            x2 = d_cls_reps[-1]

        x = torch.max(x1, x2)
        feature = self.res(x)+x
        score = self.cls(feature)

        simmat = feature					## for distillation ( features from represenation)
        #simmat = torch.cat([q_reps[-1], d_reps[-1]], dim=1)	## for distillation ( representation )
        if(value_return):
            if(self.qd):
                return score, None, simmat 
            else:
                return score, simmat, None
        else:
            return score

    def forward_without_bert(self, cls_reps, q_reps, d_reps, query_tok, doc_tok):
        x1 = self.wpool(q_reps[-1]).squeeze(dim=1)
        x2 = self.wpool(d_reps[-1]).squeeze(dim=1) 

        x = torch.max(x1, x2)
        feature = self.res(x)+x
        score = self.cls(feature)

        simmat = feature
        #simmat = torch.cat([q_reps[-1], d_reps[-1]], dim=1)	## for distilation
        return score, simmat

class ColBertRanker(TwoBertRanker):
    def __init__(self, without_bert=False, asym=False):
        super().__init__(without_bert, asym)
        self.dim = 128  # default: dim=128
        self.skiplist = self.tokenize(string.punctuation)

        self.clinear = torch.nn.Linear(
            self.BERT_SIZE, self.dim, bias=False
        )  # both for queries, documents

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, value_return=False):
        # q length default: 32  -> 20
        # d length defualt: 180 -> 510

        # 1) Prepend [Q] token to query, [D] token to document
        q_length = query_tok.shape[1]
        d_length = doc_tok.shape[1]
        num_batch_samples = doc_tok.shape[0]

        Q_tok = torch.full(
            size=(num_batch_samples, 1), fill_value=1, dtype=torch.long
        ).cuda()  # [unused0] = 1
        D_tok = torch.full(
            size=(num_batch_samples, 1), fill_value=2, dtype=torch.long
        ).cuda()  # [unused1] = 2
        one_tok = torch.full(size=(num_batch_samples, 1), fill_value=1).cuda()

        query_tok = torch.cat([Q_tok, query_tok[:, : q_length - 1]], dim=1)
        doc_tok = torch.cat([D_tok, doc_tok[:, : d_length - 1]], dim=1)
        query_mask = torch.cat([one_tok, query_mask[:, : q_length - 1]], dim=1)
        doc_mask = torch.cat([one_tok, doc_mask[:, : d_length - 1]], dim=1)

        # 2) Query augmentation with [MASK] tokens ([MASK] = 103)
        q_cls_reps, d_cls_reps, q_reps, query_mask, d_reps, doc_mask = self.encode_colbert(
            query_tok, query_mask, doc_tok, doc_mask
        )  # reps includes rep of [CLS], [SEP]
        col_q_reps = self.clinear(q_reps[-1])
        col_d_reps = self.clinear(d_reps[-1])

        # 3) skip punctuations in doc tokens
        cut_doc_tok = torch.cat([one_tok.long(), doc_tok[:, :510], one_tok.long()], dim=1)
        mask = torch.ones_like(doc_mask, dtype=torch.float).cuda()
        mask = torch.where(
            ((cut_doc_tok >= 999) & (cut_doc_tok <= 1013))
            | ((cut_doc_tok >= 1024) & (cut_doc_tok <= 1036))
            | ((cut_doc_tok >= 1063) & (cut_doc_tok <= 1066))
            | (cut_doc_tok == -1),
            torch.tensor(0.0).cuda(),
            doc_mask,
        )
        col_d_reps = col_d_reps * mask.unsqueeze(2)
        q_rep = F.normalize(col_q_reps, p=2, dim=2)
        d_rep = F.normalize(col_d_reps, p=2, dim=2)
        score = (q_rep @ d_rep.permute(0, 2, 1)).max(2).values.sum(1)
        #simmat = (q_rep @ d_rep.permute(0, 2, 1))
        #score = simmat.max(2).values.sum(1)
        simmat = torch.cat([q_rep, d_rep], dim=1)  ## for distillation
        score = score.unsqueeze(1)
        if(value_return):
            return score, None, simmat
        else:
            return score

    def forward_without_bert(self, cls_reps, q_reps, d_reps, query_tok, doc_tok):
        score = (q_reps @ d_reps.permute(0, 2, 1)).max(2).values.sum(1)
        simmat = torch.cat([q_reps, d_reps], dim=1)  ## for distillation
        score = score.unsqueeze(1)
        return score, simmat

class ColBertVRanker(BertRanker):
    def __init__(self):
        super().__init__()
        self.dim = 128  # default: dim=128
        self.skiplist = self.tokenize(string.punctuation)

        self.clinear = torch.nn.Linear(
            self.BERT_SIZE, self.dim, bias=False
        )  # both for queries, documents

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        # q length default: 32  -> 20 (+ CLS, Q, SEP)
        # d length default: 180 -> 487 (+ D, SEP)

        num_batch_samples = doc_tok.shape[0]
        one_tok = torch.full(size=(num_batch_samples, 1), fill_value=1).cuda()

        cls_reps, q_reps, query_mask, d_reps, doc_mask = self.encode_colbert(
            query_tok, query_mask, doc_tok, doc_mask
        )  # reps includes rep of [CLS], [SEP]
        col_q_reps = self.clinear(q_reps[-1])
        col_d_reps = self.clinear(d_reps[-1])

        # 3) skip punctuations in doc tokens
        cut_doc_tok = torch.cat([one_tok.long(), doc_tok[:, :487], one_tok.long()], dim=1)
        mask = torch.ones_like(doc_mask, dtype=torch.float).cuda()
        mask = torch.where(
            ((cut_doc_tok >= 999) & (cut_doc_tok <= 1013))
            | ((cut_doc_tok >= 1024) & (cut_doc_tok <= 1036))
            | ((cut_doc_tok >= 1063) & (cut_doc_tok <= 1066))
            | (cut_doc_tok == -1),
            torch.tensor(0.0).cuda(),
            doc_mask,
        )
        col_d_reps = col_d_reps * mask.unsqueeze(2)
        q_rep = F.normalize(col_q_reps, p=2, dim=2)
        d_rep = F.normalize(col_d_reps, p=2, dim=2)
        score = (q_rep @ d_rep.permute(0, 2, 1)).max(2).values.sum(1)
        score = score.unsqueeze(1)
        return score

class MultiBertRanker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.BERT_MODEL = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_MODEL)

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            state[key] = state[key].data
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
        print("load model : ", path)

    def load_cuda(self, path, device):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)), strict=False)
        print("load model set device : ", path, device)

    def get_params(self):
        params = [(k, v) for k, v in self.named_parameters() if v.requires_grad]
        non_bert_params = [v for k, v in params if '.bert.' not in k ]
        bert_params = [v for k, v in params if '.bert.' in k ]
        return non_bert_params, bert_params 

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks

class DuetBertRanker(MultiBertRanker):
    def __init__(self, sub_1, sub_2):
        super().__init__()
        self.bert_1 = sub_1.to('cuda:0')
        self.bert_2 = sub_2.to('cuda:0')
        #2-gpu case
        #self.bert_1 = sub_1.to('cuda:0')
        #self.bert_2 = sub_2.to('cuda:1')

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, value_return=False):
        if(value_return):
            score_1, cls_1, simmat_1 = self.bert_1(query_tok, query_mask, doc_tok, doc_mask, value_return=True)
            score_2, cls_2, simmat_2 = self.bert_2(query_tok, query_mask, doc_tok, doc_mask, value_return=True)
        else:
            score_1 = self.bert_1(query_tok, query_mask, doc_tok, doc_mask)
            score_2 = self.bert_2(query_tok, query_mask, doc_tok, doc_mask)
            
        score = score_1 + score_2

        if(value_return):
            return score, score_1, score_2, cls_1, simmat_1, cls_2, simmat_2
        else:
            return score, score_1, score_2

    def freeze_bert(self):
        print("freezing bert")
        self.bert_1.freeze_bert()
        self.bert_2.freeze_bert()

    def load_duet(self, path1, path2):
        print("load duet model")
        self.bert_1.load(path1)
        self.bert_2.load(path2)

class BaseBertRanker(BertRanker):
    def __init__(self):
        super().__init__()

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, q_reps, d_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        return cls_reps, q_reps, d_reps

class CBaseBertRanker(BertRanker):
    def __init__(self, device):
        super().__init__()
        self.device = device
        #self.device = 'cuda:0'
        self.dim = 128  # default: dim=128

        self.clinear = torch.nn.Linear(
            self.BERT_SIZE, self.dim, bias=False
        )  # both for queries, documents
 
    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        # q length default: 32  -> 20 (+ CLS, Q, SEP)
        # d length default: 180 -> 487 (+ D, SEP)

        num_batch_samples = doc_tok.shape[0]
        one_tok = torch.full(size=(num_batch_samples, 1), fill_value=1).cuda(self.device)

        cls_reps, q_reps, query_mask, d_reps, doc_mask = self.encode_colbert(
            query_tok, query_mask, doc_tok, doc_mask, device=self.device
        )  # reps includes rep of [CLS], [SEP]
        col_q_reps = self.clinear(q_reps[-1])
        col_d_reps = self.clinear(d_reps[-1])

        # 3) skip punctuations in doc tokens
        cut_doc_tok = torch.cat([one_tok.long(), doc_tok[:, :487], one_tok.long()], dim=1)
        mask = torch.ones_like(doc_mask, dtype=torch.float).cuda(self.device)
        mask = torch.where(
            ((cut_doc_tok >= 999) & (cut_doc_tok <= 1013))
            | ((cut_doc_tok >= 1024) & (cut_doc_tok <= 1036))
            | ((cut_doc_tok >= 1063) & (cut_doc_tok <= 1066))
            | (cut_doc_tok == -1),
            torch.tensor(0.0).cuda(self.device),
            doc_mask,
        )
        col_d_reps = col_d_reps * mask.unsqueeze(2)
        q_rep = F.normalize(col_q_reps, p=2, dim=2)
        d_rep = F.normalize(col_d_reps, p=2, dim=2)
        #score = (q_rep @ d_rep.permute(0, 2, 1)).max(2).values.sum(1)
        #score = score.unsqueeze(1)
        #return score

        return cls_reps, q_rep, d_rep

class LBaseBertRanker(TwoBertRanker):
    def __init__(self):
        super().__init__()
        self.res = torch.nn.Linear(self.BERT_SIZE, self.BERT_SIZE)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
 
    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        q_cls_reps, d_cls_reps, q_reps, d_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        
        x1 = q_cls_reps[-1]
        x2 = d_cls_reps[-1]

        x = torch.max(x1, x2)
        cls_rep = self.res(x)+x

        return [cls_rep], q_reps, d_reps

class CLBaseBertRanker(TwoBertRanker):
    def __init__(self, device):
        super().__init__()
        self.device = device
        #self.device = 'cuda:0'
        self.dim = 128  # default: dim=128

        self.clinear = torch.nn.Linear(
            self.BERT_SIZE, self.dim, bias=False
        )  # both for queries, documents

        self.res = torch.nn.Linear(self.BERT_SIZE, self.BERT_SIZE)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)
 
    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        # q length default: 32  -> 20
        # d length defualt: 180 -> 510

        # 1) Prepend [Q] token to query, [D] token to document
        q_length = query_tok.shape[1]
        d_length = doc_tok.shape[1]
        num_batch_samples = doc_tok.shape[0]

        Q_tok = torch.full(
            size=(num_batch_samples, 1), fill_value=1, dtype=torch.long
        ).cuda(self.device)  # [unused0] = 1
        D_tok = torch.full(
            size=(num_batch_samples, 1), fill_value=2, dtype=torch.long
        ).cuda(self.device)  # [unused1] = 2
        one_tok = torch.full(size=(num_batch_samples, 1), fill_value=1).cuda(self.device)

        query_tok = torch.cat([Q_tok, query_tok[:, : q_length - 1]], dim=1)
        doc_tok = torch.cat([D_tok, doc_tok[:, : d_length - 1]], dim=1)
        query_mask = torch.cat([one_tok, query_mask[:, : q_length - 1]], dim=1)
        doc_mask = torch.cat([one_tok, doc_mask[:, : d_length - 1]], dim=1)

        # 2) Query augmentation with [MASK] tokens ([MASK] = 103)
        q_cls_reps, d_cls_reps, q_reps, query_mask, d_reps, doc_mask = self.encode_colbert(
            query_tok, query_mask, doc_tok, doc_mask, device=self.device
        )  # reps includes rep of [CLS], [SEP]
        col_q_reps = self.clinear(q_reps[-1])
        col_d_reps = self.clinear(d_reps[-1])

        # 3) skip punctuations in doc tokens
        cut_doc_tok = torch.cat([one_tok.long(), doc_tok[:, :510], one_tok.long()], dim=1)
        mask = torch.ones_like(doc_mask, dtype=torch.float).cuda(self.device)
        mask = torch.where(
            ((cut_doc_tok >= 999) & (cut_doc_tok <= 1013))
            | ((cut_doc_tok >= 1024) & (cut_doc_tok <= 1036))
            | ((cut_doc_tok >= 1063) & (cut_doc_tok <= 1066))
            | (cut_doc_tok == -1),
            torch.tensor(0.0).cuda(self.device),
            doc_mask,
        )
        col_d_reps = col_d_reps * mask.unsqueeze(2)
        q_rep = F.normalize(col_q_reps, p=2, dim=2)
        d_rep = F.normalize(col_d_reps, p=2, dim=2)

        x1 = q_cls_reps[-1]
        x2 = d_cls_reps[-1]

        x = torch.max(x1, x2)
        cls_rep = self.res(x)+x

        return [cls_rep], q_rep, d_rep

##
MODEL_MAP = {
    'vbert': VanillaBertRanker,
    'twinbert': TwinBertRanker,
    'twinrbert': TwinBertResRanker,
    'colbert': ColBertRanker,
    'colvbert': ColBertVRanker,
    'duetbert': DuetBertRanker,
}

class DistilBertRanker(MultiBertRanker):
    def __init__(self, sub_1, sub_2, trainable=False, late=True, colbert=False):
        super().__init__()
        self.trainable = trainable
        self.late = late
        if(self.trainable):
            if(late):
                if(colbert): self.base_bert = CLBaseBertRanker('cuda:1').to('cuda:1') ## colbert settin
                else: self.base_bert = LBaseBertRanker().to('cuda:1')
            else:
                if(colbert): self.base_bert = CBaseBertRanker('cuda:1').to('cuda:1')  ## colbert setting
                else: self.base_bert = BaseBertRanker().to('cuda:1')
            self.bert_1 = MODEL_MAP[sub_1](without_bert=True).to('cuda:1')
            self.bert_2 = MODEL_MAP[sub_2](without_bert=True).to('cuda:1')
            self.duetbert = DuetBertRanker(MODEL_MAP[sub_1](), MODEL_MAP[sub_2]())
        else:
            if(late):
                if(colbert): self.base_bert = CLBaseBertRanker('cuda:0').to('cuda:0')
                else: self.base_bert = LBaseBertRanker().to('cuda:0')
            else:
                if(colbert): self.base_bert = CBaseBertRanker('cuda:0').to('cuda:0')  ## colbert setting
                else: self.base_bert = BaseBertRanker().to('cuda:0')
            self.bert_1 = MODEL_MAP[sub_1](without_bert=True).to('cuda:0')
            self.bert_2 = MODEL_MAP[sub_2](without_bert=True).to('cuda:0')

    def compute_loss(self, cls, cls_d, simmat, simmat_d):
        loss = 0.
        if(cls_d is not None): loss += F.mse_loss(cls[-1].to('cuda:0'), cls_d[-1]) 
        if(simmat_d is not None): loss += F.mse_loss(simmat.to('cuda:0'), simmat_d)  ## .. Warn
        return loss

    def train_forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, q_reps, d_reps = self.base_bert(query_tok.to('cuda:1'), query_mask.to('cuda:1'), doc_tok.to('cuda:1'), doc_mask.to('cuda:1')) 

        score_1, simmat_1 = self.bert_1.forward_without_bert(cls_reps, q_reps, d_reps, query_tok.to('cuda:1'), doc_tok.to('cuda:1'))
        score_2, simmat_2 = self.bert_2.forward_without_bert(cls_reps, q_reps, d_reps, query_tok.to('cuda:1'), doc_tok.to('cuda:1'))

        simmat_f = None
        if(simmat_1 is not None): simmat_f = simmat_1
        if(simmat_2 is not None): simmat_f = simmat_2

        score_d, score_d1, score_d2, cls_d1, simmat_d1, cls_d2, simmat_d2 = self.duetbert(query_tok, query_mask, doc_tok, doc_mask, value_return=True)

        ## loss
        loss_1 = self.compute_loss(cls_reps, cls_d1, simmat_f, simmat_d1)
        loss_2 = self.compute_loss(cls_reps, cls_d2, simmat_f, simmat_d2)
        
        ##
        score = score_1.to('cuda:0') + score_2.to('cuda:0')

        return score, loss_1, loss_2

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        if(self.trainable):
            cls_reps, q_reps, d_reps = self.base_bert(query_tok.to('cuda:1'), query_mask.to('cuda:1'), doc_tok.to('cuda:1'), doc_mask.to('cuda:1')) 
            score_1, simmat_1 = self.bert_1.forward_without_bert(cls_reps, q_reps, d_reps, query_tok.to('cuda:1'), doc_tok.to('cuda:1'))
            score_2, simmat_2 = self.bert_2.forward_without_bert(cls_reps, q_reps, d_reps, query_tok.to('cuda:1'), doc_tok.to('cuda:1'))
      
            score = score_1 + score_2
            return score.to('cuda:0')
        else:
            cls_reps, q_reps, d_reps = self.base_bert(query_tok, query_mask, doc_tok, doc_mask) 
            score_1, simmat_1 = self.bert_1.forward_without_bert(cls_reps, q_reps, d_reps, query_tok, doc_tok)
            score_2, simmat_2 = self.bert_2.forward_without_bert(cls_reps, q_reps, d_reps, query_tok, doc_tok)
      
            score = score_1 + score_2
            return score

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if key.startswith('duetbert'):
                del state[key]
            else:
                state[key] = state[key].data
        torch.save(state, path)

    def get_params(self):
        params = [(k, v) for k, v in self.named_parameters() if v.requires_grad and 'duetbert' not in k]
        non_bert_params = [v for k, v in params if '.bert.' not in k ]
        bert_params = [v for k, v in params if '.bert.' in k ]
        return non_bert_params, bert_params 

    def load_duet(self, path1, path2):
        self.duetbert.load_duet(path1, path2)

    def freeze_bert(self):
        print("freeze duet bert & eval mode")
        self.duetbert.freeze_bert()
        self.duetbert.eval()


MODEL_MAP['distilbert'] = DistilBertRanker
