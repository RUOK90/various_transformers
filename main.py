from config import ARGS
import spacy
import numpy as np
import torch
import dataset
from models import Transformer, get_model
from utils import LabelSmoothing, NoamOpt, MyIterator
from transformer_layers import *
import utils
import wandb
import math


# dataset
SRC, TGT, train, val, test, pad_idx = dataset.get_dataset(ARGS.dataset)
train_iter = MyIterator(train, batch_size=ARGS.batch_size, device=ARGS.device, repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=utils.batch_size_fn, train=True)
val_iter = MyIterator(val, batch_size=ARGS.batch_size, device=ARGS.device, repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=utils.batch_size_fn, train=False)
print('Done get dataset')


# model
model = Transformer(len(SRC.vocab), len(TGT.vocab), N=ARGS.n_layers, d_model=ARGS.d_model, d_ff=4*ARGS.d_model, h=ARGS.n_heads, dropout=ARGS.p_dropout).to(ARGS.device)
criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1).to(ARGS.device)


# train
if ARGS.run_mode == 'train':
    optimizer = NoamOpt(ARGS.d_model, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    iter_cnt = 1
    min_norm_val_loss = math.inf
    model.train()
    for epoch in range(ARGS.n_epochs):
        for train_batch in train_iter:
            train_batch = utils.rebatch(pad_idx, train_batch)
            train_out = model(train_batch.src, train_batch.trg, train_batch.src_mask, train_batch.trg_mask)
            train_loss = criterion(train_out.contiguous().view(-1, train_out.size(-1)), train_batch.trg_y.contiguous().view(-1)) / train_batch.ntokens

            train_loss.backward()
            optimizer.step()
            optimizer.optimizer.zero_grad()
            print(f'{iter_cnt} - train_loss: {train_loss.item()}')

            if iter_cnt % ARGS.eval_steps == 0:
                model.eval()
                val_total_loss = 0
                val_ntokens = 0
                with torch.no_grad():
                    for val_batch in val_iter:
                        val_batch = utils.rebatch(pad_idx, val_batch)
                        val_out = model(val_batch.src, val_batch.trg, val_batch.src_mask, val_batch.trg_mask)
                        val_loss = criterion(val_out.contiguous().view(-1, val_out.size(-1)), val_batch.trg_y.contiguous().view(-1))

                        val_total_loss += val_loss.item()
                        val_ntokens += val_batch.ntokens

                    norm_val_loss = val_total_loss / val_ntokens.float()
                    min_norm_val_loss = min(min_norm_val_loss, norm_val_loss)

                    if ARGS.wandb:
                        wandb.log({'Train Loss': train_loss.item(),
                                   'Norm Val Loss': norm_val_loss,
                                   'Min Norm Val Loss': min_norm_val_loss,
                                   }, step=iter_cnt)

                    model.train()

            iter_cnt += 1

# eval
elif ARGS.run_mode == 'eval':
    model = get_model('iwslt.pt').to(ARGS.device)
    with torch.no_grad():
        for i, batch in enumerate(train_iter):
            src = batch.src.transpose(0, 1)[:2]
            # src = batch.src.transpose(0, 1)
            src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
            # out = utils.greedy_decode(model, src, src_mask, max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
            out = utils.beam_search_decode(model, src, SRC.vocab, TGT.vocab)
            print("Translation:", end="\t")
            for i in range(1, out.size(1)):
                # sym = TGT.vocab.itos[out[0, i]]
                sym = out[0, i].item()
                # if sym == "</s>": break
                if sym == 3: break
                print(sym, end=" ")
            print()
            print("Target:", end="\t")
            for i in range(1, batch.trg.size(0)):
                # sym = TGT.vocab.itos[batch.trg.data[i, 0]]
                sym = batch.trg.data[i, 0].item()
                # if sym == "</s>": break
                if sym == 3: break
                print(sym, end=" ")
            print()
            break
