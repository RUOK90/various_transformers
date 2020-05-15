import time
import wandb
from config import ARGS
import utils


def run_epoch(train_iter, val_iter, model, criterion, opt, pad_idx, iter_cnt):
    model.train()
    min_val_loss = 1e9
    for train_batch in train_iter:
        train_batch = utils.rebatch(pad_idx, train_batch)
        train_out = model(train_batch.src, train_batch.trg, train_batch.src_mask, train_batch.trg_mask)
        train_loss = criterion(train_out.contiguous().view(-1, train_out.size(-1)), train_batch.trg_y.contiguous().view(-1)) / train_batch.ntokens

        train_loss.backward()
        opt.step()
        opt.optimizer.zero_grad()
        print(f'{iter_cnt} - train_loss: {train_loss.item()}')

        if iter_cnt % ARGS.eval_steps == 0:
            model.eval()
            val_total_loss = 0
            val_ntokens = 0
            for val_batch in val_iter:
                val_batch = utils.rebatch(pad_idx, val_batch)
                val_out = model(val_batch.src, val_batch.trg, val_batch.src_mask, val_batch.trg_mask)
                val_loss = criterion(val_out.contiguous().view(-1, val_out.size(-1)), val_batch.trg_y.contiguous().view(-1)) / val_batch.ntokens

                val_total_loss += val_loss.item()
                val_ntokens += val_batch.ntokens

            val_loss = val_total_loss / val_ntokens.float()
            min_val_loss = min(min_val_loss, val_loss)
            wandb.log({'Train loss': train_loss.item() / train_batch.ntokens.float(),
                        'Val loss': val_loss,
                        'Min Val loss': min_val_loss}, step=iter_cnt)

        iter_cnt += 1

    return iter_cnt


def _run_epoch(data_iter, model, criterion, opt=None):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in enumerate(data_iter):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = criterion(out.contiguous().view(-1, out.size(-1)), batch.trg_y.contiguous().view(-1)) / batch.ntokens

        if opt is not None:
            loss.backward()
            opt.step()
            opt.optimizer.zero_grad()

        loss = loss.item()
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / batch.ntokens.float(), tokens / elapsed))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens.float()
