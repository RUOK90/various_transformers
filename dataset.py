from config import ARGS
from torchtext import data, datasets


def get_dataset():
    import spacy
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"

    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD)
    train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(SRC, TGT), root=ARGS.data_path, filter_pred=lambda x: len(vars(x)['src']) <= ARGS.max_len and len(vars(x)['trg']) <= ARGS.max_len)

    SRC.build_vocab(train.src, min_freq=ARGS.min_freq)
    TGT.build_vocab(train.trg, min_freq=ARGS.min_freq)
    pad_idx = TGT.vocab.stoi["<blank>"]

    return SRC, TGT, train, val, test, pad_idx
