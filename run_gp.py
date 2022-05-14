from collections import Counter
import gc
import math
from ark_nlp.model.ner.global_pointer_bert import Dataset
from ark_nlp.model.ner.global_pointer_bert import Predictor
from ark_nlp.model.ner.global_pointer_bert import Tokenizer
from nezha.configuration_nezha import NeZhaConfig
from model import GlobalPointerModel, GlobalPointerBiLSTMModel
from nezha.modeling_nezha import NeZhaModel
from tokenizer import BertTokenizer
from utils import WarmupLinearSchedule, seed_everything, get_default_bert_optimizer
from sklearn.model_selection import train_test_split, KFold
from task import MyGlobalPointerNERTask, DistillTask
from data import read_data
from tqdm import tqdm
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
import torch
import os
import warnings

from predictor import GlobalPointerNERPredictor


def train(args):
    datalist, label_set = read_data(args.data_path)
    train_data_df = pd.DataFrame(datalist)
    train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))
    train_data_df, dev_data_df = train_test_split(
        train_data_df, test_size=0.1, shuffle=True, random_state=args.seed)

    ner_train_dataset = Dataset(train_data_df, categories=label_set)
    ner_dev_dataset = Dataset(
        dev_data_df, categories=ner_train_dataset.categories)

    tokenizer = BertTokenizer(vocab=args.model_name_or_path,
                              max_seq_len=args.max_seq_len)
    config = NeZhaConfig.from_pretrained(args.model_name_or_path,
                                         num_labels=len(ner_train_dataset.cat2id))
    encoder = NeZhaModel.from_pretrained(args.model_name_or_path,
                                         config=config)
    dl_module = GlobalPointerBiLSTMModel(config, encoder)

    ner_train_dataset.convert_to_ids(tokenizer)
    ner_dev_dataset.convert_to_ids(tokenizer)

    optimizer = get_default_bert_optimizer(dl_module, args)

    if args.warmup_ratio:
        train_steps = args.num_epochs * \
            int(math.ceil(math.ceil(len(ner_train_dataset) /
                args.batch_size) / args.gradient_accumulation_steps))
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)
    else:
        scheduler = None

    torch.cuda.empty_cache()

    model = MyGlobalPointerNERTask(
        dl_module, optimizer, 'gpce',
        scheduler=scheduler,
        ema_decay=args.ema_decay,
        device=args.device)

    model.fit(args,
              ner_train_dataset,
              ner_dev_dataset,
              epochs=args.num_epochs,
              batch_size=args.batch_size,
              save_last_model=False)


def evaluate(args):
    datalist, label_set = read_data(args.data_path)
    train_data_df = pd.DataFrame(datalist)
    train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))
    train_data_df, dev_data_df = train_test_split(
        train_data_df, test_size=0.1, shuffle=True, random_state=args.seed)

    ner_train_dataset = Dataset(train_data_df, categories=label_set)
    ner_dev_dataset = Dataset(
        dev_data_df, categories=ner_train_dataset.categories)

    tokenizer = BertTokenizer(vocab=args.model_name_or_path,
                              max_seq_len=args.max_seq_len)
    config = NeZhaConfig.from_pretrained(args.model_name_or_path,
                                         num_labels=len(ner_train_dataset.cat2id))
    encoder = NeZhaModel(config)
    dl_module = GlobalPointerBiLSTMModel(config, encoder)
    dl_module.load_state_dict(torch.load(args.predict_model))

    ner_train_dataset.convert_to_ids(tokenizer)
    ner_dev_dataset.convert_to_ids(tokenizer)

    optimizer = get_default_bert_optimizer(dl_module, args)

    torch.cuda.empty_cache()

    model = MyGlobalPointerNERTask(
        dl_module, optimizer, 'gpce',
        ema_decay=args.ema_decay,
        device=args.device)

    model.id2cat = ner_train_dataset.id2cat
    model.evaluate(ner_dev_dataset)


def predict(args):
    datalist, label_set = read_data(args.data_path)
    train_data_df = pd.DataFrame(datalist)
    train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))
    train_data_df, _ = train_test_split(
        train_data_df, test_size=0.1, shuffle=True, random_state=args.seed)

    ner_train_dataset = Dataset(train_data_df, categories=label_set)

    tokenizer = BertTokenizer(vocab=args.model_name_or_path,
                              max_seq_len=args.max_seq_len)
    config = NeZhaConfig.from_pretrained(args.model_name_or_path,
                                         num_labels=len(ner_train_dataset.cat2id))
    encoder = NeZhaModel(config)
    model = GlobalPointerBiLSTMModel(config, encoder)
    model.load_state_dict(torch.load(args.predict_model), strict=False)
    model.to(torch.device(args.device))

    ner_predictor_instance = GlobalPointerNERPredictor(
        model, tokenizer, ner_train_dataset.cat2id)

    predict_results = []

    with open(args.test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for _line in tqdm(lines):
            label = len(_line) * ['O']
            for _preditc in ner_predictor_instance.predict_one_sample(_line[:-1]):
                if 'I' in label[_preditc['start_idx']]:
                    continue
                if 'B' in label[_preditc['start_idx']] and 'O' not in label[_preditc['end_idx']]:
                    continue
                if 'O' in label[_preditc['start_idx']] and 'B' in label[_preditc['end_idx']]:
                    continue

                label[_preditc['start_idx']] = 'B-' + _preditc['type']
                label[_preditc['start_idx']+1: _preditc['end_idx']+1] = (
                    _preditc['end_idx'] - _preditc['start_idx']) * [('I-' + _preditc['type'])]

            predict_results.append([_line, label])

    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, f'{args.model_type}.txt'), 'w', encoding='utf-8') as f:
        for _result in predict_results:
            for word, tag in zip(_result[0], _result[1]):
                if word == '\n':
                    continue
                f.write(f'{word} {tag}\n')
            f.write('\n')


def train_cv(args):
    datalist, label_set = read_data(args.data_path)
    data_df = pd.DataFrame(datalist)
    data_df['label'] = data_df['label'].apply(lambda x: str(x))

    kfold = KFold(n_splits=args.fold, shuffle=True, random_state=args.seed)
    args.checkpoint = os.path.join(args.checkpoint, args.model_type)
    model_type = args.model_type
    for fold, (train_idx, dev_idx) in enumerate(kfold.split(data_df)):
        print(f'========== {fold + 1} ==========')

        args.model_type = f'{model_type}-{fold + 1}'

        train_data_df, dev_data_df = data_df.iloc[train_idx], data_df.iloc[dev_idx]
        ner_train_dataset = Dataset(train_data_df, categories=label_set)
        ner_dev_dataset = Dataset(
            dev_data_df, categories=ner_train_dataset.categories)

        tokenizer = BertTokenizer(vocab=args.model_name_or_path,
                                  max_seq_len=args.max_seq_len)
        config = NeZhaConfig.from_pretrained(args.model_name_or_path,
                                             num_labels=len(ner_train_dataset.cat2id))
        encoder = NeZhaModel.from_pretrained(args.model_name_or_path,
                                             config=config)
        dl_module = GlobalPointerBiLSTMModel(config, encoder)

        ner_train_dataset.convert_to_ids(tokenizer)
        ner_dev_dataset.convert_to_ids(tokenizer)

        optimizer = get_default_bert_optimizer(dl_module, args)

        if args.warmup_ratio:
            train_steps = args.num_epochs * \
                int(math.ceil(math.ceil(len(ner_train_dataset) /
                    args.batch_size) / args.gradient_accumulation_steps))
            scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)
        else:
            scheduler = None

        model = MyGlobalPointerNERTask(
            dl_module, optimizer, 'gpce',
            scheduler=scheduler,
            ema_decay=args.ema_decay,
            device=args.device)

        model.fit(args,
                  ner_train_dataset,
                  ner_dev_dataset,
                  epochs=args.num_epochs,
                  batch_size=args.batch_size,
                  save_last_model=False,
                  gradient_accumulation_steps=args.gradient_accumulation_steps)

        del model, tokenizer, dl_module, encoder, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()


def predict_vote(args):
    datalist, label_set = read_data(args.data_path)
    data_df = pd.DataFrame(datalist)
    data_df['label'] = data_df['label'].apply(lambda x: str(x))

    kfold = KFold(n_splits=args.fold, shuffle=True, random_state=args.seed)

    args.checkpoint = os.path.join(args.checkpoint, args.model_type)
    model_type = args.model_type

    args.save_path = os.path.join(args.save_path, model_type)
    os.makedirs(args.save_path, exist_ok=True)

    for fold, (train_idx, _) in enumerate(kfold.split(data_df)):
        print(f'========== {fold + 1} ==========')
        args.model_type = f'{model_type}-{fold + 1}'
        args.predict_model = os.path.join(
            args.checkpoint, args.model_type, 'best_model.pth')

        train_data_df = data_df.iloc[train_idx]
        ner_train_dataset = Dataset(train_data_df, categories=label_set)

        tokenizer = BertTokenizer(vocab=args.model_name_or_path,
                                  max_seq_len=args.max_seq_len)
        config = NeZhaConfig.from_pretrained(args.model_name_or_path,
                                             num_labels=len(ner_train_dataset.cat2id))
        encoder = NeZhaModel(config)
        model = GlobalPointerBiLSTMModel(config, encoder)
        model.load_state_dict(torch.load(args.predict_model), strict=False)
        model.to(torch.device(args.device))

        ner_predictor_instance = GlobalPointerNERPredictor(
            model, tokenizer, ner_train_dataset.cat2id)

        predict_results = []

        with open(args.test_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for _line in tqdm(lines):
                label = len(_line) * ['O']
                for _preditc in ner_predictor_instance.predict_one_sample(_line[:-1]):
                    if 'I' in label[_preditc['start_idx']]:
                        continue
                    if 'B' in label[_preditc['start_idx']] and 'O' not in label[_preditc['end_idx']]:
                        continue
                    if 'O' in label[_preditc['start_idx']] and 'B' in label[_preditc['end_idx']]:
                        continue

                    label[_preditc['start_idx']] = 'B-' + _preditc['type']
                    label[_preditc['start_idx']+1: _preditc['end_idx']+1] = (
                        _preditc['end_idx'] - _preditc['start_idx']) * [('I-' + _preditc['type'])]

                predict_results.append([_line, label])

        with open(os.path.join(args.save_path, f'{model_type}-{fold + 1}.txt'), 'w', encoding='utf-8') as f:
            for _result in predict_results:
                for word, tag in zip(_result[0], _result[1]):
                    if word == '\n':
                        continue
                    f.write(f'{word} {tag}\n')
                f.write('\n')


def distill(args):
    datalist, label_set = read_data(args.data_path)
    train_data_df = pd.DataFrame(datalist)
    train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))
    train_data_df, dev_data_df = train_test_split(
        train_data_df, test_size=0.1, shuffle=True, random_state=args.seed)

    ner_train_dataset = Dataset(train_data_df, categories=label_set)
    ner_dev_dataset = Dataset(
        dev_data_df, categories=ner_train_dataset.categories)

    tokenizer = BertTokenizer(vocab=args.model_name_or_path,
                              max_seq_len=args.max_seq_len)
    config = NeZhaConfig.from_pretrained(args.model_name_or_path,
                                         num_labels=len(ner_train_dataset.cat2id))
    encoder = NeZhaModel.from_pretrained(args.model_name_or_path,
                                         config=config)
    student = GlobalPointerBiLSTMModel(config, encoder)

    def teacher(path, config, device):
        encoder = NeZhaModel(config)
        model = GlobalPointerBiLSTMModel(config, encoder)
        model.load_state_dict(torch.load(path), strict=False)
        model.to(torch.device(device))
        model.eval()
        return model

    teachers = [teacher(os.path.join(args.checkpoint, args.model_type,
                                     f'{args.model_type}-{fold + 1}', 'best_model.pth'),
                        config, args.device) for fold in range(args.fold)]

    ner_train_dataset.convert_to_ids(tokenizer)
    ner_dev_dataset.convert_to_ids(tokenizer)

    optimizer = get_default_bert_optimizer(student, args)

    if args.warmup_ratio:
        train_steps = args.num_epochs * \
            int(math.ceil(math.ceil(len(ner_train_dataset) /
                args.batch_size) / args.gradient_accumulation_steps))
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)
    else:
        scheduler = None

    torch.cuda.empty_cache()

    model = DistillTask(teachers,
                        student, optimizer, 'gpce',
                        scheduler=scheduler,
                        ema_decay=args.ema_decay,
                        device=args.device)

    model.fit(args,
              ner_train_dataset,
              ner_dev_dataset,
              epochs=args.num_epochs,
              batch_size=args.batch_size,
              save_last_model=False)


def vote(args):
    path = [str(p) for p in list(Path(args.vote_path).glob('**/*.txt'))]
    all_labels = []
    chars = []

    with open(path[0], 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line != '\n':
                chars.append(line[0])
            else:
                chars.append('\n')

    for p in path:
        labels = []
        with open(p, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if line != '\n':
                    labels.append(line[2:].strip('\n'))
                else:
                    labels.append(line)
        all_labels.append(labels)

    merged_label = []
    for row in zip(*all_labels):
        label = Counter(row).most_common(n=1)
        merged_label.append(label[0][0])

    with open(os.path.join(args.extend_save_path, args.save_name), 'w', encoding='utf-8') as f:
        for char, label in zip(chars, merged_label):
            if char != '\n':
                f.write(f'{char} {label}\n')
            else:
                f.write('\n')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model_type', type=str,
                        default='nezha-cn-base')
    parser.add_argument('--model_name_or_path', type=str,
                        default='../pretrain_model/nezha-cn-base/')

    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoint')
    parser.add_argument('--data_path', type=str,
                        default='../data/train_data/train.txt')
    parser.add_argument('--test_file', type=str,
                        default='../data/preliminary_test_b/sample_per_line_preliminary_B.txt')
    parser.add_argument('--save_path', type=str, default='./submit')

    parser.add_argument('--do_predict', action='store_true', default=False)
    parser.add_argument('--do_eval', action='store_true', default=False)
    parser.add_argument('--do_train_cv', action='store_true', default=False)
    parser.add_argument('--do_predict_vote',
                        action='store_true', default=False)
    parser.add_argument('--do_vote', action='store_true', default=False)
    parser.add_argument('--do_distill', action='store_true', default=False)
    parser.add_argument('--predict_model', type=str)

    parser.add_argument('--max_seq_len', type=int, default=128)

    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--lstm_lr', type=float, default=1e-2)
    parser.add_argument('--gp_lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)

    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--early_stopping', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--use_fgm', action='store_true', default=False)
    parser.add_argument('--use_pgd', action='store_true', default=False)
    parser.add_argument('--use_awp', action='store_true', default=False)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)

    parser.add_argument('--adv_k', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--emb_name', type=str, default='word_embeddings.')
    parser.add_argument('--adv_lr', type=int, default=1)
    parser.add_argument('--adv_eps', type=int, default=0.001)

    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--extend_save_path', type=str,
                        default='./extend_data/')
    parser.add_argument('--save_name', type=str, default='vote.txt')
    parser.add_argument('--vote_path', type=str,
                        default='./submit/')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    seed_everything(args.seed)

    for k, v in vars(args).items():
        print(f'{k}: {v}')

    if args.do_predict:
        predict(args)
    elif args.do_eval:
        evaluate(args)
    elif args.do_train_cv:
        train_cv(args)
    elif args.do_predict_vote:
        predict_vote(args)
    elif args.do_vote:
        vote(args)
    elif args.do_distill:
        distill(args)
    else:
        train(args)
