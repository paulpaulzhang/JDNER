from collections import Counter
import gc
import math
from ark_nlp.model.ner.global_pointer_bert import get_default_model_optimizer
from ark_nlp.model.ner.global_pointer_bert import Dataset
from ark_nlp.model.ner.global_pointer_bert import Predictor
from ark_nlp.model.ner.global_pointer_bert import Tokenizer
from nezha.configuration_nezha import NeZhaConfig
from model import GlobalPointerModel
from nezha.modeling_nezha import NeZhaModel
from tokenizer import BertTokenizer
from utils import WarmupLinearSchedule, seed_everything
from sklearn.model_selection import train_test_split, KFold
from task import MyGlobalPointerNERTask
from data import read_data
from tqdm import tqdm
from argparse import ArgumentParser
import pandas as pd
import torch
import os
import warnings


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
    dl_module = GlobalPointerModel(config, encoder)

    ner_train_dataset.convert_to_ids(tokenizer)
    ner_dev_dataset.convert_to_ids(tokenizer)

    train_steps = args.num_epochs * \
        int(math.ceil(len(ner_train_dataset) / args.batch_size))
    optimizer = get_default_model_optimizer(dl_module)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)

    torch.cuda.empty_cache()

    model = MyGlobalPointerNERTask(
        dl_module, optimizer, 'gpce',
        scheduler=scheduler,
        ema_decay=args.ema_decay,
        cuda_device=args.cuda_device)

    model.fit(args,
              ner_train_dataset,
              ner_dev_dataset,
              lr=args.learning_rate,
              epochs=args.num_epochs,
              batch_size=args.batch_size,
              save_each_model=False)


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
    dl_module = GlobalPointerModel(config, encoder)
    dl_module.load_state_dict(torch.load(args.predict_model))

    ner_train_dataset.convert_to_ids(tokenizer)
    ner_dev_dataset.convert_to_ids(tokenizer)

    optimizer = get_default_model_optimizer(dl_module)

    torch.cuda.empty_cache()

    model = MyGlobalPointerNERTask(
        dl_module, optimizer, 'gpce',
        ema_decay=args.ema_decay,
        cuda_device=args.cuda_device)

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
    model = GlobalPointerModel(config, encoder)
    model.load_state_dict(torch.load(args.predict_model))
    model.to(torch.device(f'cuda:{args.cuda_device}'))

    ner_predictor_instance = Predictor(
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
        dl_module = GlobalPointerModel(config, encoder)

        ner_train_dataset.convert_to_ids(tokenizer)
        ner_dev_dataset.convert_to_ids(tokenizer)

        train_steps = args.num_epochs * \
            int(math.ceil(len(ner_train_dataset) / args.batch_size))
        optimizer = get_default_model_optimizer(dl_module)
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)

        model = MyGlobalPointerNERTask(
            dl_module, optimizer, 'gpce',
            scheduler=scheduler,
            ema_decay=args.ema_decay,
            cuda_device=args.cuda_device)

        model.fit(args,
                  ner_train_dataset,
                  ner_dev_dataset,
                  lr=args.learning_rate,
                  epochs=args.num_epochs,
                  batch_size=args.batch_size,
                  save_each_model=False)

        del model, tokenizer, dl_module, encoder
        gc.collect()
        torch.cuda.empty_cache()


def predict_cv(args):
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
        model = GlobalPointerModel(config, encoder)
        model.load_state_dict(torch.load(args.predict_model))
        model.to(torch.device(f'cuda:{args.cuda_device}'))

        ner_predictor_instance = Predictor(
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


def merge_cv_result(args):
    save_path = os.path.join(args.save_path, args.model_type)
    all_labels = []
    chars = []

    path = os.path.join(
        save_path, f'{args.model_type}-1.txt')
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line != '\n':
                chars.append(line[0])
            else:
                chars.append('\n')

    for fold in range(1, args.fold+1):
        labels = []
        path = os.path.join(
            save_path, f'{args.model_type}-{fold}.txt')
        with open(path, 'r', encoding='utf-8') as f:
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
                        default='gp-nezha-base')
    parser.add_argument('--model_name_or_path', type=str,
                        default='../pretrain_model/nezha-base/')

    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoint')
    parser.add_argument('--data_path', type=str,
                        default='../data/train_data/train.txt')
    parser.add_argument('--test_file', type=str,
                        default='../data/preliminary_test_a/sample_per_line_preliminary_A.txt')
    parser.add_argument('--save_path', type=str, default='./submit')

    parser.add_argument('--do_predict', action='store_true', default=False)
    parser.add_argument('--do_eval', action='store_true', default=False)
    parser.add_argument('--do_train_cv', action='store_true', default=False)
    parser.add_argument('--do_predict_cv', action='store_true', default=False)
    parser.add_argument('--do_merge', action='store_true', default=False)
    parser.add_argument('--predict_model', type=str)

    parser.add_argument('--max_seq_len', type=int, default=128)

    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--use_fgm', action='store_true', default=False)
    parser.add_argument('--use_pgd', action='store_true', default=True)

    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--adv_k', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--epsilon', type=float, default=0.3)
    parser.add_argument('--emb_name', type=str, default='word_embeddings.')

    parser.add_argument('--fold', type=int, default=10)
    parser.add_argument('--extend_save_path', type=str,
                        default='./extend_data/')
    parser.add_argument('--save_name', type=str, default='merged_res.txt')

    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    seed_everything(args.seed)

    print(args)

    if args.do_predict:
        predict(args)
    elif args.do_eval:
        evaluate(args)
    elif args.do_train_cv:
        train_cv(args)
    elif args.do_predict_cv:
        predict_cv(args)
    elif args.do_merge:
        merge_cv_result(args)
    else:
        train(args)
