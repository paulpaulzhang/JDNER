from argparse import ArgumentParser
from collections import Counter
import os
import random


def gen_train_line():
    with open('./data/contest_data/train_data/train.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    labels = []
    sentences = []
    sentence = ''
    for line in lines:
        if line != '\n':
            labels.append(line[2:].strip('\n'))
            sentence += line[0]
        else:
            sentences.append(sentence)
            sentence = ''

    with open('./data/tmp_data/sample_per_line_train.txt', 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')


def gen_stage2_data():
    with open('./data/contest_data/train_data/train.txt', 'r', encoding='utf-8') as f:
        lines1 = f.readlines()

    with open('./data/tmp_data/train_pl_4w.txt', 'r', encoding='utf-8') as f:
        lines3 = f.readlines()

    with open('./data/tmp_data/mix_train_data_8w.txt', 'w', encoding='utf-8') as f:
        for line in lines1:
            f.write(line)
        for line in lines3:
            f.write(line)


def get_unlabel_data_4w():
    with open('./data/contest_data/train_data/unlabeled_train_data.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open('./data/tmp_data/sample_per_line_extend_data_4w.txt', 'w', encoding='utf-8') as f:
        for line in lines[:40000]:
            f.write(line)

def get_unlabel_data_8w():
    with open('./data/contest_data/train_data/unlabeled_train_data.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open('./data/tmp_data/sample_per_line_extend_data_4w.txt', 'w', encoding='utf-8') as f:
        for line in lines[:80000]:
            f.write(line)


def diff(args):
    with open(os.path.join(args.extend_save_path, args.save_name), 'r', encoding='utf-8') as f:
        d_lines = f.readlines()
    with open('../data/train_data/train.txt', 'r', encoding='utf-8') as f:
        t_lines = f.readlines()

    with open('./extend_data/diff.txt', 'w', encoding='utf-8') as f:
        for d, t in zip(d_lines, t_lines):
            if d != t:
                f.write(t.strip('\n') + '\t' + d)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--gen_train_line', action='store_true', default=False)
    parser.add_argument('--gen_stage2_data',
                        action='store_true', default=False)
    parser.add_argument('--get_unlabel_data_4w',
                        action='store_true', default=False)
    parser.add_argument('--get_unlabel_data_8w',
                        action='store_true', default=False)

    args = parser.parse_args()

    if args.get_unlabel_data_4w:
        print('get_unlabel_data_4w')
        get_unlabel_data_4w()

    if args.get_unlabel_data_8w:
        print('get_unlabel_data_8w')
        get_unlabel_data_8w()

    if args.gen_stage2_data:
        print('gen_stage2_data')
        gen_stage2_data()

    if args.gen_train_line:
        print('gen_train_line')
        gen_train_line()
