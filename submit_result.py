from nezha.configuration_nezha import NeZhaConfig
from model import GlobalPointerModel
from nezha.modeling_nezha import NeZhaModel
from tokenizer import BertTokenizer
import torch
from predictor import GlobalPointerNERPredictor


cat2id = {'1': 0, '10': 1, '11': 2, '12': 3, '13': 4, '14': 5,
          '15': 6, '16': 7, '17': 8, '18': 9, '19': 10, '2': 11,
          '20': 12, '21': 13, '22': 14, '23': 15, '24': 16, '25': 17,
          '26': 18, '28': 19, '29': 20, '3': 21, '30': 22, '31': 23,
          '32': 24, '33': 25, '34': 26, '35': 27, '36': 28, '37': 29,
          '38': 30, '39': 31, '4': 32, '40': 33, '41': 34, '42': 35,
          '43': 36, '44': 37, '46': 38, '47': 39, '48': 40, '49': 41,
          '5': 42, '50': 43, '51': 44, '52': 45, '53': 46, '54': 47,
          '6': 48, '7': 49, '8': 50, '9': 51, 'O': 52}


class Args:
    def __init__(self):
        self.model_name_or_path = './'
        self.max_seq_len = 128
        self.cuda_device = 0
        self.predict_model = '/home/mw/project/checkpoint/nezha/best_model.pth'


args = Args()


def pred_BIO(path_word: str, path_sample: str, batch_size: int):
    result_path = '/home/mw/project/results.txt'

    tokenizer = BertTokenizer(vocab=args.model_name_or_path,
                              max_seq_len=args.max_seq_len)
    config = NeZhaConfig.from_pretrained(args.model_name_or_path,
                                         num_labels=len(cat2id))
    encoder = NeZhaModel(config)
    model = GlobalPointerModel(config, encoder)
    model.load_state_dict(torch.load(args.predict_model), strict=False)
    model.to(torch.device(f'cuda:{args.cuda_device}'))

    ner_predictor_instance = GlobalPointerNERPredictor(
        model, tokenizer, cat2id)

    predict_results = []

    with open(path_sample, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for _line in lines:
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

    with open(result_path, 'w', encoding='utf-8') as f:
        for _result in predict_results:
            for word, tag in zip(_result[0], _result[1]):
                if word == '\n':
                    continue
                f.write(f'{word} {tag}\n')
            f.write('\n')


if __name__ == '__main__':
    pred_BIO('../data/preliminary_test_b/sample_per_line_preliminary_B.txt',
             '../data/preliminary_test_b/sample_per_line_preliminary_B.txt', 1)
