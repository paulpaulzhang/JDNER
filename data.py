import os
from ark_nlp.factory.utils.conlleval import get_entity_bio

cat2id = {'1': 0, '10': 1, '11': 2, '12': 3, '13': 4, '14': 5,
          '15': 6, '16': 7, '17': 8, '18': 9, '19': 10, '2': 11,
          '20': 12, '21': 13, '22': 14, '23': 15, '24': 16, '25': 17,
          '26': 18, '28': 19, '29': 20, '3': 21, '30': 22, '31': 23,
          '32': 24, '33': 25, '34': 26, '35': 27, '36': 28, '37': 29,
          '38': 30, '39': 31, '4': 32, '40': 33, '41': 34, '42': 35,
          '43': 36, '44': 37, '46': 38, '47': 39, '48': 40, '49': 41,
          '5': 42, '50': 43, '51': 44, '52': 45, '53': 46, '54': 47,
          '6': 48, '7': 49, '8': 50, '9': 51, 'O': 52}

def read_data(path):
    datalist = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines.append('\n')

        text = []
        labels = []
        label_set = set()

        for line in lines:
            if line == '\n':
                text = ''.join(text)
                entity_labels = []
                for _type, _start_idx, _end_idx in get_entity_bio(labels, id2label=None):
                    entity_labels.append({
                        'start_idx': _start_idx,
                        'end_idx': _end_idx,
                        'type': _type,
                        'entity': text[_start_idx: _end_idx+1]
                    })

                if text == '':
                    continue

                datalist.append({
                    'text': text,
                    'label': entity_labels
                })

                text = []
                labels = []

            elif line == '  O\n':
                text.append(' ')
                labels.append('O')
            else:
                line = line.strip('\n').split()
                if len(line) == 1:
                    term = ' '
                    label = line[0]
                else:
                    term, label = line
                text.append(term)
                label_set.add(label.split('-')[-1])
                labels.append(label)
    return datalist, sorted(label_set)
