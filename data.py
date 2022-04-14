import os
from ark_nlp.factory.utils.conlleval import get_entity_bio


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
