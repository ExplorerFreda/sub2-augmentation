import collections
import copy
import json


def search_hyperparams(current_hparams, dictionary):
    if len(dictionary) == 0:
        yield copy.deepcopy(current_hparams)
        return
    current_dictionary = dict()
    current_key = list(dictionary.keys())[0]
    current_flag = dictionary[current_key].get('flag', None)
    if current_flag is not None:
        for key in copy.deepcopy(dictionary):
            if dictionary[key].get('flag', None) == current_flag:
                current_dictionary[key] = copy.deepcopy(dictionary[key])
                del dictionary[key]
    else:
        current_dictionary[current_key] = copy.deepcopy(
            dictionary[current_key]
        )
        del dictionary[current_key]
    num_values = len(current_dictionary[current_key]['values'])
    for key in current_dictionary:
        assert num_values == len(current_dictionary[key]['values']), \
            'hparams with the same flag must have the same #values\n' \
            'check {:s} and {:s}'.format(key, current_key)
    for i in range(num_values):
        for key in current_dictionary:
            current_hparams[key] = current_dictionary[key]['values'][i]
        for item in search_hyperparams(current_hparams, dictionary):
            yield item
    for key in current_dictionary:
        dictionary[key] = copy.deepcopy(current_dictionary[key])


class AverageMeter(object):
    """Computes and stores the average and current value"""
    val = 0
    avg = 0
    sum = 0
    count = 0
    tot_count = 0

    def __init__(self):
        self.reset()
        self.tot_count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.tot_count += n
        self.avg = self.sum / self.count


def save_result(data, fname):
    with open(fname, 'a') as fout:
        for item in data:
            print(json.dumps(item), file=fout)
    fout.close()


def collect_result(data, collect_key='lang'):
    result = collections.defaultdict(None)
    for line in open(data):
        configs, best_dev, test = json.loads(line)
        key = configs[collect_key]
        if key in result:
            if best_dev > result[key][0]:
                result[key] = (best_dev, test)
        else:
            result[key] = (best_dev, test)
    return result
