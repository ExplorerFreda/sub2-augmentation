import collections
import copy
import json
import torch 
from tqdm import tqdm


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


def slurm_job_babysit(executor, job_func, jobs, all_configs):
    resubmit_ids = list()
    new_configs = list()
    cnt_complete = 0
    bar = tqdm(jobs)
    for idx, job in enumerate(bar):
        state = job.state
        if state != 'RUNNING' and state != 'COMPLETED' and \
                state != 'PENDING' and state != 'UNKNOWN':
            new_configs.append(all_configs[idx])
            resubmit_ids.append(idx)
        elif state == 'COMPLETED':
            cnt_complete += 1
        bar.set_description(
            f'{cnt_complete} jobs finished, '
            f'{len(resubmit_ids)} jobs resubmitting'
        )
    print(f'{cnt_complete} jobs finished!')
    print(resubmit_ids)
    print([jobs[x] for x in resubmit_ids])
    if len(resubmit_ids) > 0:
        new_jobs = executor.map_array(job_func, new_configs)
        for idx, job in enumerate(new_jobs):
            jobs[resubmit_ids[idx]] = job
    return cnt_complete


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


def save_checkpoint(
            path, model, best_dev_model, best_dev_performance,
            optimizer, lr_scheduler, epoch_id
        ):
    print(f'Saving model to checkpoint path {path}.', flush=True)
    random_state = torch.get_rng_state()
    cuda_random_state = torch.cuda.get_rng_state() \
        if torch.cuda.is_available() else None
    checkpoint = {
        'random_state': random_state,
        'cuda_random_state': cuda_random_state,
        'model': model.state_dict(),
        'best_dev_model': best_dev_model,
        'best_dev_performance': best_dev_performance,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None \
            else None,
        'epoch_id': epoch_id
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, lr_scheduler):
    print(f'Loading checkpoint from {path}.', flush=True)
    state_dicts = torch.load(path)
    torch.set_rng_state(state_dicts['random_state'])
    if state_dicts['cuda_random_state'] is not None:
        torch.cuda.set_rng_state(state_dicts['cuda_random_state'])
    model.load_state_dict(state_dicts['model'])
    best_dev_model = state_dicts['best_dev_model']
    best_dev_performance = state_dicts['best_dev_performance']
    optimizer.load_state_dict(state_dicts['optimizer'])
    if state_dicts['lr_scheduler'] is not None:
        lr_scheduler.load_state_dcit(state_dicts['lr_scheduler'])
    return state_dicts['epoch_id'], best_dev_model, best_dev_performance
