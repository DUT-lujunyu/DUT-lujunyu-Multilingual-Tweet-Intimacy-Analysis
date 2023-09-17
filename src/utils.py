import torch, random
import numpy as np


def setup_seed(seed):
    '''
    Fixed random seed.
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def To_tensor(sample_batched):
    '''
    Generate the data dictionary required by Bert-model
    '''
    labels = [b['label'] for b in sample_batched]
    raw_text_batch = [b['ori_text'] for b in sample_batched]
    args = {'labels': labels,'ori_text': raw_text_batch}
    args['text'] = torch.tensor([b['text'] for b in sample_batched])
    args['mask'] = torch.tensor([b['mask'] for b in sample_batched])
    return args


def save(epoch, model, exp_name, checkpoint_path, check_num):
        '''
        Saves the pytorch model in a checkpoint file.
        '''
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, '{}ckp-{}-{}.tar'.format(checkpoint_path, exp_name, check_num))

