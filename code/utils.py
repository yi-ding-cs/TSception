import os
import time
import pprint
from networks import TSception
from eeg_dataset import *
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def set_gpu(x):
    torch.set_num_threads(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def seed_all(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def get_model(args):
    if args.model == 'TSception':
        model = TSception(
            num_classes=args.num_class, input_size=args.input_shape,
            sampling_rate=args.sampling_rate, num_T=args.T, num_S=args.T,
            hidden=args.hidden, dropout_rate=args.dropout)

    return model


def get_dataloader(data, label, batch_size, shuffle=True):
    # load the data
    dataset = eegDataset(data, label)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return loader


def get_metrics(y_pred, y_true, classes=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    if classes is not None:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    else:
        cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm


def get_trainable_parameter_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def L1Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(torch.abs(w))
    return err


def generate_TS_channel_order(original_order: list):
    """
    This function will generate the channel order for TSception
    Parameters
    ----------
    original_order: list of the channel names

    Returns
    -------
    TS: list of channel names which is for TSception
    """
    original_order_up = [item.upper() for item in original_order]
    chan_letter, chan_num = [], []
    for i, chan in enumerate(original_order_up):
        if len(chan)==2:
            chan_letter.append(chan[0])
            chan_num.append(chan[-1])
        elif len(chan)==3:
            chan_letter.append(chan[:2])
            chan_num.append(chan[-1])
    idx_pair = []
    for i, chan in enumerate(chan_letter):
        for j, chan_ in enumerate(chan_letter):
            if i!=j:
                if chan == chan_ and chan_num[i]!= 'Z' and \
                   chan_num[j] != 'Z' and int(chan_num[i]) - int(chan_num[j]) == -1:
                    idx_pair.append([i, j])
    idx_pair = np.array(idx_pair)
    idx_pair_t = idx_pair.T
    idx_pair = np.concatenate(idx_pair_t, axis=0).astype(int)
    return [original_order[item] for item in idx_pair]


if __name__=="__main__":
    # example of using generate_TS_channel_order()
    original_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                      'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6',
                      'CP2', 'P4', 'P8', 'PO4', 'O2']
    TS = generate_TS_channel_order(original_order)
    print('done')





