# This is the processing script of DEAP dataset

import _pickle as cPickle

from train_model import *


class PrepareData:
    def __init__(self, args):
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.data_path = args.data_path
        self.label_type = args.label_type
        self.original_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                               'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6',
                               'CP2', 'P4', 'P8', 'PO4', 'O2']

        self.TS = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3','O1',
                   'Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
        self.graph_type = args.graph_type

    def run(self, subject_list, split=False, expand=True, feature=False):
        """
        Parameters
        ----------
        subject_list: the subjects need to be processed
        split: (bool) whether to split one trial's data into shorter segment
        expand: (bool) whether to add an empty dimension for CNN
        feature: (bool) whether to extract features or not

        Returns
        -------
        The processed data will be saved './data_<data_format>_<dataset>_<label_type>/sub0.hdf'
        """
        for sub in subject_list:
            data_, label_ = self.load_data_per_subject(sub)
            # select label type here
            label_ = self.label_selection(label_)

            if split:
                data_, label_ = self.split(
                    data=data_, label=label_, segment_length=self.args.segment,
                    overlap=self.args.overlap, sampling_rate=self.args.sampling_rate)

            if expand:
                # expand one dimension for deep learning(CNNs)
                data_ = np.expand_dims(data_, axis=-3)
            print('Data and label prepared for sub{}!'.format(sub))
            print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
            print('----------------------')
            self.save(data_, label_, sub)

    def load_data_per_subject(self, sub):
        """
        This function loads the target subject's original file
        Parameters
        ----------
        sub: which subject to load

        Returns
        -------
        data: (40, 32, 7680) label: (40, 4)
        """
        sub += 1
        if (sub < 10):
            sub_code = str('s0' + str(sub) + '.dat')
        else:
            sub_code = str('s' + str(sub) + '.dat')

        subject_path = os.path.join(self.data_path, sub_code)
        subject = cPickle.load(open(subject_path, 'rb'), encoding='latin1')
        label = subject['labels']
        data = subject['data'][:, 0:32, 3 * 128:]  # Excluding the first 3s of baseline
        #   data: 40 x 32 x 7680
        #   label: 40 x 4
        # reorder the EEG channel to build the local-global graphs
        data = self.reorder_channel(data=data, graph=self.graph_type)
        print('data:' + str(data.shape) + ' label:' + str(label.shape))
        return data, label

    def reorder_channel(self, data, graph):
        """
        This function reorder the channel according to different graph designs
        Parameters
        ----------
        data: (trial, channel, data)
        graph: graph type

        Returns
        -------
        reordered data: (trial, channel, data)
        """
        if graph == 'TS':
            graph_idx = self.TS
        elif graph == 'O':
            graph_idx = self.original_order

        idx = []

        for chan in graph_idx:
            idx.append(self.original_order.index(chan))

        return data[:, idx, :]

    def label_selection(self, label):
        """
        This function: 1. selects which dimension of labels to use
                       2. create binary label
        Parameters
        ----------
        label: (trial, 4)

        Returns
        -------
        label: (trial,)
        """
        if self.label_type == 'A':
            label = label[:, 1]
        elif self.label_type == 'V':
            label = label[:, 0]
        if self.args.num_class == 2:
            label = np.where(label <= 5, 0, label)
            label = np.where(label > 5, 1, label)
            print('Binary label generated!')
        return label

    def save(self, data, label, sub):
        """
        This function save the processed data into target folder
        Parameters
        ----------
        data: the processed data
        label: the corresponding label
        sub: the subject ID

        Returns
        -------
        None
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        save_path = osp.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub) + '.hdf'
        save_path = osp.join(save_path, name)
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()

    def split(self, data, label, segment_length=1, overlap=0, sampling_rate=256):
        """
        This function split one trial's data into shorter segments
        Parameters
        ----------
        data: (trial, channel, data)
        label: (trial,)
        segment_length: how long each segment is (e.g. 1s, 2s,...)
        overlap: overlap rate
        sampling_rate: sampling rate

        Returns
        -------
        data:(tiral, num_segment, channel, segment_legnth)
        label:(trial, num_segment,)
        """
        data_shape = data.shape
        step = int(segment_length * sampling_rate * (1 - overlap))
        data_segment = sampling_rate * segment_length
        data_split = []

        number_segment = int((data_shape[2] - data_segment) // step)
        for i in range(number_segment + 1):
            data_split.append(data[:, :, (i * step):(i * step + data_segment)])
        data_split_array = np.stack(data_split, axis=1)
        label = np.stack([np.repeat(label[i], int(number_segment + 1)) for i in range(len(label))], axis=0)
        print("The data and label are split: Data shape:" + str(data_split_array.shape) + " Label:" + str(
            label.shape))
        data = data_split_array
        assert len(data) == len(label)
        return data, label
