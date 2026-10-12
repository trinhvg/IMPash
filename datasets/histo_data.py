import glob
import random
from collections import Counter

import collections
import json


def print_number_of_sample(train_set, valid_set, test_set):


    train_label = [train_set[i][1] for i in range(len(train_set))]

    print("train", collections.OrderedDict(sorted(Counter(train_label).items())))
    valid_label = [valid_set[i][1] for i in range(len(valid_set))]
    print("valid", collections.OrderedDict(sorted(Counter(valid_label).items())))
    test_label = [test_set[i][1] for i in range(len(test_set))]
    print("test", collections.OrderedDict(sorted(Counter(test_label).items())))
    return 0


def prepare_colon_class_data_from_json(dataset_name='k19', fold_idx=0):
    if dataset_name == 'k19':
        data_root_dir = '/data2/trinh/data/raw_data/Domain_Invariance/colon_class/NCT-CRC-HE-100K/'
        json_dir = './datasets/K19_9class_split.json'
        with open(json_dir) as json_file:
            data = json.load(json_file)

    elif dataset_name == 'k16':
        data_root_dir = '/data2/trinh/data/raw_data/Domain_Invariance/colon_class/Kather_texture_2016_image_tiles_5000'
    else:
        raise (f'dataset {dataset_name} is not implemented')


    train_set = data['train_set']
    valid_set = data['valid_set']
    test_set = data['test_set']
    train_set = [[data_root_dir + train_set[i][0], train_set[i][1]] for i in range(len(train_set))]
    valid_set = [[data_root_dir + valid_set[i][0], valid_set[i][1]] for i in range(len(valid_set))]
    test_set = [[data_root_dir + test_set[i][0], test_set[i][1]] for i in range(len(test_set))]
    print(len(train_set))
    return train_set, valid_set, test_set

def prepare_colon_class_data(dataset_name='k19', fold_idx=0):
    def load_data_info(data_name, pathname, covert_dict, parse_label=True, label_value=0):
        file_list = glob.glob(pathname)
        if data_name == 'k16':
            COMPLEX_list = glob.glob(f'{data_root_dir_k16}/03_COMPLEX/*.tif')
            file_list = [elem for elem in file_list if elem not in COMPLEX_list]

        label_list = [int(covert_dict[file_path.split('/')[-2]][1]) for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    assert fold_idx < 3, "Currently only support 5 fold, each fold is 1 TMA"

    const_kather19 = {
        'ADI': ('ADI', 0), 'BACK': ('BACK', 1), 'DEB': ('DEB', 2), 'LYM': ('LYM', 3), 'MUC': ('MUC', 4),
        'MUS': ('MUS', 5), 'NORM': ('NORM', 6), 'STR': ('STR', 7), 'TUM': ('TUM', 8)
    }

    # we don't use the complex stroma here
    const_kather16 = {
        '07_ADIPOSE': ('07_ADIPOSE', 0), '08_EMPTY': ('08_EMPTY', 1), '05_DEBRIS': ('05_DEBRIS', 2),
        '04_LYMPHO': ('04_LYMPHO', 3), '06_MUCOSA': ('06_MUCOSA', 6), '02_STROMA': ('02_STROMA', 7),
        '01_TUMOR': ('01_TUMOR', 8)
    }

    const_kather16_2 = {
        '01_TUMOR':0, '02_STROMA':1, '03_COMPLEX':7, '04_LYMPHO':2, '05_DEBRIS':3, '06_MUCOSA':4, '07_ADIPOSE':5, '08_EMPTY':6
    }
    const_crctp = {
        'Benign': ('NORM', 4), 'Debris': ('DEB', 3),
        'Inflammatory': ('LYM', 2), 'Muscle': ('MUS', 1), 'Stroma': ('STR', 1), 'Tumor': ('TUM', 0)
    }

    # Split sets is train or val sets.
    val_ratio = 0.3

    if dataset_name == 'k19':
        data_root_dir_k19 = '/data2/trinh/data/raw_data/Domain_Invariance/colon_class/NCT-CRC-HE-100K/'
        set_k19 = load_data_info(data_name='k19', pathname = f'{data_root_dir_k19}/*/*.tif', covert_dict=const_kather19)
        train_list = set_k19

    elif dataset_name == 'k16':
        data_root_dir_k16 = '/data2/trinh/data/raw_data/Domain_Invariance/colon_class/Kather_texture_2016_image_tiles_5000'
        set_k16 = load_data_info(data_name='k16', pathname=f'{data_root_dir_k16}/*/*.tif', covert_dict=const_kather16)
        train_list = set_k16
    else:
        raise (f'dataset {dataset_name} is not implemented')

    random.Random(5).shuffle(train_list)
    train_set = train_list[int(val_ratio * len(train_list)):]

    valid_set = train_list[:int(val_ratio/2 * len(train_list))]
    test_set1 = train_list[int(val_ratio/2 * len(train_list)):int(val_ratio * len(train_list))]
    return train_set, valid_set, test_set1


def prepare_colon_class_k19val_data(dataset_name='k19', fold_idx=0):
    def load_data_info(data_name, pathname, covert_dict, parse_label=True, label_value=0):
        file_list = glob.glob(pathname)
        label_list = [int(covert_dict[file_path.split('/')[-2]]) for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    assert fold_idx < 3, "Currently only support 5 fold, each fold is 1 TMA"

    const_kather19 = {
        'ADI': ('ADI', 0), 'BACK': ('BACK', 1), 'DEB': ('DEB', 2), 'LYM': ('LYM', 3), 'MUC': ('MUC', 4),
        'MUS': ('MUS', 5), 'NORM': ('NORM', 6), 'STR': ('STR', 7), 'TUM': ('TUM', 8)
    }

    # we don't use the complex stroma here
    const_kather16 = {
        '07_ADIPOSE': ('07_ADIPOSE', 0), '08_EMPTY': ('08_EMPTY', 1), '05_DEBRIS': ('05_DEBRIS', 2),
        '04_LYMPHO': ('04_LYMPHO', 3),
        '06_MUCOSA': ('06_MUCOSA', 6), '02_STROMA': ('02_STROMA', 7), '01_TUMOR': ('01_TUMOR', 8)
    }

    data_root_dir_k19_val = '/data2/trinh/data/raw_data/Domain_Invariance/colon_class/CRC-VAL-HE-7K/'
    set_k19 = load_data_info(data_name='k19', pathname = f'{data_root_dir_k19_val}/*/*.tif', covert_dict=const_kather19)
    return set_k19

def prepare_colon_class_crc_data(dataset_name='crc', sample=False,  fold_idx=0):
    """" CRC test result in the training model was wrong"""
    def load_data_info(data_root_dir, covert_dict, parse_label=True, label_value=0):
        pathname = f'{data_root_dir}/*/*.png'
        file_list = glob.glob(pathname)

        label_list = [int(covert_dict[file_path.split('/')[-2]][1]) for file_path in file_list]
        out_list = list(zip(file_list, label_list))
        out_list = [elem for elem in out_list if elem[1] != 9]
        return out_list

    assert fold_idx < 3, "Currently only support 5 fold, each fold is 1 TMA"

    #  stroma/muscle => stroma; debris/mucus => debris


    const_kather19 = {
        'ADI': ('ADI', 0), 'BACK': ('BACK', 1), 'DEB': ('DEB', 2), 'LYM': ('LYM', 3), 'MUC': ('MUC', 4),
        'MUS': ('MUS', 5), 'NORM': ('NORM', 6), 'STR': ('STR', 7), 'TUM': ('TUM', 8)
    }

    # we don't use the complex stroma here
    const_kather16 = {
        '07_ADIPOSE': ('07_ADIPOSE', 0), '08_EMPTY': ('08_EMPTY', 1), '05_DEBRIS': ('05_DEBRIS', 2),
        '04_LYMPHO': ('04_LYMPHO', 3), '06_MUCOSA': ('06_MUCOSA', 6), '02_STROMA': ('02_STROMA', 7),
        '01_TUMOR': ('01_TUMOR', 8)
    }

    const_crctp_to_kather19 = {
        'Benign': ('NORM', 6), 'Complex Stroma': ('CSTR', 9), 'Debris': ('DEB', 2),
        'Inflammatory': ('LYM', 3), 'Muscle': ('MUS', 5), 'Stroma': ('STR', 7), 'Tumor': ('TUM', 8)
    }


    data_root_dir_train = '/data2/trinh/data/raw_data/Domain_Invariance/colon_class/CRC_TP/Training'
    data_root_dir_test = '/data2/trinh/data/raw_data/Domain_Invariance/colon_class/CRC_TP/Testing'
    train_list = load_data_info(data_root_dir=data_root_dir_train, covert_dict=const_crctp_to_kather19)
    val_list = load_data_info(data_root_dir=data_root_dir_test, covert_dict=const_crctp_to_kather19)
    # Split sets is train or val sets.
    # indices = np.random.RandomState(seed=5).permutation(len(set_k19))
    val_ratio = 0.3


    random.Random(5).shuffle(train_list)
    train_set = train_list[int(val_ratio/2 * len(train_list)):]

    if sample:
        train_set = train_list[:1000]

    valid_set = train_list[:int(val_ratio/2 * len(train_list))]
    test_set1 = val_list
    print_number_of_sample(train_set, valid_set, test_set1)
    return train_set, valid_set, test_set1


def prepare_colon_class_k19crcBalance_data():
    train_pairs_k, valid_pairs_k, test_pairs_k = prepare_colon_class_data(dataset_name='k19')
    train_pairs_c, valid_pairs_c, test_pairs_c = prepare_colon_class_crc_data(dataset_name='crc')
    train_pairs = train_pairs_k + train_pairs_c
    valid_pairs = valid_pairs_k + valid_pairs_c
    new_pairs = []
    for label in range(7):
        pairs = [train_pairs[i] for i in range(len(train_pairs)) if train_pairs[i][1] == label]
        if label == 1:
            random.shuffle(pairs)
            pairs = pairs[: len(pairs) // 2]
        elif label in [5,6]:
            pairs += pairs
        new_pairs.extend(pairs)
    print_number_of_sample(new_pairs, valid_pairs, [])
    return new_pairs, valid_pairs, []


# print('a')
#
# prepare_colon_class_crc_data()
#
#
# train OrderedDict([(0, 29774), (1, 59579), (2, 17866), (3, 11824), (4, 17807)])
# valid OrderedDict([(0, 5226), (1, 10421), (2, 3134), (3, 2176), (4, 3193)])
# test OrderedDict([(0, 15000), (1, 30000), (2, 9000), (3, 6000), (4, 9000)])
