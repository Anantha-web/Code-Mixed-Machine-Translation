# Importing required Libraries
from numpy import source
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import os
import sys
from indicnlp.normalize.indic_normalize import DevanagariNormalizer
import jsonlines

# Install the dependencies using

# conda env create --file environment.yml
# conda activate cmtranslation2


# Directories pathways
BASE_DIR = f'data/preprocessed/'
DATA_DIR = f'data/'
MODEL = sys.argv[1]

assert MODEL == 'mBARTen' or MODEL == 'mBARThien', "Please Enter either 'mBARTen' or 'mBARThie'"

#Checking directories and creating if not present
if os.path.exists(BASE_DIR):
    pass
else:
    os.makedirs(BASE_DIR)

def parse_iitb_file(file_en, file_hi, data_id):
    en_data = list()
    hi_data = list()
    f2 = open(file_en,"r")
    f1 = open(file_hi,"r")
    for source, target in zip(f1, f2):
        source_word = source.strip() + '\n'
        hi_data.append(source_word)
        target_word = target.strip() + '\n'
        en_data.append(target_word)
    normalizer = DevanagariNormalizer()
    hi_data = [normalizer.normalize(x) for x in hi_data]
    assert len(en_data) == len(hi_data),"Length of English and Hindi data is not same"
    print(f'total size of {data_id} data is {len(en_data)}')
    return en_data, hi_data


# Datasets
# The below has trained data, test data
a = 'iitb_corpus/parallel/IITB.en-hi.en'
b = 'iitb_corpus/parallel/IITB.en-hi.hi'
iitb_en_train, iitb_hi_train = parse_iitb_file(DATA_DIR + a, DATA_DIR+b, 'IITB_TRAIN')
c = 'iitb_corpus/dev_test/dev.en'
d = 'iitb_corpus/dev_test/dev.hi'
iitb_en_val, iitb_hi_val = parse_iitb_file(DATA_DIR+c, DATA_DIR+d, 'IITB_VALIDATION')
e = 'iitb_corpus/dev_test/test.en'
f = 'iitb_corpus/dev_test/test.hi'
iitb_en_test, iitb_hi_test = parse_iitb_file(DATA_DIR+e, DATA_DIR+f, 'IITB_TEST')

def parse_shared(file, data_id):
    source_data = list()
    target_data = list()
    reader = jsonlines.open(f'{DATA_DIR}/processed_data/{file}.jsonl', "r") 
    for obj in reader:
        if MODEL == 'mBARThien':
            source_data.append(' '.join(obj['Hindi']) + ' ## ' + ' '.join(obj['English']) + '\n')
        else:
            source_data.append(' '.join(obj['English']) + '\n')
        target_data.append(' '.join([(x[1]) for x in obj['Devanagari_Hinglish']]) + '\n')
    print(f'total size of {data_id} data is {len(source_data)}')
    return source_data, target_data

calcs_src_train, calcs_tgt_train = parse_shared("train", "CALCS_TRAIN")

calcs_src_val, calcs_tgt_val = parse_shared("dev", "CALCS_VALIDATION")

def parse_shared_test(data_id):
    arr1 = list()
    arr2 = list()
    f = open(DATA_DIR+'mt_enghinglish/test.txt', "r")
    f_translated = open(DATA_DIR+'translated_data/test.txt', 'r')
    
    for row in f:
        english_sentence = row.strip()
        arr1.append(english_sentence)
    for row in f_translated:
        hindi_sentence = row.strip()
        arr2.append(hindi_sentence)
    assert len(arr1) == len(arr2), "Lengths should be same"
    src_data = list()
    if MODEL == 'mBARThien':
        src_data = [arr2[i] + ' ## ' + arr1[i] + '\n' for i in range(len(arr1))]
    else:
        src_data = [arr1[i] + '\n' for i in range(len(arr1))]
    
    print(f'total size of {data_id} data is {len(src_data)}')
    return src_data, src_data

calcs_src_test, calcs_tgt_test = parse_shared_test("CALCS_TEST")

file_mapping = {
    'train.en_XX': calcs_src_train,
    'train.hi_IN': calcs_tgt_train,
    'valid.en_XX': calcs_src_val,
    'valid.hi_IN': calcs_tgt_val,
    'test.en_XX': calcs_src_test,
    'test.hi_IN': calcs_tgt_test,
    'iitb.en_XX': iitb_en_train + iitb_en_val + iitb_en_test,
    'iitb.hi_IN': iitb_hi_train + iitb_hi_val + iitb_hi_test,
}

for k, v in file_mapping.items():
    with open(f'{BASE_DIR}{k}', 'w') as fp:
        fp.writelines(v)
