import json, time, random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import XLMTokenizer, AutoTokenizer

class BertData(Dataset):
    '''
    The dataset based on Bert.
    '''
    def __init__(self, data_name, data, args, language=None, add_special_tokens=True):
        self.data_name = data_name  
        self.data_file = data
        self.max_tok_len = args.max_tok_len  
        self.add_special_tokens = add_special_tokens 
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        # self.tokenizer = XLMTokenizer.from_pretrained(f"cardiffnlp/twitter-xlm-roberta-base-sentiment")      
        self.preprocess_data()
        self.data_file.reset_index(inplace=True) 

    def preprocess_data(self):
        print('Preprocessing Data {} ...'.format(self.data_name))

        data_time_start=time.time()
        self.data_file['text_idx'] = [[] for _ in range(len(self.data_file))]
        self.data_file['mask'] = [[] for _ in range(len(self.data_file))]
        text_str_lst = []
        for i, row in self.data_file.iterrows():
            row = self.data_file.iloc[i]
            ori_text = row['text']
            text_str_lst.append(ori_text)
            text = self.tokenizer(ori_text, add_special_tokens=self.add_special_tokens, truncation=True,
                                  max_length=int(self.max_tok_len), padding='max_length')
            self.data_file.at[i, 'text_idx'] = text['input_ids']
            self.data_file.at[i, 'mask'] = text['attention_mask']
            self.data_file.at[i, 'ori_text'] = ori_text
        self.data_file['ori_text'] = text_str_lst
        data_time_end = time.time()
        print("... finished preprocessing cost {} ".format(data_time_end-data_time_start))

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx, corpus=None):
        row = self.data_file.iloc[idx]

        l = float(row['label'])

        sample = {'text': row['text_idx'], 
                  'label': l,
                  'ori_text': row['ori_text'],
                  'emo_label': row['emo_label'],
                  'mask': row["mask"]
                  }

        return sample


class BatchData(DataLoader):
    '''
    A batch sampler of a dataset. 
    '''
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle 
        

        self.indices = list(range(len(data)))
        if shuffle:
            random.shuffle(self.indices)
        self.batch_num = 0

    def __len__(self):
        return int(len(self.data) / float(self.batch_size))

    def num_batches(self):
        return len(self.data) / float(self.batch_size)

    def __iter__(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.indices != []:
            idxs = self.indices[:self.batch_size]
            batch = [self.data.__getitem__(i) for i in idxs]
            self.indices = self.indices[self.batch_size:]
            return batch
        else:
            raise StopIteration

    def get(self):
        self.reset()
        return self.__next__()

    def reset(self):
        self.indices = list(range(len(self.data)))
        if self.shuffle: random.shuffle(self.indices)

def build_data(data_dir):
    '''
    Builds dev data and train data from datasets. 
    '''
    df = pd.read_csv(data_dir)
    df = df.dropna(axis=0,how='any')
    dev_df = pd.DataFrame(data=None,columns=df.columns)
    train_df = pd.DataFrame(data=None,columns=df.columns)
    language_dict = dict(df['language'].value_counts())
    languages = list(language_dict.keys())
    for language in languages:
        temp_df = df[df['language'] == language]
        temp_df.reset_index(inplace=True, drop=True)
        shuffle_list = np.random.permutation(temp_df.shape[0])
        shuffle_lists = np.array_split(shuffle_list, [500])
        dev_temp_df = temp_df.iloc[shuffle_lists[0]]
        dev_df = pd.concat([dev_df, dev_temp_df])
        train_temp_df = temp_df.iloc[shuffle_lists[1]]
        train_df = pd.concat([train_df, train_temp_df])
    train_df.reset_index(inplace=True, drop=True)
    dev_df.reset_index(inplace=True, drop=True)
    shuffle_list = np.random.permutation(train_df.shape[0])
    train_df = train_df.iloc[shuffle_list]
    shuffle_list = np.random.permutation(dev_df.shape[0])
    dev_df = dev_df.iloc[shuffle_list]
    train_df.reset_index(inplace=True, drop=True)
    dev_df.reset_index(inplace=True, drop=True)
    return train_df, dev_df