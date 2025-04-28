import torch.utils.data as data
import random
import numpy as np
import option 
args = option.paper_args()



class dataset(data.Dataset):
    def __init__(self, args, test_mode=False):
        if test_mode:
            self.data_list = args.test_list
        else:
            self.data_list = args.train_list
        
        self.test_mode = test_mode
        self.n_len = 800
        self.a_len = len(self.data_list) - self.n_len


    # def load_data(self):
    #     # Load your dataset here
    #     # For example, you can use pandas to read a CSV file
    #     # df = pd.read_csv(self.data_path)
    #     # return df.values  # or any other format you need
    #     pass

    def __len__(self):
        if self.test_mode:
            return len(self.list)
        else:
            return min(self.a_len, self.n_len)

    def __getitem__(self, index):
        if not self.test_mode:
            if index == 0:
                self.n_ind = list(range(self.a_len, len(self.list)))
                self.a_ind = list(range(self.a_len))
                random.shuffle(self.n_ind)
                random.shuffle(self.a_ind)

            nindex = self.n_ind.pop()
            aindex = self.a_ind.pop()

            path = self.list[nindex].strip('\n')
            nfeatures = np.load(path, allow_pickle=True)
            nfeatures = np.array(nfeatures, dtype=np.float32)
            nlabel = 0.0 if "Normal" in path else 1.0

            path = self.list[aindex].strip('\n')
            afeatures = np.load(path, allow_pickle=True)
            afeatures = np.array(afeatures, dtype=np.float32)
            alabel = 0.0 if "Normal" in path else 1.0

            return nfeatures, nlabel, afeatures, alabel
    
        else:
            path = self.list[index].strip('\n')
            features = np.load(path, allow_pickle=True)
            label = 0.0 if "Normal" in path else 1.0
            return features, label
