import torch.utils.data as data

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


    def load_data(self):
        # Load your dataset here
        # For example, you can use pandas to read a CSV file
        # df = pd.read_csv(self.data_path)
        # return df.values  # or any other format you need
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
