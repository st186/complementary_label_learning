import sys
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)

class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        X = X.copy()
        self.X1 = X.values.astype(np.float64) #categorical columns
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X1[idx], self.y[idx]

def generate_compl_labels(labels):
    # args, labels: ordinary labels
    K = torch.max(labels)+1
    candidates = np.arange(K)
    candidates = np.repeat(candidates.reshape(1, K), len(labels), 0)
    mask = np.ones((len(labels), K), dtype=bool)
    mask[range(len(labels)), labels.numpy()] = False
    candidates_ = candidates[mask].reshape(len(labels), K-1)  # this is the candidates without true class
    idx = np.random.randint(0, K-1, len(labels))
    complementary_labels = candidates_[np.arange(len(labels)), np.array(idx)]
    return complementary_labels

def class_prior(complementary_labels):
    return np.bincount(complementary_labels) / len(complementary_labels)

def prepare_insurance_data(batch_size):
    
    train = pd.read_csv('data/prudential-life-insurance-assessment/train.csv')
    test = pd.read_csv('data/prudential-life-insurance-assessment/test.csv')
    # train_X = train.drop(columns= ['Id', 'Product_Info_2'])
    # Y = train['Response']
    # test_X = test
    # test_X = test_X.drop(columns= ['Id', 'Product_Info_2'])
    # stacked_df = train_X.append(test_X)
    # for col in stacked_df.columns:
    #     if stacked_df[col].isnull().sum() > 10000:
    #         print("dropping", col, stacked_df[col].isnull().sum())
    #         stacked_df = stacked_df.drop(columns = [col])

    # stacked_df[['Employment_Info_4','Employment_Info_1']].describe()
    # stacked_df['Employment_Info_4'].fillna(stacked_df['Employment_Info_1'].mean(), inplace=True)
    # stacked_df['Employment_Info_1'].fillna(stacked_df['Employment_Info_1'].mean(), inplace=True)

    # X = stacked_df[0:59381]
    # test_processed = stacked_df[59381:]

    # #check if shape[0] matches original
    # print("train shape: ", X.shape, "orignal: ", train.shape)
    # print("test shape: ", test_processed.shape, "original: ", test.shape)

    # Y = LabelEncoder().fit_transform(Y)

    columns_to_drop = ['Id', 'Response', 'Medical_History_10','Medical_History_24']
    num_classes = 8

    print("Load the data using pandas")

    # combine train and test
    all_data = train.append(test)

    # Found at https://www.kaggle.com/marcellonegro/prudential-life-insurance-assessment/xgb-offset0501/run/137585/code
    # create any new variables    
    all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
    all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]

    # factorize categorical variables
    all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
    all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
    all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]

    all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']

    med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
    all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)

    print('Eliminate missing values')    
    # Use -1 for any others
    all_data.fillna(-1, inplace=True)

    # fix the dtype on the label column
    all_data['Response'] = all_data['Response'].astype(int)

    # split train and test
    X = all_data[all_data['Response']>0].copy()
    y = all_data[all_data['Response']<1].copy()


    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.10, random_state=0)

    #print(X_train)

    train_dataset = TabularDataset(X = X_train, Y = y_train)

    #print(len(train_dataset[1]))
    test_dataset = TabularDataset(X = X_val, Y = y_val)

    #print(train_dataset[1])

    batch_size = 256

    num_classes = 8

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    full_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)

    return full_train_loader, train_loader, test_loader, train_dataset, test_dataset, num_classes

def prepare_train_loaders(full_train_loader, batch_size, ordinary_train_dataset):

    for i, (data, labels) in enumerate(full_train_loader):
            K = torch.max(labels)+1 # K is number of classes, full_train_loader is full batch
    complementary_labels = generate_compl_labels(labels)
    ccp = class_prior(complementary_labels)
    complementary_dataset = torch.utils.data.TensorDataset(data, torch.from_numpy(complementary_labels).float())
    ordinary_train_loader = torch.utils.data.DataLoader(dataset=full_train_loader, batch_size=batch_size, shuffle=True)
    complementary_train_loader = torch.utils.data.DataLoader(dataset=complementary_dataset, batch_size=batch_size, shuffle=True)
    return full_train_loader, complementary_train_loader, ccp

