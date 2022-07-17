import os
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# path
Dataset_Path = '../../Data_embedding/'
Model_Path = '../../Model/Model_esm1b_all/Traditional_model/'
Result_Path = '../../Result/Result_esm1b/Result_test/'


#amino_acid = list("ACDEFGHIKLMNPQRSTVWYX")
#amino_dict = {aa: i for i, aa in enumerate(amino_acid)}

# Seed
SEED = 2333
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(1)
    torch.cuda.manual_seed(SEED)
amino_acid = list("ACDEFGHIKLMNPQRSTVWYX")
amino_dict = {aa: i for i, aa in enumerate(amino_acid)}
# Model parameters
NUMBER_EPOCHS = 1
LEARNING_RATE = 1E-4
WEIGHT_DECAY = 1E-4
BATCH_SIZE = 64
NUM_CLASSES = 1

# GCN parameters
GCN_FEATURE_DIM = 1280
GCN_HIDDEN_DIM = 256
GCN_OUTPUT_DIM = 64

# Attention parameters
DENSE_DIM = 16
ATTENTION_HEADS = 4


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def load_sequences(sequence_path):
    names, sequences, labels = ([] for i in range(3))
    for file_name in tqdm(os.listdir(sequence_path)):
        with open(sequence_path + file_name, 'r') as file_reader:
            lines = file_reader.read().split('\n')
            names.append(file_name)
            sequences.append(lines[1])
            labels.append(int(lines[2]))
    return pd.DataFrame({'names': names, 'sequences': sequences, 'labels': labels})


def load_features(sequence_name, sequence):
  
    feature_matrix = np.load(Dataset_Path + 'node/' + sequence_name + '.npy',allow_pickle=True)
    return feature_matrix





class ProDataset(Dataset):

    def __init__(self, dataframe):
        self.names = dataframe['prot'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = self.labels[index]
        # L * 91
        sequence_feature = load_features(sequence_name, sequence)
        # L * L
       
        return sequence_name, sequence, label, sequence_feature

    def __len__(self):
        return len(self.labels)



def train_one_epoch(model, data_loader, epoch):
    epoch_loss_train = 0.0
    n_batches = 0
    for data in tqdm(data_loader):
        
        sequence_names, _, labels, sequence_features = data
            
        sequence_features = torch.squeeze(sequence_features)
            
        features=np.squeeze(sequence_features,axis = 1).mean(2)
        features = Variable(features)
        y_true = Variable(labels)
        model.fit(features, labels)
        y_pred = model.predict(features)
        n_batches += 1
            #print(features.shape)  
        
    
     
        
def evaluate(model, data_loader):
    

    epoch_loss = 0.0
    n_batches = 0
    valid_pred = []
    valid_true = []
    valid_name = []

    for data in tqdm(data_loader):
        with torch.no_grad():
            sequence_names, _, labels, sequence_features = data

            sequence_features = torch.squeeze(sequence_features)
            
       
            features=np.squeeze(sequence_features,axis = 1).mean(2)
            features = Variable(features)
            y_true = Variable(labels)
            y_pred = model.predict(features)
                
            y_pred = y_pred.tolist()
            y_true = y_true.tolist()
            valid_pred.extend(y_pred)
            valid_true.extend(y_true)
            valid_name.extend(sequence_names)

          
            n_batches += 1
    

    return valid_true, valid_pred, valid_name



def test(test_dataframe):
    test_loader = DataLoader(dataset=ProDataset(test_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_result = {}

    for model_name in sorted(os.listdir(Model_Path)):
        print(model_name)
        model=torch.load(Model_Path + model_name,map_location='cpu')
        test_true, test_pred, test_name = evaluate(model, test_loader)
        result_test = analysis(test_true, test_pred)
        test_predd=np.array(test_pred)
        test_pred_label=np.empty(test_predd.shape,dtype=int)
        for i in range(0,len(test_predd),1):
            if test_predd[i]<=0.5:
                test_pred_label[i]=0
            elif test_predd[i]>0.5:
                test_pred_label[i]=1
        test_pred_label=test_pred_label.tolist()
        print("\n========== Evaluate Test set ==========")
       
        print("Test pearson:", result_test['pearson'])
        print("Test r2:", result_test['r2'])
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test auc: ", result_test['auc'])
        print("Test mcc: ", result_test['mcc'])
        print("Test sensitivity: ", result_test['sensitivity'])
        print("Test specificity: ", result_test['specificity'])

        test_result[model_name] = [
            
            result_test['pearson'],
            result_test['r2'],
            result_test['binary_acc'],
            result_test['precision'],
            result_test['recall'],
            result_test['f1'],
            result_test['auc'],
            result_test['mcc'],
            result_test['sensitivity'],
            result_test['specificity'],
        ]

        test_detail_dataframe = pd.DataFrame({'prot': test_name, 'label': test_true, 'prediction': test_pred,'pred_label':test_pred_label})
        test_detail_dataframe.sort_values(by=['prot'], inplace=True)
        test_detail_dataframe.to_csv(Result_Path + model_name + "_test_detail.csv", header=True, sep=',')

    test_result_dataframe = pd.DataFrame.from_dict(test_result, orient='index',
                                                   columns=[ 'pearson', 'r2', 'binary_acc', 'precision',
                                                            'recall', 'f1', 'auc', 'mcc', 'sensitivity', 'specificity'])
    test_result_dataframe.to_csv(Result_Path + "test_result.csv", index=True, header=True, sep=',')


def analysis(y_true, y_pred):
    binary_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]
    binary_true = [1 if true >= 0.5 else 0 for true in y_true]

    # continous evaluate
    pearson = pearsonr(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    auc = metrics.roc_auc_score(binary_true, y_pred)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)
    TN, FP, FN, TP = metrics.confusion_matrix(binary_true, binary_pred).ravel()
    sensitivity = 1.0 * TP / (TP + FN)
    specificity = 1.0 * TN / (FP + TN)

    result = {
        'pearson': pearson,
        'r2': r2,
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'mcc': mcc,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }
    return result


if __name__ == "__main__":
    test_dataframe = pd.read_csv('../../Data_embedding/protinformation_csv/All/test_split.csv')
    test(test_dataframe) 
   

