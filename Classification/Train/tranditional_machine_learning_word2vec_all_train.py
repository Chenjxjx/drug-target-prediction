from gensim.models import Word2Vec
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

Model_Path = '../../Model/Model_word2vec_all/'
Result_Path = '../../Result/Result_word2vec/Result_valid/'

#amino_acid = list("ACDEFGHIKLMNPQRSTVWYX")
#amino_dict = {aa: i for i, aa in enumerate(amino_acid)}

# Seed
SEED = 2333
np.random.seed(SEED)


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
word2vec = Word2Vec.load("../../Protein_embedding/Model_word2vec/word2vec_pall.model")


def seq_to_kmers(seq, k=3):
    N = len(seq)
    return [seq[i:i+k] for i in range(N - k + 1)]


def get_protein_embedding(model,protein):
    
    vec = np.zeros((len(protein), 200))
    i = 0
    for word in protein:
        try:
            vec[i, ] = model.wv[word]
            i += 1
        except:
            print("warning:", word)
            continue
    return vec



def word2vec_features(sequence_name, sequence):

    feature_matrix = get_protein_embedding(word2vec,seq_to_kmers(sequence))
    if len(feature_matrix)<1024:
        feature_matrix=np.concatenate((feature_matrix,np.ones((1024-len(feature_matrix),200))),axis=0)
    else:
        feature_matrix=feature_matrix[:1024,:]
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
        sequence_feature = word2vec_features(sequence_name, sequence)
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


def train(model, train_dataframe, valid_dataframe, fold=0):
    train_loader = DataLoader(dataset=ProDataset(train_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4,drop_last=True)
    valid_loader = DataLoader(dataset=ProDataset(valid_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=4,drop_last=True)

    train_losses = []
    train_pearson = []
    train_r2 = []
    train_binary_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    train_auc = []
    train_mcc = []
    train_sensitivity = []
    train_specificity = []

    valid_losses = []
    valid_pearson = []
    valid_r2 = []
    valid_binary_acc = []
    valid_precision = []
    valid_recall = []
    valid_f1 = []
    valid_auc = []
    valid_mcc = []
    valid_sensitivity = []
    valid_specificity = []

    best_val_loss = 1000
    best_epoch = 0

    for epoch in range(NUMBER_EPOCHS):

        train_one_epoch(model, train_loader, epoch + 1)
        print("========== Evaluate Train set ==========")
        train_true, train_pred, _ = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred)

        print("Train pearson:", result_train['pearson'])
        print("Train r2:", result_train['r2'])
        print("Train binary acc: ", result_train['binary_acc'])
        print("Train precision: ", result_train['precision'])
        print("Train recall: ", result_train['recall'])
        print("Train F1: ", result_train['f1'])
        print("Train auc: ", result_train['auc'])
        print("Train mcc: ", result_train['mcc'])
        print("Train sensitivity: ", result_train['sensitivity'])
        print("Train specificity: ", result_train['specificity'])


        train_pearson.append(result_train['pearson'])
        train_r2.append(result_train['r2'])
        train_binary_acc.append(result_train['binary_acc'])
        train_precision.append(result_train['precision'])
        train_recall.append(result_train['recall'])
        train_f1.append(result_train['f1'])
        train_auc.append(result_train['auc'])
        train_mcc.append(result_train['mcc'])
        train_sensitivity.append(result_train['sensitivity'])
        train_specificity.append(result_train['specificity'])

        print("========== Evaluate Valid set ==========")
        valid_true, valid_pred, valid_name = evaluate(model, valid_loader)
        result_valid = analysis(valid_true, valid_pred)
        valid_predd=np.array(valid_pred)
        valid_pred_label=np.empty(valid_predd.shape,dtype=int)
        for i in range(0,len(valid_predd),1):
            if valid_predd[i]<=0.5:
                valid_pred_label[i]=0
            elif valid_predd[i]>0.5:
                valid_pred_label[i]=1
        valid_pred_label=valid_pred_label.tolist()
        print("Valid pearson:", result_valid['pearson'])
        print("Valid r2:", result_valid['r2'])
        print("Valid binary acc: ", result_valid['binary_acc'])
        print("Valid precision: ", result_valid['precision'])
        print("Valid recall: ", result_valid['recall'])
        print("Valid f1: ", result_valid['f1'])
        print("Valid auc: ", result_valid['auc'])
        print("Valid mcc: ", result_valid['mcc'])
        print("Valid sensitivity: ", result_valid['sensitivity'])
        print("Valid specificity: ", result_valid['specificity'])


        valid_pearson.append(result_valid['pearson'])
        valid_r2.append(result_valid['r2'])
        valid_binary_acc.append(result_valid['binary_acc'])
        valid_precision.append(result_valid['precision'])
        valid_recall.append(result_valid['recall'])
        valid_f1.append(result_valid['f1'])
        valid_auc.append(result_valid['auc'])
        valid_mcc.append(result_valid['mcc'])
        valid_sensitivity.append(result_valid['sensitivity'])
        valid_specificity.append(result_valid['specificity'])


        torch.save(model, os.path.join(Model_Path, str(Model)[:-2]+ '_Fold' + str(fold) + '_best_model.pkl'))
        valid_detail_dataframe = pd.DataFrame({'gene': valid_name, 'label': valid_true, 'prediction': valid_pred, 'label_pred':valid_pred_label})
        valid_detail_dataframe.sort_values(by=['gene'], inplace=True)
        valid_detail_dataframe.to_csv(Result_Path + str(Model)[:-2]+ '_Fold' + str(fold) +"_valid_detail.csv", header=True, sep=',')

    # save calculation information
    result_all = {

        'Train_pearson': train_pearson,
        'Train_r2': train_r2,
        'Train_binary_acc': train_binary_acc,
        'Train_precision': train_precision,
        'Train_recall': train_recall,
        'Train_f1': train_f1,
        'Train_auc': train_auc,
        'Train_mcc': train_mcc,
        'Train_sensitivity': train_sensitivity,
        'Train_specificity': train_specificity,
        'Valid_pearson': valid_pearson,
        'Valid_r2': valid_r2,
        'Valid_binary_acc': valid_binary_acc,
        'Valid_precision': valid_precision,
        'Valid_recall': valid_recall,
        'Valid_f1': valid_f1,
        'Valid_auc': valid_auc,
        'Valid_mcc': valid_mcc,
        'Valid_sensitivity': valid_sensitivity,
        'Valid_specificity': valid_specificity,

    }
    result = pd.DataFrame(result_all)

    #result.to_csv(Result_Path + "Fold" + str(fold) +Model "_result.csv", sep=',')
    return result


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


def cross_validation(all_dataframe,fold_number=10):

    print("split_seed: ", SEED)
    sequence_names = all_dataframe['prot'].values
    sequence_labels = all_dataframe['label'].values
    kfold = KFold(n_splits=fold_number, shuffle=True)
    fold = 0

    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Training on", str(train_dataframe.shape[0]), "examples, Validation on", str(valid_dataframe.shape[0]),
              "examples")
        model = Model
        result=train(model, train_dataframe, valid_dataframe, fold + 1)

        if fold==0:
            Result=result
        else:
            Result=Result.append(result)

        fold += 1
    Result.to_csv(Result_Path +str(Model)[:-2]+ "_result.csv", sep=',')


if __name__ == "__main__":
    train_dataframe = pd.read_csv('../../Data_embedding/protinformation_csv/All/train_split.csv')
    Model=RandomForestClassifier()
    cross_validation(train_dataframe,fold_number=5)
    Model=DecisionTreeClassifier()
    cross_validation(train_dataframe,fold_number=5)
    Model=KNeighborsClassifier()
    cross_validation(train_dataframe,fold_number=5)
    Model=GaussianNB()
    cross_validation(train_dataframe,fold_number=5)
    Model=LogisticRegression(max_iter=1000)
    cross_validation(train_dataframe,fold_number=5)


    Model=svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,random_state=None)
    cross_validation(train_dataframe,fold_number=5)

    