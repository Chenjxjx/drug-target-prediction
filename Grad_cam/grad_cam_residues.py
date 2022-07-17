from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from train import Model
import torch
from sklearn import metrics
import numpy as np
import pandas as pd
import os
from typing import List, Tuple
from Bio import SeqIO
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr

def get_p_value(arrA, arrB):
    a = np.array(arrA)
    b = np.array(arrB)
    t, p = stats.ttest_ind(a,b)
    return p

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)


state_dict = torch.load('../Model/Model_esm1b_all/GCN_model/Fold1_best_model.pkl')
model=Model()
model.load_state_dict(state_dict)
target_layers = [model.gcn]
cam = GradCAM(model=model, target_layers=target_layers)

Result = pd.read_csv('./Biolip_information/biolip_drugtarget.csv', encoding='utf-8')
list1=[]
for root, dirs, files in os.walk("../Fasta/Positive_all_target/", topdown=False):
    for name in files:
        a=os.path.join(name)
        list1.append(a)
list_os = [s for s in list1 if 'fasta' in s] 

uniport=[]
ACC=[]
AUC_SCORE=[]
CAM_SCORE=[]
BIOLIP_SCORE=[]
pccs_score=[]
P_VALUE=[]
PRED_ONE_MEAN=[]
PRED_ZERO_MEAN=[]
for i in range(0,len(list_os),1):
    if len(Result.loc[Result['UniProt_ID']==list_os[i][:-6]])!=0:
        
        lseq=read_sequence("/home/test_zd/drugtarget/getdataset/p_all/"+list_os[i])[1]
        A=np.load('../../test/Data/edge_features/'+list_os[i][:-6]+'.npy')
        S=np.load('../../test/Data/node_features/'+list_os[i][:-6]+'.npy')
        A1=torch.from_numpy(A).to(torch.float32)
        S1=torch.from_numpy(S).to(torch.float32)
        acc=model(S1,A1)
        res=Result.loc[Result['UniProt_ID']==list_os[i][:-6]].Residue
        RES=[]
        Res=[]
        ls=res.index.values.tolist()
        for ii in ls:   

            position=res[ii].find(' ')
            Position=[]
            while position != -1:
                Position.append(position)
                position = res[ii].find(' ', position + 1)
            for iii in range(0,len(Position)-1,1):
                
                Res.append(res[ii][Position[iii]+2:Position[iii+1]])    
            Res.append(res[ii][Position[-1]+2:])
            Res.append(res[ii][1:Position[0]])
        Res = set(Res)
        if len(lseq)>1024:
            biolip=np.zeros(1024)
        else:
            biolip=np.zeros(len(lseq))
        res_position=np.array(list(map(int,Res)))
        res_position=res_position[res_position<=len(lseq)]
        res_position_numpy=res_position-1
        res_position_numpy= res_position_numpy[res_position_numpy < 1023]
        biolip[res_position_numpy]=1
        grayscale_cam = cam(S1,A1,targets=1)
        CAM=grayscale_cam
        CAM=CAM[0,:len(lseq)]
        y_scores=CAM
        y_true=biolip
        CAM=np.array(CAM)
       
        
        if np.isnan(CAM[0])==False and CAM[0]!=-np.inf:            
            pccs = pearsonr(y_scores, y_true)
            pccs_score.append(pccs)
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=1)
            AUC = metrics.auc(fpr, tpr)
            one=np.where(biolip==0)
            zero=np.where(biolip==1)
            pred_one=CAM[one]
            pred_zero=CAM[zero]
            p_value=get_p_value(pred_one,pred_zero)
            pred_one_mean=np.mean(pred_one)
            pred_zero_mean=np.mean(pred_zero)
            CAM_SCORE.append(CAM)
            BIOLIP_SCORE.append(biolip)
            PRED_ONE_MEAN.append(pred_one_mean)
            PRED_ZERO_MEAN.append(pred_zero_mean)
            P_VALUE.append(p_value)
            AUC_SCORE.append(AUC)
            uniport.append(list_os[i][:-6])
            ACC.append(acc)

CAM_SCORE_list=[i.tolist() for i in CAM_SCORE]
BIOLIP_SCORE_list=[i.tolist() for i in BIOLIP_SCORE]
result=pd.DataFrame()
result['uniport']=uniport
result['CAM_SCORE']=CAM_SCORE_list
result['BIOLIP_SCORE']=BIOLIP_SCORE_list
result['AUC_SCORE']=AUC_SCORE
result['PEARSON_SCORE']=pccs_score
result['P_VALUE']=P_VALUE
result['PRED_ONE_MEAN']=PRED_ONE_MEAN
result['PRED_ZERO_MEAN']=PRED_ZERO_MEAN
result.to_csv('./Biolip_gradcamscore_csv/result_original_CAM_sum_1.csv',encoding='utf-8')
