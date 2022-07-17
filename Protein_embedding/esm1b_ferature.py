import esm
import torch
import os
from Bio import SeqIO
from typing import List, Tuple
import string
import torch.nn.functional
import numpy as np
import random
from torch import nn

CUDA_VISIBLE_DEVICES=1

#Protein representation and contact map were obtained by Esm1b protein pretraining model 


def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

esm1b, esm1b_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
esm1b = esm1b.eval().cuda()
esm1b_batch_converter = esm1b_alphabet.get_batch_converter()
list1=[]

def protein_embedding(path):

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            a=os.path.join(name)
            list1.append(a)
    list = [s for s in list1 if 'fasta' in s] 
    for i in list:

        esm1b_data = [
        read_sequence(path+i),
        ]
        esm1b_batch_labels, esm1b_batch_strs, esm1b_batch_tokens = esm1b_batch_converter(esm1b_data)
        b =esm1b_batch_tokens.shape[1:3][0]
        if b>1024:
            tokens=esm1b_batch_tokens[:,0:1024]
        elif b<=1024:
            tokens=torch.nn.functional.pad(esm1b_batch_tokens,pad=(0,1024-b,0,0), mode='constant', value=1)
        tokens = torch.tensor(tokens.numpy())
        with torch.no_grad():
            tokens=tokens.cuda()
            results = esm1b(tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33].cpu()
            esm1b_contacts = esm1b.predict_contacts(tokens).cpu()
        torch.cuda.empty_cache()
        d=torch.tensor(np.zeros([esm1b_contacts.shape[0],1022])).unsqueeze(2)
        e=torch.cat([d,esm1b_contacts,d],dim=2)
        d=torch.tensor(np.zeros([esm1b_contacts.shape[0],1024])).unsqueeze(1)
        esm1b_contacts=torch.cat([d,e,d],dim=1)
        print(i,token_representations.shape,esm1b_contacts.shape)
        np.save('../Data_embedding/node/'+i[:-6]+'.npy',token_representations.numpy())
        np.save('../Data_embedding/edge/'+i[:-6]+'.npy',token_representations.numpy())
    torch.cuda.empty_cache()
    
    
if __name__ == "__main__":   
    protein_embedding('../Fasta/Negative_evalue/na_evalue0.001/')
    protein_embedding('../Fasta/Negative_evalue/na_evalue1/')
    protein_embedding('../Fasta/Negative_evalue/na_evalue10/')
    protein_embedding('../Fasta/Negative_pfam/')
    protein_embedding('../Fasta/Positive_all_target/')
    protein_embedding('../Fasta/Positive_approved_target/')
    


