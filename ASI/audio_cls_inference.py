from torch.utils.data import Dataset
import os 
import os.path as osp
import re 
import numpy as np 
import torch
from torch.utils.data.dataloader import default_collate
import torch.nn.utils.rnn as rnn_utils
import whisper
from tqdm.notebook import tqdm
import torch.nn as nn 
import sys
sys.path.append("..")
import random 
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm


device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch.multiprocessing.set_start_method("spawn")

#device = "cpu"

print("======> Loading Model" )
seed = 666  
vit_mode = "h"

# set seed for reproducibility 
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

class Audio_Classifier(nn.Module):

    def __init__(self, num_cls, audio_dim, hid_dim):
        super(Audio_Classifier,self).__init__()
        self.num_clas = num_cls
        self.audio_dim = audio_dim
        self.hid_dim = hid_dim
    
        self.conv1 = nn.Conv1d(512, 128,kernel_size=3,stride =3 ,padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 250, 1024) # 
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_cls)
        self.relu = nn.ReLU()

    def forward(self,x):
        #print(x.shape)
        x = self.conv1(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = x.view(-1, 128 * 250) # 展平时间维度
        #print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        feat_before_relu = self.fc2(x)
        feat_after_relu = self.relu(feat_before_relu)
        x = self.fc3(feat_after_relu)
        return x,feat_before_relu,feat_after_relu

def custom_collate_fn(batch):

    images, mask_names, cls_ids, masks, audio_specs = zip(*batch)

    images = default_collate(images)
    cls_ids = default_collate(cls_ids)
    masks = default_collate(masks)
    mask_names = default_collate(mask_names)
    audio_specs = rnn_utils.pad_sequence(audio_specs, batch_first=True, padding_value=0)

    return images, mask_names, cls_ids, masks, audio_specs


model_whisper = whisper.load_model("path/to/whisper/model").cpu()
print(
    f"Model is {'multilingual' if model_whisper.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model_whisper.parameters()):,} parameters."
)
 

audio_model = torch.load("/path/to/audio/classify/model").to(device)
audio_model.eval()

    

def preprocess_audio(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    mel = mel.unsqueeze(0)
    #print(device)
    #print(mel.device)
    #print(model_whisper.device)
    #mel = mel.to(device)
    mel = model_whisper.embed_audio(mel)
    mel = torch.autograd.Variable(mel,requires_grad = False)
    mel = mel.data
    mel = mel.squeeze()
    mel = torch.transpose(mel, 0, 1)

    return mel

def inference(audio_model,feat,device=device):

    feat = feat.to(device)
    
    return audio_model(feat)





if __name__ == "__main__":
    features = None

    print("======> Inference" )
    normal_wav_list = ["path/to/audio/file"]
    for i in normal_wav_list:
        audio_path = i
        feats = preprocess_audio(audio_path)
        result,feat_before,feat_after = inference(audio_model,feats,device)
        cls = torch.argmax(result).item()+1
        if features is None:
            features = feat_after.cpu()
        else:
            features = torch.cat((feat_after.cpu(),features),dim=0)
        
        
        print(cls)
    
    features = features.detach().numpy()
    print(features.shape)
    np.save("audio_feat.npy",features)

    





