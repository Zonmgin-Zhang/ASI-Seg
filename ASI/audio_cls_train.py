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


def custom_collate_fn(batch):

    images, mask_names, cls_ids, masks, audio_specs = zip(*batch)

    images = default_collate(images)
    cls_ids = default_collate(cls_ids)
    masks = default_collate(masks)
    mask_names = default_collate(mask_names)
    audio_specs = rnn_utils.pad_sequence(audio_specs, batch_first=True, padding_value=0)

    return images, mask_names, cls_ids, masks, audio_specs


model_whisper = whisper.load_model("/path/to/whisper/model").cpu()
print(
    f"Model is {'multilingual' if model_whisper.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model_whisper.parameters()):,} parameters."
)
 

class Endovis17Dataset_Audio(Dataset):
    def __init__(self, data_root_dir = "../data/endovis_2017", 
                 mode = "val",
                 fold = 0,  
                 vit_mode = "h",
                 version = 0):
                        
        self.vit_mode = vit_mode
        
        all_folds = list(range(1, 9))
        fold_seq = {0: [1, 3],
                    1: [2, 5],
                    2: [4, 8],
                    3: [6, 7]}
        
        if mode == "train":
            seqs = [x for x in all_folds if x not in fold_seq[fold]]     
        elif mode == "val":
            seqs = fold_seq[fold]

        self.mask_dir = osp.join(data_root_dir, str(version), "binary_annotations")
        #print("mask_dir : ",self.mask_dir)
        
        self.mask_list = []
        all_files = os.listdir(self.mask_dir)
        all_audio_file = [a for a in all_files if "wav" in a]
        for seq in seqs:
            seq_name = "seq" + str(seq) + '_'
            for mask in all_audio_file:
                if mask.startswith(seq_name):
                    self.mask_list += [mask]
            #seq_path = osp.join(self.mask_dir, f"seq{seq}")
            #self.mask_list += [f"seq{seq}/{mask}" for mask in os.listdir(seq_path)]
            
    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]
        #print(mask_name)
        # get class id from mask_name 
        # print("mask_name: ",mask_name) 
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
        # print("mask_name: ",mask_name)
        # get pre-computed sam feature 
        #feat_dir = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"),  mask_name.split("_class")[0] + ".npy")
        #sam_feat = np.load(feat_dir)
        
        # get ground-truth mask
        #mask_path = osp.join(self.mask_dir, mask_name)
        #mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # get class embedding
        #class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), mask_name.replace("png","npy"))
        #class_embedding = np.load(class_embedding_path)
        #text = ""

        audio_path = osp.join(self.mask_dir,  mask_name)
        #print("audio_path : ",audio_path)
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
        return cls_id-1,mel


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



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="endovis_2017", choices=["endovis_2018", "endovis_2017"], help='specify dataset')
parser.add_argument('--num_class', type=int, default=7, help='specify the number of label classes')
args = parser.parse_args()


print("======> Set Parameters for Training" )
dataset_name = args.dataset
num_class = args.num_class
fold = 0
thr = 0
seed = 666  
data_root_dir = f"../data/{dataset_name}"
batch_size = 16
vit_mode = "h"

# set seed for reproducibility 
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)


print("======> Load Dataset-Specific Parameters" )


num_tokens = 4
val_dataset = Endovis17Dataset_Audio(data_root_dir = data_root_dir,
                                mode = "val",
                                fold = fold, 
                                vit_mode = "h",
                                version = 0)


num_epochs = 50
lr = 0.0001
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

train_dataset = Endovis17Dataset_Audio(data_root_dir = data_root_dir,
                                        mode="train",
                                        fold = fold,
                                        vit_mode = vit_mode,
                                        version = 0)
        
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = Audio_Classifier(num_cls = num_class,audio_dim = 512, hid_dim=512).to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum=0.9)
optimizer.zero_grad()
top_acc = 0
for epoch in range(num_epochs):  
    correct_predictions = 0
    total_predictions = 0
    train_progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}")

    for i, (cls_id, feat) in enumerate(train_progress_bar): 
        #print(cls_id)
        #print(feat.shape)
        optimizer.zero_grad()
        label = cls_id.to(device)
        feat = feat.to(device)
        pred_cls,feat_before_relu,feat_after_relu = model(feat)
        output = loss(pred_cls,label)
        #print("loss: ",output.item())
        output.backward()
        optimizer.step()


        probs = torch.nn.functional.softmax(pred_cls, dim=1)
        _, predictions = torch.max(probs, 1)
        correct_predictions += (predictions == label).sum().item()
        total_predictions += label.size(0)
        batch_accuracy = (predictions == label).float().mean()
        #print("acc: ",batch_accuracy.item())

    
    total_accuracy = correct_predictions/total_predictions

    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {output.item()}, Accuracy: {total_accuracy}')

    if total_accuracy> top_acc:
        top_acc = total_accuracy
        torch.save(model,'audio_classifier.pt')
        print("save best model : audio_classifier.pt")
    
    





