from torch.utils.data import Dataset
import os 
import os.path as osp
import re 
import numpy as np 
import cv2 
import torch
from torch.utils.data.dataloader import default_collate
import torch.nn.utils.rnn as rnn_utils
import torchaudio.transforms as T
import torch.nn.functional as F
# import whisper
from tqdm import tqdm

def custom_collate_fn(batch):

    images, mask_names, cls_ids, masks, audio_specs = zip(*batch)

    images = default_collate(images)
    cls_ids = default_collate(cls_ids)
    masks = default_collate(masks)
    mask_names = default_collate(mask_names)
    audio_specs = rnn_utils.pad_sequence(audio_specs, batch_first=True, padding_value=0)

    return images, mask_names, cls_ids, masks, audio_specs


class Endovis18Dataset(Dataset):
    def __init__(self, data_root_dir = "../data/endovis_2018", 
                 mode = "val", 
                 vit_mode = "h",
                 version = 0,
                 model = ""):
        
        """Define the Endovis18 dataset

        Args:
            data_root_dir (str, optional): root dir containing all data for Endovis18. Defaults to "../data/endovis_2018".
            mode (str, optional): either in "train" or "val" mode. Defaults to "val".
            vit_mode (str, optional): "h", "l", "b" for huge, large, and base versions of SAM. Defaults to "h".
            version (int, optional): augmentation version to use. Defaults to 0.
        """
        
        self.vit_mode = vit_mode
        self.mode = mode
        self.model = model
        # self.audio_transcriptions = {}
       
        # directory containing all binary annotations
        if mode == "train":
            self.mask_dir = osp.join(data_root_dir, mode, str(version), "binary_annotations")
            self.image_dir = osp.join(data_root_dir, mode, str(version), "images")
            self.audio_dir = osp.join(data_root_dir, mode, str(version), "transcriptions")
        elif mode == "val":
            self.mask_dir = osp.join(data_root_dir, mode, "binary_annotations")
            self.image_dir = osp.join(data_root_dir, mode, "annotations")
            self.audio_dir = osp.join(data_root_dir, mode, "transcriptions")
        # put all binary masks into a list
        self.mask_list = []
        for subdir, _, files in os.walk(self.mask_dir):
            if len(files) == 0:
                continue 
            self.mask_list += [osp.join(osp.basename(subdir),i) for i in files if i.endswith('.png')]
            
        # put all binary images into a list
        self.image_list = []
        for subdir, _, files in os.walk(self.image_dir):
            if len(files) == 0:
                continue          
            self.image_list += [osp.join(osp.basename(subdir),i) for i in files if i.endswith('.png')]

        # 加载并转录所有音频文件
        # audio_files = [f for f in os.listdir(self.audio_dir) if f.endswith(".wav")]
        # for audio_file in tqdm(audio_files, desc="Loading and transcribing audio files"):
        #     if audio_file.endswith(".wav"):
        #         audio_path = osp.join(self.audio_dir, audio_file)
        #         text = model.transcribe(audio_path)["text"]
        #         self.audio_transcriptions[audio_path] = text
            
    def __len__(self):
        return len(self.mask_list)
    

    def __getitem__(self, index):
        mask_name = self.mask_list[index]
        file_name = mask_name.split('/')[-1]
        image_name = re.sub(r'_class\d+', '', file_name)
                
        # get class id from mask_name
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
        
        # TODO 加载音频文件
        if self.mode == 'train':
            audio_path = os.path.join(self.mask_dir)
            # get pre-computed sam feature 
            feat_dir = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), image_name.replace("png","npy"))
            sam_feat = np.load(feat_dir)

            # get class embedding
            class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"),file_name.replace("png","npy"))
            class_embedding = np.load(class_embedding_path)
        if self.mode == 'val':
            mask_path = str(mask_name.split('/')[-2])
            audio_path = os.path.join(self.mask_dir,mask_path)
            feat_dir = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"),mask_path, image_name.replace("png","npy"))
            sam_feat = np.load(feat_dir)
            
            # get class embedding
            class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), mask_path,file_name.replace("png","npy"))
            class_embedding = np.load(class_embedding_path)
        
        
        text_path = file_name.split(".")[0] + ".txt"
        transcription_path = os.path.join(audio_path, text_path)
        with open(transcription_path, 'r') as f:
            text = f.read()
        # audio_file_name = file_name.split(".")[0] + ".wav.txt"
        # transcription_path = os.path.join(audio_path, audio_file_name)

        # with open(transcription_path, 'r') as f:
        #     text = f.read()
        # text = self.audio_transcriptions[audio_path]

        
        # get ground-truth mask
        if self.mode == 'val':
            mask_path = str(mask_name.split('/')[-2])
            mask_path = osp.join(self.mask_dir, mask_path,file_name)
        if self.mode == 'train':
            mask_path = osp.join(self.mask_dir, file_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # TODO 返回原图images 
        # if self.mode == 'val':
        #     mask_pathway = str(mask_name.split('/')[-2])
        #     image_path = osp.join(self.image_dir, mask_pathway,image_name)
        # if self.mode == 'train':
        #     image_path = osp.join(self.image_dir, image_name)
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # image = image.astype(np.float32) / 255.0
        # image = torch.from_numpy(image).permute(2, 0, 1)  # 调整维度为 [C, H, W]

        return sam_feat, mask_name, cls_id, mask,text,class_embedding
 

class Endovis17Dataset(Dataset):
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
        
        self.mask_list = []
        for seq in seqs:
            # seq_name = "seq" + str(seq) + '_'
            # for mask in os.listdir(self.mask_dir):
            #     if mask.startswith(seq_name):
            #         self.mask_list += [mask]
            seq_path = osp.join(self.mask_dir, f"seq{seq}")
            self.mask_list += [f"seq{seq}/{mask}" for mask in os.listdir(seq_path)]
            
    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]
        
        # get class id from mask_name 
        # print("mask_name: ",mask_name) 
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))
        # print("mask_name: ",mask_name)
        # get pre-computed sam feature 
        feat_dir = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"),  mask_name.split("_class")[0] + ".npy")
        sam_feat = np.load(feat_dir)
        
        # get ground-truth mask
        mask_path = osp.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # get class embedding
        class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), mask_name.replace("png","npy"))
        class_embedding = np.load(class_embedding_path)
        text = ""
        
        return sam_feat, mask_name, cls_id, mask, text, class_embedding