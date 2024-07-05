import sys
sys.path.append("..")
import os
import os.path as osp 
import random 
import argparse
import numpy as np 
import torch 
from torch.utils.data import DataLoader
from dataset_audio import Endovis18Dataset, Endovis17Dataset
from segment_anything import sam_model_registry
from model import Learnable_Prototypes, Prototype_Prompt_Encoder
from utils import print_log, create_binary_masks, create_endovis_masks, eval_endovis, read_gt_endovis_masks
from model_forward_test import model_forward_function
from loss import DiceLoss
from pytorch_metric_learning import losses
from tqdm import tqdm
from CLIP import clip
import torch.nn as nn
import torch.nn.functional as F

print("======> Process Arguments")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="endovis_2017", choices=["endovis_2018", "endovis_2017"], help='specify dataset')
parser.add_argument('--num_class', type=int, default=7, help='specify the number of label classes')
args = parser.parse_args()

print("======> Set Parameters for Training" )
dataset_name = args.dataset
num_class = args.num_class
fold = 2
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
if "18" in dataset_name:
    num_tokens = 2
    val_dataset = Endovis18Dataset(data_root_dir = data_root_dir, 
                                   mode="val",
                                   vit_mode = "h",
                                   #model = whisper.load_model("small.en")
                                   model = '')
    
    surgicalSAM_ckp = f"../ckp/surgical_sam_test/{dataset_name}/model_ckp.pth"
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir, mode = "val")
    num_epochs = 1000
    lr = 0.0001
    # save_dir = "./work_dirs/endovis_2018_version1/"
    save_dir = "./work_dirs/endovis_2018融合_lr0.001"

elif "17" in dataset_name:  
    num_tokens = 4
    val_dataset = Endovis17Dataset(data_root_dir = data_root_dir,
                                   mode = "val",
                                   fold = fold, 
                                   vit_mode = "h",
                                   version = 0)
    
    surgicalSAM_ckp = f"../ckp/surgical_sam_test/{dataset_name}/model_ckp.pth"
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir, 
                                             mode = "val", 
                                             fold = fold)
    num_epochs = 2000
    lr = 0.0001
    save_dir = f"./work_dirs/endovis_2017/{fold}"
    
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

print("======> Load SAM" )
if vit_mode == "h":
    sam_checkpoint = "../ckp/sam/sam_vit_h_4b8939.pth"
model_type = "vit_h_no_image_encoder"
raw_model_type = "vit_h"

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)

sam_prompt_encoder, sam_decoder = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_prompt_encoder.cuda()
sam_decoder.cuda()

for name, param in sam_prompt_encoder.named_parameters():
    param.requires_grad = False
for name, param in sam_decoder.named_parameters():
    param.requires_grad = True
    

print("======> Load Prototypes and Prototype-based Prompt Encoder" )
learnable_prototypes_model = Learnable_Prototypes(num_classes = num_class, feat_dim = 256).cuda()
protoype_prompt_encoder =  Prototype_Prompt_Encoder(feat_dim = 256, 
                                                    hidden_dim_dense = 128, 
                                                    hidden_dim_sparse = 128, 
                                                    size = 64, 
                                                    num_tokens = num_tokens,
                                                    num_class=num_class).cuda()
 
with open(sam_checkpoint, "rb") as f:
    state_dict = torch.load(f)
    sam_pn_embeddings_weight = {k.split("prompt_encoder.point_embeddings.")[-1]: v for k, v in state_dict.items() if k.startswith("prompt_encoder.point_embeddings") and ("0" in k or "1" in k)}
    sam_pn_embeddings_weight_ckp = {"0.weight": torch.concat([sam_pn_embeddings_weight['0.weight'] for _ in range(num_tokens)], dim=0),
                                    "1.weight": torch.concat([sam_pn_embeddings_weight['1.weight'] for _ in range(num_tokens)], dim=0)}

    protoype_prompt_encoder.pn_cls_embeddings.load_state_dict(sam_pn_embeddings_weight_ckp)

for name, param in learnable_prototypes_model.named_parameters():
    param.requires_grad = True
    
for name, param in protoype_prompt_encoder.named_parameters():
    if "pn_cls_embeddings" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True
              
print("======> Define Optmiser and Loss")
seg_loss_model = DiceLoss().cuda()
contrastive_loss_model = losses.NTXentLoss(temperature=0.07).cuda()
optimiser = torch.optim.Adam([
            {'params': learnable_prototypes_model.parameters()},
            {'params': protoype_prompt_encoder.parameters()},
            {'params': sam_decoder.parameters()},
            
        ], lr = lr, weight_decay = 0.0001)


print("======> Set Saving Directories and Logs")
os.makedirs(save_dir, exist_ok = True) 
log_file = osp.join(save_dir, "log.txt")
print_log(str(args), log_file)


print("======> Start Training and Validation" )
best_challenge_iou_val = -100.0

version = 0
accumulation_steps = 1
optimiser.zero_grad()   

for epoch in range(num_epochs):   
    # choose the augmentation version to use for the current epoch 
    if epoch % 2 == 0 :
        version = 0 
    else:
        version = int((epoch % 80 + 1)/2)
    
    
    if "18" in dataset_name:
        train_dataset = Endovis18Dataset(data_root_dir = data_root_dir,
                                        mode="train",
                                        vit_mode = vit_mode,
                                        version = version,
                                        # model = whisper.load_model("small.en")
                                        )
        
    elif "17" in dataset_name:
        train_dataset = Endovis17Dataset(data_root_dir = data_root_dir,
                                        mode="train",
                                        fold = fold,
                                        vit_mode = vit_mode,
                                        version = version)
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    
    # training
    protoype_prompt_encoder.train()
    sam_decoder.train()
    learnable_prototypes_model.train()
    
    train_progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch}/{num_epochs-1}")


    for i, (sam_feats, _, cls_ids, masks, text, class_embeddings) in enumerate(train_progress_bar): 
        
        # Endovis 2018
        # text = ["This is a Bipolar Forceps, commonly used for coagulating tissues and vessels with precision. Its insulated shaft and fine tips allow for delicate tissue manipulation and reduced thermal spread during surgeries such as microdissections and neurosurgery.","This is a Prograsp Forceps, designed for a firm grip and precise manipulation of tissues and organs during laparoscopic procedures. Its ergonomic design ensures steady handling and its durable construction provides a reliable performance in complex surgeries.","This is a Large Needle Driver, perfect for suturing with its strong, stable grip and precise control. It's particularly useful in procedures requiring large sutures, providing surgeons with the precision and durability needed for effective tissue approximation.", "This is a Monopolar Curved Scissors, an essential tool for precise cutting and dissection, offering monopolar energy for cauterization, reducing bleeding risks. Its curved design allows for fine, detailed movements, ideal for intricate surgical areas.","This is an Ultrasound Probe, a high-resolution imaging tool crucial for diagnostic and intraoperative procedures. It provides real-time images of internal structures, guiding interventions and ensuring precision in procedures like biopsies or fluid aspirations.", "This is a Suction Instrument, designed to efficiently remove fluids and debris from the surgical site, maintaining a clear view for the surgeon. Its ergonomic design and reliable performance make it a staple in maintaining operative field clarity.","This is a Clip Applier, a vital tool for quickly and securely ligating vessels during surgery. Its precise mechanism ensures the safe and effective deployment of clips, reducing the risk of complications and promoting efficient hemostasis."]
        # Endovis 2017
        text = ["This is a Bipolar Forceps, commonly used for coagulating tissues and vessels with precision. Its insulated shaft and fine tips allow for delicate tissue manipulation and reduced thermal spread during surgeries such as microdissections and neurosurgery.","This is a Prograsp Forceps, designed for a firm grip and precise manipulation of tissues and organs during laparoscopic procedures. Its ergonomic design ensures steady handling and its durable construction provides a reliable performance in complex surgeries.","This is a Large Needle Driver, perfect for suturing with its strong, stable grip and precise control. It's particularly useful in procedures requiring large sutures, providing surgeons with the precision and durability needed for effective tissue approximation.","This is a Vessel Sealer, specifically designed for the permanent closure of blood vessels. It uses advanced energy technology to fuse vessel walls, ensuring a secure seal with minimal thermal spread. Its ergonomic design and precise control make it ideal for a variety of surgical procedures, including laparoscopic operations and complex dissections, where control and reliability are paramount.","This is a Grasping Retractor, engineered for optimal exposure and manipulation of tissues and organs during surgery. Its design allows for a firm grip and controlled retraction, minimizing tissue trauma. The device's versatility and durability make it a staple in surgeries requiring precision and care, such as abdominal and thoracic procedures.","These are Monopolar Curved Scissors, a vital tool for cutting and dissecting tissues with precision. Integrated with monopolar energy, they allow surgeons to cut and coagulate simultaneously, reducing bleeding and operation time. The curved design enhances visibility and access in tight spaces, making them indispensable in minimally invasive surgeries and complex anatomical areas.","This category includes various Other Medical Instruments tailored for specific surgeries, from diagnostic aids to tools like suction devices, clip appliers, and retractors for better exposure. Designed for precision, patient care, and meeting modern surgery's demands, these essential tools support cutting, clamping, retracting, and coagulating, showcasing surgical diversity and technological advancement."]
        
        with torch.no_grad():
            text = clip.tokenize(text).to(device)
            text_features = clip_model.encode_text(text)
        text_features = text_features.cuda()

        sam_feats = sam_feats.cuda()
        mlp_output = sam_feats
        cls_ids = cls_ids.cuda()
        masks = masks.cuda()
        class_embeddings.cuda()
        
        prototypes = learnable_prototypes_model(text_features)
        
        preds, _ = model_forward_function(protoype_prompt_encoder, sam_prompt_encoder, sam_decoder, mlp_output, prototypes, cls_ids)
        
        # compute loss
        class_embeddings = class_embeddings.to(device)
        cls_ids = cls_ids.to(device)
        indices_tensor = torch.tensor([i for i in range(1, prototypes.size()[0] + 1)], device=device)
        contrastive_loss = contrastive_loss_model(prototypes, indices_tensor, ref_emb=class_embeddings, ref_labels=cls_ids)
        seg_loss = seg_loss_model(preds, masks/255)
    
        loss = seg_loss + contrastive_loss
        loss = loss / accumulation_steps  # Normalize our loss (if averaged)
        loss.backward()  # Backpropagate the gradients
        
        if (i + 1) % accumulation_steps == 0: 
            optimiser.step()
            optimiser.zero_grad()
        print_log(f"Training - Epoch: {epoch}/{num_epochs-1}; Loss: {loss} ", log_file)

    # validation 
    binary_masks = dict()
    protoype_prompt_encoder.eval()
    sam_decoder.eval()
    learnable_prototypes_model.eval()
    val_progress_bar = tqdm(val_dataloader, desc=f"Validation Epoch {epoch}/{num_epochs-1}")


    with torch.no_grad():
        prototypes = learnable_prototypes_model(text_features)
        
        for sam_feats, mask_names, cls_ids, masks,text,_ in val_progress_bar: 
            with torch.no_grad():
                text = clip.tokenize(text).to(device)
                text_features = clip_model.encode_text(text)
            text_features = text_features.cuda()
            sam_feats = sam_feats.cuda()
            sam_feats_pooled = torch.mean(sam_feats, dim=[2, 3]).cuda()
            mlp_output = sam_feats
            
            cls_ids = cls_ids.cuda()    
            
            preds , preds_quality = model_forward_function(protoype_prompt_encoder, sam_prompt_encoder, sam_decoder, mlp_output, prototypes, cls_ids)    

            binary_masks = create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr)

    endovis_masks = create_endovis_masks(binary_masks, 1024, 1280)
    endovis_results = eval_endovis(endovis_masks, gt_endovis_masks,num_class)
                
    print_log(f"Validation - Epoch: {epoch}/{num_epochs-1}; IoU_Results: {endovis_results} ", log_file)
    
    if endovis_results["challengIoU"] > best_challenge_iou_val:
        best_challenge_iou_val = endovis_results["challengIoU"]
        
        torch.save({
            'prototype_prompt_encoder_state_dict': protoype_prompt_encoder.state_dict(),
            'sam_decoder_state_dict': sam_decoder.state_dict(),
            'prototypes_state_dict': learnable_prototypes_model.state_dict()
        }, osp.join(save_dir,'model_ckp.pth'))

        print_log(f"Best Challenge IoU: {best_challenge_iou_val:.4f} at Epoch {epoch}", log_file)        
