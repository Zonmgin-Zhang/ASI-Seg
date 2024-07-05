import sys
sys.path.append("..")
sam_path = "/data/zzm/SurgicalSAM-main"
sys.path.insert(0,sam_path)
import os
import os.path as osp 
import random 
import argparse
import numpy as np 
import torch 
from torch.utils.data import DataLoader
from dataset_audio import Endovis18Dataset, Endovis17Dataset
import segment_anything
from segment_anything import sam_model_registry
from model import Learnable_Prototypes, Prototype_Prompt_Encoder
from utils import print_log, create_binary_masks, create_endovis_masks, eval_endovis, read_gt_endovis_masks, compute_mask_IU_endovis
from model_forward_test import model_forward_function
from loss import DiceLoss
from pytorch_metric_learning import losses
from tqdm import tqdm
from CLIP import clip
import whisper
import torch.nn as nn
import torch.nn.functional as F
import re
from PIL import Image
import json

print("======> Process Arguments")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="endovis_2017", choices=["endovis_2018", "endovis_2017"], help='specify dataset')
parser.add_argument('--fold', type=int, default=3, choices=[0,1,2,3], help='specify fold number for endovis_2017 dataset')
parser.add_argument('--num_class', type=int, default=7, help='specify the number of label classes')
args = parser.parse_args()


print("======> Set Parameters for Inference" )
dataset_name = args.dataset
fold = args.fold
num_class = args.num_class
thr = 0
data_root_dir = f"../data/{dataset_name}"


print("======> Load Dataset-Specific Parameters" )
if "18" in dataset_name:
    num_tokens = 2
    dataset = Endovis18Dataset(data_root_dir = data_root_dir, 
                                mode = "val",
                                vit_mode = "h")
    surgicalSAM_ckp = f"./work_dirs/{dataset_name}/model_ckp.pth"
    
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir,
                                            mode = "val")

elif "17" in dataset_name:
    num_tokens = 4
    dataset = Endovis17Dataset(data_root_dir = data_root_dir, 
                                mode = "val",
                                fold = fold, 
                                vit_mode = "h",
                                version = 0)

    surgicalSAM_ckp = f"./work_dirs/{dataset_name}/{fold}/model_ckp.pth"
    
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir,
                                            mode = "val",
                                            fold = fold)
    
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)


print("======> Load SAM" )
sam_checkpoint = "../ckp/sam/sam_vit_h_4b8939.pth"
model_type = "vit_h_no_image_encoder"
#model_type = "vit_h"
sam_prompt_encoder, sam_decoder = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_prompt_encoder.cuda()
sam_decoder.cuda()


print("======> Load Prototypes and Prototype-based Prompt Encoder" )
# define the models
learnable_prototypes_model = Learnable_Prototypes(num_classes = num_class, feat_dim = 256).cuda()
protoype_prompt_encoder =  Prototype_Prompt_Encoder(feat_dim = 256, 
                                                    hidden_dim_dense = 128, 
                                                    hidden_dim_sparse = 128, 
                                                    size = 64, 
                                                    num_tokens = num_tokens,
                                                    num_class= num_class)
# load the weight for prototype-based prompt encoder, mask decoder, and prototypes
checkpoint = torch.load(surgicalSAM_ckp)
protoype_prompt_encoder.load_state_dict(checkpoint['prototype_prompt_encoder_state_dict'])
sam_decoder.load_state_dict(checkpoint['sam_decoder_state_dict'])
learnable_prototypes_model.load_state_dict(checkpoint['prototypes_state_dict'])


# set requires_grad to False to the whole model 
for name, param in sam_prompt_encoder.named_parameters():
    param.requires_grad = False
for name, param in sam_decoder.named_parameters():
    param.requires_grad = False
for name, param in protoype_prompt_encoder.named_parameters():
    param.requires_grad = False
for name, param in learnable_prototypes_model.named_parameters():
    param.requires_grad = False


print("======> Start Inference")
protoype_prompt_encoder.cuda()
protoype_prompt_encoder.eval()
sam_decoder.eval()
learnable_prototypes_model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)

def print_log(str_to_print, log_file, is_dict = True):
    """Print a string and meanwhile write it to a log file
    """
    #print(str_to_print)
    
    with open(log_file, "w") as file:
        if is_dict:
            file.write(json.dumps(str_to_print,indent=2))
            file.write("\n")
        else:
            file.write(str_to_print)
            file.write("\n")

def eval_endovis(endovis_masks, gt_endovis_masks,num_classes):
    """Given the predicted masks and groundtruth annotations, predict the challenge IoU, IoU, mean class IoU, and the IoU for each class
        
      ** The evaluation code is taken from the official evaluation code of paper: ISINet: An Instance-Based Approach for Surgical Instrument Segmentation
      ** at https://github.com/BCV-Uniandes/ISINet
      
    Args:
        endovis_masks (dict): the dictionary containing the predicted mask for each frame 
        gt_endovis_masks (dict): the dictionary containing the groundtruth mask for each frame 

    Returns:
        dict: a dictionary containing the evaluation results for different metrics 
    """

    endovis_results = dict()
    num_classes = num_classes
    
    all_im_iou_acc = []
    all_im_iou_acc_challenge = []
    cum_I, cum_U = 0, 0
    class_ious = {c: [] for c in range(1, num_classes+1)}
    
    for file_name, prediction in endovis_masks.items():

        #print("file_name: ",file_name)
       
        full_mask = gt_endovis_masks[file_name]

        
        im_iou = []
        im_iou_challenge = []
        target = full_mask.numpy()
        gt_classes = np.unique(target)
        gt_classes.sort()
        gt_classes = gt_classes[gt_classes > 0] 
        if np.sum(prediction) == 0:
            if target.sum() > 0: 
                all_im_iou_acc.append(0)
                all_im_iou_acc_challenge.append(0)
                for class_id in gt_classes:
                    class_ious[class_id].append(0)
            continue

        gt_classes = torch.unique(full_mask)
        #print("gt_classes: ",gt_classes)
        # loop through all classes from 1 to num_classes 
        for class_id in range(1, num_classes + 1): 

            current_pred = (prediction == class_id).astype(np.float64)
            current_target = (full_mask.numpy() == class_id).astype(np.float64)

            if current_pred.astype(np.float64).sum() != 0 or current_target.astype(np.float64).sum() != 0:
                i, u = compute_mask_IU_endovis(current_pred, current_target)     
                im_iou.append(i/u)
                cum_I += i
                cum_U += u
                class_ious[class_id].append(i/u)
                #print("class_id: ",class_id)
                if class_id in gt_classes:
                    im_iou_challenge.append(i/u)
        
        if len(im_iou) > 0:
            all_im_iou_acc.append(np.mean(im_iou))
        if len(im_iou_challenge) > 0:
            all_im_iou_acc_challenge.append(np.mean(im_iou_challenge))

    # calculate final metrics
    final_im_iou = cum_I / (cum_U + 1e-15)
    mean_im_iou = np.sum(all_im_iou_acc)
    mean_im_iou_challenge = np.sum(all_im_iou_acc_challenge)

    final_class_im_iou = torch.zeros(num_classes+2)
    cIoU_per_class = []
    for c in range(1, num_classes + 1):
        final_class_im_iou[c-1] = torch.tensor(class_ious[c]).float().mean()
        cIoU_per_class.append(round((final_class_im_iou[c-1]*100).item(), 3))
        
    mean_class_iou = torch.tensor([torch.tensor(values).float().mean() for c, values in class_ious.items() if len(values) > 0]).sum().item()
    
    endovis_results["challengIoU"] = round(mean_im_iou_challenge*100,3)
    endovis_results["IoU"] = round(mean_im_iou*100,3)
    endovis_results["mcIoU"] = round(mean_class_iou*100,3)
    endovis_results["mIoU"] = round(final_im_iou*100,3)
    
    endovis_results["cIoU_per_class"] = cIoU_per_class
    endovis_results["file"] = file_name
    
    return endovis_results
#print(gt_endovis_masks)
with torch.no_grad():
    # Endovis2018
    #text = ["This is a Bipolar Forceps, commonly used for coagulating tissues and vessels with precision. Its insulated shaft and fine tips allow for delicate tissue manipulation and reduced thermal spread during surgeries such as microdissections and neurosurgery.","This is a Prograsp Forceps, designed for a firm grip and precise manipulation of tissues and organs during laparoscopic procedures. Its ergonomic design ensures steady handling and its durable construction provides a reliable performance in complex surgeries.","This is a Large Needle Driver, perfect for suturing with its strong, stable grip and precise control. It's particularly useful in procedures requiring large sutures, providing surgeons with the precision and durability needed for effective tissue approximation.", "This is a Monopolar Curved Scissors, an essential tool for precise cutting and dissection, offering monopolar energy for cauterization, reducing bleeding risks. Its curved design allows for fine, detailed movements, ideal for intricate surgical areas.","This is an Ultrasound Probe, a high-resolution imaging tool crucial for diagnostic and intraoperative procedures. It provides real-time images of internal structures, guiding interventions and ensuring precision in procedures like biopsies or fluid aspirations.", "This is a Suction Instrument, designed to efficiently remove fluids and debris from the surgical site, maintaining a clear view for the surgeon. Its ergonomic design and reliable performance make it a staple in maintaining operative field clarity.","This is a Clip Applier, a vital tool for quickly and securely ligating vessels during surgery. Its precise mechanism ensures the safe and effective deployment of clips, reducing the risk of complications and promoting efficient hemostasis."]
    # Endovis2017
    text = ["This is a Bipolar Forceps, commonly used for coagulating tissues and vessels with precision. Its insulated shaft and fine tips allow for delicate tissue manipulation and reduced thermal spread during surgeries such as microdissections and neurosurgery.","This is a Prograsp Forceps, designed for a firm grip and precise manipulation of tissues and organs during laparoscopic procedures. Its ergonomic design ensures steady handling and its durable construction provides a reliable performance in complex surgeries.","This is a Large Needle Driver, perfect for suturing with its strong, stable grip and precise control. It's particularly useful in procedures requiring large sutures, providing surgeons with the precision and durability needed for effective tissue approximation.","This is a Vessel Sealer, specifically designed for the permanent closure of blood vessels. It uses advanced energy technology to fuse vessel walls, ensuring a secure seal with minimal thermal spread. Its ergonomic design and precise control make it ideal for a variety of surgical procedures, including laparoscopic operations and complex dissections, where control and reliability are paramount.","This is a Grasping Retractor, engineered for optimal exposure and manipulation of tissues and organs during surgery. Its design allows for a firm grip and controlled retraction, minimizing tissue trauma. The device's versatility and durability make it a staple in surgeries requiring precision and care, such as abdominal and thoracic procedures.","These are Monopolar Curved Scissors, a vital tool for cutting and dissecting tissues with precision. Integrated with monopolar energy, they allow surgeons to cut and coagulate simultaneously, reducing bleeding and operation time. The curved design enhances visibility and access in tight spaces, making them indispensable in minimally invasive surgeries and complex anatomical areas.","This category includes various Other Medical Instruments tailored for specific surgeries, from diagnostic aids to tools like suction devices, clip appliers, and retractors for better exposure. Designed for precision, patient care, and meeting modern surgery's demands, these essential tools support cutting, clamping, retracting, and coagulating, showcasing surgical diversity and technological advancement."]
    # text = ['bipolar forceps, the area represented by the bipolar forceps, bipolar forceps have a slim, elongated tweezer-like design with opposing tips, are silver-colored, made from high-quality metal, and feature an insulated shaft for controlled energy application.','prograsp forceps, the area represented by the prograsp forceps, prograsp forceps possess curved scissor-like handles, specialized grasping tips with interlocking jaws, a ratcheting mechanism, and color-coded markings for easy identification during surgery.', 'large needle driver, the area represented by the large needle driver, large needle drivers feature elongated handles, sturdy gripping surfaces, a curved or straight jaw tip for securely holding needles, and a locking mechanism to ensure precision and control.', 'vessel sealer, the area represented by the vessel sealer, vessel sealers have elongated handles, scissor-like controls, and specialized jaws with a combination of sealing and cutting surfaces, designed for securely sealing and dividing blood vessels and tissue bundles.', 'grasping retractor, the area represented by the grasping retractor, grasping retractors display elongated shafts, curved or straight jaws with serrated or smooth surfaces for gripping tissues, and a handle mechanism for precise control and retraction of the target area.',  'monopolar curved scissors, the area represented by the monopolar curved scissors, monopolar curved scissors showcase elongated handles, curved cutting edges for precise dissection, and an insulated shaft, allowing controlled application of electrical energy for cutting and coagulation.',"This category includes various Other Medical Instruments tailored for specific surgeries, from diagnostic aids to tools like suction devices, clip appliers, and retractors for better exposure. Designed for precision, patient care, and meeting modern surgery's demands, these essential tools support cutting, clamping, retracting, and coagulating, showcasing surgical diversity and technological advancement."]

    # print("train text: ",text)
    text = clip.tokenize(text).to(device)
    text_features = clip_model.encode_text(text)
    # print("text_features: ",text_features.shape)
    text_features = text_features.cuda()
    prototypes = learnable_prototypes_model(text_features)
    prototypes.cuda()
    L = []
    for sam_feats, mask_names, cls_ids, masks,text,_ in dataloader: 
        binary_masks = dict()

        #custom_sam.set_input(input_images,audio_spec,masks)
        # with torch.no_grad():
            # text = ['This is a Bipolar Forceps','This is a Prograsp Forceps','This is a Large Needle Driver', 'This is a Monopolar Curved Scissors', 'This is a Ultrasound Probe','This is a Suction Instrument','This is a Clip Applier']
        #     text = ["This is a Bipolar Forceps, commonly used for coagulating tissues and vessels with precision. Its insulated shaft and fine tips allow for delicate tissue manipulation and reduced thermal spread during surgeries such as microdissections and neurosurgery.","This is a Prograsp Forceps, designed for a firm grip and precise manipulation of tissues and organs during laparoscopic procedures. Its ergonomic design ensures steady handling and its durable construction provides a reliable performance in complex surgeries.","This is a Large Needle Driver, perfect for suturing with its strong, stable grip and precise control. It's particularly useful in procedures requiring large sutures, providing surgeons with the precision and durability needed for effective tissue approximation.", "This is a Monopolar Curved Scissors, an essential tool for precise cutting and dissection, offering monopolar energy for cauterization, reducing bleeding risks. Its curved design allows for fine, detailed movements, ideal for intricate surgical areas.","This is an Ultrasound Probe, a high-resolution imaging tool crucial for diagnostic and intraoperative procedures. It provides real-time images of internal structures, guiding interventions and ensuring precision in procedures like biopsies or fluid aspirations.", "This is a Suction Instrument, designed to efficiently remove fluids and debris from the surgical site, maintaining a clear view for the surgeon. Its ergonomic design and reliable performance make it a staple in maintaining operative field clarity.","This is a Clip Applier, a vital tool for quickly and securely ligating vessels during surgery. Its precise mechanism ensures the safe and effective deployment of clips, reducing the risk of complications and promoting efficient hemostasis."]

        #     # Endovis 2017
        #     # text = ["This is a Bipolar Forceps, commonly used for coagulating tissues and vessels with precision. Its insulated shaft and fine tips allow for delicate tissue manipulation and reduced thermal spread during surgeries such as microdissections and neurosurgery.","This is a Prograsp Forceps, designed for a firm grip and precise manipulation of tissues and organs during laparoscopic procedures. Its ergonomic design ensures steady handling and its durable construction provides a reliable performance in complex surgeries.","This is a Large Needle Driver, perfect for suturing with its strong, stable grip and precise control. It's particularly useful in procedures requiring large sutures, providing surgeons with the precision and durability needed for effective tissue approximation.","This is a Vessel Sealer, specifically designed for the permanent closure of blood vessels. It uses advanced energy technology to fuse vessel walls, ensuring a secure seal with minimal thermal spread. Its ergonomic design and precise control make it ideal for a variety of surgical procedures, including laparoscopic operations and complex dissections, where control and reliability are paramount.","This is a Grasping Retractor, engineered for optimal exposure and manipulation of tissues and organs during surgery. Its design allows for a firm grip and controlled retraction, minimizing tissue trauma. The device's versatility and durability make it a staple in surgeries requiring precision and care, such as abdominal and thoracic procedures.","These are Monopolar Curved Scissors, a vital tool for cutting and dissecting tissues with precision. Integrated with monopolar energy, they allow surgeons to cut and coagulate simultaneously, reducing bleeding and operation time. The curved design enhances visibility and access in tight spaces, making them indispensable in minimally invasive surgeries and complex anatomical areas.","This category includes various Other Medical Instruments tailored for specific surgeries, from diagnostic aids to tools like suction devices, clip appliers, and retractors for better exposure. Designed for precision, patient care, and meeting modern surgery's demands, these essential tools support cutting, clamping, retracting, and coagulating, showcasing surgical diversity and technological advancement."]
        #     # text = ['bipolar forceps, the area represented by the bipolar forceps, bipolar forceps have a slim, elongated tweezer-like design with opposing tips, are silver-colored, made from high-quality metal, and feature an insulated shaft for controlled energy application.','prograsp forceps, the area represented by the prograsp forceps, prograsp forceps possess curved scissor-like handles, specialized grasping tips with interlocking jaws, a ratcheting mechanism, and color-coded markings for easy identification during surgery.', 'large needle driver, the area represented by the large needle driver, large needle drivers feature elongated handles, sturdy gripping surfaces, a curved or straight jaw tip for securely holding needles, and a locking mechanism to ensure precision and control.', 'vessel sealer, the area represented by the vessel sealer, vessel sealers have elongated handles, scissor-like controls, and specialized jaws with a combination of sealing and cutting surfaces, designed for securely sealing and dividing blood vessels and tissue bundles.', 'grasping retractor, the area represented by the grasping retractor, grasping retractors display elongated shafts, curved or straight jaws with serrated or smooth surfaces for gripping tissues, and a handle mechanism for precise control and retraction of the target area.',  'monopolar curved scissors, the area represented by the monopolar curved scissors, monopolar curved scissors showcase elongated handles, curved cutting edges for precise dissection, and an insulated shaft, allowing controlled application of electrical energy for cutting and coagulation.',"This category includes various Other Medical Instruments tailored for specific surgeries, from diagnostic aids to tools like suction devices, clip appliers, and retractors for better exposure. Designed for precision, patient care, and meeting modern surgery's demands, these essential tools support cutting, clamping, retracting, and coagulating, showcasing surgical diversity and technological advancement."]

        #     text = clip.tokenize(text).to(device)
        #     text_features = clip_model.encode_text(text)

        # text_features = text_features.cuda()

    
        sam_feats = sam_feats.cuda()
        sam_feats_pooled = torch.mean(sam_feats, dim=[2, 3]).cuda()

        mlp_output = sam_feats
        
        cls_ids = cls_ids.cuda()
        
        preds , preds_quality = model_forward_function(protoype_prompt_encoder, sam_prompt_encoder, sam_decoder, mlp_output, prototypes, cls_ids)    

        binary_masks = create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr)

        endovis_masks = create_endovis_masks(binary_masks, 1024, 1280)
        endovis_results = eval_endovis(endovis_masks, gt_endovis_masks, num_class)
        L.append(endovis_results)
        print_log(L,log_file="/data/zzm/sam_demo/surgical_sam.log")



