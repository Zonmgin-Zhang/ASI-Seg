import torch 
import torch.nn as nn 
from einops import rearrange
import torch.nn.functional as F

class Prototype_Prompt_Encoder(nn.Module):
    def __init__(self, feat_dim=256, 
                        hidden_dim_dense=128, 
                        hidden_dim_sparse=128, 
                        size=64, 
                        num_tokens=8,
                        num_class=11):
                
        super(Prototype_Prompt_Encoder, self).__init__()
        self.dense_fc_1 = nn.Conv2d(feat_dim, hidden_dim_dense, 1)
        self.dense_fc_2 = nn.Conv2d(hidden_dim_dense, feat_dim, 1)
        self.num_class = num_class
        
        self.relu = nn.ReLU()
        # 新增，降维线性层
        self.query_proj = nn.Linear(feat_dim, feat_dim)
        self.key_proj = nn.Linear(feat_dim, feat_dim)
        self.value_proj = nn.Linear(feat_dim, feat_dim)
        self.embedding_combiner = nn.Linear(2 * feat_dim, feat_dim)


        self.sparse_fc_1 = nn.Conv1d(size*size, hidden_dim_sparse, 1)
        self.sparse_fc_2 = nn.Conv1d(hidden_dim_sparse, num_tokens, 1)
        
        pn_cls_embeddings = [nn.Embedding(num_tokens, feat_dim) for _ in range(2)] # one for positive and one for negative 

            
        self.pn_cls_embeddings = nn.ModuleList(pn_cls_embeddings)
              
    def forward(self, feat, prototypes, cls_ids):
        cls_prompts = prototypes.unsqueeze(-1)
        cls_prompts = torch.stack([cls_prompts for _ in range(feat.size(0))], dim=0)
        feat = torch.stack([feat for _ in range(cls_prompts.size(1))], dim=1)
        sim = torch.matmul(feat, cls_prompts)
        
        # compute class-activated feature
        feat =  feat + feat*sim

        feat_sparse = feat.clone()
        
        
        # compute dense embeddings
        one_hot = torch.nn.functional.one_hot(cls_ids-1,self.num_class) 
        feat = feat[one_hot ==1]
        
        feat = rearrange(feat,'b (h w) c -> b c h w', h=64, w=64)
        dense_embeddings = self.dense_fc_2(self.relu(self.dense_fc_1(feat)))
        
        # compute sparse embeddings
        feat_sparse = rearrange(feat_sparse,'b num_cls hw c -> (b num_cls) hw c')
        sparse_embeddings = self.sparse_fc_2(self.relu(self.sparse_fc_1(feat_sparse)))
        sparse_embeddings = rearrange(sparse_embeddings,'(b num_cls) n c -> b num_cls n c', num_cls=self.num_class)
        
        pos_embed = self.pn_cls_embeddings[1].weight.unsqueeze(0).unsqueeze(0) * one_hot.unsqueeze(-1).unsqueeze(-1)
        neg_embed = self.pn_cls_embeddings[0].weight.unsqueeze(0).unsqueeze(0) * (1-one_hot).unsqueeze(-1).unsqueeze(-1)    
        
        # 交叉注意力：Pos -> Neg
        pos_query = self.query_proj(pos_embed)
        neg_key = self.key_proj(neg_embed)
        neg_value = self.value_proj(neg_embed)
        
        # 计算正样本对负样本的注意力分数
        attn_scores_pos_neg = torch.matmul(pos_query, neg_key.transpose(-2, -1))  # shape: [batch_size, num_tokens, num_tokens]
        attn_scores_pos_neg = F.softmax(attn_scores_pos_neg, dim=-1)
        context_pos_neg = torch.matmul(attn_scores_pos_neg, neg_value)
        
        # 交叉注意力：Neg -> Pos
        neg_query = self.query_proj(neg_embed)
        pos_key = self.key_proj(pos_embed)
        pos_value = self.value_proj(pos_embed)
        
        # 计算负样本对正样本的注意力分数
        attn_scores_neg_pos = torch.matmul(neg_query, pos_key.transpose(-2, -1))  # shape: [batch_size, num_tokens, num_tokens]
        attn_scores_neg_pos = F.softmax(attn_scores_neg_pos, dim=-1)
        context_neg_pos = torch.matmul(attn_scores_neg_pos, pos_value)
        
        # 结合正样本和负样本的上下文
        sparse_embeddings = torch.cat((context_pos_neg, context_neg_pos), dim=-1)
        sparse_embeddings = self.embedding_combiner(sparse_embeddings)
        sparse_embeddings = rearrange(sparse_embeddings,'b num_cls n c -> b (num_cls n) c')
        return dense_embeddings, sparse_embeddings

    # def forward(self, feat, prototypes, cls_ids):
  
    #     cls_prompts = prototypes.unsqueeze(-1)
    #     cls_prompts = torch.stack([cls_prompts for _ in range(feat.size(0))], dim=0)

        
    #     feat = torch.stack([feat for _ in range(cls_prompts.size(1))], dim=1)

    #     # compute similarity matrix 
    #     #print("feat shape:", feat.shape)
    #     # print("cls_prompts shape:", cls_prompts.shape)
    #     sim = torch.matmul(feat, cls_prompts)
        
    #     # compute class-activated feature
    #     feat =  feat + feat*sim

    #     feat_sparse = feat.clone()
        
    #     # compute dense embeddings
    #     one_hot = torch.nn.functional.one_hot(cls_ids-1,self.num_class) 
    #     feat = feat[one_hot ==1]
        
    #     feat = rearrange(feat,'b (h w) c -> b c h w', h=64, w=64)
    #     dense_embeddings = self.dense_fc_2(self.relu(self.dense_fc_1(feat)))
        
    #     # compute sparse embeddings
    #     feat_sparse = rearrange(feat_sparse,'b num_cls hw c -> (b num_cls) hw c')
    #     sparse_embeddings = self.sparse_fc_2(self.relu(self.sparse_fc_1(feat_sparse)))
    #     sparse_embeddings = rearrange(sparse_embeddings,'(b num_cls) n c -> b num_cls n c', num_cls=self.num_class)
        
    #     pos_embed = self.pn_cls_embeddings[1].weight.unsqueeze(0).unsqueeze(0) * one_hot.unsqueeze(-1).unsqueeze(-1)
    #     neg_embed = self.pn_cls_embeddings[0].weight.unsqueeze(0).unsqueeze(0) * (1-one_hot).unsqueeze(-1).unsqueeze(-1)
        

    #     sparse_embeddings = sparse_embeddings + pos_embed.detach() + neg_embed.detach()
            
    #     sparse_embeddings = rearrange(sparse_embeddings,'b num_cls n c -> b (num_cls n) c')
        
    #     return dense_embeddings, sparse_embeddings
    



# class Learnable_Prototypes(nn.Module):
#     def __init__(self, num_classes=7, feat_dim=256, text_feat_dim=768):
#         super(Learnable_Prototypes, self).__init__()
#         self.class_embeddings = nn.Embedding(num_classes, feat_dim)
#         self.fusion_layer = nn.Linear(feat_dim + text_feat_dim, feat_dim) 
        
#     def forward(self, text_features=None):
        
#         prototypes = self.class_embeddings.weight 
#         combined_features = torch.cat([prototypes, text_features], dim=1)
#         # 使用融合层处理拼接后的特征
#         fused_prototypes = self.fusion_layer(combined_features)
#         return fused_prototypes
    
#     def get_batch_embeddings(self, class_ids):
#         try:
#             batch_embeddings = self.class_embeddings(class_ids)
#             return batch_embeddings
#         except RuntimeError as e:
#             print(f"Error with class_ids: {class_ids}")
#             print(f"Embedding matrix size: {self.class_embeddings.weight.size()}")
#             raise e


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        attn_output, attn_output_weights = self.attention(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights)
        return attn_output, attn_output_weights

class Learnable_Prototypes(nn.Module):
    def __init__(self, num_classes=7, feat_dim=256, text_feat_dim=768, num_heads=4, common_dim=512):
        super(Learnable_Prototypes, self).__init__()
        self.class_embeddings = nn.Embedding(num_classes, feat_dim)
        self.text_to_common = nn.Linear(text_feat_dim, common_dim)  # 从文本特征到共同维度
        self.image_to_common = nn.Linear(feat_dim, common_dim)  # 从图像特征到共同维度
        self.text_to_image_attention = MultiHeadAttention(embed_dim=common_dim, num_heads=num_heads)
        self.image_to_text_attention = MultiHeadAttention(embed_dim=common_dim, num_heads=num_heads)
        self.fusion_layer = nn.Linear(common_dim * 2, feat_dim)  # 更新fusion_layer的维度
        
    def forward(self, text_features=None):
        prototypes = self.class_embeddings.weight
        
        # 将文本特征和图像特征转换到共同维度
        text_features_common = self.text_to_common(text_features.float())  # [batch_size, common_dim]
        prototypes_common = self.image_to_common(prototypes)  # [num_classes, common_dim]
        
        # 对齐维度
        text_features_common = text_features_common.unsqueeze(1)  # [batch_size, 1, common_dim]
        prototypes_common = prototypes_common.unsqueeze(1)  # [num_classes, 1, common_dim]
        
        # 文本到图像的注意力
        text_to_image_attn, _ = self.text_to_image_attention(
            query=text_features_common,
            key=prototypes_common,
            value=prototypes_common,
        )
        
        # 图像到文本的注意力
        image_to_text_attn, _ = self.image_to_text_attention(
            query=prototypes_common,
            key=text_features_common,
            value=text_features_common,
        )
        
        # 注意力输出拼接
        combined_features = torch.cat([text_to_image_attn.squeeze(1), image_to_text_attn.squeeze(1)], dim=1)
        
        # 使用融合层处理拼接后的特征
        fused_prototypes = self.fusion_layer(combined_features)
        return fused_prototypes
