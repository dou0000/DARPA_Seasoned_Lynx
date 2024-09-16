from segment_anything import sam_model_registry
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model.sam_lora import LoRA_Sam

class SAM_pred(nn.Module):
    def __init__(self, mask_decoder_trainable=False, lora_train=False, args=None):
        super().__init__()

        self.lora_train = lora_train


        # self.sam_model = sam_model_registry['vit_h']('/root/paddlejob/workspace/env_run/vrp_sam/sam_vit_h_4b8939.pth')
        self.sam_model = sam_model_registry['vit_h']('/u/dkwark/VRP-SAM/base_ckpt/sam_ckpt/sam_vit_h_4b8939.pth')
        self.sam_model.eval()
        
        # no grad required for the model
        for name, param in self.sam_model.named_parameters():
            param.requires_grad = False

        self.mask_decoder = self.sam_model.mask_decoder

        # only make the mask decoder trainable
        if mask_decoder_trainable:
            for name, param in self.sam_model.named_parameters():
                if 'mask_decoder' in name:
                    param.requires_grad = True

        # only make the lora trainable
        if lora_train:
            self.sam_model = LoRA_Sam(args, self.sam_model, r=args.lora_r).sam


        # check if the model is freezed
        # for name, param in self.sam_model.named_parameters():
        #     print(name, param.requires_grad)
        

        # unfreeze the model
        # for param in self.sam_model.parameters():
        #     param.requires_grad = True
            

    def forward_img_encoder(self, query_img):
        query_img = F.interpolate(query_img, (1024,1024), mode='bilinear', align_corners=True)
        if self.lora_train:
            query_feats = self.sam_model.image_encoder(query_img)
        else:
            with torch.no_grad():
                query_feats = self.sam_model.image_encoder(query_img)

        # query_feats = self.sam_model.image_encoder(query_img)
        return query_feats
    
    def get_pormpt(self, protos, points_mask=None):
        if points_mask is not None :
            point_mask = points_mask

            postivate_pos = (point_mask.squeeze(0).nonzero().unsqueeze(0) + 0.5) * 64 -0.5
            postivate_pos = postivate_pos[:,:,[1,0]]
            point_label = torch.ones(postivate_pos.shape[0], postivate_pos.shape[1]).to(postivate_pos.device)
            point_prompt = (postivate_pos, point_label)
        else:
            point_prompt = None
        protos = protos
        return  protos, point_prompt

    def forward_prompt_encoder(self, points=None, boxes=None, protos=None, masks=None):
        q_sparse_em, q_dense_em = self.sam_model.prompt_encoder(
                points=points,
                boxes=None,
                protos=protos,
                masks=None)
        return  q_sparse_em, q_dense_em
    
    def forward_mask_decoder(self, query_feats, q_sparse_em, q_dense_em, ori_size=(512,512)):
        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=query_feats,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=q_sparse_em,
                dense_prompt_embeddings=q_dense_em,
                multimask_output=False)
        low_masks = F.interpolate(low_res_masks, size=ori_size, mode='bilinear', align_corners=True)
            
        # from torch.nn.functional import threshold, normalize

        # binary_mask = normalize(threshold(low_masks, 0.0, 0))
        binary_mask = torch.where(low_masks > 0, 1, 0)
        return low_masks, binary_mask
    
    def forward(self, query_img, protos, query_name=None, points_mask=None):
        B,C, h, w = query_img.shape
        
        # query_img = F.interpolate(query_img, (1024,1024), mode='bilinear', align_corners=True)
        protos, point_prompt = self.get_pormpt(protos, points_mask)
        query_feats = self.forward_img_encoder(query_img)
            

        q_sparse_em, q_dense_em = self.forward_prompt_encoder(
                points=point_prompt,
                boxes=None,
                protos=protos,
                masks=None)
            
        low_masks, binary_mask = self.forward_mask_decoder(query_feats, q_sparse_em, q_dense_em, ori_size=(h, w))

        return low_masks, binary_mask.squeeze(1)

        # low_mask_set = []
        # binary_mask_set = []

        # for idx in range(protos.shape[1]):
        #     q_sparse_em, q_dense_em = self.forward_prompt_encoder(
        #         points=point_prompt,
        #         boxes=None,
        #         protos=protos[:,idx,:].unsqueeze(1),
        #         masks=None)
            
        #     low_masks, binary_mask = self.forward_mask_decoder(query_feats, q_sparse_em, q_dense_em, ori_size=(h, w))

        #     low_mask_set.append(low_masks)
        #     binary_mask_set.append(binary_mask.squeeze(1))

        # return low_mask_set, binary_mask_set