r""" Visual Prompt Encoder of VRP-SAM """
from functools import reduce
from operator import add
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.base.resnet as models
import model.base.vgg as vgg_models
from torch.nn import BatchNorm2d as BatchNorm
from common.utils import get_stroke_preset, get_random_points_from_mask, get_mask_by_input_strokes

from pdb import set_trace as bp

from transformers import Dinov2Model, Dinov2Config, CLIPModel

from .base.transformer_decoder import transformer_decoder
from peft import LoraConfig, get_peft_model
from src.modeling import ImageEncoder

# copy from SEEM
def get_bounding_boxes(mask):
        """
        Returns:
            Boxes: tight bounding boxes around bitmasks.
            If a mask is empty, it's bounding box will be all zero.
        """
        boxes = torch.zeros(mask.shape[0], 4, dtype=torch.float32).to(mask.device)
        box_mask = torch.zeros_like(mask).to(mask.device)
        x_any = torch.any(mask, dim=1)
        y_any = torch.any(mask, dim=2)
        for idx in range(mask.shape[0]):
            x = torch.where(x_any[idx, :])[0].int()
            y = torch.where(y_any[idx, :])[0].int()
            if len(x) > 0 and len(y) > 0:
                boxes[idx, :] = torch.as_tensor(
                    [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32
                )
                x1, y1, x2, y2 = x[0], y[0], x[-1] + 1, y[-1] + 1
        
                box_mask[idx, y1:y2, x1:x2]=1
        return boxes, box_mask

def get_point_mask(mask, training, max_points=20):
        """
        Returns:
            Point_mask: random 20 point for train and test.
            If a mask is empty, it's Point_mask will be all zero.
        """
        max_points = min(max_points, mask.sum().item())
        if training:
            num_points = random.Random().randint(1, max_points) # get a random number of points 
        else:
            num_points = max_points 
        b,h,w = mask.shape
        point_masks = []

        for idx in range(b):
            view_mask = mask[idx].view(-1)
            non_zero_idx = view_mask.nonzero()[:,0] # get non-zero index of mask
            selected_idx = torch.randperm(len(non_zero_idx))[:num_points] # select id
            non_zero_idx = non_zero_idx[selected_idx] # select non-zero index
            rand_mask = torch.zeros(view_mask.shape).to(mask.device) # init rand mask
            rand_mask[non_zero_idx] = 1 # get one place to zero
            point_masks.append(rand_mask.reshape(h, w).unsqueeze(0))
        return torch.cat(point_masks, 0)

def get_scribble_mask(mask, training, stroke_preset=['rand_curve', 'rand_curve_small'], stroke_prob=[0.5, 0.5]):
        """
        Returns:
            Scribble_mask: random 20 point for train and test.
            If a mask is empty, it's Scribble_mask will be all zero.
        """
        if training:
            stroke_preset_name = random.Random().choices(stroke_preset, weights=stroke_prob, k=1)[0]
            nStroke = random.Random().randint(1, min(20, mask.sum().item()))
        else:
            stroke_preset_name = random.Random(321).choices(stroke_preset, weights=stroke_prob, k=1)[0]
            nStroke = random.Random(321).randint(1, min(20, mask.sum().item()))
        preset = get_stroke_preset(stroke_preset_name)
        
        b,h,w = mask.shape
        
        scribble_masks = []
        for idx in range(b):
            points = get_random_points_from_mask(mask[idx].bool(), n=nStroke)  
            rand_mask = get_mask_by_input_strokes(init_points=points, imageWidth=w, imageHeight=h, nStroke=min(nStroke, len(points)), **preset)
            rand_mask = (~torch.from_numpy(rand_mask)) * mask[idx].bool().cpu()
            scribble_masks.append(rand_mask.float().unsqueeze(0))
        return torch.cat(scribble_masks, 0).to(mask.device)

def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs, targets = inputs.flatten(1), targets.flatten(1)
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

class FiLM(nn.Module):
    def __init__(self, feature_dim=1024):
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(feature_dim, feature_dim)
        self.beta = nn.Linear(feature_dim, feature_dim)

    def forward(self, x, condition):
        gamma = self.gamma(condition)
        beta = self.beta(condition)
        return gamma * x + beta

class VRP_encoder(nn.Module):
    def __init__(self, args, backbone, use_original_imgsize, pretrained=False):
        super(VRP_encoder, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        self.arch_testing = args.arch_testing
        self.dino_lora = args.dino_lora
        self.clip_applied = args.clip_applied
        self.pattern_text = args.pattern_text

        if backbone == 'vgg16':
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        elif backbone == 'clip':
            # self.clip_model, _, _ = open_clip.create_model_and_transforms(
            #     'ViT-L-14', pretrained='openai', cache_dir='./.cache/open_clip'
            # )
            model_name = 'ViT-L-14'
            checkpoint_path = 'base_ckpt/finetuned.pt'

            # Load the model
            self.clip_model = ImageEncoder.load(checkpoint_path)
            
            # model_object = torch.load('base_ckpt/finetuned.pt')
            # state_dict = model_object['model']
            # self.clip_model.load_state_dict(state_dict)
            self.clip_model.eval()

            for param in self.clip_model.parameters():
                param.requires_grad = False

        elif backbone == 'dino_v2':
            # dinov2 = models.dino_v2(pretrained=pretrained)
            # self.curr_model = dinov2 # for debug purpose
            # self.layer0 = dinov2.layer0
            # self.layer1, self.layer2, self.layer3, self.layer4 = dinov2.layer1, dinov2.layer2, dinov2.layer3, dinov2.layer4
            self.dino_v2 = Dinov2Model.from_pretrained("facebook/dinov2-large")
            if args.clip_applied:
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
                for param in self.clip_model.parameters():
                    param.requires_grad = False
                self.clip_model.eval()
            # make the backbone not trainable
            # if not args.prompt_backbone_trainable:
            #     self.dino_v2.eval()
            #     # free params
            #     for param in self.dino_v2.parameters():
            #         param.requires_grad = False

            if not args.dino_lora:
                for param in self.dino_v2.parameters():
                    param.requires_grad = False
                self.dino_v2.eval()

            # Define LoRA configuration
            lora_config = LoraConfig(
                r=args.lora_r, 
                lora_alpha=1.0,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none"
            )

            self.dino_v2 = get_peft_model(self.dino_v2, lora_config)
            # lora

            for param in self.dino_v2.parameters():
                param.requires_grad = False
            
            
            for name, param in self.dino_v2.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True

            # for name, param in self.dino_v2.named_parameters():
            #     print(name, param.requires_grad)

        # elif backbone == 'clip':
        #     import openclip
        #     self.clip_model = openclip.

            

        elif backbone == 'golden_muscat':
            golden_muscat = models.golden_muscat(pretrained=pretrained)
            self.layer0 = golden_muscat.layer0
            self.layer1, self.layer2, self.layer3, self.layer4 = golden_muscat.layer1, golden_muscat.layer2, golden_muscat.layer3, golden_muscat.layer4

        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        
        if args.prompt_backbone_trainable and backbone not in ['dino_v2', 'clip']:
            self.layer0.train(), self.layer1.train(), self.layer2.train(), self.layer3.train(), self.layer4.train()
            for param in self.layer0.parameters():
                param.requires_grad = True
            for param in self.layer1.parameters():
                param.requires_grad = True
            for param in self.layer2.parameters():
                param.requires_grad = True
            for param in self.layer3.parameters():
                param.requires_grad = True
            for param in self.layer4.parameters():
                param.requires_grad = True
        elif backbone != 'dino_v2' and backbone != 'clip':
            self.layer0.eval(), self.layer1.eval(), self.layer2.eval(), self.layer3.eval(), self.layer4.eval()

        if backbone == 'vgg16':
            fea_dim = 512 + 256
        elif backbone in ['dino_v2', 'clip']:
            fea_dim = 1024 + 1024
            # fea_dim = 257 + 257
        elif backbone == 'golden_muscat':
            fea_dim = 256 + 512
        else:
            fea_dim = 1024 + 512 
            
        hidden_dim = 256

        if args.clip_applied:
            self.film_0 = FiLM(feature_dim=1024)
            self.film_1 = FiLM(feature_dim=1024)
            self.film_2 = FiLM(feature_dim=1024)
            self.film_3 = FiLM(feature_dim=1024)
            self.film_4 = FiLM(feature_dim=1024)

            self.pattern = ['horizontal lines', 'vertical lines', 'cross hatching', 'dots', 'wavy lines', 'diagonal lines',
                            'checkerboard', 'solid fill', 'brick', 'hatching']


        # self.channel_to_256 = nn.Sequential(
        #     nn.Conv1d(257, hidden_dim, kernel_size=1, padding=0, bias=False),
        #     nn.ReLU(inplace=True),
        # )

        convnd = nn.Conv2d(fea_dim, hidden_dim, kernel_size=1, padding=0, bias=False) if backbone not in ['dino_v2', 'clip'] or args.arch_testing \
           else nn.Conv1d(fea_dim, hidden_dim, kernel_size=1, padding=0, bias=False)

        dropoutnd = nn.Dropout2d(p=0.5) if backbone not in ['dino_v2', 'clip'] else nn.Dropout(p=0.5)

        self.downsample_query = nn.Sequential(
            convnd,        # nn.Conv2d(fea_dim, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            dropoutnd, #nn.Dropout2d(p=0.5)
        )
        
        convnd_1 = nn.Conv2d(hidden_dim*2+1, hidden_dim, kernel_size=1, padding=0, bias=False) # if backbone != 'dino_v2' \
           # else nn.Conv1d(hidden_dim*2+1, hidden_dim, kernel_size=1, padding=0, bias=False)
        
        self.merge_1 = nn.Sequential(
            convnd_1, #nn.Conv2d(hidden_dim*2+1, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        
        self.num_query = args.num_query

        self.transformer_decoder = transformer_decoder(args, args.num_query, hidden_dim, hidden_dim*2)

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()


    def forward(self, condition, query_img, support_img, support_mask, training, legend_img=None):

        # if training:
        #     condition = random.Random().choices(['scribble', 'point', 'box', 'mask'], weights=[0.25,0.25,0.25,0.25], k=1)[0]  
        # else:
        #     condition = condition

        if condition == 'scribble':
            support_mask_ori = get_scribble_mask(support_mask,training) # scribble_mask
        elif condition == 'point':
            support_mask_ori = get_point_mask(support_mask,training) # point_mask
        elif condition == 'box':
            boxes, support_mask_ori = get_bounding_boxes(support_mask) # box_mask
        elif condition == 'mask':
            support_mask_ori = support_mask


        # with torch.no_grad():
            # config_dino = Dinov2Config()
        if self.backbone_type == 'clip':
            patch_size = 16
            support_mask = F.interpolate(support_mask_ori.unsqueeze(1).float(), size=(patch_size, patch_size), mode='nearest')
            support_mask = support_mask.squeeze(1).unsqueeze(-1)
            # dummy_input = torch.randn(1, 3, 224, 224)
            outputs_clip_supp = self.clip_model(support_img)['hidden_states']

            supp_feat_0 = outputs_clip_supp[0]
            supp_feat_1 = outputs_clip_supp[7]
            supp_feat_2 = outputs_clip_supp[11]
            supp_feat_3 = outputs_clip_supp[20]
            supp_feat_4 = outputs_clip_supp[-1]

            outputs_clip_query = self.clip_model(query_img)['hidden_states']
            query_feat_0 = outputs_clip_query[0]
            query_feat_1 = outputs_clip_query[7]
            query_feat_2 = outputs_clip_query[11]
            query_feat_3 = outputs_clip_query[20]
            query_feat_4 = outputs_clip_query[-1]

            query_feat = torch.cat([query_feat_2, query_feat_3], 2)
            supp_feat = torch.cat([supp_feat_2, supp_feat_3], 2)

            pseudo_mask = self.get_pseudo_dino_mask(supp_feat_4, query_feat_4, support_mask)

            query_feat = query_feat[:, 1:]    #self.channel_to_256(query_feat)

            
            query_feat = self.downsample_query(query_feat.permute(0, 2, 1))
            query_feat = query_feat.view(query_feat.size(0), query_feat.size(1), patch_size, patch_size)

            supp_feat = supp_feat[:, 1:]       #self.channel_to_256(supp_feat)
            supp_feat = self.downsample_query(supp_feat.permute(0, 2, 1))
            supp_feat = supp_feat.view(supp_feat.size(0), supp_feat.size(1), patch_size, patch_size)

            support_mask = support_mask.permute(0, 3, 1, 2)




            

        elif self.backbone_type == 'dino_v2':
            # large dino v2 has up to 25 (0-24) hidden states
            
            patch_size = 16  # patch size without class token
            # just to make sure the patch size is 14x14 but the total patch num is 16x16
            support_mask = F.interpolate(support_mask_ori.unsqueeze(1).float(), size=(patch_size, patch_size), mode='nearest')
            support_mask = support_mask.squeeze(1).unsqueeze(-1)
            
            outputs_dino_supp = self.dino_v2(support_img,
                                            # bool_masked_pos=support_mask.flatten(1).bool(),
                                            output_hidden_states=True)
            
            supp_feat_0 = outputs_dino_supp['hidden_states'][0]
            supp_feat_1 = outputs_dino_supp['hidden_states'][7]
            supp_feat_2 = outputs_dino_supp['hidden_states'][11]
            supp_feat_3 = outputs_dino_supp['hidden_states'][20]
            supp_feat_4 = outputs_dino_supp['last_hidden_state']

            
            
            if self.dino_lora:
                outputs_dino_query = self.dino_v2(query_img, output_hidden_states=True) # keys : ['last_hidden_state', 'pooler_output', 'hidden_states']
                query_feat_0 = outputs_dino_query['hidden_states'][0]
                query_feat_1 = outputs_dino_query['hidden_states'][7]
                query_feat_2 = outputs_dino_query['hidden_states'][11]
                query_feat_3 = outputs_dino_query['hidden_states'][20]
                query_feat_4 = outputs_dino_query['last_hidden_state']

                if self.clip_applied:
                    assert legend_img is not None

                    clip_outputs = self.clip_model.vision_model(pixel_values=legend_img, output_hidden_states=True)
                    clip_feat_0 = clip_outputs['hidden_states'][0]
                    clip_feat_1 = clip_outputs['hidden_states'][7]
                    clip_feat_2 = clip_outputs['hidden_states'][11]
                    clip_feat_3 = clip_outputs['hidden_states'][20]
                    clip_feat_4 = clip_outputs['last_hidden_state']

                    query_feat_0 = self.film_0(query_feat_0, clip_feat_0)
                    query_feat_1 = self.film_1(query_feat_1, clip_feat_1)
                    query_feat_2 = self.film_2(query_feat_2, clip_feat_2)
                    query_feat_3 = self.film_3(query_feat_3, clip_feat_3)
                    query_feat_4 = self.film_4(query_feat_4, clip_feat_4)

                elif self.clip_applied and self.pattern_text:
                    assert legend_img is not None

                

                    
                    


            

            # else:
            #     tmp_token_holder = self.dino_v2.embeddings.cls_token
            #     # self.dino_v2.embeddings.cls_token.shape torch.Size([1, 1, 1024])
            #     query_feat_0, query_feat_1, query_feat_2, query_feat_3, query_feat_4 = [], [], [], [], []

            #     for curr_idx in range(0, query_img.size(0)):
            #         curr_dino_cls_token = outputs_dino_supp['last_hidden_state'][curr_idx:curr_idx+1, :1]
            #         self.dino_v2.embeddings.cls_token = torch.nn.Parameter(curr_dino_cls_token)
            #         outputs_dino_query = self.dino_v2(query_img[curr_idx:curr_idx+1], output_hidden_states=True)
            #         query_feat_0.append(outputs_dino_query['hidden_states'][0])
            #         query_feat_1.append(outputs_dino_query['hidden_states'][7])
            #         query_feat_2.append(outputs_dino_query['hidden_states'][11])
            #         query_feat_3.append(outputs_dino_query['hidden_states'][20])
            #         query_feat_4.append(outputs_dino_query['last_hidden_state'])
            #     query_feat_0 = torch.cat(query_feat_0, 0)
            #     query_feat_1 = torch.cat(query_feat_1, 0)
            #     query_feat_2 = torch.cat(query_feat_2, 0)
            #     query_feat_3 = torch.cat(query_feat_3, 0)
            #     query_feat_4 = torch.cat(query_feat_4, 0)

            #     self.dino_v2.embeddings.cls_token = tmp_token_holder



            
            query_feat = torch.cat([query_feat_2, query_feat_3], 2)



            
            # supp_feat_3_reshaped = supp_feat_3[:, 1:, :].reshape(batch_size, patch_size, patch_size, dim)

            # # Interpolate the support_mask_ori to match the spatial dimensions of supp_feat_3
            
            
            # supp_feat_3_masked = supp_feat_3_reshaped * support_mask
            # supp_feat_3_masked = supp_feat_3_masked.reshape(batch_size, patch_size * patch_size, dim)
            # supp_feat_3 = torch.cat([supp_feat_3[:, 0:1, :], supp_feat_3_masked], dim=1)

            # supp_feat_4 = self.dino_v2(support_img).last_hidden_state
            # supp_feat_4_reshaped = supp_feat_4[:, 1:, :].reshape(batch_size, patch_size, patch_size, dim)
            # supp_feat_4_masked = supp_feat_4_reshaped * support_mask
            # supp_feat_4_masked = supp_feat_4_masked.reshape(batch_size, patch_size * patch_size, dim)
            # supp_feat_4 = torch.cat([supp_feat_4[:, 0:1, :], supp_feat_4_masked], dim=1)

            supp_feat = torch.cat([supp_feat_2, supp_feat_3], 2)

            pseudo_mask = self.get_pseudo_dino_mask(supp_feat_4, query_feat_4, support_mask)

            # self.dino_v2(support_img, bool_masked_pos=pseudo_mask.flatten(1).bool()).keys()


            # support_mask_unsqueeze = support_mask_ori.unsqueeze(1).float()
            # support_mask = F.interpolate(support_mask_unsqueeze, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='nearest')
            # support_mask =F.interpolate(support_mask_ori.unsqueeze(1).float(), size=img_size//patch_size, mode='nearest').flatten(1)[:,:,None]
            
            # self.dino_v2.embeddings.mask_token.shape
            # torch.Size([1, 1024])
            
            # self.dino_v2.embeddings.patch_embeddings
            # Dinov2PatchEmbeddings(
            #   (projection): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14))
            # )
            

            query_feat = query_feat[:, 1:]    #self.channel_to_256(query_feat)

            if self.arch_testing:
                query_feat = query_feat.permute(0, 2, 1)
                query_feat = query_feat.view(query_feat.size(0), query_feat.size(1), patch_size, patch_size)
                query_feat = self.downsample_query(query_feat)
            else:
                # from bs, 256, 256 to bs, 256, 16, 16
                query_feat = self.downsample_query(query_feat.permute(0, 2, 1))
                query_feat = query_feat.view(query_feat.size(0), query_feat.size(1), patch_size, patch_size)
            
            
            supp_feat = supp_feat[:, 1:]       #self.channel_to_256(supp_feat)
            if self.arch_testing:
                supp_feat = supp_feat.permute(0, 2, 1)
                supp_feat = supp_feat.view(supp_feat.size(0), supp_feat.size(1), patch_size, patch_size)
                supp_feat = self.downsample_query(supp_feat)
            else:
                supp_feat = self.downsample_query(supp_feat.permute(0, 2, 1))
                supp_feat = supp_feat.view(supp_feat.size(0), supp_feat.size(1), patch_size, patch_size)
            
            # support mask from bs, 16, 16, 1 to bs, 1, 16, 16
            support_mask = support_mask.permute(0, 3, 1, 2)
            
        else:
            query_img = torch.cat([query_img, query_img], 1) if self.backbone_type == 'golden_muscat' else query_img
            query_feat_0 = self.layer0(query_img) if self.backbone_type != 'golden_muscat' \
                else self.layer0(query_img)
            
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)
            if self.backbone_type == 'vgg16' or self.backbone_type == 'golden_muscat':
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)), mode='bilinear', align_corners=True)
            query_feat = torch.cat([query_feat_2, query_feat_3], 1) 

            support_img = torch.cat([support_img, support_img], 1) if self.backbone_type == 'golden_muscat' else support_img
            supp_feat_0 = self.layer0(support_img) if self.backbone_type != 'golden_muscat' \
                else self.layer0(support_img)
            
            supp_feat_1 = self.layer1(supp_feat_0)
            supp_feat_2 = self.layer2(supp_feat_1)
            supp_feat_3 = self.layer3(supp_feat_2)


            support_mask = F.interpolate(support_mask_ori.unsqueeze(1).float(), size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='nearest')
            supp_feat_4 = self.layer4(supp_feat_3*support_mask) 
            if self.backbone_type == 'vgg16' or self.backbone_type == 'golden_muscat':
                supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)
            supp_feat = torch.cat([supp_feat_2, supp_feat_3], 1)

            pseudo_mask = self.get_pseudo_mask(supp_feat_4, query_feat_4, support_mask) # [4, 1, 16, 16]

            query_feat = self.downsample_query(query_feat) # [4, 256, 32, 32]
            supp_feat = self.downsample_query(supp_feat)

        prototype = self.mask_feature(supp_feat, support_mask) if self.backbone_type not in ['dino_v2', 'clip'] else \
                self.mask_feature_dino(supp_feat, support_mask)
        supp_feat_bin = prototype.repeat(1, 1, query_feat.shape[2], query_feat.shape[3])
    
        supp_feat_1 = self.merge_1(torch.cat([supp_feat, supp_feat_bin, support_mask*10], 1))                                                                                    
        
        if self.backbone_type == 'golden_muscat':
            pseudo_mask = F.interpolate(pseudo_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
        query_feat_1 = self.merge_1(torch.cat([query_feat, supp_feat_bin, pseudo_mask*10], 1))
        

        protos = self.transformer_decoder(query_feat_1, supp_feat_1, support_mask)
        # protos = self.transformer_decoder(query_feat, supp_feat, support_mask)
        
        return protos, support_mask_ori

    
    def mask_feature_dino(self, features, support_mask):
        mask = support_mask
        supp_feat = features# * mask
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w # / area
        return supp_feat

    def mask_feature(self, features, support_mask):
        mask = support_mask
        supp_feat = features * mask
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
        return supp_feat

    def predict_mask_nshot(self, args, batch, sam_model, nshot, input_point=None):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        protos_set = []
        for s_idx in range(nshot):
            protos_sub, support_mask = self(args.condition, batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx], False)
            protos_set.append(protos_sub)
        if nshot > 1:
            protos = torch.cat(protos_set, dim=1)
        else:
            protos = protos_sub

        low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos,input_point)
        logit_mask = low_masks
        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
        pred_mask = torch.sigmoid(logit_mask) >= 0.5

        pred_mask = pred_mask.float()
            
        logit_mask_agg += pred_mask.squeeze(1).clone()
        return logit_mask_agg, support_mask, logit_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        loss_bce = self.bce_with_logits_loss(logit_mask.squeeze(1), gt_mask.squeeze(1).float())
        loss_dice = dice_loss(logit_mask, gt_mask, bsz)
        return loss_bce + loss_dice
        

    def train_mode(self):
        self.train()
        self.apply(fix_bn)
        if self.backbone_type != 'dino_v2' and self.backbone_type != 'clip':
            self.layer0.eval(), self.layer1.eval(), self.layer2.eval(), self.layer3.eval(), self.layer4.eval()


    def get_pseudo_mask(self, tmp_supp_feat, query_feat_4, mask, mask_bool=True):
        resize_size = tmp_supp_feat.size(2)
        tmp_mask = F.interpolate(mask, size=(resize_size, resize_size), mode='bilinear', align_corners=True)

        tmp_supp_feat_4 = tmp_supp_feat * tmp_mask if mask_bool else tmp_supp_feat
        q = query_feat_4
        s = tmp_supp_feat_4
        bsize, ch_sz, sp_sz, _ = q.size()[:]

        tmp_query = q
        tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s               
        tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1) 
        tmp_supp = tmp_supp.permute(0, 2, 1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

        cosine_eps = 1e-7
        similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
        similarity = similarity.max(1)[0].reshape(bsize, sp_sz*sp_sz)
        corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
        return corr_query

    def get_pseudo_dino_mask(self, tmp_supp_feat, query_feat_4, mask):

        # Assuming tmp_supp_feat is (batch, 257, 1024)
        # and mask is (batch, 16, 16, 1)
        bsize, num_patches, dim = tmp_supp_feat.shape
        patch_size = int((num_patches - 1) ** 0.5)
        

        ############# out ########################
        # # Interpolate the mask to match the spatial dimensions of the patches
        # tmp_mask = F.interpolate(mask.permute(0, 3, 1, 2), size=(patch_size, patch_size), mode='bilinear', align_corners=True)

        # # Ensure the mask is broadcastable with tmp_supp_feat
        # tmp_mask = tmp_mask.view(bsize, -1)  # Shape (batch, num_patches)
        # tmp_mask = tmp_mask.unsqueeze(-1)  # Shape (batch, num_patches, 1)

        # # Apply the mask to tmp_supp_feat excluding the class token
        # tmp_supp_feat_4 = tmp_supp_feat[:, 1:, :] * tmp_mask[:, :num_patches-1, :]


        ########################################

        tmp_supp_feat_4 = tmp_supp_feat[:, 1:, :]

        # Reshape the features
        queue = query_feat_4[:, 1:, :]  # Excluding class token
        s_ = tmp_supp_feat_4

        sp_sz = patch_size  # Spatial size (assuming square patches)

        tmp_query = queue.view(bsize, dim, num_patches-1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s_.view(bsize, num_patches-1, dim)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        cosine_eps = 1e-7
        similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
        similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
        corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)

        return corr_query
