r""" Visual Prompt Encoder training (validation) code """
import os
import argparse

import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist

from model.sam_lora import LoRA_Sam
from model.VRP_encoder import VRP_encoder
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from SAM2pred import SAM_pred

# from ipdb import set_trace as bp
import numpy as np
import wandb
import tqdm 

from transformers import Dinov2Model, Dinov2Config
# from peft import LoraConfig, get_peft_model
# # from peft import LoraConfig, get_peft_model

import cv2
from torchmetrics.functional import jaccard_index, f1_score

from pdb import set_trace as bp

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def reduce_tensor(tensor, world_size):
    r""" Reduce the tensor across all GPUs """
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor


def train(args, epoch, model, sam_model, dataloader, optimizer, scheduler, training):
    r""" Train VRP_encoder model """

    utils.fix_randseed(args.seed + epoch) if training else utils.fix_randseed(args.seed)
    if args.parallel:
        model.module.train_mode() if training else model.module.eval()
        sam_model.module.mask_decoder.train() if training and args.mask_decoder_trainable \
                    else sam_model.module.mask_decoder.eval()
        if args.lora:
            sam_model.module.train() if training else sam_model.module.eval()
        elif args.mask_decoder_trainable:
            sam_model.module.mask_decoder.train() if training else sam_model.module.mask_decoder.eval()
        else:
            sam_model.module.eval()
    else:
        model.train_mode() if training else model.eval()
        if args.lora:
            sam_model.train() if training else sam_model.eval()
        elif args.mask_decoder_trainable:
            sam_model.mask_decoder.train() if training and args.mask_decoder_trainable \
                        else sam_model.mask_decoder.eval()
        else:
            sam_model.eval()

    # average_meter = AverageMeter(dataloader.dataset)
    # average_meter = AverageMeter(args.bsz)

    total_loss = []
    total_iou = []
    total_f1 = []

    tqdm_dataloader = tqdm.tqdm(dataloader)
    for idx, batch in enumerate(tqdm_dataloader):
    # for idx, batch in enumerate(dataloader):
        
        batch = utils.to_cuda(batch)

        # query_img : map_patch
        # support_imgs : support_patches
        # support_masks : support_masks


        curr_input_img = batch['query_img'] if args.backbone != 'dino_v2' \
                            else batch['query_img_dino']
        curr_support_img = batch['support_imgs'].squeeze(1) if args.backbone != 'dino_v2' \
                            else batch['support_img_dino'].squeeze(1)
        
        if args.clip_applied:
            additional_legend_support_img = batch['support_imgs'].squeeze(1)
        else:
            additional_legend_support_img = None

        protos, _ = model(args.condition, 
                          curr_input_img,       
                          curr_support_img,
                          batch['support_masks'].squeeze(1), 
                          training,
                          additional_legend_support_img)

        low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos)
        logit_mask = low_masks
        
        pred_mask = torch.sigmoid(logit_mask) > 0.5
        pred_mask = pred_mask.float()

        loss = model.module.compute_objective(logit_mask, batch['query_mask']) if args.parallel \
            else model.compute_objective(logit_mask, batch['query_mask'])

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # tp, fp, fn = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
        # average_meter.update(tp, fp, fn, loss.detach().clone())

        # Calculate F1 score for the current batch
        # gt_mask = batch['query_mask'].cpu().numpy().flatten()
        # pred_mask_np = pred_mask.cpu().numpy().flatten()
        # batch_f1 = f1_score(gt_mask, pred_mask_np, average='binary') * 100
        iou = jaccard_index(pred_mask, batch['query_mask'], task='binary')
        f1 = f1_score(pred_mask, batch['query_mask'].int(), task='binary', average='macro')

        # avg_loss = loss.item()
        # avg_iou = iou.item()
        # avg_f1 = f1.item()

        # total_loss.append(avg_loss)
        # total_iou.append(avg_iou)
        # total_f1.append(avg_f1)
        total_loss.append(loss)
        total_iou.append(iou)
        total_f1.append(f1)


        # folder name
        folder_name = f'viz_mask' # 4 is default
        os.makedirs(f'./viz/{folder_name}', exist_ok=True)

        # print(f'Current Batch F1: {batch_f1:.2f}')
        if utils.is_main_process():
            train_val = 'Training' if training else 'Validation'
            if idx % 10 == 0:
                avg_loss = loss.item()
                avg_iou = iou.item()
                avg_f1 = f1.item()
                
                # gpu suages
                if idx % 1000 == 0:
                    os.system('nvidia-smi')
                
                print(f"Epoch: {epoch}, Batch: {idx}, Loss: {loss:.2f}, IoU: {iou:.2f}, F1: {f1:.2f}")
        
                
                wandb.log({
                    f"{train_val}_loss": avg_loss,
                    f"{train_val}_iou": avg_iou,
                    f"{train_val}_f1_score": avg_f1,
                })

            if idx % 50 == 0:
                
                input_images, ground_truth_masks, predicted_masks = [], [], []
                for i in range(batch['query_img'].shape[0]):
                    img = batch['query_img'][i].cpu().numpy().transpose(1, 2, 0)
                    mask = batch['query_mask'][i].cpu().numpy().squeeze()
                    curr_pred = pred_mask[i].cpu().numpy().squeeze()

                    target_size = (img.shape[1] // 2, img.shape[0] // 2)

                    # Normalize and convert to uint8
                    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
                    img = (img * 255).astype('uint8')
                    mask = (mask * 255).astype('uint8')
                    curr_pred = (curr_pred * 255).astype('uint8')

                    # Convert grayscale masks to 3-channel images for stacking
                    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    curr_pred = cv2.cvtColor(curr_pred, cv2.COLOR_GRAY2BGR)

                    # Resize images to half their size
                    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_AREA)
                    curr_pred = cv2.resize(curr_pred, target_size, interpolation=cv2.INTER_AREA)

                    input_images.append(img)
                    ground_truth_masks.append(mask)
                    predicted_masks.append(curr_pred)

                # Concatenate images horizontally
                input_row = np.hstack(input_images)
                ground_truth_row = np.hstack(ground_truth_masks)
                prediction_row = np.hstack(predicted_masks)

                # Stack the rows vertically
                combined_image = np.vstack((input_row, ground_truth_row, prediction_row))

                wandb.log({
                    f"{train_val}_image": wandb.Image(combined_image, caption=f"Epoch {epoch} - Batch {idx}"),
                })









        
        ############ IoU evaluation ############
        # area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
        # # average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        # average_meter.update(area_inter, area_union, loss.detach().clone())
        # average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # average_meter.write_result('Training' if training else 'Validation', epoch)
    # avg_loss = utils.mean(average_meter.loss_buf)
    # f1 = average_meter.compute_f1()

    total_loss_tensor = torch.stack(total_loss).mean()
    total_iou_tensor = torch.stack(total_iou).mean()
    total_f1_tensor = torch.stack(total_f1).mean()

    if args.parallel:
        world_size = dist.get_world_size()
        total_loss_tensor = reduce_tensor(total_loss_tensor, world_size)
        total_iou_tensor = reduce_tensor(total_iou_tensor, world_size)
        total_f1_tensor = reduce_tensor(total_f1_tensor, world_size)

    avg_loss = total_loss_tensor.item()
    avg_iou = total_iou_tensor.item()
    avg_f1 = total_f1_tensor.item()

    print(f"Epoch: {epoch}, {'Training' if training else 'Validation'} Loss: {avg_loss:.2f}, IoU: {avg_iou:.2f}, F1: {avg_f1:.2f}")

    return avg_loss, avg_f1, avg_iou

    # avg_loss = sum(total_loss) / len(total_loss)
    # f1 = sum(total_f1) / len(total_f1)
    # iou = sum(total_iou) / len(total_iou)

    # print(f"Epoch: {epoch}, {'Training' if training else 'Validation'} Loss: {avg_loss:.2f}, IoU: {iou:.2f}, F1: {f1:.2f}")

    # return avg_loss, f1, iou


    # miou, fb_iou = average_meter.compute_iou()

    # return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/root/paddlejob/workspace/env_run/datsets/')
    parser.add_argument('--benchmark', type=str, default='map', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=2) # batch size = num_gpu * bsz default num_gpu = 4
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--condition', type=str, default='scribble', choices=['point', 'scribble', 'box', 'mask'])
    parser.add_argument('--use_ignore', type=bool, default=True, help='Boundaries are not considered during pascal training')
    parser.add_argument('--local-rank', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--num_query', type=int, default=50)
    
    parser.add_argument('--backbone', type=str, default='dino_v2', choices=['vgg16', 'resnet50', 'resnet101', 'dino_v2', 'golden_muscat'])
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--mask_decoder_trainable', type=bool, default=False)
    parser.add_argument('--exp_name', type=str, default='exp_name')
    parser.add_argument('--starter', type=bool, default=False)
    parser.add_argument('--lora', type=bool, default=False)

    parser.add_argument('--if_encoder_lora_layer', type=bool, default=False) # [0,1,10,11]
    parser.add_argument('--encoder_lora_layer', type=list, default=[], help='depth of blocks to add lora layer, if [], it will add at each layer')
    parser.add_argument('--if_decoder_lora_layer', type=bool, default=False)
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--prompt_backbone_trainable', type=bool, default=False)
    parser.add_argument('--dino_lora', type=bool, default=False)
    parser.add_argument('--arch_testing', type=bool, default=False)
    parser.add_argument('--img_size_for_models', type=int, default=1024)

    parser.add_argument('--clip_applied', type=bool, default=False)
    parser.add_argument('--overlap', type=str, default=None)
    parser.add_argument('--pattern_text', type=bool, default=False)

    args = parser.parse_args()

    # Distributed setting
    local_rank = args.local_rank
    # args.world_size = torch.distributed.get_world_size()
    args.world_size = torch.cuda.device_count()
    args.logpath = args.exp_name

        
    if args.parallel:
        dist.init_process_group(backend='nccl')
        print('local_rank: ', local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda')
    
    if utils.is_main_process():
        exp_name = args.exp_name
        wandb.init(project='VRP-SAM', name=exp_name, config=args)
        Logger.initialize(args, training=True)  
    utils.fix_randseed(args.seed)

    # Model initialization
    model = VRP_encoder(args, args.backbone, use_original_imgsize=False, pretrained=True)
    if utils.is_main_process():
        Logger.log_params(model)
        

    sam_model = SAM_pred(args.mask_decoder_trainable, lora_train=args.lora, args=args)

    sam_model.to(device)
    model.to(device)

    if torch.cuda.device_count() > 1 and args.parallel:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        sam_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(sam_model)
        find_unused_param_check = False if args.lora and args.encoder_lora_layer and args.decoder_lora_layer else True
        # Device setup
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=find_unused_param_check)
        sam_model = torch.nn.parallel.DistributedDataParallel(sam_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=find_unused_param_check)

    # bp()

    if args.backbone != 'dino_v2':
        if args.parallel and not args.prompt_backbone_trainable:
            for param in model.module.layer0.parameters():
                param.requires_grad = False
            for param in model.module.layer1.parameters():
                param.requires_grad = False
            for param in model.module.layer2.parameters():
                param.requires_grad = False
            for param in model.module.layer3.parameters():
                param.requires_grad = False
            for param in model.module.layer4.parameters():
                param.requires_grad = False

        elif not args.prompt_backbone_trainable:
            for param in model.layer0.parameters():
                param.requires_grad = False
            for param in model.layer1.parameters():
                param.requires_grad = False
            for param in model.layer2.parameters():
                param.requires_grad = False
            for param in model.layer3.parameters():
                param.requires_grad = False
            for param in model.layer4.parameters():
                param.requires_grad = False

        elif args.parallel and args.prompt_backbone_trainable:
            for param in model.module.layer0.parameters():
                param.requires_grad = True
            for param in model.module.layer1.parameters():
                param.requires_grad = True
            for param in model.module.layer2.parameters():
                param.requires_grad = True
            for param in model.module.layer3.parameters():
                param.requires_grad = True
            for param in model.module.layer4.parameters():
                param.requires_grad = True

        elif args.prompt_backbone_trainable:
            for param in model.layer0.parameters():
                param.requires_grad = True
            for param in model.layer1.parameters():
                param.requires_grad = True
            for param in model.layer2.parameters():
                param.requires_grad = True
            for param in model.layer3.parameters():
                param.requires_grad = True
            for param in model.layer4.parameters():
                param.requires_grad = True
        
        # else:
        #     for param in model.

    if not args.dino_lora:
        opt_params = [
            {'params': model.module.transformer_decoder.parameters()},
            {'params': model.module.downsample_query.parameters(), "lr": args.lr},
            {'params': model.module.merge_1.parameters(), "lr": args.lr},
            ] if args.parallel else [
            {'params': model.transformer_decoder.parameters()},
            {'params': model.downsample_query.parameters(), "lr": args.lr},
            {'params': model.merge_1.parameters(), "lr": args.lr},
            ]
    else:
        opt_params = []
    
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    if args.mask_decoder_trainable:
        if args.parallel:
            opt_params.append({'params': sam_model.module.mask_decoder.parameters(), "lr": args.lr})
        else:
            opt_params.append({'params': sam_model.mask_decoder.parameters(), "lr": args.lr})
    if args.lora:
        if args.parallel:
            opt_params.append({'params': filter(lambda p: p.requires_grad, sam_model.module.parameters()), "lr": args.lr})
        else:
            opt_params.append({'params': filter(lambda p: p.requires_grad, sam_model.parameters()), "lr": args.lr})
    if args.dino_lora:
        if args.parallel:
            opt_params.append({'params': filter(lambda p: p.requires_grad, model.module.parameters()), "lr": args.lr})
        else:
            opt_params.append({'params': filter(lambda p: p.requires_grad, model.parameters()), "lr": args.lr})
    
    if args.prompt_backbone_trainable:
        if args.parallel:
            opt_params.append({'params': model.module.layer0.parameters(), "lr": args.lr})
            opt_params.append({'params': model.module.layer1.parameters(), "lr": args.lr})
            opt_params.append({'params': model.module.layer2.parameters(), "lr": args.lr})
            opt_params.append({'params': model.module.layer3.parameters(), "lr": args.lr})
            opt_params.append({'params': model.module.layer4.parameters(), "lr": args.lr})
        else:
            opt_params.append({'params': model.layer0.parameters(), "lr": args.lr})
            opt_params.append({'params': model.layer1.parameters(), "lr": args.lr})
            opt_params.append({'params': model.layer2.parameters(), "lr": args.lr})
            opt_params.append({'params': model.layer3.parameters(), "lr": args.lr})
            opt_params.append({'params': model.layer4.parameters(), "lr": args.lr})


    # bp()

    if args.parallel:
        optimizer = optim.AdamW(opt_params, 
                                lr=args.lr,
                                weight_decay=args.weight_decay, betas=(0.9, 0.999))
        
    else:
        optimizer = optim.AdamW(opt_params, 
                                lr=args.lr,
                                weight_decay=args.weight_decay, betas=(0.9, 0.999))

        
    Evaluator.initialize(args)

    # Dataset initialization
    if args.backbone == 'golden_muscat':
        img_size_for_models = 256
    elif args.img_size_for_models != 512:
        img_size_for_models = args.img_size_for_models
    else:
        img_size_for_models = 256 if args.backbone == 'golden_musca' else 512

    FSSDataset.initialize(img_size=img_size_for_models, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn', args)

    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val', args)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= args.epochs * len(dataloader_trn))
    
    checkpoint_dir = args.logpath
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training 
    checkpoint_path = os.path.join(args.logpath, 'best_model_checkpoint.pt')
    if args.starter:
        checkpoint_path = os.path.join('golden_muscat_query_50_no_mask', 'best_model_checkpoint.pt')

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load the model state dict
        if args.starter:
            if args.parallel:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
        else:
            if args.parallel:
                model.module.load_state_dict(checkpoint['model_state_dict'])
                sam_model.module.load_state_dict(checkpoint['sam_model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
                sam_model.load_state_dict(checkpoint['sam_model_state_dict'])
        
        # Load the optimizer state dict and scheduler state dict
        if not args.starter:
            if not 'fixed' in args.exp_name:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # Load the last epoch and best validation F1 score
                start_epoch = checkpoint['epoch'] + 1
                best_val_f1 = checkpoint['best_val_f1']
                
                # If using EMA or other custom states, load them here
                # if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
                #     ema.load_state_dict(checkpoint['ema_state_dict'])

                print(f"Resuming training from epoch {start_epoch} with best validation F1: {best_val_f1:.4f}")
        else:
            start_epoch = 0
            best_val_f1 = float('-inf')
    else:
        start_epoch = 0
        best_val_f1 = float('-inf')

    # check the param names in optimizer
    # for name, param in optimizer.named_parameters():
    #     print(name, param.requires_grad)


    for epoch in range(start_epoch, args.epochs):

        # trn_loss, trn_miou, trn_fb_iou = train(args, epoch, model, sam_model, dataloader_trn, optimizer, scheduler, training=True)
        # with torch.no_grad():
        #     val_loss, val_miou, val_fb_iou = train(args, epoch, model, sam_model, dataloader_val, optimizer, scheduler, training=False)

        trn_loss, trn_f1, trn_miou = train(args, epoch, model, sam_model, dataloader_trn, optimizer, scheduler, training=True)
        with torch.no_grad():
            val_loss, val_f1, val_miou = train(args, epoch, model, sam_model, dataloader_val, optimizer, scheduler, training=False)

        # Save the best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if utils.is_main_process():
                # Logger.save_model_miou(model, epoch, val_f1, 'prompt_generator')
                # Logger.save_model_miou(sam_model, epoch, val_f1, 'finetuned_sam')
                # # Logger.save_model_miou()

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if args.parallel else model.state_dict(),
                    'sam_model_state_dict': sam_model.module.state_dict() if args.parallel else sam_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_f1': best_val_f1,
                    # 'ema_state_dict': ema.state_dict() if ema else None,  # Uncomment if using EMA
                }, os.path.join(args.logpath, 'best_model_checkpoint.pt'))
                print(f"Checkpoint saved at epoch {epoch} with best validation F1: {val_f1:.4f}")

        if utils.is_main_process():
            Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
            Logger.tbd_writer.add_scalars('data/f1', {'trn_f1': trn_f1, 'val_f1': val_f1}, epoch)
            Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
            # Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            Logger.tbd_writer.flush()

            wandb.log({
                "train_loss_per_epoch": trn_loss,
                "train_miou_per_epoch": trn_miou,
                "train_f1_per_epoch": trn_f1,
                "val_loss_per_epoch": val_loss,
                "val_miou_per_epoch": val_miou,
                "val_f1_per_epoch": val_f1,
            })

    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')