r""" Logging during training/testing """
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch
from .utils import is_main_process, save_on_master, reduce_metric

class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, bsz=2):
        self.bsz = bsz
        self.tp_buf = torch.zeros(bsz).float().cuda()
        self.fp_buf = torch.zeros(bsz).float().cuda()
        self.fn_buf = torch.zeros(bsz).float().cuda()
        self.loss_buf = []

    def update(self, tp, fp, fn, loss):
        batch_tp = tp.float().sum(0)
        batch_fp = fp.float().sum(0)
        batch_fn = fn.float().sum(0)
        
        self.tp_buf += batch_tp
        self.fp_buf += batch_fp
        self.fn_buf += batch_fn
        
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)

        # Compute and print F1 score for the current batch
        precision = batch_tp / torch.max(batch_tp + batch_fp, torch.tensor(1.0).cuda())
        recall = batch_tp / torch.max(batch_tp + batch_fn, torch.tensor(1.0).cuda())
        f1 = 2 * (precision * recall) / torch.max(precision + recall, torch.tensor(1.0).cuda())
        
        batch_f1 = f1.mean() * 100

        print(f'Current Batch F1: {batch_f1:.2f}')

    def compute_f1(self):
        precision = self.tp_buf / torch.max(self.tp_buf + self.fp_buf, torch.tensor(1.0).cuda())
        recall = self.tp_buf / torch.max(self.tp_buf + self.fn_buf, torch.tensor(1.0).cuda())
        f1 = 2 * (precision * recall) / torch.max(precision + recall, torch.tensor(1.0).cuda())

        mean_f1 = f1.mean() * 100

        return mean_f1

    def write_result(self, split, epoch):
        self.tp_buf, self.fp_buf, self.fn_buf = self.reduce_metrics([self.tp_buf, self.fp_buf, self.fn_buf], False)
        f1 = self.compute_f1()

        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch if epoch != -1 else ''
        if epoch != -1:
            loss_buf = torch.stack(self.loss_buf)
            loss_buf = self.reduce_metrics([loss_buf])[0]
            msg += 'Avg L: %6.5f  ' % loss_buf.mean()
        msg += 'F1: %5.2f   ' % f1
        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx + 1, datalen)
            precision = self.tp_buf / torch.max(self.tp_buf + self.fp_buf, torch.tensor(1.0).cuda())
            recall = self.tp_buf / torch.max(self.tp_buf + self.fn_buf, torch.tensor(1.0).cuda())
            f1 = 2 * (precision * recall) / torch.max(precision + recall, torch.tensor(1.0).cuda())
            if epoch != -1:
                loss_buf = torch.stack(self.loss_buf)
                msg += 'L: %6.5f  ' % loss_buf[-1]
                msg += 'Avg L: %6.5f  ' % loss_buf.mean()
            msg += 'F1: %5.2f' % f1.mean().item()
            Logger.info(msg)

    def reduce_metrics(self, metrics, average=True):
        reduced_metrics = []
        for m in metrics:
            reduce_metric(m, average)
            reduced_metrics.append(m)
        return reduced_metrics


# class AverageMeter:
#     r""" Stores loss, evaluation results """
#     def __init__(self, bsz=2):
#         self.intersection_buf = torch.zeros(bsz).float().cuda()
#         self.union_buf = torch.zeros(bsz).float().cuda()
#         self.ones = torch.ones_like(self.union_buf)
#         self.loss_buf = []

#     def update(self, inter_b, union_b, loss):
#         self.intersection_buf += inter_b.float().sum(0)
#         self.union_buf += union_b.float().sum(0)
#         if loss is None:
#             loss = torch.tensor(0.0)
#         self.loss_buf.append(loss)

#     def compute_iou(self):
#         iou = self.intersection_buf.float() / torch.max(self.union_buf, self.ones)
#         miou = iou.mean() * 100

#         fb_iou = (self.intersection_buf.sum() / self.union_buf.sum()) * 100

#         return miou, fb_iou

#     def write_result(self, split, epoch):
#         self.intersection_buf, self.union_buf = self.reduce_metrics([self.intersection_buf, self.union_buf], False)
#         iou, fb_iou = self.compute_iou()

#         msg = '\n*** %s ' % split
#         msg += '[@Epoch %02d] ' % epoch if epoch != -1 else ''
#         if epoch != -1:
#             loss_buf = torch.stack(self.loss_buf)
#             loss_buf = self.reduce_metrics([loss_buf])[0]
#             msg += 'Avg L: %6.5f  ' % loss_buf.mean()
#         msg += 'mIoU: %5.2f   ' % iou
#         msg += 'FB-IoU: %5.2f   ' % fb_iou

#         msg += '***\n'
#         Logger.info(msg)

#     def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
#         if batch_idx % write_batch_idx == 0:
#             msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
#             msg += '[Batch: %04d/%04d] ' % (batch_idx + 1, datalen)
#             iou, fb_iou = self.compute_iou()
#             if epoch != -1:
#                 loss_buf = torch.stack(self.loss_buf)
#                 msg += 'L: %6.5f  ' % loss_buf[-1]
#                 msg += 'Avg L: %6.5f  ' % loss_buf.mean()
#             msg += 'mIoU: %5.2f  |  ' % iou
#             msg += 'FB-IoU: %5.2f' % fb_iou
#             Logger.info(msg)

#     def reduce_metrics(self, metrics, average=True):
#         reduced_metrics = []
#         for m in metrics:
#             reduce_metric(m, average)
#             reduced_metrics.append(m)
#         return reduced_metrics

# class AverageMeter:
#     r""" Stores loss, evaluation results """
#     def __init__(self, dataset):
#         self.benchmark = dataset.benchmark
#         self.class_ids_interest = dataset.class_ids
#         self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()

#         if self.benchmark == 'pascal':
#             self.nclass = 20
#         elif self.benchmark == 'coco':
#             self.nclass = 80
#         elif self.benchmark == 'fss':
#             self.nclass = 1000
#         elif self.benchmark == 'map':
#             self.nclass = 2

#         self.intersection_buf = torch.zeros([2, self.nclass]).float().cuda()
#         self.union_buf = torch.zeros([2, self.nclass]).float().cuda()
#         self.ones = torch.ones_like(self.union_buf)
#         self.loss_buf = []

#     def update(self, inter_b, union_b, class_id, loss):
#         self.intersection_buf.index_add_(1, class_id, inter_b.float())
#         self.union_buf.index_add_(1, class_id, union_b.float())
#         if loss is None:
#             loss = torch.tensor(0.0)
#         self.loss_buf.append(loss)

#     def compute_iou(self):
#         iou = self.intersection_buf.float() / \
#               torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
#         iou = iou.index_select(1, self.class_ids_interest)
#         miou = iou[1].mean() * 100

#         fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
#                   self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100

#         return miou, fb_iou

#     def write_result(self, split, epoch):
#         self.intersection_buf, self.union_buf = self.reduce_metrics([self.intersection_buf, self.union_buf], False)
#         iou, fb_iou = self.compute_iou()

#         # loss_buf = torch.stack(self.loss_buf)
#         msg = '\n*** %s ' % split
#         msg += '[@Epoch %02d] ' % epoch if epoch != -1 else ''
#         if epoch != -1:
#             loss_buf = torch.stack(self.loss_buf)
#             loss_buf = self.reduce_metrics([loss_buf])[0]
#             msg += 'Avg L: %6.5f  ' % loss_buf.mean()
#         msg += 'mIoU: %5.2f   ' % iou
#         msg += 'FB-IoU: %5.2f   ' % fb_iou

#         msg += '***\n'
#         Logger.info(msg)

#     def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
#         if batch_idx % write_batch_idx == 0:
#             msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
#             msg += '[Batch: %04d/%04d] ' % (batch_idx + 1, datalen)
#             iou, fb_iou = self.compute_iou()
#             if epoch != -1:
#                 loss_buf = torch.stack(self.loss_buf)
#                 msg += 'L: %6.5f  ' % loss_buf[-1]
#                 msg += 'Avg L: %6.5f  ' % loss_buf.mean()
#             msg += 'mIoU: %5.2f  |  ' % iou
#             msg += 'FB-IoU: %5.2f' % fb_iou
#             Logger.info(msg)

#     def reduce_metrics(self, metrics, average=True):
#         reduced_metrics = []
#         for m in metrics:
#             reduce_metric(m, average)
#             reduced_metrics.append(m)
#         return reduced_metrics


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training):
        # logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logtime = datetime.datetime.now().strftime('_%m%d_%H%M%S')
        if training:
            logpath = args.logpath if args.logpath else logtime
        else:
            if args.load:
                logpath = '_TEST_' + args.load.split('/')[-2].split('.')[0] + logtime
            else:
                logpath = '_TEST_' + logtime
        

        # cls.logpath = os.path.join('logs', logpath + '.log')
        cls.logpath = os.path.join('logs', logpath)
        cls.benchmark = args.benchmark

        log_dir = os.path.dirname(cls.logpath)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # make log dir
        if not os.path.exists(cls.logpath):
            os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        logging.info('\n:=========== Few-shot Seg. with VRP-SAM ===========')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':==================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_miou(cls, model, epoch, val_miou, type='prompt_generator'):
        if type == 'prompt_generator':
            torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model_prompt_generator.pt'))
        elif type == 'prompt_generator_backbone':
            torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model_prompt_generator_backbone.pt'))
        elif type == 'finetuned_sam':
            torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model_finetuned_sam.pt'))
        else:
            assert False, 'Invalid model type.'
        # torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. mIoU: %5.2f.\n' % (epoch, val_miou))

    @classmethod
    def log_params(cls, model):
        total_param = 0
        learnable_param = 0

        for v in model.parameters():
            n_param = v.numel()
            total_param += n_param
            if v.requires_grad:
                learnable_param += n_param

        Logger.info('Total # param.: %d' % total_param)
        Logger.info('Learnable # param.: %d' % learnable_param)



    # def log_params(cls, model):
    #     backbone_param = 0
    #     learner_param = 0
    #     for k in model.state_dict().keys():
    #         n_param = model.state_dict()[k].view(-1).size(0)
    #         if [i for i in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4'] if i in k]:
    #             backbone_param += n_param
    #         else:
    #             learner_param += n_param
    #     Logger.info('Backbone # param.: %d' % backbone_param)
    #     Logger.info('Learnable # param.: %d' % learner_param)
    #     Logger.info('Total # param.: %d' % (backbone_param + learner_param))

