import collections
import json
import logging
import os
import pdb
from turtle import width

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

from models.loss import  mlm_loss, ivc_loss, cma_loss, div_loss, cvc_loss
from utils import TimeMeter, AverageMeter


def info(msg):
    print(msg)
    logging.info(msg)


class MainRunner:
    def __init__(self, args):
        self.args = args
        self._build_dataset()

        self.args['model']['config']['vocab_size'] = self.train_set.vocab_size
        self.args['model']['config']['max_epoch'] = self.args['train']['max_num_epochs']

        self.flag = True

        self._build_model()

        if 'train' in args:
            self._build_optimizer()
            self.num_updates = 0
            self.rank_num_updates = 0



    def train(self):
        best_T2M_results = None
        best_T2V_results = None
        info('Eval Initially at Epoch {}'.format(0))
        # best_T2M_results = self._eval_ranking_model(epoch=0)



        for epoch in range(1, self.args['train']['max_num_epochs']+1):
            info('Start Epoch {}'.format(epoch))
            if epoch >= self.args['train']['rank_training_start_epochs'] and self.args['train']['rank_training_start_epochs'] != -1: 
                info('Training Metric Encoder at Epoch {}'.format(epoch))
                self.model_saved_path = self.args['train']['model_saved_path']
                os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
                save_path = os.path.join(self.model_saved_path, 'model-{}.pt'.format(epoch))
                txt_acc, vid_acc = self._train_ranking_model(epoch=epoch)
                self._save_model(save_path)
                results = self._eval_ranking_model(epoch=epoch)
                if best_T2M_results is None or results['R@1,mIoU'].avg > best_T2M_results['R@1,mIoU'].avg:
                    best_T2M_results = results
                    os.system('cp %s %s'%(save_path, os.path.join(self.model_saved_path, 'model-t2m-best.pt')))
                    info('Best results of T2M have been updated.')
            else:
                self.model_saved_path = self.args['train']['model_saved_path']
                os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
                save_path = os.path.join(self.model_saved_path, 'model-{}.pt'.format(epoch))
                self._train_model(epoch=epoch)

        msg = '|'.join([' {} {:.4f} '.format(k, v.avg) for k, v in best_T2M_results.items()])
        info('Best T2M results:')
        info('|'+msg+'|')

    def _train_model(self, epoch, **kwargs):
        self.model.train()
        self.rank_model.eval()

        def print_log():
            msg = 'Epoch {}, Batch {}, lr = {:.5f}, {:.3f} seconds/batch'.format(epoch, bid, curr_lr, 1.0 / time_meter.avg)
            for k, v in loss_meter.items():
                msg += '{} = {:.4f}, '.format(k, v.avg)
                v.reset()
            info(msg)

        display_n_batches, bid = 200, 0
        time_meter = TimeMeter()
        loss_meter = collections.defaultdict(lambda: AverageMeter())
        total_txt_acc = 0
        total_vid_acc = 0

        for bid, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training epoch {}".format(epoch)):
            self.optimizer.zero_grad()
            net_input = move_to_cuda(batch['net_input'])
            bsz = net_input["frames_feat"].shape[0]
            output = self.model.forward_train(epoch=epoch, **net_input)
            num_props = self.model.num_props
            
            loss = 0
            loss_dict = {}

            ## Semantic Loss
            rec_loss, rec_loss_dict, txt_acc, vid_acc = mlm_loss(**output, num_props=num_props, use_min=True,**self.args['loss'])
            loss_dict.update(rec_loss_dict)
            loss = loss + rec_loss

            # Concept Loss 
            conc_loss, conc_loss_dict = cma_loss(**output, **self.args['loss'], num_props=num_props, num_concepts=1)
            loss_dict.update(conc_loss_dict)
            loss = loss + conc_loss

            # cross-video contrastive Loss
            cvcl_loss, cvcl_loss_dict = cvc_loss(**output, num_props=num_props, use_div_loss=True, **self.args['loss'])
            loss_dict.update(cvcl_loss_dict)
            loss = loss + cvcl_loss

            
            # Diversity Loss
            diver_loss, diver_loss_dict = div_loss(**output, **self.args['loss'], num_props=num_props, num_concepts=self.model.num_concepts)
            loss_dict.update(diver_loss_dict)
            loss = loss + diver_loss

            # Intra-video Loss
            rnk_loss, rnk_loss_dict = ivc_loss(**output, num_props=num_props, use_div_loss=True, **self.args['loss'])
            loss_dict.update(rnk_loss_dict)
            loss = loss + rnk_loss


            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()
            self.num_updates += 1
            curr_lr = self.lr_scheduler.step_update(self.num_updates)

            time_meter.update() 

            total_txt_acc += txt_acc
            total_vid_acc += vid_acc


            for k, v in loss_dict.items():
                loss_meter[k].update(v)

            if bid % display_n_batches == 0:
                print_log()

        if bid % display_n_batches != 0:
            print_log()
        return total_txt_acc/len(self.train_loader), total_vid_acc/len(self.train_loader)


    def _train_ranking_model(self, epoch, **kwargs):
        self.model.eval()
        self.rank_model.train()

        def print_log():
            msg = 'Epoch {}, Batch {}, lr = {:.5f}, {:.3f} seconds/batch'.format(epoch, bid, curr_lr, 1.0 / time_meter.avg)
            for k, v in loss_meter.items():
                msg += '{} = {:.4f}, '.format(k, v.avg)
                v.reset()
            info(msg)

        display_n_batches, bid = 200, 0
        time_meter = TimeMeter()
        loss_meter = collections.defaultdict(lambda: AverageMeter())
        total_txt_acc = 0
        total_vid_acc = 0

        for bid, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training epoch {}".format(epoch)):
            self.rank_optimizer.zero_grad()
            net_input = move_to_cuda(batch['net_input'])

            durations = np.asarray([i[1] for i in batch['raw']])
            gt = np.asarray([i[2] for i in batch['raw']])

            net_input = move_to_cuda(batch['net_input'])

            bsz = len(durations)
            num_props = self.model.num_props
            k = min(num_props, 5)


            vid_output = self.model.forward_vid(epoch=epoch, **net_input)
            txt_output = self.model.forward_txt(epoch=epoch, **net_input)
                    

            width = vid_output['width'].view(bsz, num_props)
            center = vid_output['center'].view(bsz, num_props)

            output = self.rank_model.forward_rank_train(epoch=epoch, center=center, width=width, **net_input)
            
            loss = 0
            loss_dict = {}

            ## Semantic Loss
            rec_loss, rec_loss_dict, txt_acc, vid_acc = mlm_loss(**output, num_props=num_props, use_min=True,**self.args['loss'])
            loss_dict.update(rec_loss_dict)
            loss = loss + rec_loss

            # Concept Loss 
            conc_loss, conc_loss_dict = cma_loss(**output, **self.args['loss'], num_props=num_props, num_concepts=1)
            loss_dict.update(conc_loss_dict)
            loss = loss + conc_loss

            # cross-video contrastive Loss
            cvcl_loss, cvcl_loss_dict = cvc_loss(**output, num_props=num_props, use_div_loss=True, **self.args['loss'])
            loss_dict.update(cvcl_loss_dict)
            loss = loss + cvcl_loss

            
            # Intra-video Loss
            rnk_loss, rnk_loss_dict = ivc_loss(**output, num_props=num_props, use_div_loss=True, **self.args['loss'])
            loss_dict.update(rnk_loss_dict)
            loss = loss + rnk_loss


            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.rank_optimizer.step()
            self.rank_num_updates += 1
            curr_lr = self.rank_lr_scheduler.step_update(self.rank_num_updates)

            time_meter.update() 

            total_txt_acc += txt_acc
            total_vid_acc += vid_acc


            for k, v in loss_dict.items():
                loss_meter[k].update(v)

            if bid % display_n_batches == 0:
                print_log()

        if bid % display_n_batches != 0:
            print_log()
        return total_txt_acc/len(self.train_loader), total_vid_acc/len(self.train_loader)


    def _eval_model(self, save=None, epoch=0):
        self.model.eval()
        self.rank_model.eval()
        with torch.no_grad():
            metrics_logger = collections.defaultdict(lambda: AverageMeter())

            with torch.no_grad():

                for bid, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc="T2M Eval epoch {}".format(epoch)):
                    durations = np.asarray([i[1] for i in batch['raw']])
                    gt = np.asarray([i[2] for i in batch['raw']])

                    net_input = move_to_cuda(batch['net_input'])

                    bsz = len(durations)
                    num_props = self.model.num_props
                    k = min(num_props, 5)


                    vid_output = self.model.forward_vid(epoch=epoch, **net_input)
                    txt_output = self.model.forward_txt(epoch=epoch, **net_input)
                    
                    pos_vid_concept = vid_output['pos_vid_concepts']
                    txt_concepts = txt_output['txt_concepts']

                    num_concept = pos_vid_concept.shape[1]
                    proposal = F.normalize(pos_vid_concept,dim=-1, p=2).reshape(bsz*num_props, num_concept, -1)
                    txt_conc = F.normalize(txt_concepts,dim=-1, p=2).unsqueeze(1).expand(bsz, num_props, num_concept, -1).reshape(bsz*num_props, num_concept, -1)

                    pos_samilarity = torch.diagonal(torch.bmm(proposal, txt_conc.transpose(1, 2)), dim1=-2, dim2=-1).sum(dim=-1)
                    pos_samilarity = pos_samilarity.reshape(bsz, num_props)
                    idx = pos_samilarity.argsort(dim=-1, descending=True)

                    width = vid_output['width'].view(bsz, num_props).gather(index=idx, dim=-1)
                    center = vid_output['center'].view(bsz, num_props).gather(index=idx, dim=-1)

                    selected_props = torch.stack([torch.clamp(center-width/2, min=0), torch.clamp(center+width/2, max=1)], dim=-1)

                    selected_props = selected_props.cpu().numpy()
                    gt = gt / durations[:, np.newaxis]


                    res = top_1_metric(selected_props[:, 0], gt)
                    
                    for key, v in res.items():
                        metrics_logger['R@1,'+key].update(v, bsz)
                    
                    res = top_n_metric(selected_props[:, :k].transpose(1, 0, 2), gt)
                    for key, v in res.items():
                        metrics_logger['R@%d,'%(k)+key].update(v, bsz)

            msg = '|'.join([' {} {:.4f} '.format(k, v.avg) for k, v in metrics_logger.items()])
            info('|'+msg+'|')
            return metrics_logger

    def _eval_ranking_model(self, save=None, epoch=0):
        self.model.eval()
        self.rank_model.eval()
        with torch.no_grad():
            metrics_logger = collections.defaultdict(lambda: AverageMeter())

            with torch.no_grad():
                pred_res_rec = []
                
                for bid, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc="T2M Eval epoch {}".format(epoch)):
                    durations = np.asarray([i[1] for i in batch['raw']])
                    gt = np.asarray([i[2] for i in batch['raw']])

                    net_input = move_to_cuda(batch['net_input'])

                    bsz = len(durations)
                    num_props = self.model.num_props
                    k = min(num_props, 5)


                    vid_output = self.model.forward_vid(epoch=epoch, **net_input)
                    txt_output = self.model.forward_txt(epoch=epoch, **net_input)

                    width = vid_output['width'].view(bsz, num_props)
                    center = vid_output['center'].view(bsz, num_props)

                    vid_output = self.rank_model.forward_vid_rank(epoch=epoch, center=center, width=width, **net_input)
                    txt_output = self.rank_model.forward_txt(epoch=epoch, **net_input)

                    pos_vid_concept = vid_output['pos_vid_concepts']
                    txt_concepts = txt_output['txt_concepts']

                    num_concept = pos_vid_concept.shape[1]
                    proposal = F.normalize(pos_vid_concept,dim=-1, p=2).reshape(bsz*num_props, num_concept, -1)
                    txt_conc = F.normalize(txt_concepts,dim=-1, p=2).unsqueeze(1).expand(bsz, num_props, num_concept, -1).reshape(bsz*num_props, num_concept, -1)

                    pos_samilarity = torch.diagonal(torch.bmm(proposal, txt_conc.transpose(1, 2)), dim1=-2, dim2=-1).sum(dim=-1)
                    pos_samilarity = pos_samilarity.reshape(bsz, num_props)
                    idx = pos_samilarity.argsort(dim=-1, descending=True)

                    width = vid_output['width'].view(bsz, num_props).gather(index=idx, dim=-1)
                    center = vid_output['center'].view(bsz, num_props).gather(index=idx, dim=-1)

                    selected_props = torch.stack([torch.clamp(center-width/2, min=0), torch.clamp(center+width/2, max=1)], dim=-1)

                    selected_props = selected_props.cpu().numpy()
                    gt = gt / durations[:, np.newaxis]


                    res = top_1_metric(selected_props[:, 0], gt)
                    
                    for key, v in res.items():
                        metrics_logger['R@1,'+key].update(v, bsz)
                    
                    res = top_n_metric(selected_props[:, :k].transpose(1, 0, 2), gt)
                    for key, v in res.items():
                        metrics_logger['R@%d,'%(k)+key].update(v, bsz)

            

            msg = '|'.join([' {} {:.4f} '.format(k, v.avg) for k, v in metrics_logger.items()])
            info('|'+msg+'|')
            return metrics_logger



    def _build_dataset(self):
        import datasets as da
        import pickle
        from torch.utils.data import DataLoader
        args = self.args['dataset']
        cls = getattr(da, args['dataset'], None)
        with open(args['vocab_path'], 'rb') as fp:
            vocab = pickle.load(fp)
        self.train_set = cls(data_path=args['train_data'], vocab=vocab, args=args, is_training=True, split='train')
        self.test_set = cls(data_path=args['test_data'], vocab=vocab, args=args, split='test')
        self.val_set = cls(data_path=args['val_data'], vocab=vocab, args=args, split='val') if args['val_data'] else None

        cls = getattr(da, args['dataset']+'_Video', None)
        self.video_set = cls(data_path=args['test_data'], vocab=vocab, args=args, split='test')
        cls = getattr(da, args['dataset']+'_Query', None)
        self.query_set = cls(data_path=args['test_data'], vocab=vocab, args=args, split='test')

        info('train: {} samples, test: {} samples, {} videos, {} queries'.format(len(self.train_set), len(self.test_set), len(self.video_set), len(self.query_set)))
        batch_size = self.args['train']['batch_size']

        def worker_init_fn(worker_id):
            def set_seed(seed):
                import random
                import numpy as np
                import torch

                random.seed(seed)
                np.random.seed(seed + 1)
                torch.manual_seed(seed + 3)
                torch.cuda.manual_seed(seed + 4)
                torch.cuda.manual_seed_all(seed + 4)

            set_seed(8 + worker_id)

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True,
                                       collate_fn=self.train_set.collate_data, num_workers=4,
                                       worker_init_fn=worker_init_fn)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False,
                                      collate_fn=self.test_set.collate_data,
                                      num_workers=4)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False,
                                     collate_fn=self.val_set.collate_data,
                                     num_workers=4) if args['val_data'] else None

        self.test_vid_loader = DataLoader(self.video_set, batch_size=batch_size, shuffle=False,
                                      collate_fn=self.video_set.collate_data,
                                      num_workers=4)
        self.test_txt_loader = DataLoader(self.query_set, batch_size=batch_size, shuffle=False,
                                      collate_fn=self.query_set.collate_data,
                                      num_workers=4)

    def _build_model(self):
        model_config = self.args['model']
        import models

        self.model = getattr(models, model_config['name'], None)(model_config['config'])
        self.model = self.model.cuda()
        self.rank_model = getattr(models, model_config['name'], None)(model_config['config'])
        self.rank_model = self.rank_model.cuda()
        print(self.model)
        

    def _build_optimizer(self):
        from optimizers import AdamOptimizer
        from optimizers.lr_schedulers import InverseSquareRootSchedule
        
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        args = self.args['train']["optimizer"]
        self.optimizer = AdamOptimizer(args, parameters)
        self.lr_scheduler = InverseSquareRootSchedule(args, self.optimizer)

        rank_parameters = list(filter(lambda p: p.requires_grad, self.rank_model.parameters()))
        args = self.args['train']["optimizer"]
        self.rank_optimizer = AdamOptimizer(args, rank_parameters)
        self.rank_lr_scheduler = InverseSquareRootSchedule(args, self.rank_optimizer)

    def _save_model(self, path):
        state_dict = {
            'config': self.args,
            'num_updates': self.num_updates,
            'model_parameters': self.model.state_dict(),
            'rank_num_updates': self.rank_num_updates,
            'rank_model_parameters': self.rank_model.state_dict(),
        }
        torch.save(state_dict, path)
        info('save model to {}, num_updates {}, rank_num_updates {}.'.format(path, self.num_updates, self.rank_num_updates))

        

    def _load_model(self, path):
        state_dict = torch.load(path)
        self.num_updates = state_dict['num_updates']
        self.lr_scheduler.step_update(self.num_updates)
        parameters = state_dict['model_parameters']
        self.model.load_state_dict(parameters)

        self.rank_num_updates = state_dict['rank_num_updates']
        self.rank_lr_scheduler.step_update(self.rank_num_updates)
        rank_parameters = state_dict['rank_model_parameters']
        self.rank_model.load_state_dict(rank_parameters)

        info('load model from {}, num_updates {}, rank_num_updates {}.'.format(path, self.num_updates, self.rank_num_updates))


def calculate_IoU_batch(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


# [nb, 2], [nb, 2]
def top_n_metric(preds, label):
    result = {}
    bsz = preds[0].shape[0]
    top_iou = []
    for pred in preds:
        iou = calculate_IoU_batch((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
        top_iou.append(iou)
    iou = np.max(np.stack(top_iou, 1), 1)
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result

def top_1_metric(pred, label):
    result = {}
    bsz = pred.shape[0]
    iou = calculate_IoU_batch((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)
