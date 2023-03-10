import random
from tkinter.messagebox import NO
import numpy as np
import torch
from torch.utils.data import Dataset
import pdb

from utils import load_json
import nltk

class BaseDataset(Dataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        self.vocab = vocab
        self.args = args
        self.data = load_json(data_path)
        
        self.ori_data = self.data
        self.max_num_frames = args['max_num_frames']
        self.max_num_words = args['max_num_words']

        self.keep_vocab = dict()
        for w, _ in vocab['counter'].most_common(args['vocab_size']):
            self.keep_vocab[w] = self.vocab_size

    def _load_frame_features(self, vid):
        raise NotImplementedError

    def _sample_frame_features(self, frames_feat):
        num_clips = self.num_clips
        keep_idx = np.arange(0, num_clips + 1) / num_clips * len(frames_feat)
        keep_idx = np.round(keep_idx).astype(np.int64)
        keep_idx[keep_idx >= len(frames_feat)] = len(frames_feat) - 1
        frames_feat1 = []
        for j in range(num_clips):
            s, e = keep_idx[j], keep_idx[j + 1]
            assert s <= e
            if s == e:
                frames_feat1.append(frames_feat[s])
            else:
                frames_feat1.append(frames_feat[s:e].mean(axis=0))
        return np.stack(frames_feat1, 0)

    @property
    def num_clips(self):
        return self.max_num_frames

    @property
    def vocab_size(self):
        return len(self.keep_vocab) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            vid, duration, timestamps, sentence, weak_labels = self.data[index]
        except:
            vid, duration, timestamps, sentence = self.data[index]
            weak_labels = None
        
        duration = float(duration)
        if weak_labels != None:
            if isinstance(weak_labels, list):
                glance = float(weak_labels[0]/duration)
            else:
                glance = float(weak_labels/duration)
        else:
            glance = None

        weights = [] # Probabilities to be masked
        words = []
        for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(sentence)):
            word = word.lower()
            if word in self.keep_vocab:
                if 'NN' in tag:
                    weights.append(2)
                elif 'VB' in tag:
                    weights.append(2)
                elif 'JJ' in tag or 'RB' in tag:
                    weights.append(2)
                else:
                    weights.append(1)
                words.append(word)
        words_id = [self.keep_vocab[w] for w in words]
        words_feat = [self.vocab['id2vec'][self.vocab['w2id'][words[0]]].astype(np.float32)] # placeholder for the start token
        words_feat.extend([self.vocab['id2vec'][self.vocab['w2id'][w]].astype(np.float32) for w in words])
        words_feat.append(self.vocab['id2vec'][self.vocab['w2id'][words[0]]].astype(np.float32)) # placeholder for the pred_vec token
        frames_feat = self._sample_frame_features(self._load_frame_features(vid))
        return {
            'frames_feat': frames_feat,
            'words_feat': words_feat,
            'words_id': words_id,
            'weights': weights,
            'raw': [vid, duration, timestamps, sentence],
            'glance': glance
        }


def build_collate_data(max_num_frames, max_num_words, frame_dim, word_dim):
    def collate_data(samples):
        bsz = len(samples)
        batch = {
            'raw': [sample['raw'] for sample in samples],
        }

        frames_len = []
        words_len = []

        for i, sample in enumerate(samples):
            frames_len.append(min(len(sample['frames_feat']), max_num_frames))
            words_len.append(min(len(sample['words_id']), max_num_words))

        frames_feat = np.zeros([bsz, max_num_frames, frame_dim]).astype(np.float32)
        words_feat = np.zeros([bsz, max(words_len) + 2, word_dim]).astype(np.float32)
        words_id = np.zeros([bsz, max(words_len)]).astype(np.int64)
        weights = np.zeros([bsz, max(words_len)]).astype(np.float32)
        glance = np.zeros([bsz]).astype(np.float32)

            

        for i, sample in enumerate(samples):
            frames_feat[i, :len(sample['frames_feat'])] = sample['frames_feat']
            keep = min(len(sample['words_feat']), words_feat.shape[1])
            words_feat[i, :keep] = sample['words_feat'][:keep]
            keep = min(len(sample['words_id']), words_id.shape[1])
            words_id[i, :keep] = sample['words_id'][:keep]
            keep = min(len(sample['weights']), weights.shape[1])
            tmp = np.exp(sample['weights'][:keep])
            weights[i, :keep] = tmp / np.sum(tmp)
            glance[i] = sample['glance']
        
        batch.update({
                'net_input': {
                    'frames_feat': torch.from_numpy(frames_feat),
                    'frames_len': torch.from_numpy(np.asarray(frames_len)),
                    'words_feat': torch.from_numpy(words_feat),
                    'words_id': torch.from_numpy(words_id),
                    'weights': torch.from_numpy(weights),
                    'words_len': torch.from_numpy(np.asarray(words_len)),
                    'glance': torch.from_numpy(glance),
                }
            })
        return batch

    return collate_data


def build_collate_video_data(max_num_frames, frame_dim):
    def collate_data(samples):
        bsz = len(samples)
        batch = {
            'raw': [sample['raw'] for sample in samples],
        }

        frames_len = []

        for i, sample in enumerate(samples):
            frames_len.append(min(len(sample['frames_feat']), max_num_frames))

        frames_feat = np.zeros([bsz, max_num_frames, frame_dim]).astype(np.float32)

            

        for i, sample in enumerate(samples):
            frames_feat[i, :len(sample['frames_feat'])] = sample['frames_feat']
        
        batch.update({
                'net_input': {
                    'frames_feat': torch.from_numpy(frames_feat),
                    'frames_len': torch.from_numpy(np.asarray(frames_len)),
                }
            })
        return batch

    return collate_data



def build_collate_query_data(max_num_words, word_dim):
    def collate_data(samples):
        bsz = len(samples)
        batch = {
            'raw': [sample['raw'] for sample in samples],
        }

        words_len = []

        for i, sample in enumerate(samples):
            words_len.append(min(len(sample['words_id']), max_num_words))

        words_feat = np.zeros([bsz, max(words_len) + 2, word_dim]).astype(np.float32)
        words_id = np.zeros([bsz, max(words_len)]).astype(np.int64)
        weights = np.zeros([bsz, max(words_len)]).astype(np.float32)
        glance = np.zeros([bsz]).astype(np.float32)

        for i, sample in enumerate(samples):
            keep = min(len(sample['words_feat']), words_feat.shape[1])
            words_feat[i, :keep] = sample['words_feat'][:keep]
            keep = min(len(sample['words_id']), words_id.shape[1])
            words_id[i, :keep] = sample['words_id'][:keep]
            keep = min(len(sample['weights']), weights.shape[1])
            tmp = np.exp(sample['weights'][:keep])
            weights[i, :keep] = tmp / np.sum(tmp)
            glance[i] = sample['glance']
        
        batch.update({
                'net_input': {
                    'words_feat': torch.from_numpy(words_feat),
                    'words_id': torch.from_numpy(words_id),
                    'weights': torch.from_numpy(weights),
                    'words_len': torch.from_numpy(np.asarray(words_len)),
                    'glance': torch.from_numpy(glance),
                }
            })
        return batch

    return collate_data