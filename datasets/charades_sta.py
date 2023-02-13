import h5py
import numpy as np

from datasets.base import BaseDataset, build_collate_data, build_collate_video_data, build_collate_query_data

import nltk


class CharadesSTA(BaseDataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        super().__init__(data_path, vocab, args, **kwargs)
        self.collate_fn = build_collate_data(args['max_num_frames'], args['max_num_words'],
                                             args['frame_dim'], args['word_dim'])

    def _load_frame_features(self, vid):
        with h5py.File(self.args['feature_path'], 'r') as fr:
            return np.asarray(fr[vid]).astype(np.float32)

    def collate_data(self, samples):
        return self.collate_fn(samples)



class CharadesSTA_Video(BaseDataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        super().__init__(data_path, vocab, args, **kwargs)
        self.collate_fn = build_collate_video_data(args['max_num_frames'], args['frame_dim'] )
        self._data_modify()

    def _data_modify(self):
        vid_dict = set()
        for line in self.data:
            vid_dict.add(line[0])
        self.data = list(vid_dict)
        
    def _load_frame_features(self, vid):
        with h5py.File(self.args['feature_path'], 'r') as fr:
            return np.asarray(fr[vid]).astype(np.float32)

    def __getitem__(self, index):
        vid = self.data[index]
        frames_feat = self._sample_frame_features(self._load_frame_features(vid))
        return {
            'frames_feat': frames_feat,
            'raw': [vid],
        }

    def collate_data(self, samples):
        return self.collate_fn(samples)

class CharadesSTA_Query(BaseDataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        super().__init__(data_path, vocab, args, **kwargs)
        self.collate_fn = build_collate_query_data(args['max_num_words'], args['word_dim'])

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
        return {
            'words_feat': words_feat,
            'words_id': words_id,
            'weights': weights,
            'raw': [vid, duration, timestamps, sentence],
            'glance': glance
        }

    def collate_data(self, samples):
        return self.collate_fn(samples)