from typing import Text, Dict
import os
import json
from dst.dataset.dataset import Ontology, Dataset
from vocab import Vocab
from tqdm import tqdm
from embeddings import GloveEmbedding, KazumaCharEmbedding
from pprint import pformat
from dst.net import GladModel
import logging
from pprint import pprint

class DialogueStateTrackingLearner(object):
    def __init__(self, configure):
        self.configure = configure
        self.model_path = self.configure['GENERAL']['model_path']

        args = self.configure['ALGORITHM']['hyperparameter']
        args['dout'] = os.path.join(args['dexp'], args['model'], args['nick'])
        self.train_args = args
        self.training = True
        self.ontology: Ontology = Ontology()
        self.vocab: Vocab = Vocab()
        self.dataset: Dict = {}
        self.embeddings = None

        if not self.training:
            self.load(self.model_path)

        else:
            if 'raw' in self.configure['DATA']:
                data_paths = self.configure['TRAINER']['raw']
                self.process_raw_dataset(models_path=self.configure['GENERAL']['model_path'],
                                         **data_paths)
            else:
                data_paths = self.configure['DATA']['ann']
                self.process_ann_dataset(models_path=self.configure['GENERAL']['model_path'],
                                         **data_paths)
        self.glad_model = GladModel(
            args=self.train_args,
            ontology=self.ontology,
            vocab=self.vocab)
        self.glad_model.save_config()
        self.glad_model.load_emb(self.embeddings)

    def train(self):
        """

        Args:
            **kwargs:

        Returns:

        """
        self.glad_model = self.glad_model.to(self.glad_model.device)
        print('Starting train')
        self.glad_model.run_train(
            self.dataset['train'],
            self.dataset['dev'],
            self.train_args
            )

        self.glad_model = self.glad_model.to(self.glad_model.device)
        logging.info('Running dev evaluation')
        dev_out = self.glad_model.run_eval(self.dataset['dev'], self.train_args)
        pprint(dev_out)
        exit()

    def load(self, models_path):
        """

        Args:
            models_path:

        Returns:

        """
        if os.path.isfile(os.path.join(models_path, 'ontology.json')):
            with open(os.path.join(models_path, 'ontology.json')) as f:
                self.ontology = Ontology.from_dict(json.load(f))

        if os.path.isfile(os.path.join(models_path, 'vocab.json')):
            with open(os.path.join(models_path, 'vocab.json')) as f:
                self.vocab = Vocab.from_dict(json.load(f))

        if os.path.isfile(os.path.join(models_path, 'emb.json')):
            with open(os.path.join(models_path, 'emb.json')) as f:
                self.embeddings = json.load(f)

    def process_raw_dataset(
            self,
            models_path="models",
            train_path: Text = None,
            dev_path: Text = None,
            test_path: Text = None,
    ):
        """
        data path

        Args:
            train_path:
            dev_path:
            test_path:
            models_path:
        Returns:

        """
        splits_path = {}

        if train_path:
            splits_path.update({'train': train_path})
        if dev_path:
            splits_path.update({'dev': dev_path})
        if test_path:
            splits_path.update({'test': test_path})

        for name, path in splits_path.items():
            self.dataset[name] = Dataset.annotate_raw(path)
            self.dataset[name].numericalize_(self.vocab)
            self.ontology += self.dataset[name].extract_ontology()

            ann_path = path[:-5] + "_ann.json"
            with open(ann_path, 'wt') as f:
                json.dump(self.dataset[name].to_dict(), f, indent=4)

        self.ontology.numericalize_(self.vocab)
        with open(os.path.join(models_path, 'ontology.json'), 'wt') as f:
            json.dump(self.ontology.to_dict(), f, indent=4)
        with open(os.path.join(models_path, 'vocab.json'), 'wt') as f:
            json.dump(self.vocab.to_dict(), f, indent=4)

        # Generate embedding file
        self.embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
        E = []
        for w in tqdm(self.vocab._index2word):
            e = []
            for emb in self.embeddings:
                e += emb.emb(w, default='zero')
            E.append(e)
        with open(os.path.join(models_path, 'emb.json'), 'wt') as f:
            json.dump(E, f)

    def process_ann_dataset(
            self,
            models_path="models",
            train_path: Text = None,
            dev_path: Text = None,
            test_path: Text = None,
    ):
        """

        Args:
            models_path:
            train_path:
            dev_path:
            test_path:

        Returns:

        """
        splits_path = {}

        if train_path:
            splits_path.update({'train': train_path})
        if dev_path:
            splits_path.update({'dev': dev_path})
        if test_path:
            splits_path.update({'test': test_path})

        with open(os.path.join(models_path, 'ontology.json')) as f:
            self.ontology = Ontology.from_dict(json.load(f))
        with open(os.path.join(models_path, 'vocab.json')) as f:
            self.vocab = Vocab.from_dict(json.load(f))
        with open(os.path.join(models_path, 'emb.json')) as f:
            self.embeddings = json.load(f)
        for name, path in splits_path.items():
            with open(path) as f:
                logging.warning('loading split {}'.format(path))
                self.dataset[name] = Dataset.from_dict(json.load(f))

        logging.info(
            'dataset sizes: {}'.format(pformat({k: len(v) for k, v in self.dataset.items()})))
