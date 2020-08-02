#-*-coding:utf-8-*-
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from utils import *

path = "./data"

class config():
    def __init__(self):
        self._configs = {}

        self._configs['path']       = path + '/aminer/views.txt' ### view path
        self._configs['nodes']      = path + '/aminer/nodes.txt'
        self._configs['labels']     = path + '/aminer/author_label.txt'
        self._configs['dims']       = 256
        self._configs['lr']         = 0.01
        self._configs['epochs']     = 50
        self._configs['layers']     = 8
        self._configs['order']      = 2
        self._configs['cuda']       = '0'
        self._configs['rate']       = 0.25 # dimmension assignment strategy
        self._configs['mode']       = 0 # 0 for consensus stages first, 1 for complement stages first
        self._configs['gamma']      = 20 # gamma
        self._configs['init']       = 'nmf' ### rd or nmf
        self._configs['dw_matrix']  = 'qiu' # qiu or yang
        self._configs['self_loop']  = False
        self._configs['row_norm']   = False
        self._configs['scale']      = False
        self._configs['views']      = None
        self._configs['adj_mat']    = None
        self._configs['look_up']    = None
        self._configs['node_list']  = None
        self._configs['node_label'] = None

    @property
    def path(self):
        return self._configs['path']

    @property
    def nodes(self):
        return self._configs['nodes']

    @property
    def labels(self):
        return self._configs['labels']

    @property
    def save(self):
        return self._configs['save']

    @property
    def dims(self):
        return self._configs['dims']

    @property
    def lr(self):
        return self._configs['lr']

    @property
    def epochs(self):
        return self._configs['epochs']

    @property
    def layers(self):
        return self._configs['layers']

    @property
    def order(self):
        return self._configs['order']

    @property
    def cuda(self):
        return self._configs['cuda']

    @property
    def rate(self):
        return self._configs['rate']

    @property
    def mode(self):
        return self._configs['mode']

    @property
    def clf_ratio(self):
        return self._configs['clf_ratio']

    @property
    def gamma(self):
        return self._configs['gamma']

    @property
    def init(self):
        return self._configs['init']

    @property
    def dw_matrix(self):
        return self._configs['dw_matrix']

    @property
    def self_loop(self):
        return self._configs['self_loop']

    @property
    def row_norm(self):
        return self._configs['row_norm']

    @property
    def scale(self):
        return self._configs['scale']

    @property
    def multilabel(self):
        return self._configs['multilabel']

    @property
    def views(self):
        return self._configs['views']

    @property
    def adj_mat(self):
        return self._configs['adj_mat']

    @property
    def look_up(self):
        return self._configs['look_up']

    @property
    def node_list(self):
        return self._configs['node_list']

    @property
    def node_label(self):
        return self._configs['node_label']

    @property
    def time(self):
        return self._configs['time']

    def update_config(self,key,value):
        if key in self._configs.keys():
            self._configs[key] = value
        else:
            raise RuntimeError('Update_Config_Error')

    def save_cfg(self,logger):
        for key in self._configs.keys():
            if key not in ['views','adj_mat','node_list','node_label','look_up']:
                logger.info('{} : {}'.format(key,self._configs[key]))

cfg = config()
node_list,look_up = read_node(cfg)
cfg.update_config('node_list',node_list)
cfg.update_config('look_up',look_up)
views,adj_mat = read_edgelist(cfg)
cfg.update_config('views',views)
cfg.update_config('adj_mat',adj_mat)
node_label = read_label(cfg)
cfg.update_config('node_label',node_label)