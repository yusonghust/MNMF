#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
from mf_config import cfg
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
from sklearn.decomposition import NMF
import time
from classifier import Classifier
import logging
from log import setup_logger
from scipy.linalg import interpolative

time_now = setup_logger('mf', 'mf_' + cfg.labels.split('/')[-1].split('.')[0], level=logging.INFO, screen=True, tofile=True)
logger = logging.getLogger('mf')

global time_cost

class MNMF(object):
    def __init__(self):
        pass

    def initializer(self,dims,adj_mat):
        m = len(cfg.node_list)
        n = int(dims)
        view_nums = len(cfg.views)
        if cfg.init == 'rd':
            temp_U = np.random.uniform(size=(m,n),low=0.0,high=0.02).astype(np.float32)
            temp_V = dict()
            for k in range(view_nums):
                temp_V[k] = np.random.uniform(size=(n,m),low=0.0,high=0.02).astype(np.float32)
        else:
            temp_U = np.zeros(shape=(m,n))
            temp_V = dict()
            for k in range(view_nums):
                model = NMF(n_components=n, init='random', random_state=0)
                U = model.fit_transform(adj_mat[k])
                V = model.components_
                temp_U = temp_U + U
                temp_V[k] = V.astype(np.float32)
            temp_U = (temp_U/view_nums).astype(np.float32)
        alpha = np.ones(shape=(view_nums))/float(view_nums)
        return temp_U,temp_V,alpha

    def coo_opt(self,optimizer,loss,var_list):
        opt_list = []
        for var in var_list:
            opt = optimizer.minimize(loss=loss,var_list=[var])
            opt_list.append(opt)
        return opt_list

    def update_U(self,tf_inputs,U,V,alpha,view_nums):
        x = 0
        y = 0
        for k in range(view_nums):
            x += tf.pow(alpha[k],cfg.gamma) * tf.matmul(tf_inputs[k],V[k],transpose_b=True)
            y += tf.pow(alpha[k],cfg.gamma) * tf.matmul(tf.matmul(U,V[k]),V[k],transpose_b=True)
        res = tf.multiply(U,tf.divide(x,y+1e-10))
        return res

    def update_V(self,tf_inputs,U,V,k):
        x = tf.matmul(U,tf_inputs[k],transpose_a=True)
        y = tf.matmul(tf.matmul(U,U,transpose_a=True),V[k])
        res = tf.multiply(V[k],tf.divide(x,y+1e-10))
        return res

    def update_alpha(self,W,view_nums):
        res = []
        xs = []
        gamma = cfg.gamma
        y = 0
        for k in range(view_nums):
            x = tf.pow(gamma*W[k],1/(1-gamma))
            xs.append(x)
            y += x
        for k in range(view_nums):
            res.append(tf.divide(xs[k],y+1e-10))
        return res

    def cal_loss(self,tf_inputs,U,V,alpha,view_nums):
        Ws = []
        cost = 0
        for k in range(view_nums):
            UV = tf.matmul(U, V[k])
            W = tf.reduce_mean(tf.pow(tf_inputs[k]-UV, 2))
            cost = cost + alpha[k]*W
            Ws.append(W)
        return Ws,cost


    def consensus_nmf_layer(self,inputs,dims,scope):
        m = len(cfg.node_list)
        n = int(dims)
        view_nums = len(cfg.views)
        tf_inputs = []
        for k in range(view_nums):
            tf_inputs.append(tf.constant(inputs[k],dtype=tf.float32))
        with tf.variable_scope(scope) as scope:
            temp_U,temp_V,temp_alpha = self.initializer(dims,inputs)
            U = tf.Variable(temp_U,dtype=tf.float32)
            V = dict()
            alpha = dict()
            for k in range(view_nums):
                A = tf_inputs[k]
                V[k] = tf.Variable(temp_V[k],dtype=tf.float32)
                alpha[k] = tf.Variable(temp_alpha[k],dtype=tf.float32)
            Ws,cost = self.cal_loss(tf_inputs,U,V,alpha,view_nums)
            opt_list = []
            new_U = self.update_U(tf_inputs,U,V,alpha,view_nums)
            opt_list.append(U.assign(new_U))
            for k in range(view_nums):
                new_V = self.update_V(tf_inputs,U,V,k)
                opt_list.append(V[k].assign(new_V))
            new_alphas = self.update_alpha(Ws,view_nums)
            for k in range(view_nums):
                new_alpha = new_alphas[k]
                opt_list.append(alpha[k].assign(new_alpha))
            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.allow_growth = True
            last_costvalue = 0.0
            count = 0
            with tf.Session(config=gpu_config) as sess:
                sess.run(tf.global_variables_initializer())
                for idx in range(cfg.epochs):
                    for opt in opt_list:
                        sess.run(opt)
                    costValue = sess.run(cost)
                    if (((idx+1)%10) == 0) or (idx==0):
                        end = time.time()
                        current_costvalue = round(costValue,5)
                        if current_costvalue<=last_costvalue:
                            count += 1
                            if count>5:
                                break
                        else:
                            count = 0
                        last_costvalue = round(costValue,5)
                U_ = sess.run(U)
                V_ = dict()
                for k in range(view_nums):
                    V_[k] = sess.run(V[k])
                alpha_ = dict()
                for k in range(view_nums):
                    alpha_[k] = sess.run(alpha[k])
                    logger.info('alpha {:d} = {:.6f}'.format(k,alpha_[k]))
        tf.reset_default_graph()
        return U_, V_

    def complement_nmf(self,inputs,dims):
        m = len(cfg.node_list)
        n = int(dims)
        view_nums = len(cfg.views)

        U_ = []
        V_ = []
        for k in range(view_nums):
            model = NMF(n_components=n, init='random', random_state=0, max_iter=50,solver='cd')
            U = model.fit_transform(inputs[k])
            V = model.components_
            U_.append(U)
            V_.append(V)
        return U_,V_

    def res_matrix(self,inputs,U,V):
        view_nums = len(cfg.views)
        res_mat = []
        for k in range(view_nums):
            res = inputs[k] - np.dot(U,V[k])
            res = res.clip(0)
            res_mat.append(res)
        return res_mat

    def res_matrix_c(self,inputs,U,V):
        view_nums = len(cfg.views)
        res_mat = []
        for k in range(view_nums):
            res = inputs[k] - np.dot(U[k],V[k])
            res = res.clip(0)
            res_mat.append(res)
        return res_mat


    def mnmf(self):
        inputs = cfg.adj_mat
        if cfg.mode==0:
            U_list = []
            for k in range(cfg.layers):
                print('current consensus layer = %d'%(k+1))
                dims = int(cfg.rate*cfg.dims/cfg.layers)
                U,V = self.consensus_nmf_layer(inputs,dims,scope='consensus_%d'%k)
                inputs = self.res_matrix(inputs,U,V)
                U_list.append(U)
            U_ = []
            for k in range(cfg.layers):
                print('current complement layer = %d'%(k+1))
                dims = int((1-cfg.rate)*cfg.dims/(len(cfg.views)*cfg.layers))
                U,V = self.complement_nmf(inputs,dims)
                inputs = self.res_matrix_c(inputs,U,V)
                U_ += U
        else:
            U_ = []
            for k in range(cfg.layers):
                print('current complement layer = %d'%(k+1))
                dims = int((1-cfg.rate)*cfg.dims/(len(cfg.views)*cfg.layers))
                U,V = self.complement_nmf(inputs,dims)
                inputs = self.res_matrix_c(inputs,U,V)
                U_ += U
            U_list = []
            for k in range(cfg.layers):
                print('current consensus layer = %d'%(k+1))
                dims = int(cfg.rate*cfg.dims/cfg.layers)
                U,V = self.consensus_nmf_layer(inputs,dims,scope='consensus_%d'%k)
                inputs = self.res_matrix(inputs,U,V)
                U_list.append(U)
        U = np.concatenate(U_list + U_,axis=1)
        return U

if __name__ == '__main__':
    s_t = time.time()
    M = MNMF()
    U = M.mnmf()
    e_t = time.time()
    logger.info('time cost = {:.2f}'.format(e_t-s_t))
    Classifier(U,cfg,logger)
