# The derived class of convnet

import numpy as np
import numpy.random as nr
import random as r
from python_util.util import *
from python_util.data import *
from python_util.options import *
from python_util.gpumodel import *
import sys
import math as m
import layer as lay
from convdata import ImageDataProvider, CIFARDataProvider, DummyConvNetLogRegDataProvider
from os import linesep as NL
import copy as cp
import os
import convnet 
# BEGIN MY modules ---Lisijin
import iconfig
import dhmlpe_utils as dutils
# END MY modules
class IConvNetError(Exception):
    pass
class MMDP(object):
    """
    Maximum Margin Data Provider
    """
    def __init__(self, dp):
        """
        Ensure dp is a train data provider
        """
        self.batch_meta = dp.batch_meta
        self.max_depth = dp.max_depth
        self.num_joints = dp.num_joints
        self.feature_range = dp.feature_range
    def generate_candidate_set(self, ndata, num_candidate, train=True):
        """
        return [candidate_set, indexes (in feature_list)]
        """
        gt_data = self.batch_meta['feature_list'][0] / self.max_depth
        train_range = self.feature_range
        indexes = np.random.randint(low=0, high=len(train_range), size=ndata * num_candidate)
        res = gt_data[..., train_range[indexes]]
        data_dim = res.shape[0]
        return res.reshape((data_dim, num_candidate, ndata),order='F'), train_range[indexes]
    def assemble_candidate(self, ori_data, candidate_set, candidate_indexes,
                               idx_range, train=True):
        """
        Here only idx 2 is for candidate set
        requirements:
            idx 0 should be the ground-truth data
        """
        # if candidate_set.shape[0] != ori_data[2][2].shape[0]:
        #     raise IConvNetError('The dimension of input is not consistent')
        K_candidate = candidate_set.shape[1]
        num = len(idx_range) * K_candidate
        alldata = []
        for i,t in enumerate(ori_data[2]):
            e = t[..., idx_range]
            data_dim = t.shape[0]
            if len(idx_range) == 1:
                e = e.reshape((-1,1),order='F')
            alldata += [np.require(np.tile(e, [K_candidate, 1]).reshape((data_dim, \
                                                                         K_candidate * e.shape[1]),order='F'), dtype=np.single, requirements='C')]
        alldata[2] = np.require(candidate_set[...,idx_range].reshape((alldata[2].shape[0],
                                                                    alldata[2].shape[-1]),
                                                                   order='F'), 
                                dtype=np.single, requirements='C')
        return alldata
    def re_calc_data(self, alldata, candidate_indexes, step=1,start=0):
        num_joints = self.num_joints
        candidate_jt = self.batch_meta['feature_list'][0][..., candidate_indexes]
        candidate_jt = candidate_jt / self.max_depth
        s = self.calc_margin(alldata[0][...,start::step]-candidate_jt, num_joints)
        alldata[3][...,start::step] = np.require(s, dtype=np.single, requirements='C')
    def calc_margin(self, residuals, num_joints):
        mpjpe = dutils.calc_mpjpe_from_residual(residuals, num_joints)
        return mpjpe
class MMDP2(MMDP):
    def generate_candidate_set(self, ndata, num_candidate, train=True):
        """
        return [candidate_set, indexes (in feature_list)]
        """
        candidate_feature = self.batch_meta['feature_list'][2]
        train_range = self.feature_range
        indexes = np.random.randint(low=0, high=len(train_range), size=ndata * num_candidate)
        res = candidate_feature[..., train_range[indexes]]
        data_dim = res.shape[0]
        return res.reshape((data_dim, num_candidate, ndata),order='F'), train_range[indexes]
class IConvNet(convnet.ConvNet):
    ######Structure learning training module##########################################
    def generate_candidate_set(self, ndata, num_candidate, train=True):
        """
        generate candidate set by sampling the ground-truth data
        requirements:
        
        batch_meta['feature_list'][0] is ground-truth x
        
        The score prediction data provider should be the derived class for
        ()
        I will always use train_data_provider
        """
        return self.mmdp.generate_candidate_set(ndata, num_candidate, train)
    def assemble_candidate(self, ori_data, candidate_set, candidate_indexes, \
                           idx_range, train=True):
        K_candidate = candidate_set.shape[1]
        alldata = self.mmdp.assemble_candidate(ori_data, candidate_set, \
                                               candidate_indexes, idx_range,train)
        self.re_calc_data(alldata, candidate_indexes[idx_range[0]*K_candidate:\
                                                     (idx_range[-1] + 1)*K_candidate],
                          step=1,start=0
        )
        return ori_data[0], ori_data[1], alldata
    def find_most_violated(self,data, train=True):
        """
        data is [epoch, batchnum, alldata]
        The output data components
        0: ground-truth joint (y)
        1: imgfeature         (x)
        2: candidate          (y_c)
        3: score <---- Will be overrided by margin
        4: mpjpe <---- 
        data
        """
        ## init all require const
        num_candidate = 200
        max_num = 20  # The number of data to be processed at one batch
        ndata = data[2][0].shape[-1]
        # candidate set will be a dim * num_candidate * ndata array
        # candidate_indexes is the index in the feature_list for the candidate 
        candidate_set, candidate_indexes = self.generate_candidate_set(ndata, num_candidate, train)
        score_list = [] # save the "extended" score for most violated point
        selected_set = [] # save the most violated point matrix for each local batch

        num_batch =  int((ndata - 1)/max_num) + 1
        feature_dim = 1
        feature_name = 'scorepred2' # The name for score + margin : Use as default

        idx_range = range(0, min(max_num, ndata)) 
        cur_data = self.assemble_candidate(data, candidate_set, candidate_indexes, \
                                           idx_range, train)
        pre_data_num = len(idx_range)
        buf = np.require(np.zeros((len(idx_range)*num_candidate, feature_dim),\
                                   dtype=np.single), requirements='C')
        most_violated_indexes = []
        
        for b in range(num_batch):
            self.libmodel.startFeatureWriter(cur_data[2], [buf], [feature_name])
            cur_candidate_indexes = candidate_indexes[idx_range[0]*num_candidate:\
                                                      (idx_range[-1]+1)*num_candidate]
            # print 'Len of cur_candidate_indexes is {}'.format(len(cur_candidate_indexes))
            if b < num_batch - 1:
                idx_range = range((b + 1)*max_num, min((b+2)*max_num, ndata)) 
                next_data = self.assemble_candidate(data, candidate_set, candidate_indexes, \
                                                    idx_range, train)
            IGPUModel.finish_batch(self)
            # keep the selected pose
            predscore = buf.flatten().reshape((num_candidate, pre_data_num),order='F')
            index = np.argmax(predscore, axis=0).flatten() + \
              np.asarray(range(0,pre_data_num)) * num_candidate
            selected_set += [cur_data[2][2][...,index]]
            # print 'cur_data[2][2].shape = {}'.format(cur_data[2][2].shape)
            most_violated_indexes += cur_candidate_indexes[index].tolist()
            # score_list += np.max(predscore, axis=0).flatten().tolist()
            score_list += predscore.flatten(order='F')[index].tolist()
            if pre_data_num != len(idx_range):
                    buf = np.require(np.zeros((len(idx_range)*num_candidate, feature_dim), \
                                        dtype=np.single), requirements='C')
            pre_data_num = len(idx_range)
            cur_data = next_data
        # print 'aug score range candidate %.6f  %.6f' % (min(score_list), max(score_list))
        if len(selected_set) == 1:
            most_violated = np.require(selected_set[0], dtype=np.single, requirement='C')
        else:
            most_violated = np.require(np.concatenate(selected_set, axis=1), dtype=np.single, \
                          requirements='C')
        alldata = []
        for i,e in enumerate(data[2]):
            data_dim = e.shape[0]
            alldata += [np.require(np.tile(e, [2, 1]).reshape((data_dim, 2*ndata),order='F'), \
                                   dtype=np.single, requirements='C')]
        s1range = range(1, ndata * 2, 2) # set target feature for most violated data
        alldata[2][...,s1range] = most_violated
        # s0range = range(0, ndata *2,2) # put ground-truth data
        # alldata[2][..., s0range] = alldata[0][...,s0range]
        
        self.re_calc_data(alldata, most_violated_indexes,step=2,start=1)
        alldata[3][...,0::2] = 0 # set ground-truth margin to zero
        # print '------margin for most violated len = (%d)--------' % alldata[2].size 
        # print alldata[3][0,:10]
        # print '------score for the most violated len = (%d)-----' % len(score_list)
        # print score_list[0:10]
        # print '-------------------------------------' 
        return data[0], data[1], alldata
    def re_calc_data(self, alldata, candidate_indexes, step=1, start=0):
        """
        Calc the correct "margin" for each sample
        """
        return self.mmdp.re_calc_data(alldata, candidate_indexes,step, start)
    def simple_test_score(self, most_violated_data):
        """
        Just for testing 
        """
        feature_name = 'scorepred2'
        data = most_violated_data[2]
        ndata = data[0].shape[-1]
        buf = np.require(np.zeros((ndata,1),dtype=np.single), requirements='C')
        self.libmodel.startFeatureWriter(data, [buf], [feature_name])
        IGPUModel.finish_batch(self)
        print buf.shape, '----------\t ndata = ', ndata
        s = buf.flatten() 
        print data[3].flatten()[:10]
        s1 = s[range(0,s.size,2)]
        s2 = s[range(1,s.size,2)]
        print('gt range [%.6f, %.6f](%.6f)\t most violsated range [%.6f, %.6f](%.6f)' % \
              (np.min(s1), np.max(s1), np.min(s1), np.min(s2), np.max(s2), np.min(s2)))
        diff = s2 - s1
        print '%.6f%% > 0' % (np.sum(diff > 0)*100.0/diff.size)
        print '==============\n\n\n\n'
    def simple_test_activation(self, most_violated_data):
        """
        just for testing
        """
        import iutils as iu
        feature_name = 'fc_f1'
        feature_dim = self.model_state['layers'][feature_name]['outputs']
        print 'feature_dim:\t', feature_dim
        data = most_violated_data[2]
        ndata = data[0].shape[-1]
        buf = np.require(np.zeros((ndata,feature_dim),dtype=np.single), requirements='C')
        self.libmodel.startFeatureWriter(data, [buf], [feature_name])
        IGPUModel.finish_batch(self)
        iu.print_common_statistics(buf, 'Analysis:\t%s' % feature_name)
        print '==============\n\n\n\n'
    def slp_train(self):
        print "========================="
        print "Training Structure Learning Netework %s" % self.model_name
        print "Requirement: MaxMarginPairCost"
        self.op.print_values()
        print "========================="
        self.print_model_state()
        print "Running on CUDA device(s) %s" % ", ".join("%d" % d for d in self.device_ids)
        print "Current time: %s" % asctime(localtime())
        print "Saving checkpoints to %s" % self.save_file
        mode_params = self.parse_params(self.mode_params)
        mmdp_dic = {'mmdp':MMDP, 'mmdp2':MMDP2}
        # Use train data provider for mmdp initialization 
        self.mmdp = mmdp_dic[mode_params[0]](self.train_data_provider)
        print "========================="
        next_data = self.get_next_batch()
        while self.epoch <= self.num_epochs:
            data = next_data
            self.epoch, self.batchnum = data[0], data[1]
            self.print_iteration() # Maybe change it later
            print '---Begin to calculate candidate result',
            sys.stdout.flush()
            compute_time_py = time()
            most_violated_data = self.find_most_violated(data)
            print '\t---End\t',
            self.print_elapsed_time(time() - compute_time_py)
            # self.simple_test_score(most_violated_data)
            self.simple_test_activation(most_violated_data)
            # now prepare the training for update
            compute_time_py = time()
            inner_iter_max = 1
            for inner_i in range(inner_iter_max):
                self.start_batch(most_violated_data)
                # load the next batch while the current one is computing
                if inner_i == 0:
                    next_data = self.get_next_batch()
                batch_output = self.finish_batch()
                self.train_outputs += [batch_output]
                self.print_train_results()

                num_batch_done = self.get_num_batches_done() * inner_iter_max + inner_i
                if  num_batch_done % self.testing_freq == 0:
                    self.sync_with_host()
                    self.test_outputs += [self.slp_get_test_error()]
                    self.print_test_results()
                    self.print_test_status()
                    self.conditional_save()
            self.print_elapsed_time(time() - compute_time_py)
    def slp_get_test_error(self):
        next_data = self.get_next_batch(train=False)
        test_outputs = []
        while True:
            data = next_data
            start_time_test = time()
            most_violdated_data = self.find_most_violated(data,train=False)
            self.start_batch(most_violdated_data, train=False)
            load_next = (not self.test_one or self.test_only) and data[1] < self.test_batch_range[-1]
            if load_next:
                next_data = self.get_next_batch(train=False)
            test_outputs +=[self.finish_batch()]
            if self.test_only:
                print "batch%d: %s" % (data[1], str(test_outputs[-1])),
                self.print_elapsed_time(time() - start_time_test)
            if not load_next:
                break
            sys.stdout.flush()
        return self.aggregate_test_outputs(test_outputs)
        
    def start(self):
        if self.mode:
            mode_dic = {'slp-train': lambda : self.slp_train()}
            mode_dic[self.mode]()
            self.cleanup()
            if self.force_save:
                self.save_state().join()
            sys.exit(0)
        else:
            convnet.Convnet.start(self)
    @classmethod
    def parse_params(cls, s):
        l = s.split(',')
        res_l = []
        for x in l:
            if x.find('@') != -1:
                a = x.split('@')
                res_l += [(a[0], int(a[1]))]
            else:
                res_l += [x]              
        return res_l
    @classmethod
    def get_options_parser(cls):
        op = convnet.ConvNet.get_options_parser()
        op.add_option('mode', 'mode', StringOptionParser, 'The training mode', default="")
        op.add_option('mode-params', 'mode_params', StringOptionParser, 'The training mode params', default="")
        return op
    
if __name__ == "__main__":
#    nr.seed(6)

    op = IConvNet.get_options_parser()
    op, load_dic = IGPUModel.parse_options(op)
    model = IConvNet(op, load_dic)
    model.start()


