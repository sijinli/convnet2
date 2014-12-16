# The derived class of convnet

import numpy as n
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
# END MY modules
class IConvNetError(Exception):
    pass
class IConvNet(convnet.ConvNet):
    ######Structure learning training module##########################################
    def generate_candidate_set(self, ndata, num_candidate, Train=True):
        """
        generate candidate set by sampling the ground-truth data
        requirements:
        
        batch_meta['feature_list'][0] is ground-truth x
        
        The score prediction data provider should be the derived class for
        ()
        I will always use train_data_provider
        """
        dp = self.train_data_provider 
        gt_data = dp.batch_meta['feature_list'][0] / dp.max_depth
        train_range = dp.feature_range
        indexes = np.random.randint(low=0, high=len(train_range), size=ndata * num_candidate)
        res = gt_data[..., train_range[indexes]]
        data_dim = res.shape[0]
        return res.reshape((data_dim, ndata*num_candidate),order='F')
    def assemble_candidate(self, ori_data, candidate_set, idx_range, Train=True):
        """
        Here only idx 2 is for candidate set
        requirements:
            idx 0 should be the ground-truth data
        """
        if candidate_set.shape[0] != ori_data[2][2].shape[0]:
            raise IConvNetError('The dimension of input is not consistent')
        epoch = ori_data[0]
        K_candidate = candidate_set.shape[1]
        num = len(idx_range) * K_candidate
        alldata = []
        for i,t in enumerate(ori_data[2]):
            e = t[..., idx_range]
            data_dim = t.shape[0]
            if len(idx_range) == 1:
                e = e.reshape((-1,1),order='F')
            alldata += [np.tile(e, [K_candidate, 1]).reshape((data_dim, \
                                                        K_candidate * e.shape[1]),order='F')]
        alldata[2] = candidate_set[..., idx_range].reshape((alldata[2].shape[0],\
                                                  alldata[2].shape[-1]),order='F')
        self.re_calc_data(alldata)
        return epoch, ori_data[1], alldata
    def find_most_violated(self,data, Train=True):
        """
        data is [epoch, batchnum, alldata]
        
        0: ground-truth
        1: imgfeature
        2: candidate
        data
        """
        raise Exception('How to do that if in test mode')
        ## init all require const
        num_candidate = 200
        ndata = data[2][0].shape[-1]
        # candidate set will be a dim * num_candidate * ndata array
        candidate_set = self.generate_candidate_set(ndata, num_candidate, Train)
        score_list = []
        selected_set = []
        max_num = 20
        num_batch =  int((ndata - 1)/max_num) + 1
        feature_dim = 1
        feature_name = 'scorepred2' # The name for score + margin : Use as default

        idx_range = range(0, min(max_num, ndata)) 
        cur_data = self.assemble_candidate(data, candidate_set, idx_range, Train)
        pre_data_num = len(idx_range)
        buf = np.require(np.zeros((len(idx_range)*num_candidate, feature_dim),\
                                   dtype=np.single), requirement='C') 
        for b in range(num_batch):
            self.libmodel.startFeatureWriter(cur_data, [buf], [feature_name])
            if b < num_batch - 1:
                idx_range = range((b + 1)*max_num, min((b+2)*max_num, ndata)) 
                cur_data = self.assemble_candidate(data, candidate_set, idx_range, Train)
                if pre_data_num != len(idx_range):
                    buf = np.require(np.zeros((len(idx_range)*num_candidate, feature_dim), \
                                        dtype=np.single), requirement='C')
            IGPUModel.finish_batch(self)
            score_list += buf.flatten().tolist()
            # keep the selected pose
            predscore = buf.flatten().reshape((num_candidate, len(idx_range)),order='F')
            index = np.argmax(predscore, axis=0).flatten() + \
              np.asarray(range(0,len(idx_range))) * num_candidate
            selected_set += [cur_data[2][2][...,index]]
        # data_dim2 = self.train_data_provider.get_data_dims(2)
        if len(selected_set) == 1:
            most_violated = np.require(selected_set[0], dtype=np.single, requirement='C')
        else:
            most_violated = np.require(np.concatenate(selected_set, axis=1), dtype=np.single, \
                          requirement='C')
        alldata = []
        for i,e in enumerate(data[2]):
            data_dim = e.shape[0]
            alldata += [np.tile(e, [2, 1]).reshape((data_dim, 2*ndata),order='F')]
        s2range = range(1, ndata * 2, 2)
        alldata[2][...,s2range] = most_violated
        self.re_calc_data(alldata) 
        return data[0], data[1], alldata
    def re_calc_data(self, alldata):
        raise IConvNetError('Unfinished yet')
        pass
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
        print "========================="
        next_data = self.get_next_batch()
        while self.epoch <= self.num_epochs:
            data = next_data
            self.epoch, self.batchnum = data[0], data[1]
            self.print_iteration() # Maybe change it later
            print '---Begin to calculate candidate result'
            sys.stdout.flush()
            compute_time_py = time()
            most_violated_data = self.find_most_violated(data)
            print '---End',
            self.print_elapsed_time(time() - compute_time_py)

            # now prepare the training for update
            compute_time_py = time()            
            self.start_batch(most_violated_data)
            # load the next batch while the current one is computing
            next_data = self.get_next_batch()
            batch_output = self.finish_batch()
            self.train_outputs += [batch_output]
            self.print_train_results()

            if self.get_num_batches_done() % self.testing_freq == 0:
                self.sync_with_host()
                self.test_outputs += [self.slp_get_test_error()]
                self.print_test_results()
                self.print_test_status()
                self.conditional_save()
            
            self.print_elapsed_time(time() - compute_time_py)
    def slp_get_test_error():
        data = self.get_next_batch(train=False)
        test_outputs = []
        while True:
            data = next_data
            start_time_test = time()
            most_violdated_data = self.find_most_violated(data,train=False)
            load_next = (not self.test_one or self.test_only) and data[1] < self.test_batch_range[-1]
            self.start_batch(most_violdated_data, train=False)
            if self.test_only:
                print "batch%d: %s" % (data[1], str(test_outputs[-1])),
                self.print_elapsed_time(time() - start_time_test)
            if not load_next:
                break
            sys.stdout.flush()
            
        
    def start(self):
        if self.mode:
            mode_dic = ['slp_train':self.slp_train]
            mode_dic[self.mode]()
            self.cleanup()
            if self.force_save:
                self.save_state().join()
            sys.exit(0)
        else:
            convnet.Convnet.start(self)
    @classmethod
    def get_options_parser(cls):
        op = convnet.ConvNet.get_options_parser()
        op.add_option('mode', 'mode', StringOptionParser, 'The training mode')
        op.add_option('mode-params', 'mode_params', StringOptionParser, 'The training mode params')
        return op
if __name__ == "__main__":
#    nr.seed(6)

    op = IConvNet.get_options_parser()
    op, load_dic = IGPUModel.parse_options(op)
    model = IConvNet(op, load_dic)
    model.start()


