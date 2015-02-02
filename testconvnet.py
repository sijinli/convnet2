import numpy as np
import sys
import getopt as opt
from python_util.util import *
from math import sqrt, ceil, floor
import os
from python_util.gpumodel import IGPUModel
import random as r
import numpy.random as nr
from convnet import ConvNet
from python_util.options import *
import pylab as pl
import matplotlib.pyplot as plt
import iutils as iu
import dhmlpe_utils as dutils
sys.path.append('/home/grads/sijinli2/Projects/DHMLPE/Python/src/')
sys.path.append('/media/M_FILE/cscluster/Projects/DHMLPE/Python/src/')
class TestConvNetError(Exception):
    pass
class TestConvNet(ConvNet):
    def __init__(self, op, load_dict):
        ConvNet.__init__(self, op, load_dic)
        self.statistics = dict()
        self.temp_data = dict()
    def get_gpus(self):
        self.need_gpu = False
        if self.op.get_value('mode'):
            mode_value = self.op.get_value('mode')
            flag = mode_value in ['do-score-prediction', 'slp-server', 'mpjpe-eva', \
                                  'write-feature']
            self.need_gpu |= flag
        #self.need_gpu |= self.op.get_value('ubd_image_folder') is not None
    def init_data_providers(self):
        class Dummy:
            def advance_batch(self):
                pass
        self.get_gpus()
        if self.need_gpu:
            ConvNet.init_data_providers(self)
        else:
            self.train_data_provider = self.test_data_provider = Dummy()
    def import_model(self):
        if self.need_gpu:
            ConvNet.import_model(self)
    def init_model_state(self):
        pass
    def init_model_lib(self):
        if self.need_gpu:
            ConvNet.init_model_lib(self)
    def write_feature(self):
        import iread.myio as mio
        if not self.mode_params or len(self.mode_params) < 2:
            raise TestConvNetError('write_feature need parameters')
        params = self.parse_params(self.mode_params)
        feature_name = params[0]
        save_folder = params[1]
        if len(params) == 3:
            Y_idx = int(params[2])
        else:
            Y_idx = 1
        import scipy.io as sio
        testdp = self.test_data_provider
        num_batches = len(testdp.batch_range)
        print 'There are ' + str(testdp.get_num_batches(self.data_path)) + ' in directory'
        if self.test_data_provider.batch_size > 0:
            num_batches = (num_batches - 1)/ self.test_data_provider.batch_size + 1
        if self.test_one:
            num_batches = min(num_batches, 1)
        print 'There are ' + str( num_batches ) + ' in range'
        iu.ensure_dir(save_folder)
        feature_dim = self.model_state['layers'][feature_name]['outputs']
        print 'Feature dim is %d' % feature_dim
        for b in range(num_batches):
            epoch, b_num, data = self.get_next_batch(train=False)
            print '   Start writing batch......\t' + str(b_num)
            num_data = data[0].shape[-1]
            buf =  np.zeros((num_data, feature_dim), dtype=np.single)
            save_name = 'batch_feature_' + str(b_num) + '_' + feature_name 
            save_path = iu.fullfile(save_folder, save_name)
            self.libmodel.startFeatureWriter(data, [buf], [feature_name])
            IGPUModel.finish_batch(self)
            d = dict()
            d['X'] = buf.transpose()
            d['batch_num'] = b_num
            d['Y'] = data[Y_idx]
            cur_batch_indexes = self.test_data_provider.data_dic['cur_batch_indexes']
            print d['Y'].shape
            d['cur_batch_indexes'] = cur_batch_indexes
            print 'The len of data is ' + str(len(data))
            print 'The shape of X is' + str(d['X'].shape)
            print 'The shape of Y is' + str(d['Y'].shape)
            ##sio.savemat(save_path, d)
            mio.pickle(save_path, d)
    def plot_cost(self):
        if not self.mode_params:
            raise TestConvNetError('plot cost need parameters')
        params = self.parse_params(self.mode_params)
        if type(params[0]) is not list:
            cost_name = params[0]
            cost_idx = 0
        else:
            cost_name = params[0][0]
            cost_idx = params[0][1]
        if cost_name not in self.train_outputs[0][0]:
            raise TestConvNetError("Cost layer with name {} not defined by given convnet.".format(cost_name))
        train_errors = [o[0][cost_name][cost_idx]/o[1] for o in self.train_outputs]
        test_errors = [o[0][cost_name][cost_idx]/o[1] for o in self.test_outputs]
        numbatches = len(self.train_batch_range)
        test_errors = np.row_stack(test_errors)
        test_errors = np.tile(test_errors, (1, self.testing_freq))
        test_errors = list(test_errors.flatten())
        test_errors += [test_errors[-1]] * max(0,len(train_errors) - len(test_errors))
        test_errors = test_errors[:len(train_errors)]
        if self.batch_size == -1:
            numepochs = len(train_errors) / int(numbatches)
        else:
            numepochs = len(train_errors) * self.batch_size  / int(len(self.train_batch_range))
        pl.figure(1)
        x = range(0, len(train_errors))

        print 'numepochs=%d' % numepochs
        pl.plot(x, train_errors, 'k-', label='Training set')
        pl.plot(x, test_errors, 'r-', label='Test set')
        print test_errors[-10:]
        log_scale = False
        if log_scale:
            pl.gca().set_yscale('log')
        else: 
            pl.ylim([0, np.median(test_errors[-len(test_errors)/10:-1])*5])
            # print np.median(test_errors[-len(test_errors)/10:-1])*2, ',,,,,,,,'
        
        pl.legend()
        if self.batch_size == -1:
            ticklocs = range(numbatches, len(train_errors) - len(train_errors) % numbatches + 1, numbatches)
        else:
            t = np.ceil(numbatches / self.batch_size)
            ## approximate the time for change 
            ticklocs = range(int(t), len(train_errors), int(numbatches/self.batch_size))
        epoch_label_gran = int(ceil(numepochs / 20.)) # aim for about 20 labels
        epoch_label_gran = int(ceil(float(epoch_label_gran) / 10) * 10) # but round to nearest 10
        if self.batch_size == -1:
            ticklabels = map(lambda x: str((x[1] / numbatches)) if x[0] % epoch_label_gran == epoch_label_gran-1 else '', enumerate(ticklocs))
        else:
            t = np.ceil( numbatches / self.batch_size) * epoch_label_gran
            ticklabels = map(lambda x: str(x[1] * self.batch_size/numbatches) if np.floor(x[1] * self.batch_size/numbatches) % epoch_label_gran == 0  else '', enumerate(ticklocs))
    
        # pl.plot(x, test_errors, 'r-', label='Test set')
        pl.xticks(ticklocs, ticklabels)
        pl.xlabel('Epoch')

        pl.ylabel(cost_name)
        pl.title(cost_name)
    @classmethod
    def parse_params(cls, s):
        l = s.split(',')
        res_l = []
        for x in l:
            if x.find('@') != -1:
                a = x.split('@')
                res_l += [[a[0], int(a[1])]]
            else:
                res_l += [x]              
        return res_l
    def calc_MPJPE(self, est, gt, num_joints, is_relskel=False):
        """
        est, gt will be dim X ndata matrix
        dim will be dim_data (2 or 3) x num_joints
        """
        ndata = gt.shape[-1]
        est = est.reshape((-1, num_joints, ndata),order='F')
        gt = gt.reshape((-1, num_joints, ndata),order='F')
        print est[:,[0,1,2],0]
        print gt[:,[0,1,2],0]
        return [np.sum(np.sqrt(np.sum((est - gt) ** 2,axis=0)).flatten())/num_joints, ndata]
    def calc_MPJPE_raw(self, est, gt, num_joints, is_relskel=False):
        """
        est, gt will be dim X ndata matrix
        dim will be dim_data (2 or 3) x num_joints
        """
        ndata = gt.shape[-1]
        est = est.reshape((-1, num_joints, ndata),order='F')
        gt = gt.reshape((-1, num_joints, ndata),order='F')
        print est[:,[0,1,2],0]
        print gt[:,[0,1,2],0]
        res = np.sum(np.sqrt(np.sum((est - gt) ** 2,axis=0)),axis=0)/num_joints
        print res.size, ndata
        return res.tolist()
    def convert_pairwise2rel_simple(self, mat):
        import dhmlpe_features
        return dhmlpe_features.convert_pairwise2rel_simple(mat, 3)
    def evaluate_output(self):
        import scipy.io as sio
        next_data=self.get_next_batch(train=False)[2]
        test_outputs = []
        num_cases = []
        params = self.parse_params(self.op.get_value('mode_params'))
        save_path = None
        # output_layer_idx = self.get_layer_idx(params[0])
        output_layer_name = params[0]
        if len(params) == 1:
            target_type = 'h36m_body'
            gt_idx = 1
        else:
            target_type = params[1]
            gt_idx = int(params[2])
        data_dim = self.model_state['layers'][output_layer_name]['outputs']
        test_outputs= []
        tosave_pred = []
        tosave_indexes = []
        err_list = []
        rel_list = ['RelativeSkel_Y3d_mono_body']
        if 'feature_name_3d' not in dir(self.test_data_provider):
            is_relskel = False
        else:
            is_relskel = (self.test_data_provider.feature_name_3d in rel_list)
        # print 'I am using %s' % ('RelSkel' if is_relskel else 'Rel')
        convert_dic = {'h36m_rel':lambda x:x,\
                       'h36m_body':self.convert_relskel2rel, \
                       'humaneva_body':self.convert_relskel2rel_eva,
                       'people_count':lambda X: X * self.test_data_provider.maximum_count, \
                       'h36m_pairwise_simple': lambda X: self.convert_pairwise2rel_simple(X)}
        # if is_relskel == False and target_type in ['h36m_body']:
        #     raise Exception('target|dp does''t match')
        try:
            max_depth = self.test_data_provider.max_depth
            num_joints = self.test_data_provider.num_joints
        except AttributeError:
            max_depth = 1200
            num_joints = 17
            print 'Assigning default max_depth'
        while True:
            data = next_data
            num_cases += [data[0].shape[-1]]
            buf = np.require(np.zeros((data[0].shape[-1], data_dim),\
                                      dtype=np.single), \
                             requirements='C')
            self.libmodel.startFeatureWriter(data, [buf], [output_layer_name])
            cur_batch_indexes = self.test_data_provider.data_dic['cur_batch_indexes']
            next_start_batch_idx = self.test_data_provider.curr_batchnum
            load_next = (not self.test_one) and (next_start_batch_idx!=0)
            if load_next:
                next_data = self.get_next_batch(train=False)[2]
            IGPUModel.finish_batch(self)
            print '-------max %.6f min %.6f-------' % (np.max(buf.flatten()), \
                                                       np.min(buf.flatten()))
            if target_type in convert_dic:
                est = convert_dic[target_type](buf.T)
                gt =  convert_dic[target_type](data[gt_idx])
            else:
                est = buf.T
                gt = data[gt_idx]
            if target_type in ['h36m_rel', 'h36m_body', 'humaneva_body', \
                               'h36m_pairwise_simple']:
                test_outputs += [self.calc_MPJPE(est, gt, num_joints)]
                err_list += self.calc_MPJPE_raw(est, gt, num_joints)
            elif target_type == 'h36m_body_len':
                test_outputs += [self.calc_MPJPE(est, gt, num_joints-1)]
            elif target_type == 'people_count':
                test_outputs += [self.calc_absdiff_count(est,gt)]
                err_list += (est - gt).flatten().tolist() 
            print test_outputs[-1]
            if self.save_path:
                tosave_pred += [est]
                tosave_indexes += cur_batch_indexes.flatten().tolist()
            if not load_next:
                break
            sys.stdout.flush()
        a = 0
        b = 0
        for x in test_outputs:
           a = a + x[0]
           b = b + x[1]
        if target_type in ['h36m_rel', 'h36m_body', 'humaneva_body', 'h36m_pairwise_simple']:
            print 'max_depth = %6f' % max_depth
            print 'MPJPE is %.6f, a, b = %.6f, %.6f' % ((a/b) * max_depth, a,b)
            arr = np.asarray(err_list).flatten()*max_depth
            print 'MPJPE is %.6f, std =%.6f ' % (np.mean(arr), np.std(arr))
        elif target_type in [ 'people_count']:
            print 'Average counting error is %.6f (%d patches)' % (a/b, b)
            err_arr = np.abs(np.asarray(err_list))
            print 'Average counting error is %.6f (std=%.6f)' % (np.mean(err_arr), \
                                                                 np.std(err_arr))
        if save_path:
            saved = dict()
            if target_type in ['h36m_body', 'humaneva_body','h36m_rel', \
                               'h36m_pairwise_simple']:
                saved['prediction'] = np.concatenate(tosave_pred, axis=-1) * max_depth
            else:
                saved['prediction'] = np.concatenate(tosave_pred, axis=-1)
            saved['indexes'] = tosave_indexes
            if len(self.test_data_provider.images_path) != 0:
                saved['images_path'] = [self.test_data_provider.images_path[x] for x in tosave_indexes]
            saved['oribbox'] = self.test_data_provider.batch_meta['oribbox'][...,tosave_indexes].reshape((4,-1),order='F')
            sio.savemat(save_path, saved)
    def do_score_prediction(self):
        """
        IN the current version, I will not take parameters from outside
        THis will be improved in the future.
        """
        
        import iread.myio as mio
        from igui.score_canvas import ScoreCanvas
        exp_name = 'JPS_act_12_exp_4_accv_half_fc_j2'
        exp_name_base = 'ASM_act_12_exp_4'
        exp_base_folder = '/opt/visal/tmp/for_sijin/Data/H36M/H36MExp'
        exp_path = iu.fullfile(exp_base_folder, 'folder_%s' % exp_name, 'batches.meta')
        meta_base_path = iu.fullfile(exp_base_folder, 'folder_%s' % exp_name_base, 'batches.meta')
        meta = mio.unpickle(exp_path)
        meta_base = mio.unpickle(meta_base_path)
        images_path = meta_base['images_path']
        
        pred_pose = meta['feature_list'][0]
        gt_pose = meta['random_feature_list'][0]
        ntotal = gt_pose.shape[-1]
        print 'gt_pose_shape',gt_pose.shape
        print 'pred_pose_shape', pred_pose.shape
        ref_frame = 7600 # This is the index in test range
        ## ref_frame = 2600 # This is the index in test range
        test_range = self.test_data_provider.feature_range
        ref_idx = test_range[ref_frame]
 
        n_to_show = 1000
        
        idx_to_show = np.random.choice(ntotal, n_to_show - 1)
        idx_to_show = [ref_idx] + idx_to_show.tolist()  
        idx_to_show = np.asarray(idx_to_show, dtype=np.int).flatten()
        
        ref_pose =  pred_pose[...,ref_idx].reshape((-1,1),order='F')      
        pose_to_eval =gt_pose[...,idx_to_show]
        output_feature_name = 'fc_2' # <------------------Parameter
        output_layer_idx = self.get_layer_idx(output_feature_name)

        # do it once <------------- Maybe it can support multiple batch in the future
        data_dim = self.model_state['layers'][output_layer_idx]['outputs']
        print 'data_dim', data_dim
        
        cur_data = [np.require(np.tile(ref_pose, [1,n_to_show]), \
                               dtype=np.single,requirements='C'), \
                    np.require(pose_to_eval.reshape((-1,n_to_show),order='F'),\
                               dtype=np.single,requirements='C'), \
                    np.require(np.zeros((1,n_to_show),dtype=np.single), \
                               requirements='C'),
                    np.require(np.zeros((n_to_show,data_dim),dtype=np.single), \
                               requirements='C')]
        residuals = cur_data[1][...,0].reshape((-1,1),order='F') - cur_data[1]
        dp = self.test_data_provider
        mpjpe = dutils.calc_mpjpe_from_residual(residuals, dp.num_joints)

        gt_score = dp.calc_score(mpjpe, dp.mpjpe_factor/dp.max_depth,\
                              dp.mpjpe_offset/dp.max_depth).reshape((1,n_to_show)).flatten()
        self.libmodel.startFeatureWriter(cur_data, output_layer_idx)
        self.finish_batch()
        score = cur_data[-1].T
        print 'dim score', score.shape, 'dim gt_score', gt_score.shape
        score = score.flatten()
        # score = gt_score.flatten()
        def my_sort_f(k):
            if k == 0:
                return 10000000
            else:
                return score[k]
        sorted_idx = sorted(range(n_to_show), key=my_sort_f,reverse=True)
        s_to_show = [idx_to_show[k] for k in sorted_idx]
        sorted_score = np.asarray( [score[k] for k in sorted_idx])
        
        pose_to_plot = self.convert_relskel2rel(cur_data[1])
        sorted_pose = pose_to_plot[...,sorted_idx]
        class ScorePoseCanvas(ScoreCanvas):
            def __init__(self,data_dic):
                import iread.h36m_hmlpe as h36m
                ScoreCanvas.__init__(self,data_dic)
                self.pose_data = data_dic['pose_data']
                self.limbs = h36m.part_idx
                self.tmp = 0
            def show_image(self,ax):
                # ScoreCanvas.show_image(self,ax)
                # return
                import Image
                idx =self.cur_data_idx
                if idx == 0:
                    self.tmp = self.tmp + 1
                    if self.tmp == 1:
                        img = self.load_image(idx)
                        ax.imshow(np.asarray(img))
                        return
                print 'Current data idx %d ' % self.cur_data_idx
                # params = {'elev':-89, 'azim':-107}
                # params = {'elev':-69, 'azim':-107}
                params = {'elev':-81, 'azim':-91} # frontal view
                fig = plt.figure(100)
                from mpl_toolkits.mplot3d import Axes3D
                import imgproc
                # new_ax = self.fig.add_axes( rng_rel,projection='polar')
                new_ax = fig.add_subplot(111,projection='3d')
                imgproc.turn_off_axis(new_ax)
                cur_pose = self.pose_data[...,idx].reshape((3,-1),order='F')
                dutils.show_3d_skeleton(cur_pose.T,\
                                        self.limbs, params)
                xmin,xmax = np.min(cur_pose[0]),np.max(cur_pose[0])
                ymin,ymax = np.min(cur_pose[1]),np.max(cur_pose[1])
                zmin,zmax = np.min(cur_pose[2]),np.max(cur_pose[2])
                def extent(x,y,ratio):
                    x = x + (x-y) * ratio
                    y = y + (y-x) * ratio
                    return -0.4,0.4
                r = 0.1
                new_ax.set_xlim(extent(xmin,xmax,r))
                new_ax.set_ylim(extent(ymin,ymax,r))
                new_ax.set_ylim(extent(zmin,zmax,r))
                tmp_folder = '/public/sijinli2/ibuffer/2014-CVPR2015/tmp/images'
                save_path = iu.fullfile(tmp_folder, 'tmp_image.png')
                plt.savefig(save_path)
                img = Image.open(save_path)
                plt.close(100)
                img_arr = np.asarray(img)
                s = np.int(img_arr.shape[0]/5.0)
                e = np.int(img_arr.shape[0] - s)
                s  = 0
                e = img_arr.shape[0]
                img_arr = img_arr[s:e,:,:]
                ax.imshow(np.asarray(img_arr))
                # ax.plot([1,0,0],[0,1,0],[0,0,1])


        sc = ScorePoseCanvas({'x': np.asarray(range(len(idx_to_show))), 'y':sorted_score,\
                          'images_path': [images_path[k] for k in s_to_show], \
                          'pose_data':sorted_pose})
        sc.start()
        print 'max score is ' , sorted_score.max()
        gt_sort_idx = sorted(range(n_to_show), key=lambda k:gt_score[k], reverse=True)
        sorted_gt_score = np.asarray([gt_score[k] for k in gt_sort_idx])
        sorted_score_by_gt = [score[k] for k in gt_sort_idx]
        pl.plot(np.asarray(range(n_to_show)), sorted_gt_score, 'r', label='gt_score')
        pl.plot(np.asarray(range(n_to_show)), sorted_score_by_gt, 'g', label='pred_score')
        
    
    def start_SLP_server(self):
        """
        This function is used for structure learning project.
        It will read data from the inbox and produce file in outbox
        """
        import slp
        
        s = self.op.get_value('mode_params')
        if len(s) == 0:
            print 'No parameter received'
            params = {}
        else:
            l_p = self.parse_params(s)
            params = {'action':[np.int(l_p[0])]}
            if len(l_p) == 2:
                params['image_feature_layer_name'] = l_p[1]
        slpserver = slp.SLPServerTrial(self, params)
        slpserver.start()
    def get_backtrack_layer_list(self, layer_idx):
        """
        This function can be used for generating the back trackted layer list
        """
        res_list =  [self.model_state['layers'][layer_idx]]
        while ('inputLayers' in res_list[0]):
            ### If there are multiple input, take the 0-th layer
            l = res_list[0]['inputLayers'][0]
            if l['type'] == 'data':
                break
            res_list = [l] + res_list
        return res_list
    def start(self):
        self.op.print_values()
        if self.test_only:
            self.test_outputs += [self.get_test_error()]
            self.print_test_results()
            sys.exit(0)
        if self.mode:
            if self.mode == 'do-score-prediction':
                self.do_score_prediction()
            elif self.mode == 'slp-server':
                self.start_SLP_server()
            elif self.mode == 'mpjpe-eva':
                self.evaluate_output()
            elif self.mode == 'show-cost':
                self.plot_cost()
            elif self.mode == 'write-feature':
                self.write_feature()
        plt.show()
        sys.exit(0)
    @classmethod
    def convert_relskel2rel(cls, x):
        import dhmlpe_features as df
        return df.convert_relskel2rel(x)
    def convert_relskel2rel_eva(cls,x):
        import dhmlpe_features as df
        import humaneva_meta as hm
        return df.convert_relskel2rel_base(x, hm.limbconnection)
    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        for option in list(op.options):
            if option not in ('gpu', 'load_file', 'train_batch_range', 'test_batch_range', 'data_path', 'minibatch_size', 'layer_params', 'batch_size', 'test_only', 'test_one', 'shuffle_data', 'crop_one_border', 'external_meta_path'):
                op.delete_option(option)
        op.add_option('mode', 'mode', StringOptionParser, 'Determine what to do next')
        op.add_option('mode-params', 'mode_params', StringOptionParser, 'Determine what to do next', default='')
        op.add_option('save-path', 'save_path', StringOptionParser, 'Specify the path for results', default='')
        op.options['load_file'].default = None
        return op
    

if __name__ == "__main__":
    try:
        op = TestConvNet.get_options_parser()
        op, load_dic = IGPUModel.parse_options(op)
        model = TestConvNet(op, load_dic)
        model.start()
    except (UnpickleError, TestConvNetError, opt.GetoptError), e:
        print '-----------------'
        print "error"
        print e
        print '           '
