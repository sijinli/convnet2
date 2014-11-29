# Copyright (c) 2013, Li Sijin (lisijin7@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from ibasic_convdata import *
##
import dhmlpe
import indicatormap
###

class DHMLPEDataProviderError(Exception):
    pass
class CroppedDHMLPEJointDataProvider(CroppedImageDataProvider):
    """
    This data provider will provide
        [data, joints, joints_indicator_map]
        joints is the relateive joints
    """
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedImageDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        jt_filter_size = self.batch_meta['joint_indicator_map']['filter_size']
        jt_stride = self.batch_meta['joint_indicator_map']['stride']
        if ('joint_indmap_type' not in dp_params):
            ## default 
            self.joint_indmap = indicatormap.IndicatorMap(self.input_image_dim, \
                                                        jt_filter_size, \
                                                        jt_stride,
                                                        create_lookup_table=True)
        else:
            self.joint_indmap = indicatormap.IndMapDic[dp_params['joint_indmap_type']](\
                                                        self.input_image_dim, \
                                                        jt_filter_size, \
                                                        jt_stride,
                                                        create_lookup_table=True) 
        self.num_joints = self.batch_meta['num_joints']
        self.feature_name_3d = 'Relative_Y3d_mono_body'
        if 'max_depth' in self.batch_meta:
            self.max_depth = self.batch_meta['max_depth']
        else:
            self.max_depth = 1200 
        # self.max_depth = 1
    def get_next_batch(self):
        if self.data_dic is None or len(self.batch_range) > 1:
            self.data_dic = self.get_batch(self.curr_batchnum)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        ndata = self.data_dic['data'].shape[-1]
        alldata = [np.require(self.data_dic['data'].reshape((-1,ndata),order='F'),dtype=np.single, requirements='C')]
        ## Add joints here
        alldata += [np.require(self.data_dic['joints_3d'].reshape((-1,ndata),order='F'),dtype=np.single, requirements='C')/self.max_depth]
        ## ADD joint indicator here
        alldata += [np.require(self.data_dic['joints_indicator_map'].reshape((-1,ndata),order='F'),dtype=np.single, requirements='C')]
        return epoch, batchnum, alldata
    def get_batch(self, batch_num):
        dic = CroppedImageDataProvider.get_batch(self,batch_num)
        ## ADD joint data here
        #print self.batch_meta['occ_body'].shape
        #print max(dic['cur_batch_indexes']), min(dic['cur_batch_indexes'])
        # Require square image
        dic['occ_body'] = np.concatenate(map(lambda x:self.batch_meta['occ_body'][...,x].reshape((-1,1)), dic['cur_batch_indexes']), axis=1)        
        dic['joints_2d'] = np.concatenate(map(lambda x:self.batch_meta['Y2d_bnd_body'][...,x].reshape((-1,1)), dic['cur_batch_indexes']), axis=1)
        dic['joints_3d'] = np.concatenate(map(lambda x:self.batch_meta[self.feature_name_3d][...,x].reshape((-1,1),order='F'),dic['cur_batch_indexes']),axis=1)
        ## generate joint indicator map
        pts = dic['joints_2d'].reshape((2,-1),order='F')
        #  subtract offset to get the cropped coordinates 
        offset = np.tile(np.concatenate([np.asarray(self.cur_offset_c).reshape((1,-1)), np.asarray(self.cur_offset_r).reshape((1,-1))],axis=0), [self.num_joints,1]).reshape((2,-1),order='F')
        pts = pts - offset # 0-index coordinates
        dic['joints_2d'] = pts.reshape((2*self.num_joints, -1),order='F')
        allmaps = self.joint_indmap.get_joints_indicatormap(pts.T)
        mdim = self.joint_indmap.mdim
        ndata = len(dic['cur_batch_indexes'])
        dic['joints_indicator_map'] = allmaps
        return dic
    def get_plottable_data(self, imgdata):
        ## it is different with base class's method
        # just to be consistent with previous testconvnet.py functions
        # Without transpose and scaling
        ndata = imgdata.shape[-1]
        dimX = imgdata.shape[0]
        res = imgdata.copy() +self.cropped_mean_image.reshape((dimX,1),order='F')
        imgdim = list(self.input_image_dim) + [ndata]
        return res.reshape(imgdim, order='F')
    def get_data_dims(self,idx=0):
        if idx == 0:
            return iprod(self.input_image_dim)
        elif idx == 1:
            return self.num_joints * 3
        else:
            return iprod(self.joint_indmap.mdim) * self.num_joints


        
class CroppedDHMLPEJointOccDataProvider(CroppedDHMLPEJointDataProvider):
    def get_batch(self, batch_num):
        dic = CroppedDHMLPEJointDataProvider.get_batch(self, batch_num)
        dic['joints_indicator_map'][..., dic['occ_body']] = False
        return dic


class CroppedDHMLPEDepthJointDataProvider(CroppedImageDataProvider):
    """
    This data provider will output:
      [data, joints, depth_joints_indicator_map]
    """
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedImageDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        jt_filter_size2d = self.batch_meta['joint_indicator_map']['filter_size']
        jt_stride2d = self.batch_meta['joint_indicator_map']['stride']
        self.max_depth = self.batch_meta['max_depth']
        ## 
        dimz = self.max_depth * 2
        ## Require those fields
        win_z = self.batch_meta['win_z']
        stride_z = self.batch_meta['stride_z']
        depthdim = [self.input_image_dim[1], self.input_image_dim[0], dimz]
        self.depth_joint_indmap = indicatormap.DepthIndicatorMap(depthdim, jt_filter_size2d, jt_stride2d, win_z, stride_z, create_lookup_table=True)
        self.num_joints = self.batch_meta['num_joints']
    def get_next_batch(self):
        if self.data_dic is None or len(self.batch_range) > 1:
            self.data_dic = self.get_batch(self.curr_batchnum)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        ndata = self.data_dic['data'].shape[-1]
        alldata = [np.require(self.data_dic['data'].reshape((-1,ndata),order='F'),dtype=np.single, requirements='C')]
        ## Add joints here
        alldata += [np.require(self.data_dic['joints_3d'].reshape((-1,ndata),order='F'),dtype=np.single, requirements='C')/self.max_depth]
        ## ADD depth joint indicator here
        alldata += [np.require(self.data_dic['depth_joints_indicator_map'].reshape((-1,ndata),order='F'),dtype=np.single, requirements='C')]
        return epoch, batchnum, alldata
    def get_batch(self, batch_num):
        dic = CroppedImageDataProvider.get_batch(self,batch_num)
        dic['occ_body'] = np.concatenate(map(lambda x:self.batch_meta['occ_body'][...,x].reshape((-1,1)), dic['cur_batch_indexes']), axis=1)
        dic['joints_2d'] = np.concatenate(map(lambda x:self.batch_meta['Y2d_bnd_body'][...,x].reshape((-1,1)), dic['cur_batch_indexes']), axis=1)        
        dic['joints_3d'] = np.concatenate(map(lambda x:self.batch_meta['Relative_Y3d_mono_body'][...,x].reshape((-1,1)),dic['cur_batch_indexes']),axis=1)
        ## Get rectified 2d coordinates
        pts = dic['joints_2d'].reshape((2,-1),order='F')
        offset = np.tile(np.concatenate([np.asarray(self.cur_offset_c).reshape((1,-1)), np.asarray(self.cur_offset_r).reshape((1,-1))],axis=0), [self.num_joints,1]).reshape((2,-1),order='F')
        pts = pts - offset # 0-index coordinates
        dic['joints_2d'] = pts.reshape((self.num_joints*2, -1),order='F')
        depth_pts = np.concatenate([pts, dic['joints_3d'].reshape((3,-1))[2,:].reshape((1,-1),order='F')],axis=0).T        
        allmaps = self.depth_joint_indmap.get_joints_indicatormap(depth_pts)
        dic['depth_joints_indicator_map'] = allmaps
        return dic
    def get_plottable_data(self, imgdata):
        ## it is different with base class's method
        # just to be consistent with previous testconvnet.py functions
        # Without transpose and scaling
        ndata = imgdata.shape[-1]
        dimX = imgdata.shape[0]
        res = imgdata.copy() +self.cropped_mean_image.reshape((dimX,1),order='F')
        imgdim = list(self.input_image_dim) + [ndata]
        return res.reshape(imgdim, order='F')
    def get_data_dims(self,idx=0):
        if idx == 0:
            return iprod(self.input_image_dim)
        elif idx == 1:
            return self.num_joints * 3
        else:
            return iprod(self.depth_joint_indmap.mdim) * self.num_joints

class CroppedDHMLPERelSkelJointDataProvider(CroppedDHMLPEJointDataProvider):
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedDHMLPEJointDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.feature_name_3d = 'RelativeSkel_Y3d_mono_body'
class CroppedDHMLPERelSkelJointLenDataProvider(CroppedDHMLPERelSkelJointDataProvider):
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedDHMLPERelSkelJointDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)    
    def get_next_batch(self):
        epoch, batchnum, alldata = CroppedDHMLPEJointDataProvider.get_next_batch(self)
        # Need to add Limb_length_3d
        alldata += [np.require(self.data_dic['Limb_length_3d'], dtype=np.single, requirements='C')/self.max_depth]
        return epoch, batchnum, alldata
    def get_batch(self, batch_num):
        dic = CroppedDHMLPERelSkelJointDataProvider.get_batch(self, batch_num)
        ndata = len(dic['cur_batch_indexes'])
        dic['Limb_length_3d'] = self.batch_meta['Limb_length_3d'][...,dic['cur_batch_indexes']].reshape((-1,ndata),order='F')
        return dic
    def get_data_dims(self,idx=0):
        if idx == 0:
            return iprod(self.input_image_dim)
        elif idx == 1:
            return self.num_joints * 3
        elif idx == 2:
            return iprod(self.joint_indmap.mdim) * self.num_joints
        else:
            return self.num_joints - 1

        
