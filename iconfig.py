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
from dhmlpe_convdata import *
from python_util.options import *
def register_data_provider(DataProvider):
    DataProvider.register_data_provider('croppeddhmlperelskeljt', 'CROPPEDDHMLPERELSKELJOINTDATAPROVIDER', CroppedDHMLPERelSkelJointDataProvider)
    DataProvider.register_data_provider('memfeat', 'MEMORYFEATUREDATAPROVIDER', MemoryFeatureDataProvider)
    DataProvider.register_data_provider('spsimple', 'SPSIMPLEDATAPROVIDER', SPSimpleDataProvider)
def add_options(op):
    op.add_option("shuffle-data", "shuffle_data", IntegerOptionParser, "whether to shullfe the data", default=0)
    op.add_option("batch-size", "batch_size", IntegerOptionParser, "Determine how many data can be loaded in a batch. Note: only valid for data providing loading images directly", default=-1) 
    op.add_option("crop-border", "crop_border", IntegerOptionParser, "Cropped DP: crop border size", default=-1, set_once=True)
    op.add_option("crop-one-border", "crop_one_border", IntegerOptionParser, "Cropped side: crop  size", default=-1, set_once=True)
    
def add_dp_params(dp_params, op):
    dp_params['crop_border'] = op.get_value('crop_border')
    dp_params['batch_size'] = op.get_value('batch_size')
    dp_params['crop_one_border'] = op.get_value('crop_one_border')
    try:
        dp_params['shuffle_data'] = op.get_value('shuffle_data')
    except:
        dp_params['shuffle_data'] = 0
