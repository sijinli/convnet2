"""
Usage
initWFunc=initweight.gwfc('path', checkpath)

initWfunc=initweight.constant_initializer(c)
"""
import numpy as np
from python_util.gpumodel import *
checkpoint_path_dic = {}
def prod(X):
    r = 1
    for x in X:
        r = r * x
    return r
class InitWeightException(Exception):
    pass

def get_layer_idx(layername, model_state):
    # Alex Krizhevsky's code
    try:
        layer_idx = [ l['name'] for l in model_state['layers']].index(layername)
    except ValueError:
        raise InitWeightException('Layer with name %s not defined' % layername)
    return layer_idx

def get_wb_from_checkpoints(type_name, name, idx, shape, params=None):
# params[0] = 'netid' | 'path'
# params[1] =    id   |checkpath
# params[2] =   Empty | name

# If params[2] is not given, it will copy the weights with the same name
# from the referenced network
# type_name = 'weights' or 'biases'
    if params is None or len(params) < 2:
        raise InitWeightException('Invalid parameers (none or incorrect len')
    if str(params[0]) == 'netid':
        if params[1] in checkpoint_path_dic:
            checkpath = checkpoint_path_dic[str(params[1])]
        else:
            raise InitWeightException('No network named %s is found' % str(params[1]))
    else:
        checkpath = str(params[1])
    filepath = os.path.join(checkpath, sorted(os.listdir(checkpath), key=alphanum_key)[-1])
    saved = unpickle(filepath)
    model_state = saved['model_state']
    if len(params) == 2:
        layername = name
    else:
        layername = str(params[2])
    print 'layer %s (Target) << layer %s (Source) =====\n'  % (name, layername)
    if type(model_state['layers']) is list:
        print '                  copy from convnet saved model'
        layer_idx = get_layer_idx(layername, model_state)
        layer = model_state['layers'][layer_idx]
    else:
        layer = model_state['layers'][layername]
    ### Currently I don't care about the idx's value
    #sidx = 0 if len(params) <= 3 else int( params[3])
    sidx = idx
    if type_name == 'weights':
        print '--------Current sidx = %d' % sidx
        if shape != layer[type_name][sidx].shape:
            raise InitWeightException('The shape %s doesn''t match %s' % (str(shape), str(layer[type_name][sidx].shape)))
        else:
            # if np.sum((np.isnan(layer[type_name][sidx])).flatten()) !=0:
            #     raise Exception('I find nan');
            return layer[type_name][sidx]
    else:
        # because there is only one biase matrix!!! Pay more attention
        if shape != layer[type_name].shape:
            raise InitWeightException('The shape %s doesn''t match %s' % (str(shape), str(layer[type_name].shape)))
        else:
            return layer[type_name]
def get_biases_from_checkpoints(name, shape, params = None):
    return get_wb_from_checkpoints('biases', name, 0, shape, params)
def get_weights_from_checkpoints(name, idx, shape, params = None):
    return get_wb_from_checkpoints('weights', name, idx, shape, params)
def gwfc(name,idx,shape, params=None):
    return get_weights_from_checkpoints(name, idx, shape, params)
def gbfc(name, shape, params= None):
    return get_biases_from_checkpoints(name,  shape, params)

###
def get_wb_from_dict(name, idx, shape, params):
    """
    name: is the layer name
    idx : is the idx in the matrix
    shape: the shape of the matrix to be initialized
    params[0]: path
    params[1]: 'weights' | 'biases'
    params[2]: name_source

    """
    print 'For name = %s ==========='% name
    d = unpickle(params[0])
    value_category = str(params[1])
    if prod(shape) != prod(d[params[2]][value_category].shape):
        raise InitWeightException('The target shape %s doesn''t match source shape %s' % (str(shape), str(d[params[2]][value_category].shape)))
    ncol = shape[1]
    return d[params[2]][value_category].reshape((-1,ncol),order='F')
def gwfd(name,idx,shape,params=None):
    """
    params= path [name_source]
    """
    if params is None:
        raise InitWeightException('Please specify the dict path ') 
    if len(params) == 1:
        params += ['weights', name]
    elif len(params) == 2:
        params = [params[0], 'weights', params[1]]
    return get_wb_from_dict(name, idx, shape, params)

def gbfd(name,shape,params=None):
    """
    params= path [name_source]
    """
    if params is None:
        raise InitWeightException('Please specify the dict path ') 
    if len(params) == 1:
        params += ['biases', name]
    elif len(params) == 2:
        params = [str(params[0]), 'biases', str(params[1])]
    return get_wb_from_dict(name, 0, shape, params)
def constant_initializer(name, idx, shape, params=None):
    # initialize the weights/biaes with constant value
    print params ,'<================================='
    if params is None:
        constant = 0
    else:
        constant = float(params[0])
    print 'Init to constant value %.6f' % constant
    t =  np.ones(shape,dtype=np.float32) * constant
    return t
def init_cont_biases(name, shape, params=None):
    return constant_initializer(name, -1, shape, params)
def init_cont_weights(name, idx, shape, params=None):
    return constant_initializer(name, idx, shape, params)
