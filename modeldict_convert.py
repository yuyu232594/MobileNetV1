import json
import tensorflow as tf
import os
from MobileNet.mobilenet_v1 import MobileNet
import numpy as np
import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
CHECKPOINT_PATH='root/mobilenet_v1_1.0_224.ckpt'

# write the json file
def new_dict(checkpoint_path,json_path):
    reader=tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
    weights_shape =reader.get_variable_to_shape_map()
    print('the layer',weights_shape['MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean'])
    length=len(weights_shape['MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean'])
    # print(length)
    if not os.path.exists(json_path):
        weights_small = {n: 1 for (n, _) in reader.get_variable_to_shape_map().items()}
        keys_list=list(weights_small.keys())
        for key_ in keys_list:
            if "/ExponentialMovingAverage" in key_:
                del weights_small[key_]
            elif "/RMSProp" in key_:
                del weights_small[key_]
        with open(json_path, 'w') as writer:
            json.dump(weights_small, fp=writer, sort_keys=True)
    else:
        print('the json file has been write!')

# get convBn_dict
def get_convbn_convert_dict(layer_num):
    convert_dict={
        'mobelnet.'+str(layer_num)+'.convBn.0.weight':'MobilenetV1/Conv2d_'+str(layer_num)+'/weights',
        'mobelnet.'+str(layer_num)+'.convBn.1.weight':'MobilenetV1/Conv2d_'+str(layer_num)+'/BatchNorm/beta',
        'mobelnet.'+str(layer_num)+'.convBn.1.bias':'MobilenetV1/Conv2d_'+str(layer_num)+'/BatchNorm/gamma',
        'mobelnet.'+str(layer_num)+'.convBn.1.running_mean':'MobilenetV1/Conv2d_'+str(layer_num)+'/BatchNorm/moving_mean',
        'mobelnet.'+str(layer_num)+'.convBn.1.running_var':'MobilenetV1/Conv2d_'+str(layer_num)+'/BatchNorm/moving_variance'
    }
    return convert_dict

# get depthWise_dict
def get_dpwise_convert_dict(layer_num):
    convert_dict={
        'mobelnet.'+str(layer_num)+'.convDepthwise.0.weight':
            'MobilenetV1/Conv2d_'+str(layer_num)+'_depthwise/depthwise_weights',
        'mobelnet.'+str(layer_num)+'.convDepthwise.1.weight':
            'MobilenetV1/Conv2d_'+str(layer_num)+'_depthwise/BatchNorm/beta',
        'mobelnet.'+str(layer_num)+'.convDepthwise.1.bias':
            'MobilenetV1/Conv2d_'+str(layer_num)+'_depthwise/BatchNorm/gamma',
        'mobelnet.'+str(layer_num)+'.convDepthwise.1.running_mean':
            'MobilenetV1/Conv2d_'+str(layer_num)+'_depthwise/BatchNorm/moving_mean',
        'mobelnet.'+str(layer_num)+'.convDepthwise.1.running_var':
            'MobilenetV1/Conv2d_'+str(layer_num)+'_depthwise/BatchNorm/moving_variance',
        'mobelnet.'+str(layer_num)+'.convDepthwise.3.weight':
            'MobilenetV1/Conv2d_'+str(layer_num)+'_pointwise/weights',
        'mobelnet.'+str(layer_num)+'.convDepthwise.4.weight':
            'MobilenetV1/Conv2d_'+str(layer_num)+'_pointwise/BatchNorm/beta',
        'mobelnet.' + str(layer_num) + '.convDepthwise.4.bias':
            'MobilenetV1/Conv2d_' + str(layer_num) + '_pointwise/BatchNorm/gamma',
        'mobelnet.' + str(layer_num) + '.convDepthwise.4.running_mean':
            'MobilenetV1/Conv2d_' + str(layer_num) + '_pointwise/BatchNorm/moving_mean',
        'mobelnet.' + str(layer_num) + '.convDepthwise.4.running_var':
            'MobilenetV1/Conv2d_' + str(layer_num) + '_pointwise/BatchNorm/moving_variance'
    }
    return convert_dict

# get conversion_dict
def get_model_dict(layers_num):
    merge = lambda dict1, dict2: {**dict1, **dict2}
    conversion_table = {}
    convBn_dict=get_convbn_convert_dict(0)
    conversion_table=merge(conversion_table,convBn_dict)
    for i in range(1,layers_num):
        dpWise_dict=get_dpwise_convert_dict(i)
        conversion_table=merge(conversion_table,dpWise_dict)
    # load_parameter(CHECKPOINT_PATH,conversion_table)
    return conversion_table
def write_json(conversion_table,json_path):
    if not os.path.exists(json_path):
        with open(json_path, 'w') as writer:
            json.dump(conversion_table, fp=writer, sort_keys=True)
    else:
        print('the conversion table has been wirten!')

def load_parameter(conversion_table):
    module=MobileNet()
    original_model_dict=module.state_dict()
    pth_list=list(conversion_table.keys())
    ckpt_list=list(conversion_table.values())
    assert len(pth_list)==len(ckpt_list) ,('the length is not right!')
    reader=tf.compat.v1.train.NewCheckpointReader(CHECKPOINT_PATH)
    for i,ckpt_name in enumerate(ckpt_list):
        ckpt_name_value=tf.compat.v1.train.load_variable(CHECKPOINT_PATH,ckpt_name)
        if 'Conv2d' in ckpt_name and 'weights' in ckpt_name:
            ckpt_name_value=np.transpose(ckpt_name_value,(3,2,0,1))
            if 'depthwise' in ckpt_name:
                ckpt_name_value=np.transpose(ckpt_name_value,(1,0,2,3))
        elif 'BatchNorm' in ckpt_name and ckpt_name_value.ndim==1:
            # ckpt_name_value=np.transpose(ckpt_name_value)
            ckpt_name_value=ckpt_name_value
        pytorch_dict_key=pth_list[i]
        original_model_dict[pytorch_dict_key].data=torch.from_numpy(ckpt_name_value)

    torch.save(original_model_dict,'root/MobileNet/tf_to_torch.pth')
    return original_model_dict

if __name__ == '__main__':
    conversion_table=get_model_dict(14)
    dic_mobel=load_parameter(conversion_table)
    # print(dic_mobel['mobelnet.1.convDepthwise.0.weight'].shape)
    model=MobileNet()
    model.load_state_dict(torch.load('root/MobileNet/tf_to_torch.pth'))
    pretrained_dict=model.state_dict()
    print(pretrained_dict['mobelnet.1.convDepthwise.0.weight'].shape)

