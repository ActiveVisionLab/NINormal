import os
import numpy as np

def get_pt_clouds_path(datafolder_read, obj_names, noise_type, noise_intensity='none'):
    if noise_type is 'none':
        result = [os.path.join(datafolder_read, name) for name in obj_names]
    elif noise_type is 'gradient':
        result = [os.path.join(datafolder_read, name + '_ddist_minmax') for name in obj_names]
    elif noise_type is 'striped':
        result = [os.path.join(datafolder_read, name + '_ddist_minmax_layers') for name in obj_names]
    elif noise_type is 'white':
        result = [os.path.join(datafolder_read, name + '_noise_white_' + '{0:.2e}'.format(float(noise_intensity))) for name in obj_names]
    elif noise_type is 'brown':
        result = [os.path.join(datafolder_read, name + '_noise_brown_' + '{0:.2e}'.format(float(noise_intensity))) for name in obj_names]
    else:
        print('noise type not implemented. exit now')
        exit()
    return result
