# *_*coding:utf-8 *_*
import os
import sys
import socket

import torch

## gain linux ip
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(('10.0.0.1',8080))
        ip= s.getsockname()[0]
    finally:
        s.close()
    return ip

### Change the following DIRs to your own data directory before use. ###

DATASET_ROOT = os.environ.get(
    "PEROUTER_DATASET_ROOT",
    '/home/xuanqin/Comp/datasets/ComP_clean',
)

DATA_DIR = { 
	'CMUMOSI': os.path.join(DATASET_ROOT, 'CMUMOSI'),
	'CMUMOSEI': os.path.join(DATASET_ROOT, 'CMUMOSEI'),
	'IEMOCAPSix': os.path.join(DATASET_ROOT, 'IEMOCAP'),
	'IEMOCAPFour': os.path.join(DATASET_ROOT, 'IEMOCAP'),
}
SAVED_ROOT = os.environ.get(
	'PEROUTER_SAVED_ROOT',
	'/home/xuanqin/PERouter/runs/comP_saved/',
)


### Your don't need to change the following. ###
PATH_TO_FEATURES = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'features'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'features'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'features'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'features'),
}
PATH_TO_LABEL = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'CMUMOSI_features_raw_2way.pkl'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'CMUMOSEI_features_raw_2way.pkl'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'IEMOCAP_features_raw_6way.pkl'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'IEMOCAP_features_raw_4way.pkl'),
}


DATA_DIR = os.path.join(SAVED_ROOT, 'data')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
NPZ_DIR = os.path.join(SAVED_ROOT, 'npz')


