# -*- coding: utf-8 -*-
"""
From project https://github.com/QuantumLiu/tf_gpu_manager, License: MIT

Created on Mon Aug  7 19:38:30 2017

@author: Quantum Liu
"""

import re
import os
import sys


def _get_timestr():
    import datetime
    dt = datetime.datetime.now()
    return '{}-{}-{}-{}'.format(dt.month, dt.day, dt.hour, dt.minute)


def check_gpus():
    '''
    GPU available check
    reference : http://feisky.xyz/machine-learning/tensorflow/gpu_list.html
    '''
# =============================================================================
#     all_gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
# =============================================================================
    out = os.popen('nvidia-smi --query-gpu=index --format=csv,noheader').readlines()
    if len(out) < 1:
        return False
    first_gpus = out[0].strip()
    if not first_gpus=='0':
        print('This script could only be used to manage NVIDIA GPUs,but no GPU found in your device')
        return False
    elif not 'NVIDIA System Management' in os.popen('nvidia-smi -h').read():
        print("'nvidia-smi' tool not found.")
        return False
    return True


def parse(line,qargs):
    '''
    line:
        a line of text
    qargs:
        query arguments
    return:
        a dict of gpu infos
    Pasing a line of csv format text returned by nvidia-smi
    解析一行nvidia-smi返回的csv格式文本
    '''
    numeric_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']#可计数的参数
    power_manage_enable=lambda v:(not 'Not Support' in v)#lambda表达式，显卡是否滋瓷power management（笔记本可能不滋瓷）
    to_numeric=lambda v:float(v.upper().strip().replace('MIB','').replace('W',''))#带单位字符串去掉单位
    process = lambda k,v:((int(to_numeric(v)) if power_manage_enable(v) else 1) if k in numeric_args else v.strip())
    dct = dict(zip(qargs, line.strip().split(',')))
    if any(v.find('Error') != -1 for _,v in dct.items()):
        print('======== Skipping faulty device', line, file=sys.stderr)
        return None
    return {k: process(k,v) for k,v in dct.items()}


def query_gpu(qargs=[]):
    '''
    qargs:
        query arguments
    return:
        a list of dict
    Querying GPUs infos
    查询GPU信息
    '''
    qargs = ['index', 'gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit', 'fan.speed'] + qargs
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()
    results = [parse(line,qargs) for line in results]
    return [r for r in results if r is not None]


def by_power(d):
    '''
    helper function fo sorting gpus by power
    '''
    power_infos=(d['power.draw'],d['power.limit'])
    if any(v==1 for v in power_infos):
        print('Power management unable for GPU {}'.format(d['index']))
        return 1
    return float(d['power.draw'])/d['power.limit']


class GPUManager():
    '''
    qargs:
        query arguments
    A manager which can list all available GPU devices
    and sort them and choice the most free one.Unspecified 
    ones pref.
    GPU设备管理器，考虑列举出所有可用GPU设备，并加以排序，自动选出
    最空闲的设备。在一个GPUManager对象内会记录每个GPU是否已被指定，
    优先选择未指定的GPU。
    '''
    def __init__(self,qargs=[]):
        '''
        '''
        self.qargs=qargs
        self.gpus=query_gpu(qargs)
        for gpu in self.gpus:
            gpu['specified']=False
        self.gpu_num=len(self.gpus)

    def _sort_by_memory(self,gpus,by_size=False):
        if by_size:
            print('Sorted by free memory size')
            return sorted(gpus,key=lambda d:d['memory.free'],reverse=True)
        else:
            print('Sorted by free memory rate')
            return sorted(gpus,key=lambda d:float(d['memory.free'])/ d['memory.total'],reverse=True)

    def _sort_by_power(self,gpus):
        return sorted(gpus,key=by_power)
    
    def _sort_by_custom(self,gpus,key,reverse=False,qargs=[]):
        if isinstance(key,str) and (key in qargs):
            return sorted(gpus,key=lambda d:d[key],reverse=reverse)
        if isinstance(key,type(lambda a:a)):
            return sorted(gpus,key=key,reverse=reverse)
        raise ValueError("The argument 'key' must be a function or a key in query args,please read the documention of nvidia-smi")

    def auto_choice(self,mode=0):
        '''
        mode:
            0:(default)sorted by free memory size
        return:
            a TF device object
        Auto choice the freest GPU device,not specified
        ones 
        自动选择最空闲GPU
        '''
        for old_infos,new_infos in zip(self.gpus,query_gpu(self.qargs)):
            old_infos.update(new_infos)
        unspecified_gpus=[gpu for gpu in self.gpus if not gpu['specified']] or self.gpus
        
        if mode==0:
            print('Choosing the GPU device has largest free memory...')
            chosen_gpu=self._sort_by_memory(unspecified_gpus,True)[0]
        elif mode==1:
            print('Choosing the GPU device has highest free memory rate...')
            chosen_gpu=self._sort_by_power(unspecified_gpus)[0]
        elif mode==2:
            print('Choosing the GPU device by power...')
            chosen_gpu=self._sort_by_power(unspecified_gpus)[0]
        else:
            print('Given an unaviliable mode,will be chosen by memory')
            chosen_gpu=self._sort_by_memory(unspecified_gpus)[0]
        chosen_gpu['specified']=True
        index=chosen_gpu['index']
        print('Using GPU {i}:\n{info}'.format(i=index,info='\n'.join([str(k)+':'+str(v) for k,v in chosen_gpu.items()])))
        return tf.device('/gpu:{}'.format(index))


def get_devices(n_gpus):
    if not check_gpus():
        return [-1]
    gm = GPUManager()
    gpus = gm._sort_by_memory(gm.gpus, True)[:n_gpus]
    for g in gpus:
        print('- Selected GPU {i}:\n\t{info}'.format(i=g['index'],
            info='\n\t'.join([str(k)+':'+str(v) for k,v in g.items()])))
    return [i['index'] for i in gpus]


def safe_path_str(raw):
    return re.sub('[^\w\-_\.]', '_', raw)


def task_id(task):
    return hex(abs(hash(task.cmd) % int(1e10)))[2:]


import threading


class AtomicCounter:

    def __init__(self, init=0):
        self.value = init
        self._lock = threading.Lock()

    def inc(self):
        with self._lock:
            self.value += 1
            return self.value

