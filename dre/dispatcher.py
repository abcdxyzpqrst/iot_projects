import os
import logging
import multiprocessing
from multiprocessing import Pool
import time
import argparse
import yaml
import subprocess


module_logger = logging.getLogger(__name__)
print(module_logger)


#get args
parser = argparse.ArgumentParser()

parser.add_argument('--runfile', default='main.py')
parser.add_argument('--type', choices=['cpu', 'gpu'], default='gpu')
parser.add_argument('--n_workers', type=int, default=1)
args = parser.parse_args()

config_dir = 'choose_configs'

# Make GPU queue
manager = multiprocessing.Manager()
GPUqueue = manager.Queue()

runfileName = args.runfile

if args.type == 'cpu':
    n_rsc = args.n_workers
elif args.type == 'gpu':
    import torch
    n_rsc = torch.cuda.device_count()

for times in range(n_rsc):
    # how many jops run on each gpu ?
    GPUqueue.put(times)
    #GPUqueue.put(times)
    #GPUqueue.put(times)
    #GPUqueue.put(times)


def launch_experiment(x):

    ROOT_DIR = os.getcwd()
    config_name = x
    config_path = os.path.join(ROOT_DIR, config_dir)
    subprocess.run(args=['python3', os.path.join(ROOT_DIR, runfileName),
        "--config=" + os.path.join(config_path, config_name)])
    launch_experiment.GPUq.put(int(os.environ['CUDA_VISIBLE_DEVICES']))
    return x

#def launch_data_generation(x):
#    ROOT_DIR = os.getcwd()
#    config_name = x
#    config_path = os.path.join(ROOT_DIR, config_dir)
#    subprocess.run(args=['python3', os.path.join(ROOT_DIR, 'generate_synthetic.py'), os.path.join(config_path,config_name)])
#    return x


def distribute_gpu(q):
    launch_experiment.GPUq = q
    num = q.get()
    print("process id = {0}: using gpu {1}".format(os.getpid(), num))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(num)






ROOT_DIR = os.getcwd()



todo = next(os.walk(os.path.join(ROOT_DIR, config_dir)))[2]  #get yaml names


print(todo)


# Launch processes for data generation
'''
pool = Pool(processes=4)
a = pool.map(launch_data_generation, todo)
print("Data is ready.")
'''

print("Using {0} workers.".format(n_rsc))

# Launch processes for experiment
pool = Pool(processes=n_rsc, initializer=distribute_gpu, initargs=(GPUqueue,), maxtasksperchild=1)
a = pool.map(launch_experiment, todo)


