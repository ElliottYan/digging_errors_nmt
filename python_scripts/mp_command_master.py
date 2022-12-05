import argparse
import sys
import subprocess
import copy
import tempfile
import subprocess
import os
import re
import random
import time
import multiprocessing as mp
from collections import defaultdict
from itertools import repeat
from functools import partial

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        usage="mp_command_master.py [<args>] [-h | --help]"
    )

    # in moses format
    parser.add_argument("--n_gpu", type=int, default=8,
                        help="Number of gpus to use")
    parser.add_argument("--per_gpu_process", type=int, default=5,
                        help="Number of workers to place on one gpu")
    parser.add_argument("--command_file", type=str, default="",
                        help="File that contains all commands.")
    parser.add_argument("--shuffle_commands", action='store_true', help="Shuffle the commands.")

    return parser.parse_args(args)

# def call_command_test(command, resource_q, resource_item):
#     import time
#     sleep_time = 5 if resource_item[0] > 4 else 10
#     # sleep_time = random.randint(0,5)
#     # print('Sleep time: {}'.format(sleep_time))
#     print('Resource : {}'.format(resource_item))
#     time.sleep(sleep_time)
#     resource_q.put(resource_item)
#     return sleep_time

def call_command(command, resource_q, resource_item):
    gpu_id, per_idx = resource_item
    cuda_envs = 'CUDA_VISIBLE_DEVICES={}'.format(gpu_id)
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # command = 'ls -l'
    # cuda_command = "{} {}".format(cuda_envs, command)
    # print(cuda_command)
    cuda_command_list = command.split(' ')
    # subprocess.run('echo $CUDA_VISIBLE_DEVICES', env=my_env)
    subprocess.run(cuda_command_list, env=my_env)
    resource_q.put(resource_item)
    return True
    
def main(args):
    if args.command_file != "":
        with open(args.command_file, 'r', encoding='utf8') as f:
            command_lines = f.readlines()
        # assert len(command_lines) > args.n_gpu * args.per_gpu_process
        commands = [line.strip() for line in command_lines]
    else:
        commands = [1] * 100
    
    resource_q = mp.Queue()
    for i in range(args.n_gpu):
        for j in range(args.per_gpu_process):
            resource_q.put((i, j))

    if args.shuffle_commands is True:
        import random
        random.shuffle(commands)

    all_process = []
    while commands:
        print(resource_q.qsize())
        item = resource_q.get(block=True)
        command = commands.pop(0)
        p = mp.Process(target=call_command, args=(command, resource_q, item))

        all_process.append(p)
        p.start()
    
    for p in all_process:
        while p.is_alive is True:
            time.sleep(0.5)
            
    return 

def get_empty_gpus(gpus, pg):
    for gpu in gpus:
        if gpu >= pg:
            continue
        else:
            return gpu
    return -1

if __name__ == "__main__":
    args = parse_args()
    main(args)
