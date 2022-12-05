import sys
import os
input_dir = sys.argv[1]
splits = 10
beam = 100

output_dirs = [input_dir + '.{}'.format(i) for i in range(splits)]
    
for i, file_name in os.listdir(input_dir):
    cur_path = os.path.join(input_dir, file_name)
    
    