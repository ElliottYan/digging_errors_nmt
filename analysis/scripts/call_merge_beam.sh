root=`your root path`

input_file=$root/../../results/sp_eval/greedy_reg/test.out
output_file=$input_file.merge

python3 $root/merge_outputs.py $input_file $output_file 10