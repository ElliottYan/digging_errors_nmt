root=`your root path`
script_path=$root/../model_errors
ref_file=$root/../model_errors/test.de.tok.detok

sig=wmt14.en-de.transformer_new
beam_dir=$root/../archive/ende_beam_$sig/ende_beam_$sig
# beam_dir=$root/../archive/ende_min_heap_beam_$sig/ende_min_heap_beam_$sig
dfs_dir=$root/../archive/ende_dfstopk_$sig/ende_dfstopk_$sig

nla=5 # dfs
nlb=5 # beam

tmp_dir=$root/tmp
mkdir -p $tmp_dir 
export TMPDIR=$tmp_dir

cd $root

export LC_ALL=en_US.UTF-8

python3 $root/compute_search_errors.py --beam_input_dir $beam_dir --dfs_input_dir $dfs_dir --reference_file $ref_file --num_lines_a $nla --num_lines_b $nlb --script_path $script_path