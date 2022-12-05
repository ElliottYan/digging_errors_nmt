root=`your root path`
script_path=$root/../model_errors
ref_file=$root/../model_errors/test.de.tok.detok

sig=wmt14.en-de.transformer
beam_dir=$root/../model_errors/ende_beam_$sig
dfs_dir=$root/../model_errors/ende_dfstopk_$sig

beam=5

tmp_dir=$root/tmp
mkdir -p $tmp_dir 
export TMPDIR=$tmp_dir

cd $root

python3 $root/compute_edit_distances.py --beam_input_dir $beam_dir --dfs_input_dir $dfs_dir --reference_file $ref_file --beam $beam --script_path $script_path