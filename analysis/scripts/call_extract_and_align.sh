root`your root path`
script_path=$root/../model_errors
# ref_file=$root/../model_errors/test.de.tok.detok
# src_file=$root/../model_errors/test.en
src_file=$root/../../checkpoints/wmt14.en-de.transformer_new/test.en

# dfs directories
dfs_input_dir_1=$root/../archive/ende_dfstopk_wmt14.en-de.transformer_new_beam_10_bpe

output_prefix=$root/../archive/sp_input/all/ende

beam=5

tmp_dir=$root/tmp
mkdir -p $tmp_dir 
export TMPDIR=$tmp_dir

cd $script_path

export LC_ALL=en_US.UTF-8

python3 $root/extra_and_align.py --dfs_input_dir $dfs_input_dir_1 --source_file $src_file --beam 5 --output_prefix $output_prefix
# python3 $root/compute_exact_match.py --dfs_input_dir $dfs_input_dir_2 --beam_input_dir $beam_input_dir_2 --beam 10
# python3 $root/compute_exact_match.py --dfs_input_dir $dfs_input_dir_3 --beam_input_dir $beam_input_dir_3 --beam 10
# python3 $root/compute_exact_match.py --dfs_input_dir $dfs_input_dir_4 --beam_input_dir $beam_input_dir_4 --beam 5
# python3 $root/compute_exact_match.py --dfs_input_dir $dfs_input_dir_5 --beam_input_dir $beam_input_dir_5 --beam 5
# python3 $root/compute_exact_match.py --dfs_input_dir $dfs_input_dir_6 --beam_input_dir $beam_input_dir_6 --beam 5
