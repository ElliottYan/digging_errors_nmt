root=`your root path`
script_path=$root/../model_errors
ref_file=$root/../model_errors/test.de.tok.detok

# dfs directories
dfs_input_dir_1=$root/../model_errors/ende_dfstopk_checkpoints_beam_10
dfs_input_dir_2=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.nolb
dfs_input_dir_3=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.para_ft
dfs_input_dir_4=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.big
dfs_input_dir_5=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.dp0.2
dfs_input_dir_6=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.dp0.3

# beam directories
beam_input_dir_1=$root/../model_errors/ende_min_heap_beam_checkpoints_beam_10
beam_input_dir_2=$root/../archive/ende_beam_wmt14.en-de.transformer.nolb
beam_input_dir_3=$root/../archive/ende_beam_wmt14.en-de.transformer.para_ft
beam_input_dir_4=$root/../archive/ende_beam_wmt14.en-de.transformer.big
beam_input_dir_5=$root/../archive/ende_beam_wmt14.en-de.transformer.dp0.2
beam_input_dir_6=$root/../archive/ende_beam_wmt14.en-de.transformer.dp0.3

beam=5

tmp_dir=$root/tmp
mkdir -p $tmp_dir 
export TMPDIR=$tmp_dir

cd $root

export LC_ALL=en_US.UTF-8

python3 $root/compute_exact_match.py --dfs_input_dir $dfs_input_dir_1 --beam_input_dir $beam_input_dir_1 --beam 5
# python3 $root/compute_exact_match.py --dfs_input_dir $dfs_input_dir_2 --beam_input_dir $beam_input_dir_2 --beam 10
# python3 $root/compute_exact_match.py --dfs_input_dir $dfs_input_dir_3 --beam_input_dir $beam_input_dir_3 --beam 10
# python3 $root/compute_exact_match.py --dfs_input_dir $dfs_input_dir_4 --beam_input_dir $beam_input_dir_4 --beam 5
# python3 $root/compute_exact_match.py --dfs_input_dir $dfs_input_dir_5 --beam_input_dir $beam_input_dir_5 --beam 5
# python3 $root/compute_exact_match.py --dfs_input_dir $dfs_input_dir_6 --beam_input_dir $beam_input_dir_6 --beam 5
