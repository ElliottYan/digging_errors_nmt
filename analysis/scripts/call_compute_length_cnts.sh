root=`your root path`
script_path=$root/../model_errors
ref_file=$root/../model_errors/test.de.tok.detok

# sig=wmt14.en-de.transformer
# beam_dir=$root/../model_errors/ende_beam_$sig
# dfs_dir=$root/../model_errors/ende_dfstopk_$sig

# dfs directories
dfs_input_dir_1=$root/../archive/ende_dfstopk_wmt14.en-de.transformer_new_beam_100
# wmt14.en-de.transformer.12+6/
# wmt14.en-de.transformer.18+6/
# dfs_input_dir_2=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.768_beam_10
# dfs_input_dir_3=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.18+6_beam_10
# dfs_input_dir_4=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.big
# dfs_input_dir_5=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.dp0.2
# dfs_input_dir_6=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.dp0.3

# beam directories
# beam_input_dir_1=$root/../archive/ende_beam_wmt14.en-de.transformer
# beam_input_dir_2=$root/../archive/ende_beam_wmt14.en-de.transformer.nolb
# beam_input_dir_3=$root/../archive/ende_beam_wmt14.en-de.transformer.para_ft
# beam_input_dir_4=$root/../archive/ende_beam_wmt14.en-de.transformer.big
# beam_input_dir_5=$root/../archive/ende_beam_wmt14.en-de.transformer.dp0.2
# beam_input_dir_6=$root/../archive/ende_beam_wmt14.en-de.transformer.dp0.3


beam=5

tmp_dir=$root/tmp
mkdir -p $tmp_dir 
export TMPDIR=$tmp_dir

cd $root

python3 $root/count_length.py --input_dir $dfs_input_dir_1 --beam $beam #--script_path $script_path #--paint
# python3 $root/compute_edit_distances.py --dfs_input_dir $dfs_input_dir_2 --reference_file $ref_file --beam $beam --script_path $script_path #--paint
# python3 $root/compute_edit_distances.py --dfs_input_dir $dfs_input_dir_3 --reference_file $ref_file --beam $beam --script_path $script_path #--paint
# python3 $root/compute_edit_distances.py --dfs_input_dir $dfs_input_dir_4 --reference_file $ref_file --beam $beam --script_path $script_path #--paint
# python3 $root/compute_edit_distances.py --dfs_input_dir $dfs_input_dir_5 --reference_file $ref_file --beam $beam --script_path $script_path #--paint
# python3 $root/compute_edit_distances.py --dfs_input_dir $dfs_input_dir_6 --reference_file $ref_file --beam $beam --script_path $script_path #--paint