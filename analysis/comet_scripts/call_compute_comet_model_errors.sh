# root=/app/analysis/scripts
root=/apdcephfs/share_47076/elliottyan/beam-search/beam-search-decoding/analysis/scripts
# script_root=$root
# root=/Users/elliottyan/weixin/beam_search/beam-search-decoding
script_path=$root/../model_errors
ref_file=$root/../model_errors/test.de.tok.detok
src_file=$root/../model_errors/newstest2014-deen-src.en.sgm.txt
# ref_file=$root/../model_errors/ref.test.out
# input_dir_1=$root/../archive/ende_dfstopk_wmt14.en-de.transformer_new_beam_100
# input_dir_1=$root/../archive/ende_dfstopk_wmt14.en-de.transformer_new
# input_dir_2=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.12+6_beam_10
# input_dir_3=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.18+6_beam_10
# input_dir_4=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.768_beam_10
# input_file=$root/../../results/sp_eval/greedy_reg/test.out.merge

input_dir_1=$root/../archive/ende_dfstopk_wmt14.en-de.transformer_new
input_dir_2=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.nolb
input_dir_3=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.dp0.2
input_dir_4=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.dp0.3
input_dir_5=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.para_ft
input_dir_6=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.para_bt
input_dir_7=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.big
input_dir_8=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.12+6_beam_10
input_dir_9=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.18+6_beam_10
input_dir_10=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.768_beam_10

beam=5

tmp_dir=$root/tmp
mkdir -p $tmp_dir 
export TMPDIR=$tmp_dir

# NOTE: change beam !!!
cd $root
export LC_ALL=en_US.UTF-8
python3 $root/compute_model_errors.py --input_dir $input_dir_1 --reference_file $ref_file --beam $beam --script_path $script_path --disable_cache --score_type comet --source_file $src_file
python3 $root/compute_model_errors.py --input_dir $input_dir_2 --reference_file $ref_file --beam 10 --script_path $script_path --disable_cache --score_type comet --source_file $src_file
python3 $root/compute_model_errors.py --input_dir $input_dir_3 --reference_file $ref_file --beam $beam --script_path $script_path --disable_cache --score_type comet --source_file $src_file
python3 $root/compute_model_errors.py --input_dir $input_dir_4 --reference_file $ref_file --beam $beam --script_path $script_path --disable_cache --score_type comet --source_file $src_file
python3 $root/compute_model_errors.py --input_dir $input_dir_5 --reference_file $ref_file --beam 10 --script_path $script_path --disable_cache --score_type comet --source_file $src_file
python3 $root/compute_model_errors.py --input_dir $input_dir_6 --reference_file $ref_file --beam $beam --script_path $script_path --disable_cache --score_type comet --source_file $src_file
python3 $root/compute_model_errors.py --input_dir $input_dir_7 --reference_file $ref_file --beam $beam --script_path $script_path --disable_cache --score_type comet --source_file $src_file
python3 $root/compute_model_errors.py --input_dir $input_dir_8 --reference_file $ref_file --beam $beam --script_path $script_path --disable_cache --score_type comet --source_file $src_file
python3 $root/compute_model_errors.py --input_dir $input_dir_9 --reference_file $ref_file --beam $beam --script_path $script_path --disable_cache --score_type comet --source_file $src_file
python3 $root/compute_model_errors.py --input_dir $input_dir_10 --reference_file $ref_file --beam $beam --script_path $script_path --disable_cache --score_type comet --source_file $src_file
