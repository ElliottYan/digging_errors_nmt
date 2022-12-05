root=`your root path`
script_path=$root/../model_errors
ref_file=$root/../model_errors/test.de.tok.detok

sig=wmt14.en-de.transformer.dp0.3
beam_dir=$root/../archive/ende_beam_$sig
dfs_dir=$root/../archive/ende_dfstopk_$sig


input_dir_1=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.768_beam_10
# input_dir_2=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.12+6_beam_10
# input_dir_3=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.18+6_beam_10
# input_dir_4=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.big
# input_dir_5=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.dp0.2
# input_dir_6=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.dp0.3
# input_dir_7=$root/../archive/ende_dfstopk_wmt14.en-de.transformer.para_bt


beam=5

tmp_dir=$root/tmp
mkdir -p $tmp_dir 
export TMPDIR=$tmp_dir

cd $root

python3 $root/count_empty.py --input_dir $input_dir_1 --beam 5 
# python3 $root/count_empty.py --input_dir $input_dir_2 --beam 5
# python3 $root/count_empty.py --input_dir $input_dir_3 --beam 5
# python3 $root/count_empty.py --input_dir $input_dir_4 --beam 5
# python3 $root/count_empty.py --input_dir $input_dir_5 --beam 5
# python3 $root/count_empty.py --input_dir $input_dir_6 --beam 5
# python3 $root/count_empty.py --input_dir $input_dir_7 --beam 5
