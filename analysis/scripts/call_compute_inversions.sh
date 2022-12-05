root=`your root path`
script_path=$root/../model_errors
ref_file=$root/../model_errors/test.de.tok.detok
input_dir_1=$root/../model_errors/ende_dfstopk_wmt14.en-de.transformer

beam=5
# beam=10

tmp_dir=$root/tmp
mkdir -p $tmp_dir 
export TMPDIR=$tmp_dir

cd $root

python3 $root/compute_inversions.py --input_dir  $input_dir_1 --reference_file $ref_file --beam $beam --script_path $script_path --disable_cache