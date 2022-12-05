root=`your root path`
script_path=$root/../model_errors
ref_file=$root/../model_errors/test.de.tok.detok
# ref_file=$root/../model_errors/ref.test.out
input_dir_1=$root/../archive/ende_dfstopk_wmt14.en-de.transformer_new

beam=5

tmp_dir=$root/tmp
mkdir -p $tmp_dir 
export TMPDIR=$tmp_dir

cd $root

python3 $root/compute_model_errors.py --input_path $input_dir_1 --reference_file $ref_file --beam $beam --script_path $script_path --disable_cache