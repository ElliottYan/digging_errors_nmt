root=`your root path`
script_path=$root/../model_errors
ref_file=$root/../model_errors/test.de.tok.detok
ref_tok=$root/../model_errors/test.de.tok
input_dir_1=$root/../archive/ende_beam_wmt14.en-de.transformer_new_beam_100
# input_dir_1=$root/../model_errors/ende_dfstopk_wmt14.en-de.transformer.para_ft

beam=5
# beam=10

tmp_dir=$root/tmp
mkdir -p $tmp_dir 
export TMPDIR=$tmp_dir

cd $root

export LC_ALL=en_US.UTF-8
python3 $root/compute_top1_bleu.py --input_dir  $input_dir_1 --beam $beam --script_path $script_path 

output=$input_dir_1.outs.top1
cd $script_path
bash ./delbpe.sh $output
cat $output.delbpe | perl ./detokenizer.perl > $output.delbpe.detok

cat $output.delbpe.detok | sacrebleu $ref_file

echo "Test multi-bleu"
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $ref_tok > ${ref_tok}.atat
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $output.delbpe > $output.delbpe.atat
./multi-bleu.perl ${ref_tok}.atat < ${output}.delbpe.atat
# ./multi-bleu.perl ${ref_tok} < ${output}.delbpe