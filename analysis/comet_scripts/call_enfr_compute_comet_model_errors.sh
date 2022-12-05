# root=/app/analysis/scripts
root=/apdcephfs/share_47076/elliottyan/beam-search/beam-search-decoding/analysis/scripts
# root=/Users/elliottyan/weixin/beam_search/beam-search-decoding
script_path=$root/../model_errors
ref_file=$root/../model_errors/newstest2014.en-fr.fr.delbpe.detok
source_file=$root/../model_errors/newstest2014-fren-src.en.sgm.txt

input_dir_1=$root/../enfr_archive/enfr_dfstopk_wmt14.en-fr.transformer_beam_10
input_dir_2=$root/../enfr_archive/enfr_dfstopk_wmt14.en-fr.transformer.nolb_beam_10
input_dir_3=$root/../enfr_archive/enfr_dfstopk_wmt14.en-fr.transformer.paraft_beam_10
input_dir_4=$root/../enfr_archive/enfr_dfstopk_wmt14.en-fr.transformer.12+6_beam_10
input_dir_5=$root/../enfr_archive/enfr_dfstopk_wmt14.en-fr.transformer.18+6_beam_10
input_dir_6=$root/../enfr_archive/enfr_dfstopk_wmt14.en-fr.transformer.d768_beam_10
input_dir_7=$root/../enfr_archive/enfr_dfstopk_wmt14.en-fr.transformer.d1024_beam_10
# input_dir_8=$root/../enfr_archive/enfr_dfstopk_wmt14.en-fr.transformer.paraft_beam_10
# input_dir_8=$root/../zhen_archive/zhen_beam_wmt19.zh-en.transformer.d1024_beam_10/

beam=5
# beam=10

tmp_dir=$root/tmp
mkdir -p $tmp_dir 
export TMPDIR=$tmp_dir

cd $root
export LC_ALL=en_US.UTF-8


command_file=$root/../comet_scripts/command.enfr.txt
rm -rf $command_file

echo "python3 $root/compute_model_errors.py --input_dir $input_dir_1 --reference_file $ref_file --beam $beam --script_path $script_path --disable_cache --tgt_lang 'fr' --source_file $source_file --score_type comet --output_file ${input_dir_1}.eval " >> $command_file
echo "python3 $root/compute_model_errors.py --input_dir $input_dir_2 --reference_file $ref_file --beam $beam --script_path $script_path --disable_cache --tgt_lang 'fr' --source_file $source_file --score_type comet --output_file ${input_dir_2}.eval " >> $command_file
echo "python3 $root/compute_model_errors.py --input_dir $input_dir_3 --reference_file $ref_file --beam $beam --script_path $script_path --disable_cache --tgt_lang 'fr' --source_file $source_file --score_type comet --output_file ${input_dir_3}.eval " >> $command_file
echo "python3 $root/compute_model_errors.py --input_dir $input_dir_4 --reference_file $ref_file --beam $beam --script_path $script_path --disable_cache --tgt_lang 'fr' --source_file $source_file --score_type comet --output_file ${input_dir_4}.eval " >> $command_file
echo "python3 $root/compute_model_errors.py --input_dir $input_dir_5 --reference_file $ref_file --beam $beam --script_path $script_path --disable_cache --tgt_lang 'fr' --source_file $source_file --score_type comet --output_file ${input_dir_5}.eval " >> $command_file
echo "python3 $root/compute_model_errors.py --input_dir $input_dir_6 --reference_file $ref_file --beam $beam --script_path $script_path --disable_cache --tgt_lang 'fr' --source_file $source_file --score_type comet --output_file ${input_dir_6}.eval " >> $command_file
echo "python3 $root/compute_model_errors.py --input_dir $input_dir_7 --reference_file $ref_file --beam $beam --script_path $script_path --disable_cache --tgt_lang 'fr' --source_file $source_file --score_type comet --output_file ${input_dir_7}.eval " >> $command_file
# python3 $root/compute_model_errors.py --input_dir $input_dir_8 --reference_file $ref_file --beam $beam --script_path $script_path --disable_cache --tgt_lang 'fr'

N_GPUS=1
PRO_EACH_GPU=1

python3 $root/../../python_scripts/mp_command_master.py \
--command_file $command_file \
--n_gpu $N_GPUS \
--shuffle_commands \
--per_gpu_process $PRO_EACH_GPU