root=`your root path`

output_prefix=$root/../archive/sp_input/all/tmp

in=$root/../archive/ende_dfstopk_wmt14.en-de.transformer_new_beam_10_bpe/test.out.SHARD.237
in2=$root/../../checkpoints/wmt14.en-de.transformer_new/split/test.en.SHARD.237

python $root/extract_from_moses.py $in

mv $in.text $output_prefix.tgt
python $root/dup.py $in2 10
mv $in2.dup10 $output_prefix.src 
