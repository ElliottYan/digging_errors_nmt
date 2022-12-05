root=/apdcephfs/share_47076/elliottyan/beam-search/beam-search-decoding/analysis/scripts
ref_file=$root/../model_errors/test.de.tok.atat
hypo_dir=$root/../archive/ende_beam_wmt14.en-de.transformer_new_beam_5

output=$hypo_dir.outs.top1
output=${output}.delbpe.atat

python3 sent_level_bleu.py --input_file $output --ref_file $ref_file --output $output.sent_bleu_scores.txt --n_processes 20