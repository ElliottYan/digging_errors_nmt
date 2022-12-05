root=`your root path`
input_dir=$root/../archive/ende_dfstopk_wmt14.en-de.transformer_new_beam_100_bpe
output_dir=$root/../archive/ende_dfstopk_wmt14.en-de.transformer_new_beam_10_bpe

mkdir -p $output_dir

for file in $input_dir/*; do
    base_name="$(basename -- $file)"
    python3 $root/extract_top_k.py $file $output_dir/$base_name 100 10
done