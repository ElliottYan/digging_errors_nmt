root="your fairseq path"
export PYTHONPATH=$root:$PYTHONPATH

if [ -z "$2" ]; then
	data_signature=wmt14_en_de
else
	data_signature=$2
fi
ori_data_signature=wmt14_en_de
signature=$1

output_dir=$root/checkpoints/$data_signature/$signature

#data_dir=$root/data/$data_signature
data_dir=$root/data/$ori_data_signature

result_dir=$root/results/$data_signature/$signature
mkdir -p $result_dir

for ckpt in $output_dir/checkpoint_*_*.pt; do
	echo "decoding $ckpt" 
	python $root/fairseq_cli/generate.py \
	    $root/data-bin/$data_signature \
            --gen-subset valid \
	    --results-path $result_dir/valid.out \
	    --path $ckpt \
	    --lenpen 0.6 \
	    --beam 4 

	LC_ALL=en_US.UTF-8 python $root/scripts/extract_generate_output.py \
	    --output $result_dir/valid.out/generate-valid --srclang de --tgtlang en $result_dir/valid.out/generate-valid.txt

	# cd $data_dir
    ckpt_base="$(basename -- $ckpt)"
	bash eval_test13.sh $result_dir/valid.out/generate-valid.de > $result_dir/valid.result.${ckpt_base}
done

