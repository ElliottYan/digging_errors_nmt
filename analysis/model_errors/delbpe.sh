out_file=$1
#sed -r "s/(@@ )|(@@ ?$)//g" $out_file > ${out_file}.delbpe
sed -E "s/(@@ )|(@@ ?$)//g" $out_file > ${out_file}.delbpe
