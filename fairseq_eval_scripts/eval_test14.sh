ref=test.de.tok

./delbpe.sh $1
#./detruecase.perl < ${1}.delbpe > ${1}.delbpe.dt
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $ref > ${ref}.atat
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < ${1}.delbpe > ${1}.delbpe.atat
./multi-bleu.perl ${ref}.atat < ${1}.delbpe.atat
