# Digging Errors in NMT: Evaluating and Understanding Model Errors from Partial Hypothesis Space
This is the official code repository released for EMNLP 2022 main track paper "Digging Errors in NMT: Evaluating and Understanding Model Errors from Partial Hypothesis Space".

# Exact Top-k Decoding
Decoding library based on SGNMT: https://github.com/ucam-smt/sgnmt. See their [docs](http://ucam-smt.github.io/sgnmt/html/) for setting up a fairseq model to work with the library.

Dependencies
```
fairseq==0.10.1
scipy==1.5.4
numpy==1.19.4
Cython==0.29.21
sortedcontainers==2.3.0
subword-nmt==0.3.7
```

To compile the datastructure classes, run:
```
pip install -e .
```

To compile the statistics classes, navigate to the `runstats` submodule:
```
cd runstats
python setup.py install
```


## Getting Started
We recommend starting with the pretrained models available from fairseq. Download any of the models from, e.g., their NMT examples, unzip, and place model checkpoints in `data/ckpts`. You'll have to preprocess the dictionary files to a format that the library expects. Using the [pre-trained convolutional English-French WMT‘14 model](https://github.com/pytorch/fairseq/tree/master/examples/translation) an example:

```
curl https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2 | tar xvjf -
cat wmt14.en-fr.fconv-py/dict.en.txt | awk 'BEGIN{print "<epsilon> 0\n<s> 1\n</s> 2\n<unk> 3"}{print $1" "(NR+3)}' > wmap.en
cat wmt14.en-fr.fconv-py/dict.fr.txt | awk 'BEGIN{print "<epsilon> 0\n<s> 1\n</s> 2\n<unk> 3"}{print $1" "(NR+3)}' > wmap.fr
``` 

Tokenization (for input) and detokenization (for output) should be performed with the [mosesdecoder library](https://github.com/moses-smt/mosesdecoder.git). If the model uses BPE, you'll have to preprocess the input file to put words in byte pair format. Given your bpecodes listed in e.g., file `bpecodes`, the entire pre-processing of input file `input_file.txt` in English (en) can be done as follows. Again using the convolutional English-French WMT‘14 model with the [`newstest` test set](http://statmt.org/wmt14/test-full.tgz) as an example input file:

#### Remove special formatting from newstest set
```
grep '<seg id' test-full/newstest2014-fren-src.en.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > newstest_cleaned.txt
```
#### Tokenize and apply BPE
```
cat newstest_cleaned.txt | perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -l en > out
subword-nmt apply-bpe -c wmt14.en-fr.fconv-py/bpecodes -i out -o newstest_bpe.txt
```

Alternatively, one can play around with the toy model in the test scripts. Outputs are not meaningful but it is deterministic and useful for debugging.

### Beam Search

Basic beam search can be performed on a fairseq model translating from English to French as follows:

```
 python decode.py  --fairseq_path wmt14.en-fr.fconv-py/model.pt --fairseq_lang_pair en-fr --src_wmap wmap.en --trg_wmap wmap.fr --input_file newstest_bpe.txt --preprocessing word --postprocessing bpe@@ --decoder beam --beam 10 
 ```
By default, decoders only return the best solution. Set `--early_stopping False` if you want the entire set.

A basic example of outputs can be seen when using the test suite:

 ```
 python test.py --decoder beam --beam 10 
 ```

 Additionally, you can run
 ```
 python decode.py --help
 ```
 to see descriptions of all available arguments.

### DFS search

To output with file, just add `outputs` and `output_path` to configs.
In addition, you can use `--outputs nbest` to output candidates and corresponding scores.

```
LC_ALL=en_US.UTF-8 python decode.py  --fairseq_path wmt14.en-fr.fconv-py/model.pt --fairseq_lang_pair en-fr --src_wmap wmap.en --trg_wmap wmap.fr --input_file newstest_bpe.txt --preprocessing word --postprocessing bpe@@ --decoder simpledfs --outputs text --output_path $output_path
```

For our accelerated dfs topk search, try
```
python decode.py  --fairseq_path checkpoints/wmt14.en-fr.fconv-py/model.pt --fairseq_lang_pair en-fr --src_wmap wmap.en --trg_wmap wmap.fr --input_file  newstest_bpe.txt --preprocessing word --postprocessing bpe@@ --decoder batchdfstopk --beam $beam --output_path $dfs_output_file --outputs nbest_lower_bounds --nbest $beam --early_stopping False --remove_eos False --score_lower_bounds_file $lower_bound_file --dfstopk_batchsize 50 --simpledfs_topk $beam --max_len_factor -1 --num_log $beam --nbest $beam 
```

### Lower bounds
We support using lower bound file to accelerate decoding for dfs and dfs topk. One additional argument `--score_lower_bounds_file` is needed. Conventionally, we use the outputs of beam search as our lower bounds. The output type is nbest_lower_bounds.

For dfs with topk, we need to use beam search or min heap augmented beam search as our lower bounds. Decoder argument can be beam or min_heap_beam.
```
LC_ALL=en_US.UTF-8 python decode.py  --fairseq_path wmt14.en-fr.fconv-py/model.pt --fairseq_lang_pair en-fr --src_wmap wmap.en --trg_wmap wmap.fr --input_file newstest_bpe.txt --preprocessing word --postprocessing bpe@@ --decoder beam --beam $topk --output_path $beam_output_file --outputs nbest_lower_bounds --nbest $beam --early_stopping False --remove_eos False --max_len_factor 3
```
A example format of lower bound file can be found in lower_bounds/*.

### Fast Decoding
In the original implementation of SGNMT and UID-Decoding, multi-gpu setting is not supported. Here, we provide a multiprocessing script python_scripts/mp_command_master.py for supporting multi-gpu setting.
Specifically, we split the input files into pieces with 5 sentences per file and store the decoding commands like `CUDA_VISIBLE_DEVICES=0 bash decode.sh $input_file.1$` into a command file. And then, we use the mp_command_master.py for controling the actual decoding on different GPUs.

### DFS-Topk
We provide a sample script for dfs topk.
```
python decode.py  --fairseq_path checkpoints/wmt14.en-fr.fconv-py/model.pt --fairseq_lang_pair en-fr --src_wmap checkpoints/wmap.en --trg_wmap checkpoints/wmap.fr --input_file checkpoints/wmt14.en-fr.fconv-py/newstest_bpe.txt.head10 --preprocessing word --postprocessing bpe@@ --decoder simpledfstopk --beam 10 --output_path $result_dir/test.out --outputs nbest_sep --num_log 100 --nbest 100 --simpledfs_topk 10
```

### Evaluation Scripts
The evaluation scripts lie in analysis/scripts. These bash scripts are used for various kind of analysis for topk results. To reproduce the results, p1lease replace $root with your $topk_path/analysis/scripts first. 
Here, we provide an example of evaluating our model outputs with one simple line of code.
```
cd analysis/scripts && bash call_compute_model_errors.sh
```
Then, it can compute the model error metrics for our results `ende_dfstopk_wmt14.en-de.transformer_new` in analysis/archive directory.  

For reproduce our results with COMET as reported in the paper, we refer the readers to analysis/comet_scripts/*.sh
