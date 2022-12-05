## Evaluating WMT'14 English-German models

We follow these steps: 
1. Make sure your WMT data and data-bin is well orgnized under $root directory, e.g. $root/data/wmt14_en_de, $root/data-bin/wmt14_en_de
2. Make sure your model files are saved in $root/checkpoints/wmt14_en_de/$model_name
3. Replace $root variable with your own root path in the following scripts: evaluate_ende_template_2013.sh, evaluate_ende_template.sh .
4. ``` bash test_14.sh `input your model name' '''
5. Outputs should be in $root/results/wmt14_en_de/$model_name/