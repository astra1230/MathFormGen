#### Model for visual edit model

The following lua libraries are required for the main model.

* tds
* class 
* nn
* nngraph
* cunn
* cudnn
* cutorch

The following Python libraries are required for the main model.

* Pillow
* numpy
* python-Levenshtein
* matplotlib
* Distance

Other tools
pdflatex : https://www.tug.org/texlive/
ImageMagick convert :http://www.imagemagick.org/script/index.php
Perl : https://www.perl.org/

### Preprocess
```
python scripts/preprocessing/preprocess_images.py --input-dir  --output-dir 
```
```
python scripts/preprocessing/preprocess_formulas.py --mode normalize --input-file  --output-file 
```
```
python scripts/preprocessing/preprocess_filter.py --filter --image-dir  --label-path  --data-path  --output-path 

```
python scripts/preprocessing/generate_latex_vocab.py --data-path data/1w/train_filter.lst --label-path data/1w/formulas.norm.lst --output-file data/1w/latex_vocab.txt
```
### Train
The most important parameters for training are `data_base_dir`, which specifies where the images live; `data_path`, the training file; `label_path`, the LaTeX formulas, `val_data_path`, the validation file; `vocab_file`, the vocabulary file with one token per each line.

```
th src/train.lua -phase train -gpu_id 1 \
-model_dir model/1w \
-input_feed -prealloc \
-data_base_dir data/1w/images_processed/ \
-data_path data/1w/train_filter.lst \
-val_data_path data/1w/validate_filter.lst \
-label_path data/1w/formulas.norm.lst \
-vocab_file data/1w/latex_vocab.txt \
-max_num_tokens 150 -max_image_width 500 -max_image_height 160 \
-batch_size 5 -beam_size 1 -num_epochs 10
```
Now we can load the model and test on test set. Note that in order to output the predictions, a flag `-visualize` must be set.

```
th src/train.lua -phase test -gpu_id 1 -load_model -model_dir model/1w_o -visualize \
-data_base_dir data/1w/images_processed/ \
-data_path data/1w/test_filter.lst \
-label_path data/1w/formulas.norm.lst \
-output_dir results/1w_o \
-vocab_file data/1w/latex_vocab.txt \
-max_num_tokens 500 -max_image_width 800 -max_image_height 800 \
-batch_size 10 -beam_size 5 
```


