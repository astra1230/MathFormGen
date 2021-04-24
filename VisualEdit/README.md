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
```
python scripts/preprocessing/generate_latex_vocab.py --data-path data/1w/train_filter.lst --label-path data/1w/formulas.norm.lst --output-file data/1w/latex_vocab.txt
```

### Train

```
th src/train.lua -phase train -gpu_id 1 \
-model_dir  \
-input_feed -prealloc \
-data_base_dir  \
-data_path  \
-val_data_path  \
-label_path  \
-vocab_file  \
-max_num_tokens 150 -max_image_width 500 -max_image_height 160 \
-batch_size 5 -beam_size 1 -num_epochs 10
```

```
th src/train.lua -phase test -gpu_id 1 -load_model -model_dir  \
-data_base_dir  \
-data_path  \
-label_path  \
-output_dir  \
-vocab_file  \
-max_num_tokens 500 -max_image_width 800 -max_image_height 800 \
-batch_size 10 -beam_size 5 
```


