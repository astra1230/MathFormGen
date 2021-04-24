#### Model for visual edit model

Our tool applies an encoder-decoder structure for screenshot transcription, inwhich the encoder uses DenseNet, and the decoder uses LSTM with attention mechanism. Theoverall structure is shown in following figure.To get the feature maps of the input images, DenseNet is firstly used in the encoder to extract feature map 𝑉. Unlike traditional Convolutional Neural Network (CNN), DenseNet connects eachlayer to every subsequent layer. The network structure of DenseNet we used is shown in Fig. The output features of DenseNet contain sequential order information, thus we use another RNN encoder to re-encode each row of DenseNet’s output feature map. As shown in Fig, after running row encoder across each row of the 𝑉, the new feature map 𝑇 is created.

Based on the feature map 𝑇, we use LSTM as decoder to generate a sequence of predicted LaTeXtokens. The context vector𝑐𝑡considers the whole feature map 𝑇 with which to capture contextinformation. Most parts of the feature grid may be irrelevant to the current predicted LaTeX token,thus the model should know which part of the feature map is important. In other words, the modelshould pay attention to the important parts.  We use an attention mechanism to achieve this goal.

<!-- <div style="color:#0000FF" align="center"> -->
<p align="center">
<img src="figures/stru.png" width="70%"/> 
</p><p align="center">Fig. Structure of our visual edit model<p align="center">
<!-- </div> -->
  
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


