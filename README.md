# Mathematical Formula Markup Generation for Collaborative Q&A Sites
Collaborative editing questions and answers plays an important role in quality control in Mathematics Stack Exchange which is a math Q&A Site. Our study of post edits in Mathematics Stack Exchange shows that a large number of edits are about converting the blurred math formula screenshots to LaTeX sequence which is further rendered to display math with better readability and accessibility. Despite its importance, editing one formula screenshot into LaTeX is time-consuming and error-prone even for experienced users. 
To assist post owners and editors to do this editing, we have developed an edit-assistance tool, MathFormGen, to generate the LaTeX sequences, given the formula screenshot. We formulate this formula editing task as a translation problem, in which an original image is translated to a LaTeX sequence. MathFormGen implements a deep learning based approach including an encoder-decoder model with domain-specific inference and visualization. 
It is trained on 1 million screenshot-formula pairs which are synthesized with LaTeX formulas collected from Mathematics Stack Exchange. Our evaluation of MathFormGen not only demonstrates the accuracy of our model, but also the usefulness of MathFormGen in editing real-world posts which are accepted in Mathematics Stack Exchange.

#### Model

The following lua libraries are required for the main model.

* tds
* class 
* nn
* nngraph
* cunn
* cudnn
* cutorch

Note that currently we only support **GPU** since we use cudnn in the CNN part.

#### Preprocess

Python

* Pillow
* numpy

Optional: We use Node.js and KaTeX for preprocessing [Installation](https://nodejs.org/en/)

##### pdflatex [Installaton](https://www.tug.org/texlive/)

Pdflatex is used for rendering LaTex during evaluation.

##### ImageMagick convert [Installation](http://www.imagemagick.org/script/index.php)

Convert is used for rending LaTex during evaluation.

#### Evaluate

Python image-based evaluation

* python-Levenshtein
* matplotlib
* Distance


##### Perl [Installation](https://www.perl.org/)

Perl is used for evaluating BLEU score.

### Preprocess
The images in the dataset contain a LaTeX formula rendered on a full page. To accelerate training, we need to preprocess the images. 
```
python scripts/preprocessing/preprocess_images.py --input-dir data/1w/images --output-dir data/1w/images_processed
```
The above command will crop the formula area, and group images of similar sizes to facilitate batching.

Next, the LaTeX formulas need to be tokenized or normalized.
```
python scripts/preprocessing/preprocess_formulas.py --mode normalize --input-file data/1w/formulas.lst --output-file data/1w/formulas.norm.lst
```
The above command will normalize the formulas. Note that this command will produce some error messages since some formulas cannot be parsed by the KaTeX parser.

Then we need to prepare train, validation and test files. We will exclude large images from training and validation set, and we also ignore formulas with too many tokens or formulas with grammar errors.
```
python scripts/preprocessing/preprocess_filter.py --filter --image-dir data/1w/images_processed --label-path data/1w/formulas.norm.lst --data-path data/1w/train.lst --output-path data/1w/train_filter.lst 
```

```
python scripts/preprocessing/preprocess_filter.py --filter --image-dir data/1w/images_processed --label-path data/1w/formulas.norm.lst --data-path data/1w/validate.lst --output-path data/1w/validate_filter.lst 
```

```
python scripts/preprocessing/preprocess_filter.py --no-filter --image-dir data/1w/images_processed --label-path data/1w/formulas.norm.lst --data-path data/1w/test.lst --output-path data/1w/test_filter.lst 
```

Finally, we generate the vocabulary from training set. All tokens occuring less than (including) 1 time will be excluded from the vocabulary.

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

# Acknowlegement
Our implementation is based on HarvardNLP NMT implementation im2markup.

