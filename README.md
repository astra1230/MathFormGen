# Latexify Math: Mathematical Formula Markup Revision toAssist Collaborative Editing in Math Q&A Sites
Collaborative editing questions and answers plays an important role in quality control of Mathematics StackExchange which is a math Q&A Site. Our study of post edits in Mathematics Stack Exchange shows thatthere is a large number of math-related edits about latexifying formulas, revising LaTeX and convertingthe blurred math formula screenshots to LaTeX sequence. Despite its importance, manually editing onemath-related post especially those with complex mathematical formulas is time-consuming and error-proneeven for experienced users. To assist post owners and editors to do this editing, we have developed an edit-assistance tool,MathLatexEditfor formula latexification, LaTeX revision and screenshot transcription. Weformulate this formula editing task as a translation problem, in which an original post is translated to a revisedpost.MathLatexEditimplements a deep learning based approach including two encoder-decoder models fortextual and visual LaTeX edit recommendation with math-specific inference. The two models are trained onlarge-scale historical original-edited post pairs and synthesized screenshot-formula pairs. Our evaluation ofMathLatexEditnot only demonstrates the accuracy of our model, but also the usefulness ofMathLatexEditin editing real-world posts which are accepted in Mathematics Stack Exchange.


#### Model for converting images into LaTeX formulas

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

