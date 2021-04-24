The textual LaTeX edit recommendation can be treated as a machine translation problem bytreating the original post sentence as input and edited sentence as output. Therefore, we adopt neural machine translation model to learn the mapping from the source sentence to the targetsentence.
Attention based transformer architecture is used for formula latexification andLaTeX revision. Given the source word tokens (𝑥1, ...,𝑥𝑁), the goal is to predict the target word tokens(𝑦1, ...,𝑦𝑇). The source word tokens are the original post sentence, and the predicted outputword tokens are the edited post sentence. In the encoder part, a stack of 6 identical encoder blocks encodes the input sentence. Each block contains two sub-layers, a multi-head self-attention andposition-wise fully connected feed-forward layers. Followed by layer normalization, a residual connection around each of the two sub-layers are used in each encoder block. Just like the encoder, astack of 6 identical decoder blocks, which contain multi-head attention with feed-forward networks,are used in the decoder for target hidden states. However, in addition to the two sub-layers in each encoder layer, an extra attention layer over the encoder’s hidden states is used in each decoderblock, which is the third layer. Similar to the encoder, residual connections are employed between each of the sub-layers, followed by a normalization layer.
#### training
```
python train.py -src_data data/english.txt -trg_data data/french.txt -src_lang en -trg_lang fr -epochs 10
```
#### testing
```
python translate.py -load_weights weights
```
