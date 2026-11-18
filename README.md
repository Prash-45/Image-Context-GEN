# Image Context Generator

**An Experimental Study of CNN--LSTM, Attention-Based Captioning, and Transformer Decoding**

The project was developed in three iterations. Each version keeps a
pretrained ResNet-50 as the visual feature extractor while changing
how image information is conditioned on and translated into language:

1- V1: CNN + LSTM
2- V2: CNN + Bahdanau Attention + LSTM
3- V3: CNN + Transformer Decoder

The objective was not just to generate a sentence from an image, but to
understand the architectural trade-offs between a global visual
representation, spatial attention, and token-level cross-attention.


This project separates the task into two main components:
1. Visual encoding: ResNet-50 extracts image features.
2. Language decoding: A sequence model predicts the caption one
   token at a time, conditioned on the image representation and
   previously generated tokens.


# Architecture Evolution

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/09669009-be3c-4066-8c84-edeb80486749" />

## V1 ---> CNN + LSTM

ResNet-50 extracts a 2048-dimensional image representation. Linear
layers transform this feature into the LSTM's initial hidden and cell
states. The caption decoder receives the previous word embedding and
predicts the next token.

### Data flow : Image → ResNet-50 → 2048-D feature → initial LSTM states → LSTM → vocabulary projection → next token

Key implementation dimensions:
- Image input: 3 × 224 × 224
- Image feature: 2048-D
- Word embedding: 256-D
- LSTM hidden size: 512
- Vocabulary size: 5000

Limitation addressed in the next version: A single global image
vector compresses spatial information. The decoder does not explicitly
select different image regions as it generates each word.


<img width="1600" height="875" alt="image" src="https://github.com/user-attachments/assets/d841dc62-6446-43fa-a9d3-24c2e81e354d" />

## V2 ---> CNN + Bahdanau Attention + LSTM

Instead of using only one global feature vector, the CNN produces a
7 × 7 × 2048 feature map. It is reshaped into 49 spatial feature
vectors, each with 2048 dimensions.

At each decoding step, Bahdanau additive attention combines the
image-region features with the current LSTM hidden state. A softmax
produces attention weights over the 49 regions. Their weighted sum forms
a context vector, which is projected and concatenated with the current
word embedding before entering the LSTM.

### Data flow : Image → ResNet-50 → 7×7×2048 feature map → 49 regions

49 regions + current LSTM hidden state → Bahdanau Attention → attention weights → context vector

Context vector + previous-word embedding → LSTM → vocabulary projection → next token

Key implementation dimensions:
- Spatial features: 49 × 2048
- Attention hidden dimension: 256
- Context projection: 2048 → 256
- Word embedding: 256-D
- LSTM input: 512-D (context + word embedding)
- LSTM hidden size: 512
 
<img width="1860" height="756" alt="image" src="https://github.com/user-attachments/assets/de79e20c-6e24-46a7-89a9-91c059d49bff" />

What changed: The decoder can assign different weights to different
image regions at different time steps. The attention maps can also be
visualized to inspect which regions received higher weights during
generation.


<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/492b935f-99e8-49a3-a7fa-03d0dacb6c63" />

## V3 --- CNN + Transformer Decoder
Purpose: Replace recurrent sequence decoding with Transformer-based
sequence modeling.

The ResNet-50 feature map is reshaped into 49 visual tokens. A linear
projection maps each 2048-dimensional visual token into the
Transformer's 512-dimensional model space. Learnable positional
embeddings are added to the image tokens.

The previous caption tokens are embedded and combined with learnable
text positional embeddings. The Transformer decoder then applies:
1. Masked self-attention: Each text position can attend only to
   current and preceding text positions, preventing access to future
   tokens.
2. Cross-attention: Text representations attend to the encoded
   image tokens.
3. Feed-forward network: Further transforms the decoder
   representations before vocabulary prediction.
   
A final linear layer maps the decoder output to vocabulary logits.

**During inference, the model generates tokens autoregressively until the
end token or maximum caption length is reached.**

### Data flow : Image → ResNet-50 → 7×7×2048 → 49 visual tokens → projection to 512-D → image positional embeddings

Previous caption tokens → token embeddings + text positional embeddings

Text tokens → masked self-attention → cross-attention over image tokens → feed-forward network → vocabulary projection → next token

Key implementation dimensions:
- Visual tokens: 49 × 2048
- Projected visual tokens: 49 × 512
- Model dimension: 512
- Attention heads: 8
- Transformer decoder layers: 4
- Feed-forward dimension: 2048
- Dropout: 0.1
- Maximum text positional length: 50

What changed: V2 uses the recurrent hidden state to guide visual
attention. V3 uses text representations to query visual tokens through
cross-attention, while masked self-attention models relationships across
the generated text sequence.

# Experimental Observations

The notebook includes a qualitative comparison of generated captions and
a BLEU calculation for one selected dataset example.

<img width="2560" height="1214" alt="image" src="https://github.com/user-attachments/assets/494f1523-04e5-473a-9b13-5a3cb7309925" />

Interpretation: On this selected example, the attention-based and
Transformer versions scored higher than the baseline. V3 achieved the
highest score in this particular comparison.

Important evaluation limitation: These values are produced by
sentence_bleu using one reference caption and one generated caption
for a selected dataset item. They are not corpus-level BLEU scores and
do not establish that V3 generalizes better.
