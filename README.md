# bimodal-speech-recognition

Multimodal signal processing has become an important topic of research for overcoming certain problems of audio-only speech processing. Audio-visual speech recognition is one area with great potential.

## background

In the original paper, [ngiam2011](https://people.csail.mit.edu/khosla/papers/icml2011_ngiam.pdf), they demonstrated **cross modality feature learning**, where better features for one modality can be learned if multiple modalities are present at feature learning time. In their experiment, the overall task was divided into three phases:
* feature learning
* supervised training (*the simple linear classifier*)
* testing (*the simple linear classifier*)

In terms of different learning settings, they considered the following options:
* multimodal fusion
* cross-modality learning
* shared representation learning

The detailed difference between these three settings are shown as below:
|                               | feature learning | supervised learning | testing |
| --                            | --               | --                  | --      |
| classic deep learning BASELINE| A or V           | A or V              | A or V  |
| multimodal fusion             | A + V            | A + V               | A + V   |
| cross-modality learning       | A + V            | V                   | V       |
|                               | A + V            | A                   | A       |
| shared representation learning| A + V            | A                   | V       |
|                               | A + V            | V                   | A       |


## autoencoder setup

From the similar example, [decoupled bimodal learning](https://github.com/Jakobovski/decoupled-multimodal-learning/blob/master/README.md), vocal features were converted into 64x64 spectrograms and visual features into 28x28 pixels. Then, the dimensionality reduction was completed via a deep denoising autoencoder of [2048, 1024, 256, 64] for visual features, and [4096, 512, 64] for vocal features.

In the CUAVE dataset, video frames are 75x50 pixels and raw audio waveforms are 534 in length.

| DATA                      | SHAPE         |
| concat data 1 / 2 (A + V) | 4942, 180

## learning architecture

Separately train a RBM model for audio and video, and this model as a baseline for the later models.

**multimodal fusion**

Train a multimodal model by concatenating audio and video data.

> While this approach jointly models the distribution of the audio and video data, it is limited as a shallow model. Since the correlations between the audio and video data are highly non-linear, it is hard for a RBM to learn these correlations and form multimodal representations. In particular, they (Ngiam2011) found the learning a shallow bimodal RBM results in hidden units that have a strong connections to variables from individual modality but few units that connect across the modalities.


**cross-modality learning**

Train a deep autoencoder on both modalities in feature learning, but in supervised learning task, only one modality is used.

> By representing the data through learned first layer representations, it can be easier for the model to learn higher-order correlations across modalities. There is no explicit objective for the models to discover correlations across the modalities. Moreover, the models are clumsy to use in a cross modality learning setting where only one modality is present during supervised learning and testing (with only one single modality present, one would need to integrate out the unobserved visible variables to perform inference). 


**shared representation learning**

> Inspired by denoising autoencoders, the proposed bimodal deep autoencoder is trained using an augmented but noisy dataset with additional examples that have zeros values for one of the input modality (e.g. video). Due to initialization using sparse RBMs, the hidden units are reported to have low activation, even after the deep autoencoder training. 


## Experiment Results

| dataset   | architecture      | modality      | classification | acc    | recall | F1   |
| --        | --                | --            | --             | --     | --     | --   |
| CUAVE     | baseline          | MFCC          | RF             |     |
| CUAVE     | baseline          | A             | RF             |     |
| CUAVE     | baseline          | V             | RF             |     |
| AVLetters | baseline          | MFCC          | RF             |     |
| AVLetters | baseline          | V             | RF             |     |
| CUAVE     | bimodal fusion    | A + V         | RF (A + V)     |     |
| CUAVE     | bimodal fusion    | A + V         | RF (MFCC + V)  |     |
| AVLetters | bimodal fusion    | MFCC + V      | RF (MFCC + V)  |     |
| CUAVE     | cross-modality    | A + V         | RF (A)         |     |
| CUAVE     | cross-modality    | A + V         | RF (V)         |     |
| CUAVE     | cross-modality    | MFCC + V      | RF (MFCC)      |     |
| CUAVE     | cross-modality    | MFCC + V      | RF (V)         |     |
| AVLetters | cross-modality    | MFCC + V      | RF (MFCC)      |     |
| AVLetters | cross-modality    | MFCC + V      | RF (V)         |     |
| CUAVE     | shared-repres     | A + V         | RF (A / V)     |     |
| CUAVE     | shared-repres     | A + V         | RF (V / A)     |     |
| CUAVE     | shared-repres     | MFCC + V      | RF (MFCC / V)  |     |
| CUAVE     | shared-repres     | MFCC + V      | RF (V / MFCC)  |     |
| AVLetters | shared-repres     | MFCC + V      | RF (MFCC / V)  |     |
| AVLetters | shared-repres     | MFCC + V      | RF (V / MFCC)  |     |