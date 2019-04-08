# bimodal-speech-recognition

Multimodal signal processing has become an important topic of research for overcoming certain problems of audio-only speech processing. Audio-visual speech recognition is one area with great potential.

## Autoencoder setup

From the similar example, [decoupled bimodal learning](https://github.com/Jakobovski/decoupled-multimodal-learning/blob/master/README.md), vocal features were converted into 64x64 spectrograms and visual features into 28x28 pixels. Then, the dimensionality reduction was completed via a deep denoising autoencoder of [2048, 1024, 256, 64] for visual features, and [4096, 512, 64] for vocal features.

In the CUAVE dataset, video frames are 75x50 pixels and raw audio waveforms are 534 in length.