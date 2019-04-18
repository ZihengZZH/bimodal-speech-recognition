from src.utility import load_cuave, load_avletter, visualize_frame, visualize_spectrogram
from src.autoencoder import Autoencoder

def main():
    load_cuave(verbose=True)
    # visualize_frame('cuave_2')
    # visualize_spectrogram('cuave', write=True)
    # ae = Autoencoder('cuave','spectrogram')
    # ae.build_model()
    # ae.load_model()
    # ae.vis_model()

if __name__ == "__main__":
    main()