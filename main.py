from src.utility import load_cuave, load_avletter, visualize_frame, visualize_spectrogram
from src.autoencoder import Autoencoder

def main():
    # load_cuave(verbose=True)
    # visualize_spectrogram('cuave', write=True)
    ae = Autoencoder('cuave','frame')
    ae.build_model()
    ae.train_model()
    ae.vis_model()

if __name__ == "__main__":
    main()