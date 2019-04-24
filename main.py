import numpy as np
import matplotlib.pyplot as plt
from src.utility import load_cuave, load_avletter, concatenate_data, load_concatenate_data, pca_frame, contiguous
from src.boltzmann import RBM, BBRBM, GBRBM
from src.autoencoder import Autoencoder
from src.linearsvm import LinearSVM
from src.forest import tree_models


def audio_only():
    '''
    2136 (712) for CUAVE
    '''
    _, audio, _, _, labels = load_cuave()
    audio = contiguous(audio)
    print(audio.shape)
    bbrbm = BBRBM(use_tqdm=True)
    # errs = bbrbm.fit(audio)
    # bbrbm.save_weights('audio_only_cuave')
    # plt.plot(errs)
    # plt.show()
    bbrbm.load_weights('audio_only_cuave')
    repres = bbrbm.transform(audio)
    tree_models(repres, labels[:,0], repres.shape[1], 4)



def video_only():
    '''
    15000 (5000) for CUAVE
    8192 (2048) for CUAVE PCA
    '''
    _, _, frame_1, frame_2, labels = load_cuave()
    print(frame_1.shape, frame_2.shape)
    frames = contiguous(np.vstack((frame_1, frame_2)))
    print(frames.shape)
    bbrbm = BBRBM(use_tqdm=True)
    # errs = bbrbm.fit(frames)
    # bbrbm.save_weights('video_only_cuave')
    # plt.plot(errs)
    # plt.show()
    bbrbm.load_weights('video_only_cuave')
    repres = bbrbm.transform(frames)
    tree_models(repres, labels[:,0], repres.shape[1], 4)


def bimodal_fusion(dataset):
    """
    180 (36) for CUAVE (frames PCA & mfcc)
    4852 (1000) for AVLetters
    """
    concat_data = load_concatenate_data(dataset)
    print(concat_data.shape)
    bbrbm = BBRBM(use_tqdm=True)
    # errs = bbrbm.fit(concat_data)
    # bbrbm.save_weights('avletter_fusion')
    # plt.plot(errs)
    # plt.show()
    bbrbm.load_weights('avletter_fusion')
    repres = bbrbm.transform(concat_data)
    
    if dataset == 'cuave':
        _, _, _, _, label = load_cuave()
    else:
        _, _, label = load_avletter()

    tree_models(repres, label[:,0], repres.shape[1], 26)


def cross_modality(dataset):
    concat_data = load_concatenate_data(dataset)
    print(concat_data.shape)
    if dataset == 'cuave':
        _, _, _, _, label = load_cuave()
    else:
        _, _, label = load_avletter()
    
    # label = label[:,0]
    # label = np.hstack((label, label))
    # label = np.reshape(label, (len(label), 1))

    ae = Autoencoder(dataset, 'cross_modality', concat_data, label)
    ae.build_model()
    # ae.train_model()
    ae.load_model()
    repres = ae.transform(concat_data)
    print(concat_data.shape, repres.shape, label.shape)
    tree_models(repres, label, 50, 26)


def shared_repre():
    pass


def main():
    # audio_only()
    # video_only()
    # bimodal_fusion('avletter')
    # concatenate_data('cuave')
    # pca_frame('cuave')
    # concatenate_data('avletter')
    cross_modality('avletter')


if __name__ == "__main__":
    main()