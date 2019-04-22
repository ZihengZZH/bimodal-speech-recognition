import numpy as np
import matplotlib.pyplot as plt
from src.utility import load_cuave, load_avletter, concatenate_data, load_concatenate_data, pca_frame
from src.boltzmann import RBM, BBRBM, GBRBM
from src.linearsvm import LinearSVM

def bimodal_fusion():
    concat_data = load_concatenate_data('cuave')
    bbrbm = BBRBM(use_tqdm=True)
    # errs = bbrbm.fit(concat_data)
    # bbrbm.save_weights('cuave_fusion')
    # plt.plot(errs)
    # plt.show()
    bbrbm.load_weights('cuave_fusion')
    repres = bbrbm.transform(concat_data)
    
    _, _, _, _, label = load_cuave()
    label = np.vstack((label, label))
    label = label[:,0]
    
    svm = LinearSVM('cuave', repres, label)
    svm.tune()
    svm.train()
    svm.test()


def cross_modality():
    pass


def shared_repre():
    pass


def main():
    bimodal_fusion()
    # concatenate_data('cuave')


if __name__ == "__main__":
    main()