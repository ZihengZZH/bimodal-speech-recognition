import os
import json
import numpy as np
import pandas as pd
import skimage.io as io
from scipy.io import loadmat
from scipy.signal import spectrogram
import matplotlib.pyplot as plt


data_config = json.load(open('./config/config.json', 'r'))


# load the CUAVE dataset
def load_cuave(verbose=False):
    # para verbose: whether or not to print more 
    processed_dir = data_config['data']['processed']['cuave']

    if len(os.listdir(processed_dir)) == 6:
        print("processed data exist\nstart loading processed data")
        mfcc = np.load(os.path.join(processed_dir, 'mfccs.npy'))
        audio = np.load(os.path.join(processed_dir, 'audio.npy'))
        spec = np.load(os.path.join(processed_dir, 'spectrogram.npy'))
        frame_1 = np.load(os.path.join(processed_dir, 'frames_1.npy'))
        frame_2 = np.load(os.path.join(processed_dir, 'frames_2.npy'))
        label = np.load(os.path.join(processed_dir, 'labels.npy'))

    else:
        mfcc, audio, spec, frame_1, frame_2, label = [], [], [], [], [], []
        mfcc_temp, audio_temp, spec_temp, frame_1_temp, frame_2_temp = [], [], [], [], []
        every_five = 1
        for index in range(1, 23):
            filename = "g%s_aligned.mat" % str(index).zfill(2)
            each = loadmat(os.path.join(data_config['data']['cuave'], filename))
            # load the label
            label.extend(each['labels'][0])
            each_mfcc = each['mfccs']
            each_audio = each['audioIndexed']
            fs = each['fs'][0][0]
            each_frame_1 = each['video'][0][0]
            each_frame_2 = each['video'][0][1]
            for i in range(len(each['labels'][0])):
                # load vocal features (mfcc)
                mfcc_temp.append(each_mfcc[:,i])
                # load vocal modality data (along with spectrogram)
                audio_temp.append(each_audio[:,i])
                f, t, Sxx = spectrogram(each_audio[:,i], fs=fs)
                spec_temp.append((f, t, Sxx))
                # load visual modality data
                frame_1_temp.append(each_frame_1[:,:,i])
                frame_2_temp.append(each_frame_2[:,:,i])

                if every_five == 5:
                    # push features at a freq of five
                    mfcc.append(mfcc_temp)
                    audio.append(audio_temp)
                    spec.append(spec_temp)
                    frame_1.append(frame_1_temp)
                    frame_2.append(frame_2_temp)
                    mfcc_temp, audio_temp, spec_temp, frame_1_temp, frame_2_temp = [], [], [], [], []
                    every_five = 1
                else:
                    every_five += 1

            print(filename, "read")

        if verbose:
            print(label[:10])
            print(mfcc[:10])
            print(audio[:10])
            print(spec[:10])
            print(frame_1[:10])
            print(frame_2[:10])

        # 5 contiguous frames to use as input
        label = np.array(label).reshape(int(len(label)/5), 5)

        drop_row_idx = []
        for k in range(label.shape[0]):
            if np.unique(label[k]).size == 2:
                drop_row_idx.append(k)

        label = np.delete(label, (drop_row_idx), axis=0)
        mfcc = np.delete(mfcc, (drop_row_idx), axis=0)
        audio = np.delete(audio, (drop_row_idx), axis=0)
        spec = np.delete(spec, (drop_row_idx), axis=0)
        frame_1 = np.delete(frame_1, (drop_row_idx), axis=0)
        frame_2 = np.delete(frame_2, (drop_row_idx), axis=0)

        # write processed data to external files
        print("no processed data exist\nstart writing processed data")
        np.save(os.path.join(processed_dir, 'mfccs'), mfcc)
        np.save(os.path.join(processed_dir, 'audio'), audio)
        np.save(os.path.join(processed_dir, 'spectrogram'), spec)
        np.save(os.path.join(processed_dir, 'frames_1'), frame_1)
        np.save(os.path.join(processed_dir, 'frames_2'), frame_2)
        np.save(os.path.join(processed_dir, 'labels'), label)

    return mfcc, audio, spec, frame_1, frame_2, label


# load the AVLetters dataset
def load_avletter(verbose=False):
    # para verbose: whether or not to print more info
    processed_dir = data_config['data']['processed']['avletter']

    if len(os.listdir(processed_dir)) == 4:
        print("processed data exist\nstart loading processed data")
        mfcc = np.load(os.path.join(processed_dir, 'mfccs.npy'))
        frame = np.load(os.path.join(processed_dir, 'frames.npy'))
        label = np.load(os.path.join(processed_dir, 'labels.npy'))

    else:
        base = ord('A')
        mfcc, frame, label = [], [], []
        names = data_config['data']['avletter']['name']
        mfcc_file = data_config['data']['avletter']['vocal']
        frame_file = data_config['data']['avletter']['visual']
        for i in range(26):
            for name in names:
                for num in [1, 2, 3]:
                    # load mfccs
                    mfcc_temp = np.loadtxt(os.path.join(mfcc_file, '%s%d_%s.mat' % (chr(base+i), num, name)))
                    # load frames
                    frame_temp = loadmat(os.path.join(frame_file, '%s%d_%s-lips.mat' % (chr(base+i), num, name)))
                    vid_temp = frame_temp['vid']
                    (h, w, no_frame) = frame_temp['siz'][0]
                    # reshape the matrix
                    while mfcc_temp.shape[0] != int(no_frame) * 2:
                        mfcc_temp = np.vstack((mfcc_temp, mfcc_temp[-1,:]))
                    mfcc_temp_reshape = np.reshape(mfcc_temp, (int(no_frame*4), 13), order='F')
                    for j in range(int(no_frame)):
                        vid_temp_reshape, temp_1, temp_2 = np.reshape(vid_temp[:,j], (int(h), int(w))), [], []
                        for k in range(1, 5):
                            temp_1.append(vid_temp_reshape[:, 20*(k-1):20*k])
                            temp_2.append(mfcc_temp_reshape[4*int(no_frame-1)+k-1, :])
                        frame.append(temp_1)
                        mfcc.append(temp_2)
                        label.append(i) # ASCII value for label
            print("%s character loaded" % chr(base+i))

        if verbose:
            print(label[:20])
            print(mfcc[:20])
            print(frame[:20])

        # write processed data to external files
        print("no processed data exist\nstart writing processed data")
        np.save(os.path.join(processed_dir, 'mfccs'), mfcc)
        np.save(os.path.join(processed_dir, 'frames'), frame)
        np.save(os.path.join(processed_dir, 'labels'), label)

    return mfcc, frame, label


# helper function to visualize frames
def visualize_frame(dataset, write=False):
    # para dataset: name of dataset (cuave / avletter)
    # para write: whether or not to write images to external files
    filename = data_config['data']['visualized']['frames'][dataset]
    frames = np.load(filename)
    frame_1, frame_2, frame_3, frame_4 = frames[10], frames[1], frames[20], frames[50]
    if dataset == 'avletter':
        frame_1, frame_2, frame_3, frame_4 = frame_1.transpose(), frame_2.transpose(), frame_3.transpose(), frame_4.transpose()

    plt.subplot(221)
    plt.title('1st frame')
    plt.axis('off')
    io.imshow(frame_1)

    plt.subplot(222)
    plt.title('2nd frame')
    plt.axis('off')
    io.imshow(frame_2)

    plt.subplot(223)
    plt.title('3rd frame')
    plt.axis('off')
    io.imshow(frame_3)

    plt.subplot(224)
    plt.title('4th frame')
    plt.axis('off')
    io.imshow(frame_4)

    if write:
        io.imsave('images/%s_frame_1.png' % dataset, frame_1)
        io.imsave('images/%s_frame_2.png' % dataset, frame_2)
        io.imsave('images/%s_frame_3.png' % dataset, frame_3)
        io.imsave('images/%s_frame_4.png' % dataset, frame_4)
    else:
        plt.show()


# helper function to visualize spectrogram
def visualize_spectrogram(dataset, write=False):
    # para dataset: name of dataset (cuave / avletter)
    # para write: whether or not to write spectrogram to external files
    filename = data_config['data']['visualized']['spectrogram'][dataset]
    specs = np.load(filename)
    f_1, t_1, Sxx_1 = specs[0]
    f_2, t_2, Sxx_2 = specs[1000]
    f_3, t_3, Sxx_3 = specs[2000]
    f_4, t_4, Sxx_4 = specs[3000]

    fig = plt.figure(figsize=(15, 15))

    plt.subplot(221)
    plt.title('1st spectrogram')
    plt.xlabel('time')
    plt.ylabel('freq')
    plt.pcolormesh(t_1, f_1, Sxx_1)

    plt.subplot(222)
    plt.title('2nd spectrogram')
    plt.xlabel('time')
    plt.ylabel('freq')
    plt.pcolormesh(t_2, f_2, Sxx_2)

    plt.subplot(223)
    plt.title('3rd spectrogram')
    plt.xlabel('time')
    plt.ylabel('freq')
    plt.pcolormesh(t_3, f_3, Sxx_3)

    plt.subplot(224)
    plt.title('4th spectrogram')
    plt.xlabel('time')
    plt.ylabel('freq')
    plt.pcolormesh(t_4, f_4, Sxx_4)

    if write:
        fig.savefig('./images/%s_spectrogram.png' % dataset)
    else:
        plt.show()