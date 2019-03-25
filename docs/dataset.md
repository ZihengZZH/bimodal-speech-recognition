# CUAVE dataset

> a new, flexible, and fairly comprehensive audiovisual database

The CUAVE database is a speaker independent corpus of over 7,000 utterances of both connected and isolated digits. 

## description

36 speakers saying the digits *0* to *9*. The *normal* portion of the database contains frontal facing speakers saying each digits 5 times. 

In the ```cuave-group-aligned``` directory, there are extracted and aligned faces, along with the audio features for all "group" sequences in the CUAVE database.

Every individual sequence contains a 2 element cell array containing grayscale video frames, one 75\*50\*n frame matrix for each person. It also has frame indexed raw audio and MFCCs in variables ```audioIndexed``` and ```mfccs```. Ground truth labeling is in the variable "labels". 

The elements contained in each ```g_aligned.mat``` are listed as follows.
* video { [75\*50\*957] [75\*50\*957] }
* audioIndexed
* mfccs
* labels
* frameNumbers
* fps
* fs
* flowX
* flowY
* flowStats



## tree-structure

```tree
|__ dataset
|   |__ cuave-group-aligned
|   |   |__ g[01:22]_aligned.mat
|   |__ dataset2.mat
```