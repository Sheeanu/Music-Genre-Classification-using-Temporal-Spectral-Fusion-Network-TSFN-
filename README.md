# Music-Genre-Classification-using-Temporal-Spectral-Fusion-Network-TSFN-
 Music Genre Classification using Temporal-Spectral Fusion Network (TSFN) is a deep learning approach that separates tonal and timbral audio features into dual processing streams. Using cross-attention and gated fusion, the model effectively captures feature interactions and achieves 84.5% accuracy on the GTZAN dataset.




## Overview

Music Genre Classification is an important task in Music Information Retrieval that enables computers to automatically identify the genre of an audio track. This project proposes a **Temporal-Spectral Fusion Network (TSFN)**, a deep learning architecture designed to improve genre classification by modeling the relationship between tonal and timbral characteristics of music.

Unlike traditional approaches that combine all extracted audio features into a single vector, the proposed model processes tonal and timbral information separately and learns their interactions through a cross-attention mechanism. This allows the network to capture richer musical representations and achieve higher classification performance.


## Problem Statement

Most existing music genre classification methods treat all audio features equally by concatenating them into a single feature vector. This approach ignores the structural differences between tonal features, which describe pitch-related information, and timbral features, which describe sound texture and quality.

The objective of this work is to develop a model that can explicitly learn these distinct characteristics and their interactions, thereby improving genre classification accuracy.


## Dataset

The model is trained and evaluated using the GTZAN Dataset, a widely used benchmark dataset for music genre classification research. The dataset contains 1,000 audio tracks distributed equally across ten music genres. To ensure reliable evaluation and avoid data leakage, the dataset was divided into 800 training tracks and 200 testing tracks.


## Feature Extraction

Audio signals were processed into 3-second segments, and a total of 57 audio features were extracted from each segment. These features include MFCC coefficients, chromagram features, spectral centroid, spectral contrast, zero-crossing rate, and tempo-related information.

The extracted features were grouped into two categories:

* **Tonal Features:** Represent pitch and harmonic information.
* **Timbral Features:** Represent sound texture and spectral characteristics.

This separation forms the foundation of the proposed architecture.


## Proposed Methodology

The Temporal-Spectral Fusion Network consists of two parallel processing streams. One stream processes tonal features, while the other processes timbral features. Each stream applies feature encoding and channel-wise recalibration through Squeeze-and-Excitation blocks to identify the most informative feature channels.

A cross-attention mechanism is then employed to allow both streams to exchange information and learn relationships between tonal and timbral characteristics. The resulting feature representations are combined using a gated fusion module that adaptively determines the contribution of each stream. Finally, the fused representation is passed to a classifier for genre prediction.


## Training Strategy

The model was trained using the AdamW optimizer together with the OneCycleLR learning rate scheduler to improve convergence. CutMix augmentation was incorporated to enhance generalization and reduce overfitting. In addition, a LightGBM-based ensemble classifier was used during the final prediction stage to improve classification performance.


## Results

The proposed TSFN model achieved a track-level classification accuracy of **84.5%**, outperforming several traditional and deep learning approaches reported in the literature. The results demonstrate that explicitly modeling tonal-timbral interactions contributes significantly to improved genre recognition.

Compared with conventional methods such as Gaussian Mixture Models, Multi-Layer Perceptrons, and standard Convolutional Neural Networks, TSFN provides better feature representation and more effective learning of musical characteristics.


## Key Contributions

* Introduced a dual-stream architecture for separate tonal and timbral processing.
* Developed a cross-attention mechanism for feature interaction learning.
* Applied gated fusion for adaptive combination of audio representations.
* Achieved improved classification accuracy on the GTZAN dataset.
* Reduced the limitations of conventional single-vector feature representations.


## Applications

The proposed system can be applied in music recommendation platforms, playlist generation systems, music library organization, content-based retrieval systems, and other intelligent audio analysis applications.


## Future Work

Future research may focus on integrating physiological signals such as EEG data for emotion-aware music analysis. The development of lightweight and real-time implementations for mobile and edge devices also represents an important direction for practical deployment.


## Technologies Used

* Python
* PyTorch
* Librosa
* NumPy
* Pandas
* Scikit-learn
* LightGBM


## Author

**Anushri Pramanik**
B.Tech CSE (IoT, CS, BT)
University of Engineering and Management, Kolkata


## License

This project is intended for academic and research purposes.
