# Removing Rain in Videos: A Large-scale Database and A Two-stream ConvLSTM Approach

Tie Liu, Mai Xu, Zulin Wang

[[Paper Link](https://arxiv.org/abs/1906.02526)] (ICME'19)

Rain removal has recently attracted increasing research attention, as it is able to enhance the visibility of rain videos. However, the existing learning based rain removal approaches for videos suffer from insufficient training data, especially when applying deep learning to remove rain. In this paper, we establish a large-scale video database for rain removal (LasVR), which consists of 316 rain videos. Then, we observe from our database that there exist the temporal correlation of clean content and similar patterns of rain across video frames. According to these two observations, we propose a two-stream convolutional long- and short- term memory (ConvLSTM) approach for rain removal in videos. The first stream is composed of the subnet for rain detection, while the second stream is the subnet of rain removal that leverages the features from the rain detection subnet. Finally, the experimental results on both synthetic and real rain videos show the proposed approach performs better than other state-of-the-art approaches.

<p align="center">
<img src="src_img/fig1.png"/>
</p>

## LasVR-Dataset
![Video_class](./src_img/dataset_example.png)

**A large-scale video database for rain removal (LasVR)**. The videos of our database contain a wide range of content categories, e.g., animal, nature landscapes, human scenes, action sports, man-made object and so forth. Additionally, the rendered streaks vary from light drizzling to heavy rain storm and vertical rain to slash line. The training, validation and testing sets consist of 261, 27 and 28 rain videos.

**Download LasVR-Dataset:**

[[Dropbox](https://www.dropbox.com/s/cf1q283dvwvpby9/LasVR.zip?dl=0)]     [[BaiduYun](https://pan.baidu.com/s/1uyrCiK7rtFzOYKGBaf_D5w)] (passpord:bwg0)

## Framework
![Video_class](./src_img/framework.png)

      @article{liu2019removing,
        title={Removing Rain in Videos: A Large-scale Database and A Two-stream ConvLSTM Approach},
        author={Liu, Tie and Xu, Mai and Wang, Zulin},
        journal={arXiv preprint arXiv:1906.02526},
        year={2019}
        }

       @INPROCEEDINGS{8784876,
       author={T. {Liu} and M. {Xu} and Z. {Wang}},
       booktitle={2019 IEEE International Conference on Multimedia and Expo (ICME)},
       title={Removing Rain in Videos: A Large-Scale Database and a Two-Stream ConvLSTM Approach},
       year={2019},
       volume={},
       number={},
       pages={664-669},
       keywords={convolutional neural nets;feature extraction;image enhancement;image sequences;learning (artificial intelligence);rain;recurrent neural nets;video databases;video signal processing;video frames;rain detection;ConvLSTM;rain removal;rain video visibility enhancement;convolutional long-and short-term memory;Rain;Videos;Databases;Feature extraction;Correlation;Deep learning;Convolution;Rain removal;convolutional LSTM},
       doi={10.1109/ICME.2019.00120},
       ISSN={1945-7871},
       month={July},}


## Contact
liutie@buaa.edu.cn, tieliu0@gmail.com (Tie Liu)
