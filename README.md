# A-dual-branch-network-for-ultrasound-image-segmentation

Ultrasonic imaging has become one of the most commonly used imaging modalities in medical diagnosis due to its advantages such as relative safety, low cost, non-invasiveness, and real-time display. However,
ultrasonic waves produce speckle noise due to reflections from different types of tissue. Additionally, uneven reflections of the signal may cause artifacts, further reducing image quality. These challenges
pose significant obstacles to the ultrasound image segmentation task Convolutional Neural Networks (CNNs) have been proven effective for ultrasound image segmentation tasks. This paper proposes a
Dual-Branch Ultrasound Image Segmentation Network (DBUNet) based on Unet architecture. The network consists of four main components: an enhanced branch, an original branch, a feature aggregation module, and a decoding module. The enhanced branch combines signal processing techniques such as filtering and histogram equalization with attention-based denoising. The original branch extracts
comprehensive inter-image information to supplement the details lost by the enhancement operation. A Deep Feature Aggregation Module (DFAM) is designed to efficiently fuse deep features from different
branches. In DFAM, a channel reconstitution module is used to refine channel-level features. Next is cross-fusion to facilitate cross-feature information exchange and generate fusion attention maps to guide feature fusion generation. In addition, a Shallow Feature Optimization Module (SFOM) is proposed to retain important information and suppress unimportant information using a separation-reconstruction strategy to achieve spatial redundancy optimization. An attention mechanism is introduced to achieve feature denoising. This network was compared with state-of-the-art segmentation methods to evaluate its segmentation performance using five quantitative evaluation indicators on the publicly available ultrasound breast cancer dataset BUSI and the thyroid ultrasound dataset DDTI. Experimental results show that the proposed method outperforms state-of-the-art ultrasound image segmentation methods。

## Cite
Zhu, Z., Zhang, Z., Qi, G., Li, Y., Li, Y., & Mu, L. (2025). A dual-branch network for ultrasound image segmentation. Biomedical Signal Processing and Control, 103, 107368.
\\\\\\
@article{zhu2025dual,
  title={A dual-branch network for ultrasound image segmentation},
  author={Zhu, Zhiqin and Zhang, Zimeng and Qi, Guanqiu and Li, Yuanyuan and Li, Yuzhen and Mu, Lan},
  journal={Biomedical Signal Processing and Control},
  volume={103},
  pages={107368},
  year={2025},
  publisher={Elsevier}
}



