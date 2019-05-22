# Low light enhance for driving scene
Images of road scene in low-illumination environment are often loss the details revealing the essence of images which increases the danger when driving at night, especially in those areas without any light. Therefore, an efficient deblurring, enhancement algorithm for low-illumination images is necessary. Here is a CNN based model to restore the low-illumination image.
![low_illumination_disp](pictures/display.png)

## Methodology
Our model was inspired by U-net, we used some 1x1 conv to replace 3x3 conv which make it more efficient. Besides, we add (["group normalization"](https://arxiv.org/abs/1803.08494)) in each layer before activation function. During training, we randomly add gaussian blur, gaussian noise to enhance the generalization performance of the model.
<div align=center><img width="600" height="212" src="pictures/net_structure.png"></div>
