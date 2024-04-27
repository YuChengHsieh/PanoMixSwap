## PanoMixSwap

PanoMixSwap introduces a novel data augmentation approach, wherein indoor panoramic images are segmented into three components: background style, layout, and foreground furniture. These segments are then mixed to produce diverse samples. The method leverages mofel from [SEAN](https://github.com/ZPdesu/SEAN) and integrates Panostretch technique from [HorizonNet](https://github.com/sunset1995/HorizonNet) to enhance its efficacy. You can find more details in the original paper linked [here](https://arxiv.org/abs/2309.09514).

## Setting Environment
1. Clone this project
   ```bash
    git clone https://github.com/YuChengHsieh/PanoMixSwap
   ```
3. Create a virtual environment with `conda` and activate it
   ```bash
   conda create -n PanoMixSwap python=3.7.15
   conda activate PanoMixSwap
   ```
5. Install required packages
   ```bash
   pip install -r requirements.txt
   ```

## Usage
We only support 3D Indoor panoramic Dataset "Structured3D" and "Stanford2D3D". Just execute the command below
```
python PanoMixSwap.py 
    --dataset Structured3D or Stanford2D3D 
    --layout_path path/to/layout 
    --image_path path/to/image
    --results_root path/to/results
```

## Citation
```
@article{park2021nerfies,
  title     = {PanoMixSwap – Panorama Mixing via Structural Swapping for Indoor Scene Understanding},
  author    = {Yu-Cheng Hsieh and Cheng Sun and Suraj Dengale and Min Sun},
  journal   = {British Machine Vision Conference (BMVC)},
  year      = {2023},
}
```