<p align="center">
    <img src="https://github.com/user-attachments/assets/0c03d65c-31ef-4abe-bfae-e3d59ee029fb" width="100%"><br>
    <a href="https://arxiv.org/abs/2506.21348" target="_blank">📖 <b>Paper</b></a> • 
    <a href="https://youtu.be/f1YY1XJJ-HA" target="_blank">🎬 <b>Overview</b></a> • 
    <a href="#cite">🔖 <b>Cite</b></a>
</p>

<br>

Official implementation of `PanSt3R: Multi-view Consistent Panoptic Segmentation`.
Presented at **ICCV 2025**.

<p align="center">
    <img src="https://github.com/user-attachments/assets/956de4bb-4173-4b62-a8d0-e883660c4128" height="300">
    <img src="https://github.com/user-attachments/assets/add7456a-8358-4d17-8e96-c611c41c92bd" height="300">
    <br/>
    PanSt3R: Panoptic 3D reconstruction examples
</p>


## Table of Contents

- [License](#license)
- [Getting started](#getting-started)
    - [Installation](#installation)
    - [Running the demo](#running-the-demo)
    - [Checkpoints](#checkpoints)
- [Training](#training)
    - [Preparing the data](#preparing-the-data)
    - [Training](#training-1)
- [Cite](#cite)


## License
PanSt3R is released under the PanSt3R Non-Commercial License. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for more information.  
[NOTICE](NOTICE) also contains information about the datasets used to train the checkpoints.

## Getting started

### Installation

Setup tested on **Python 3.11**.

1. Clone repository
    ```bash
    git clone https://github.com/naver/panst3r.git
    cd panst3r
    ```

2. Install PyTorch (follow [official instructions](https://pytorch.org/get-started/locally/) for your platform)
    ```bash
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
    ```

3. (optional) Install xFormers (for memory-efficient attention):
    ```bash
    pip install -U xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu126 # Use appropriate wheel for your setup
    ```

4. Install PanSt3R with dependecies
    ```bash
    pip install -e . 
    ```

5. (optional) Install cuRoPE extension
    Make sure you have appropriate CUDA versions installed and in your `$PATH` and `$LD_LIBRARY_PATH`. Then you can build the extension by:
    ```bash
    pip install --no-build-isolation "git+https://github.com/naver/croco.git@croco_module#egg=curope&subdirectory=curope"
    ```

### Running the demo

We include a Gradio demo for running PanSt3R inference. You can run it with the following command.

```bash
python gradio_panst3r.py --weights /path/to/model/weights
```

<details>

<summary><strong>Optional arguments</strong></summary>

- `--weights`: Path to model weights for inference
- `--retrieval`: Path to retrieval weights for open-vocabulary segmentation
- `--server_name`: Specify the server URL (default: `127.0.0.1`)
- `--server_port`: Choose the port for the Gradio app (default: auto-select starting at `7860`)
- `--viser_port`: Choose the port for the embedded viser visualizer (default: auto-select starting at `5000`)
- `--image_size`: Sets input image size (select according to the model you use)
- `--encoder`: Override config for the MUSt3R encoder
- `--decoder`: Override config for the MUSt3R decoder
- `--camera_animation`: Enable camera animation controls in the visualizer
- `--allow_local_files`: Allow loading local files in the app
- `--device`: PyTorch device to use (`cuda`, `cpu`, etc.; default: `cuda`)
- `--tmp_dir`: Directory for temporary files
- `--silent` / `--quiet` / `-q`: Suppress verbose output
- `--amp`: Use Automatic Mixed Precision (`bf16`, `fp16`, or `False`; default: `False`)

</details>
<br>

**Using the demo**
1. Upload images
2. Select parameters (number of keyframes, prediction classes, and the postprocessing approach)
3. Click "Run"
4. After prediction and postprocessing is complete, the 3D reconstruction will appear in the "Visualizer" panel.

![Demo Instructions](https://github.com/user-attachments/assets/938131ea-8549-494a-8ba0-c50fae2634bf)

> [!NOTE]
> Keyframes are selected at the start and processed together. Predictions for the remaining frames are performed individually, based on the memory features and decoded queries from the keyframes.
>
> For typical use, set the number of keyframes equal to the number of input images. For large image collections (more than 50 images), reducing the number of keyframes helps manage memory usage. 

> [!WARNING]
> The demo currently does not properly implement the support for multiple sessions.

> [!CAUTION]
> Using `--allow_local_files` allows users to load images from a local path where the server is hosted. Use with caution.

### Checkpoints

The following checkpoints are available for download. We include PQ scores obtained via direct multi-view prediction on the rendered test images (without LUDVIG).

| Model | hypersim | replica | scannet | MD5 Checksum |
|-------|----------|---------|---------|--------------|
| [PanSt3R_512_5ds](https://download.europe.naverlabs.com/ComputerVision/PanSt3R/panst3r_v1_512_5ds.pth) | 56.5 | 62.0 | 65.7 | `c3836c108f1bf441fe53776e825cd1ac` |

## Training

We also provide an example of training (preprocessing, dataloaders, training code) on ScanNet++.

### Preparing the data

We include a preprocessing script, based on the preprocessing script for MUSt3R. It prepares the related 3D targets (camera parameters, depthmaps) and panoptic masks. Depthmaps and panoptic masks are rendered using *pyrender*.

1. Download the ScanNet++V2 data
2. Prepare image pairs (e.g. random sampling with ensured overlap).   
   Precomputed pairs for ScanNet++V2 avaliable for download [here](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/scannetpp_v2_pairs.zip).
3. Run the preprocessing script:
```bash
python tools/preprocess_scannetpp.py \
--root_dir /path/to/scannetppv2 \
--pairs_dir /path/to/pairs_dir \
--output_dir /path/to/output_dir
```

> [!WARNING]
> We use a fork of `pyrender` with an added shader for rendering instance masks (without anti-aliasing). If you followed the installation steps in a fresh environment is should be installed automatically. Manual installation is possible via `pip install "git+https://github.com/lojzezust/pyrender.git"`.

> [!TIP]
> Use `--pyopengl-platform "egl"` if running in headless mode (e.g. on a remote server).


### Training

After the data is prepared, you can run training via the provided training script and configurations.

> [!NOTE]
> Set paths to your preprocessed data and checkpoints in `config/base.yaml`.  
> - If training from scratch provide a MUST3R checkpoint (can be found [here](https://github.com/naver/must3r#checkpoints)).  
> - If fine-tuning from PanSt3R, provide a path to the PanSt3R checkpoint.

```bash
# Limit CPU threading to prevent oversubscription in multi-GPU training
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

Single GPU training (or debug):

```bash
python train.py --config-name=base
```

For distributed training use `torchrun`:

```bash
# 2 GPUs on one node
torchrun --nnodes=1 --nproc_per_node=2 \
    train.py --config-name=distributed
```

## Cite

```bibtex
@InProceedings{zust2025panst3r,
  title={PanSt3R: Multi-view Consistent Panoptic Segmentation},
  author={Zust, Lojze and Cabon, Yohann and Marrie, Juliette and Antsfeld, Leonid and Chidlovskii, Boris and Revaud, Jerome and Csurka, Gabriela},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2025}
}
```
