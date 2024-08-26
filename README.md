[TOC]
## Contents 
 - src
    - <font color="#1936C9">net.py</font>: contain three model classes 
        - ==PhysicsInformedNN==: 
            - Original PINN, 
            - Training includes warm-up stage and physics stage. 
            - Source *f* is **not** included in PDE loss. 
        - ==Embed_PINN==: 
            - Train **without** warm-up stage.
            - Source *f* **is** included in PDE loss.
            - Init_loss is included. 
            - Requires a pre-trained warm-up checkpoint (.ckpt) to aid training.
        - ==Ultra_PINN==: 
            - Train **without** warm-up stage
            - source *f* **is** included in PDE loss. 
            - No init_loss.
    - <font color="#1936C9">pinn_homo.py</font>: orginal pinn for homogeneous case
    ```bash
    python3 pinn_homo.py --name 'OUTPUT FOLDER NAME' -- cuda 'CUDA DEVICE ID' --data 'GROUND TRUTH DATA PATH'
    ```
    - <font color="#1936C9">pinn_inhomo.py</font>: orginal pinn for inhomogeneous case
    ```bash
    python3 pinn_inhomo.py --name 'OUTPUT FOLDER NAME' -- cuda 'CUDA DEVICE ID' --data 'GROUND TRUTH DATA PATH' --map 'SOS MAP FILE PATH'
    ```
    - <font color="#1936C9">embedd_pinn.py</font>: embedded pinn for homogeneous case
    ```bash
    python3 embedd_pinn.py --name 'OUTPUT FOLDER NAME' -- cuda 'CUDA DEVICE ID' --data 'GROUND TRUTH DATA PATH' --model 'PRETRAINED MODEL CKPT PATH'
    ```
    - <font color="#1936C9">ultra_pinn.py</font>: ultra pinn for homogeneous case
    ```bash
    python3 ultra_pinn.py --name 'OUTPUT FOLDER NAME' -- cuda 'CUDA DEVICE ID' --data 'GROUND TRUTH DATA PATH'
    ```
     - <font color="#1936C9">utils.py</font>: contain utilily functions such as plotting and bilinear interpolation.
    
- visualize
    - <font color="#1936C9">animate.py</font>: create mp4 file based on the input .npz file. Must be shape of ('TIME', 'WIDTH','LENGTH')
    ```bash
    python3 animate.py --file '.npz file path to visualize'
    ```
    - <font color="#1936C9">readmat.py</font>: read .mat file to .npz
