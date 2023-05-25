# ANCL-master
This is official code implementation of the &lt;Achieving a Better Stability-Plasticity Trade-off via Auxiliary Networks in Continual Learning> in CVPR 2023

## How To Use
We implement our code based on FACIL available online https://github.com/mmasana/FACIL

Below instructions are from FACIL directory and no additional library is required to run our code. Please refer to above link to find more information to run code with different setting.

<details>
  <summary>Optionally, create an environment to run the code (click to expand).</summary>

  ### Using a requirements file
  The library requirements of the code are detailed in [requirements.txt](requirements.txt). You can install them
  using pip with:
  ```
  python3 -m pip install -r requirements.txt
  ```

  ### Using a conda environment
  Development environment based on Conda distribution. All dependencies are in `environment.yml` file.

  #### Create env
  To create a new environment check out the repository and type: 
  ```
  conda env create --file environment.yml --name FACIL
  ```
  *Notice:* set the appropriate version of your CUDA driver for `cudatoolkit` in `environment.yml`.

  #### Environment activation/deactivation
  ```
  conda activate FACIL
  conda deactivate
  ```

</details>

## Citations 

Please cite our paper if it is helpful to your work:

```bibtex
@InProceedings{Kim_2023_CVPR,
    author    = {Kim, Sanghwan and Noci, Lorenzo and Orvieto, Antonio and Hofmann, Thomas},
    title     = {Achieving a Better Stability-Plasticity Trade-Off via Auxiliary Networks in Continual Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {11930-11939}
}
```

