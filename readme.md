### Ensure all requirements are installed from requirements.txt, by running `pip install -r requirements.txt`
### Imprtant: Ensure Cuda GPU is enabled for max performance, otherwise will run on CPU which is orders slower.

### Open the denoiser.ipynb notebook using jupyter notebooks and run the code from there to perfrom video denoising: 
- Open command line in the current directory and run `jupyter notebook`

### The noisy frames should be in the "./Source/" folder named "00000.png", "00001.png" and so on

### The output denoised image sequence will be saved under "Denoised/" named "00000.png", "00001.png" and so on

- The image seqeunce can then be exported to a video file using the appropriate settings you require, e.g. using FFMPEG