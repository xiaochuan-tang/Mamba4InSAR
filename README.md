Running Instructions

1. Create a Conda Environment
   Create and activate a Python 3.8 Conda environment:
   
2. Install Dependencies
   Install the required packages:
   pip install -r requirements.txt
   
3. Go to the Mamba official repository/Release page and download the precompiled wheel (.whl) files for mamba\_ssm and causal\_conv1d that match your Python version, PyTorch version, and CUDA version.
   Install the downloaded files inside your Conda environment, for example: 

   pip install mamba\_ssm-xxx.whl
   pip install causal\_conv1d-xxx.whl

   After installation, replace the mamba\_simple.py file in the mamba\_ssm module with ./models/Mamba/mamba\_simple.py.
   
4. Run the Project
   Activate the environment and run the bash script:
   conda activate mamba\_env
   bash scripts/run.sh
