This folder includes the yml file defining the conda python environment that I have set up on the server.
This yml file can be used to create a new environment on local that matches the one on the server. Note that the original
yml file contained a lot of details both on packages but also on OS specific versions of these packages. For the local
yml file I have removed these details so that the file would be trully crossplatform.


Create yml file from existing environment:
"conda env export > environment.yml"

Create conda environment from existing yml file:
"conda env create -f environment.yml"