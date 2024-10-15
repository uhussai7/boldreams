# BOLDreams
boldreams is a suite of tools to train neural encoding models using the <a href=https://naturalscenesdataset.org/> Natural Scenes Dataset (NSD)</a>. The family of encoding models we deal with are the ones based on a pre-trained visual or text backbone. Here, a particular focus is on the interpretability and explainbility (xAI) of these models.

### Configuration and data handling
After downloading the dataset, we need to define some paths as environment variables, these can be found in `/configs/config.py`. An approach we take here is to create a preprocessed dataset that only contains the visual cortex voxels. This is done with the `nsdhandling` class, sample usage can be found in `/scripts/make_preprocessed_data.py`.

### Training and prediction
Training is done with <a href='https://lightning.ai/docs/pytorch/stable/'> lightning </a> and is configurable with `.yaml` files. Various examples of such config files can be found in `/configs/`. A simple example is `alexnet.yaml`. Here one can define various parameters of the training, for example, which layers to use for feature extraction, `LAYERS_TO_EXTRACT`, and the percentage of filters to use per layer, `PERCENT_OF_CHANNELS`. A typical training script can be found in `/scripts/training_script.py`. The prediction are relatively straightforward and can be conducted using `/scripts/training_script.py`.    
