# BOLDreams
boldreams is a suite of tools to train neural encoding models using the <a href=https://naturalscenesdataset.org/>Natural Scenes Dataset(NSD)</a>. The family of encoding models we deal with are the ones based on a pre-trained visual or text backbone. Here, a particular focus is on the interpretability and explainbility (xAI) of these models.

### Configuration and data handling
After downloading the dataset, we need to define some paths as environment variables, these can be found in `/configs/config.py`. An approach we take here is to create a preprocessed dataset that only contains the visual cortex voxels. This is done with the `nsdhandling` class, sample usage can be found in `/scripts/make_preprocessed_data.py`.

### Training and prediction
Training is done with <a href='https://lightning.ai/docs/pytorch/stable/'>lightning</a> and is configurable with `.yaml` files. Various examples of such config files can be found in `/configs/`. A simple example is `alexnet.yaml`. Here one can define various parameters of the training, for example, which layers to use for feature extraction, `LAYERS_TO_EXTRACT`, and the percentage of filters to use per layer, `PERCENT_OF_CHANNELS`. A typical training script can be found in `/scripts/training_script.py`. The predictions are relatively straightforward and can be conducted using `/scripts/prediction_script.py`. 

### Dreams
We adapt the <a href='https://github.com/greentfrapp/lucent'>lucent</a> package to generate dreams with objective functions involving brain regions and voxels. One natural objective is to maximally activate a particule ROI, although any objective function can be defined. In `/scripts/dreams/abstract.py` we show typical usuage with objectives that maximally activate an ROI and also an objective function that promotes diversity in the dreams. 

Here is an example of the dreams for `subj01` that maximize activation in the face related areas,

<img src='https://github.com/uhussai7/images/blob/main/dreams.png' align='center' width='1080'>

Each coloumn shows the backbone used, for example, `Alexnet-25-False` denotes the Alexnet backbone, 25% of filters per layer and finetuning off. The second row shows a word cloud of top nouns where the similarity score is predicted using <a href='https://github.com/openai/CLIP'>CLIP</a>.

Here is an example for the same subject, dreams that maximize activation in the place related areas,
<img src='https://github.com/uhussai7/images/blob/main/places.png' align='center' width='1080'>

We see that the `CLIP` backbone (RN50x4) creates the most elborate dreams. 

Here is an example with the `CLIP` backbone (RN50x4) and a small diversity term in the objective,
<img src='https://github.com/uhussai7/images/blob/main/places_diversity.png' align='center' width='640'>

We can also generate dreams for retinotopic eccentricity ROIs,
<img src='https://github.com/uhussai7/images/blob/main/ecc_dream.png' align='center' width='640'>


### Implicit attention
We use the <a href='https://arxiv.org/abs/1703.01365'>integrated gradients</a> approach to calculated saliency maps for each ROI. Here are some examples of faces, bodies and places ROIs respectively,

<img src='https://github.com/uhussai7/images/blob/main/attention.png' align='center' width='640'>

 
