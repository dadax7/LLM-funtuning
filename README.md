# LLM-funtuning
This project fine-tunes the weights of the pre-trained DistilBERT model from huggingface. Datasets from Google GoEmotions (https://github.com/google-research/google-research/tree/master/goemotions), which contains about 58000 from Reddit comments, It is annotated as 27 emotions and neutral emotions.
# install
```bash
pip install -r requirements.txt
```
# train
python train.py
# inference
python inference.py

### Considerations
To train your own dataset, change the associated label definition and path in config

## Configuration instructions

The following parameters can be adjusted in 'config.py' :

### Dataset configuration
- 'dataset_name' : dataset name
- 'max_samples' : Maximum number of samples (None means all)
- 'num_labels' : The number of class labels
- 'label_names' : List of label names

### Model Configuration
- 'model_name' : The name of the pre-trained model
- 'local_model_path' : the local model path
- 'use_local_model' : Whether to use a local model
- 'max_length' : Maximum sequence length

### Training configuration
- 'batch_size' : the batch size
- 'learning_rate' : The learning rate
- 'num_epochs' : number of training rounds
- 'warmup_steps' : The number of steps to warmup
