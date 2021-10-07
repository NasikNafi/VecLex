
# VecLex Reward-based Summarization of Disaster-Related Document

This repository contains the code for the following ISCRAM 2020 paper:\
*[Abstractive Text Summarization of Disaster-Related Documents](http://idl.iscram.org/files/nasikmuhammadnafi/2020/2279_NasikMuhammadNafi_etal2020.pdf)*.

If you use this code, please cite our paper:
```
@inproceedings{nafi2020abstractive,
  title={Abstractive Text Summarization of Disaster-Related Document},
  author={Nafi, Nasik Muhammad and Bose, Avishek and Khanal, Sarthak and Caragea, Doina and Hsu, William H},
  booktitle={ISCRAM 2020 Conference Proceedings--17th International Conference on Information Systems for Crisis Response and Management},
  year={2020}
}
```

Our code is largely based on *[this](https://github.com/ChenRocks/fast_abs_rl)* implementation and the corresponding paper is available *[here](https://arxiv.org/abs/1805.11080)*.
  
  
  
## Dependencies

-  **Python 3**

-  [PyTorch](https://github.com/pytorch/pytorch) (GPU and CUDA enabled installation is preferred)
-  [cytoolz](https://github.com/pytoolz/cytoolz)
-  [gensim](https://github.com/RaRe-Technologies/gensim)
-  [pandas](https://pandas.pydata.org/)
-  [scikit-learn](https://scikit-learn.org/stable/install)
-  [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)

You can use any type of python package manager (*pip/conda*) to install the dependencies.
The code is tested on the *Linux* operating system using python 3.8 and pytorch 1.8.1.

This code provides the implementation of our best model which uses Word2Vec-based vector similarity and lexicon matching reward to train the summarizer.



## Decoding summaries using the pre-trained model

Download the pre-trained RL-based model from *[here](https://drive.google.com/file/d/1RsP8O13B3zYEHFNrRSBXC3aOqwtGMEFf/view?usp=sharing)*. 

You need to have your data preprocessed according to the CNN/DailyMail dataset preprocessing instructions available *[here](https://github.com/ChenRocks/cnn-dailymail)*. Then you can extract the disaster-related documents using your own method of choice. 

When you are ready with the above steps, please configure the **data path** by setting the following environment variable:

`export DATA=[path/to/extracted/data]`

To generate summaries using the pre-trained model, run:
```
python decode_full_model.py --path=[path/to/save/generated/summary] --model_dir=[path/to/pretrained/model] --beam=[beam_size] [--test/--val]
```

In our experiment, we have used *beam_szie=1*. You can explicitly keep your preprocessed data in a folder named *test* inside the data path and use the option *- - test* if you are using your own dataset other than CNN/Dailymail. 

  
## Evaluating average VecLex score of your generated summaries

First, you need to download the Google’s pre-trained word2vec model from *[here](https://drive.google.com/file/d/12DzHGI-Ollv5gDy2O48FVx4yZf1hNIW8/view?usp=sharing)* and the CrisisLex lexicon file from *[here](http://crisislex.org/crisis-lexicon.html#collapseOne)*. Then, configure the paths in *path_config.py* file.

**Word2Vec Model Path:** Update the path to the downloaded pre-trained word2vec file.\
`word2vec_filepath = [path/to/the/downloaded/word2vec/file]`

**Lexicon File Path:** Update the path to the downloaded lexicon file path.\
`lexicon_filepath = [path/to/the/downloaded/crisislex/file]`

\
Next, create the reference (ground truth) files for evaluation:
```
python make_eval_references.py
```

Finally, to evaluate, run:
```
python eval_vec_lex.py --reference_dir=[/path/to/reference/summaries] --summary_dir=[path/to/generated/summaries]
```

## Training your own model from the scratch

Please follow the same instructions to download and preprocess the CNN/DailyMail dataset. Then, to train a version of our best model follow the steps mentioned below:


**STEP 1:**
Download the Google's pre-trained word2vec model and the lexicon file for VecLex score calculation. Configure all the paths (data/word2vec/lexfile) as mentioned earlier. 

**STEP 2:**
Train a local word2vec word embedding to train extractor and abstractor in the next step:
```
python train_word2vec.py --path=[path/to/save/word2vec/model]
```
Additionally, you may specify the word vector dimension (the default is 128).

**STEP 3:**
Construct the pseudo-labels.
```
python make_extraction_labels.py
```

**STEP 4:**
Train abstractor and extractor model using the Maximum Likelihood objectives.
```
python train_abstractor.py --path=[path/to/save/the/abstractor/model] --w2v=[path/to/trained/word2vec/word2vec.128d.135k.bin]
```

```
python train_extractor_ml.py --path=[path/to/save/the/extractor/model] --w2v=[path/to/trained/word2vec/word2vec.128d.135k.bin]
```
Check the corresponding files to learn more about the additonal arguments. 

**STEP 5:**
Train the final RL-based model using the VecLex reward.
```
python train_full_rl.py --path=[path/to/save/the/model] --abs_dir=[path/to/pretrained/abstractor/model] --ext_dir=[path/to/pretrained/extractor/model]
```

Now, if you want to decode and evaluate using the newly trained model, follow the instructions from the previous sections.