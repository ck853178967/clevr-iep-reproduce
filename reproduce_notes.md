
# reproduce notes
## v0. the pth version for the original repo is too old
```
ERROR: torch-0.1.11.post5-cp35-cp35m-linux_x86_64.whl is not a supported wheel on this platform.
```
> the error message still emerges even if we re-create a env with python 3.5

and no torch==0.1 version available now.
```
# pip install torch==
ERROR: Could not find a version that satisfies the requirement torch== (from versions: 1.4.0, 1.5.0, 1.5.1, 1.6.0, 1.7.0, 1.7.1, 1.8.0, 1.8.1, 1.9.0, 1.9.1, 1.10.0, 1.10.1, 1.10.2, 1.11.0, 1.12.0, 1.12.1, 1.13.0)
```

## v1, 202212, pytorch version 1.8.0

### fork and git clone
original repo : https://github.com/facebookresearch/clevr-iep

### 1. scripts/ relative import
[final solution] just `ln -s ../iep scripts/`

### setup, env and requirements
instead of create new virtual env, we just use the default conda env for test.

requirements 
```
numpy
Pillow
scipy
torchvision
h5py
```
are all met. the torch version is 1.8.0 instead of 0.1.11, we may need to modify some codes later.

### pretrained models
the models are downloaded before in ../iep_pretrained_models, then
`ln -s ../iep_pretrained_models models`


### run example

```
python scripts/run_model.py \
  --program_generator models/CLEVR/program_generator_18k.pt \
  --execution_engine models/CLEVR/execution_engine_18k.pt \
  --image img/CLEVR_val_000013.png \
  --question "Does the small sphere have the same color as the cube left of the gray cube?"
```

#### bugs
##### 1. ImportError: cannot import name 'imread' from 'scipy.misc'
use pil image instead

and mod code in run_single_example in run_model.py

##### 2. IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)

`y.size(1)` -> `y.view(y.size(0), -1).size(1)`

##### 3. y_embed view t dim even it's 1.
```
  File "/data/hanfei.ck/coding/clevr-iep/clevr-iep-fork/scripts/iep/models/seq2seq.py", line 98, in decoder
    rnn_input = torch.cat([encoded_repeat, y_embed], 2)
RuntimeError: Tensors must have same number of dimensions: got 2 and 3
```

-> `y_embed = self.decoder_embed(y).view(N, T_out, D)`


##### 4. softmax specify dim
```
/data/hanfei.ck/coding/clevr-iep/clevr-iep-fork/scripts/iep/models/seq2seq.py:172: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
  probs = F.softmax(logprobs.view(N, -1)) # Now N x V
  
...
```

-> `probs = F.softmax(logprobs.view(N, -1), dim=-1)`

##### 5. type is variable -> tensor
```
  File "/data/hanfei.ck/coding/clevr-iep/clevr-iep-fork/scripts/iep/models/module_net.py", line 238, in forward
    raise ValueError('Unrecognized program format')
ValueError: Unrecognized program format
```

-> `elif type(program) is torch.Tensor and program.dim() == 2:`

##### 6. .item()
```
  File "/data/hanfei.ck/coding/clevr-iep/clevr-iep-fork/scripts/iep/models/module_net.py", line 186, in _forward_modules_ints_helper
    fn_str = self.vocab['program_idx_to_token'][fn_idx]
KeyError: tensor(5, device='cuda:0')
```

-> `fn_idx.item()`

> and some similar bugs like 

```
IndexError: invalid index of a 0-dim tensor.
```

`predicted_answer_idx[0]` -> `predicted_answer = vocab['answer_idx_to_token'][predicted_answer_idx.item()] ## debug`

##### 7 contiguous view
```
  File "/data/hanfei.ck/coding/clevr-iep/clevr-iep-fork/scripts/iep/models/layers.py", line 53, in forward
    return x.view(x.size(0), -1)
RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```
-> `contiguous().view(...)`


> single sample test finish.

----

### training (and validation)

#### preprocessing clevr
- download data
- extract image features
- preprocess questions

these steps finish in previous reproduce process. prepare the data folder by link and copy. 

```
[11:06:02 root@jupyter-hanfei-2eck:data]# ll
total 0
lrwxrwxrwx 1 root root 41 Dec 13 11:03 CLEVR_v1.0 -> /data/hanfei.ck/datasets/CLEVR/CLEVR_v1.0
lrwxrwxrwx 1 root root 58 Dec 13 11:04 test_features.h5 -> /root/hanfei.ck/coding/ns-2022/data_cache/test_features.h5
lrwxrwxrwx 1 root root 59 Dec 13 11:06 test_questions.h5 -> /root/hanfei.ck/coding/ns-2022/data_cache/test_questions.h5
lrwxrwxrwx 1 root root 59 Dec 13 11:04 train_features.h5 -> /root/hanfei.ck/coding/ns-2022/data_cache/train_features.h5
lrwxrwxrwx 1 root root 60 Dec 13 11:05 train_questions.h5 -> /root/hanfei.ck/coding/ns-2022/data_cache/train_questions.h5
lrwxrwxrwx 1 root root 57 Dec 13 11:04 val_features.h5 -> /root/hanfei.ck/coding/ns-2022/data_cache/val_features.h5
lrwxrwxrwx 1 root root 58 Dec 13 11:05 val_questions.h5 -> /root/hanfei.ck/coding/ns-2022/data_cache/val_questions.h5
lrwxrwxrwx 1 root root 52 Dec 13 11:05 vocab.json -> /root/hanfei.ck/coding/ns-2022/data_cache/vocab.json
[11:06:04 root@jupyter-hanfei-2eck:data]# 
```

#### training on clevr
- train pg
- train ee
- jointly train
- test

in previous reproduce, train pg show well results, while train ee failed (~50% acc) thus later jointly train also failed.

then we modify steps to:
- test downloaded model weights
- train ee using (downloaded pg model weights)/(trained pg in previous reproduce)

and before train_model, we add some tensorboard log codes and argparse items into the main scripts.

##### step 1: test downloaded pg and ee
```
python scripts/run_model.py \
  --program_generator models/CLEVR/program_generator_18k.pt \
  --execution_engine models/CLEVR/execution_engine_18k.pt \
  --input_question_h5 data/val_questions.h5 \
  --input_features_h5 data/val_features.h5
```

##### bugs
##### 1. keyerror -> tensor.item()
##### 2. warning: softmax not specify dim -> softmax(..., dim=-1)


##### results
Got 132973 / 149991 = 88.65 correct

##### test other downloaded models
```
python scripts/run_model.py \
  --program_generator models/CLEVR/program_generator_700k.pt \
  --execution_engine models/CLEVR/execution_engine_700k_strong.pt \
  --input_question_h5 data/val_questions.h5 \
  --input_features_h5 data/val_features.h5
```

Got 135653 / 149991 = 90.44 correct

> the difference from paper results may decrease after joint finetune? these models are not released.

> while it's still much better than our simply trained ee before, the training of ee might need carefully parameters tuning.

#### test on humans first

#### preprocessing on humans
- download data
- preprocess questions

link from previous data cache (omit the test set temp). [n samples 17817/ 7202/ 7145]

#### test on humans
train clevr finetune humans
```
python scripts/run_model.py \
  --program_generator models/CLEVR-Humans/human_program_generator.pt \
  --execution_engine models/CLEVR/execution_engine_18k.pt \
  --input_question_h5 data/val_human_questions.h5 \
  --input_features_h5 data/val_features.h5
```
Got 4622 / 7202 = 64.18 correct


train clevr (link val_human_questions_no_expand.h5 before, which is produced with old input vocab)
```
python scripts/run_model.py \
  --program_generator models/CLEVR/program_generator_18k.pt \
  --execution_engine models/CLEVR/execution_engine_18k.pt \
  --input_question_h5 data/val_human_questions_no_expand.h5 \
  --input_features_h5 data/val_features.h5
```
Got 3805 / 7202 = 52.83 correct


#### train ee on clevr
link pre trained pg (in previous reproduce )
```
...:log]# ll
total 0
lrwxrwxrwx 1 root root 67 Dec 13 17:30 202211241611_CLEVR_PG_18k -> /root/hanfei.ck/coding/ns-2022/saved_logs/202211241611_CLEVR_PG_18k
```

then modify train_model.py to add some checkpoint args and tensorboard records.

and train ee
```
python scripts/train_model.py \
  --model_type EE \
  --program_generator_start_from log/202211241611_CLEVR_PG_18k/program_generator_18k.pt \
  --num_iterations 100001 \
  --checkpoint_path [log_path]/execution_engine.pt \
  --log_model_name 202212131832_CLEVR_EE 
  
```

> no num_train_samples 18000 here?

<!--
self notes:
start check: (iter 1) train acc 0.209 val acc 0.212

1300 iter / ~0.5h on 1080, -> (infer) 10w iter around 2d
2.4G gpu mem cost, batch size 64

quake cmd
```
python scripts/train_model.py \
  --model_type EE \
  --program_generator_start_from /nas/coding/neural_symbolic/clevr-iep/log/202211241611_CLEVR_PG_18k/program_generator_18k.pt \
  --num_iterations 100001 \
  --checkpoint_path [log_path]/execution_engine.pt \
  --log_model_name 202212141220_CLEVR_EE \
  --log_path /nas/coding/neural_symbolic/clevr-iep/log/ \
  --data_cache_from_nas /nas/coding/neural_symbolic/data_cache
  
```
-->