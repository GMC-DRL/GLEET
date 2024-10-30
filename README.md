# GLEET
 
An python implementation for GLEET: "Auto-configuring Exploration-Exploitation Tradeoff in Evolutionary Computation via Deep Reinforcement Learning", which is accepted by GECCO 2024.

### Requirement
```
pip install -r requirements.txt
```

### Running

* **To train GLEET**
```
python run.py --train --problem=pro_id --dim=problem_dimension --backbone=backbone_algorithm [--run_name=run_name_to_specify --resume=resume_path]
```
where `--train` switch the running mode to training. `--problem` is the function id for training, which range from `1` to `10` in CEC2021 dataset. `--problem_dimension` is the dimension of running function, you can specify it as `10` or `30`. `--backbone` specify the backbone algorithm for GLEET, where you can specify it to be `PSO` or `DMSPSO`. `--run_name` specify the running name for this run for better identification. `--resume` specify the loading path for GLEET if you want to resume training base on any former model.

An example can be:
```
python run.py --train --problem=1 --dim=10 --backbone=PSO --run_name=train_problem1_10dim
```
which means we start training GLEET with PSO as backbone algorithm in 10 dimensional 1st problem (Bent cigar to be specific) from scratch.

In addition, the training output model will be saved in ./outputs, and you can load from it if needed. The training logging output will be saved in ./logs.


* **To rollout GLEET or the backbone algorithm**
```
python run.py --test --problem=pro_id --dim=problem_dimension --backbone=backbone_algorithm [--load_path=path to load for GLEET]
```
where `--test` switch the running mode to testing. The meanings of `problem`, `dim`, `backbone` are the same as before. `--load_path` specify the loading path of some former trained model. If the `load_path` is given, the backbone algorithm will be running under the control of the loaded agent. If the `load_path` isn't given, the backbone algorithm will be running under the default configuration without any control.

An example can be:
```
python run.py --test --problem=1 --dim=10 --backbone=DMSPSO --load_path=outputs/func1_10.pt
```
which means we firstly load a pre_trained model from load_path and use it to control DMSPSO in 10 dimensional 1st problem (Bent cigar to be specific).


## Citing GLEET

The PDF version of the paper is available [here](https://arxiv.org/pdf/2404.08239). If you find our GLEET useful, please cite it in your publications or projects.

```latex
@inproceedings{ma2024auto,
  title={Auto-configuring exploration-exploitation tradeoff in evolutionary computation via deep reinforcement learning},
  author={Ma, Zeyuan and Chen, Jiacheng and Guo, Hongshu and Ma, Yining and Gong, Yue-Jiao},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference},
  pages={1497--1505},
  year={2024}
}
```
