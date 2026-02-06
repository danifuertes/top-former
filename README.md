# TOP-Former: A Multi-Agent Transformer Approach for the Team Orienteering Problem

![](images/python-3.8.svg)
![](images/torch-1.13.1.svg)
![](images/cuda-11.7.svg)
![](images/cudnn-8.5.svg)

![TSP100](images/top.gif)

## Paper
Solve a variant of the Orienteering Problem (OP) called the Team Orienteering Problem (TOP) with a cooperative
multi-agent system based on Transformer Networks, called TOP-Former. For more details, please see our [paper](https://doi.org/10.48550/arXiv.2311.18662). If this repository is
useful for your work, please cite our paper:

```
@ARTICLE{fuertes2025,
  author={Fuertes, Daniel and del-Blanco, Carlos R. and Jaureguizar, Fernando and García, Narciso},
  title={TOP-Former: A Multi-Agent Transformer Approach for the Team Orienteering Problem}, 
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  volume={26},
  number={9},
  pages={13799-13810},
  year={2025},
  doi={10.1109/TITS.2025.3566157}
}
``` 

## Dependencies

* Python >= 3.8
* NumPy
* SciPy
* Numba
* [PyTorch](http://pytorch.org/) >= 1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib

## Usage

First, it is necessary to create test and validation sets:
```bash
python generate_data.py --name test --seed 1234 --graph_sizes 20 20 20 35 35 35 50 50 50 75 75 75 100 100 100 --max_length 1.5 2 2.5 1.5 2 2.5 1.5 2 2.5 1.5 2 2.5 1.5 2 2.5
python generate_data.py --name val --seed 4321 --graph_sizes 20 20 20 35 35 35 50 50 50 75 75 75 100 100 100 --max_length 1.5 2 2.5 1.5 2 2.5 1.5 2 2.5 1.5 2 2.5 1.5 2 2.5
```

To train a Transformer model (`attention`) use:
```bash
python run.py --problem top --model attention --val_dataset data/1depots/const/20/val_seed4321_L2.0.pkl --graph_size 20 --data_distribution const --num_agents 2 --max_length 2.0 --baseline rollout
```

and change the environment conditions (number of agents, graph size, max length, reward distribution...)
at your convenience.

Pretrained weights are available
[here](https://upm365-my.sharepoint.com/:u:/g/personal/d_fcoiras_upm_es/ESsdHv4PULNAlN2lqA9bThIBjrzr_rMHZCLqIG7Ar3HkOg?e=ko9IE5). You can unzip the file
with `unzip` (`sudo apt-get install unzip`):

```bash
unzip pretrained.zip
```

[Pointer Network](https://arxiv.org/abs/1506.03134) (`pointer`),
[Graph Pointer Network](https://arxiv.org/abs/1911.04936) (`gpn`) and
[GAMMA](https://doi.org/10.1109/TNNLS.2022.3159671)
(`gamma`) can also be trained with the `--model` option. To resume training, load your last saved model with the
`--resume` option. Additionally, pretrained models are provided inside the folder `pretrained`.

Evaluate your trained models with:
```bash
python eval.py data/1depots/const/20/test_seed1234_L2.0.pkl --model outputs/top_const20/attention_... --num_agents 2
```
If the epoch is not specified, by default the last one in the folder will be used.

Baselines algorithms like Ant Colony Optimization (`aco`), Particle Swarm Optimization (`pso`), or Genetic Algorithm
(`opga`) can be executed as follows:
```bash
python -m problems.top.eval_baselines --method aco --multiprocessing True --datasets data/1depots/const/20/test_seed1234_L2.0.pkl
```

Finally, you can visualize an example using:
```bash
python visualize.py --graph_size 20 --num_agents 2 --max_length 2 --data_distribution const --model outputs/top_const20/attention_...
python visualize.py --graph_size 20 --num_agents 2 --max_length 2 --data_distribution const --model aco
```

### Other options and help
```bash
python run.py -h
python eval.py -h
python -m problems.op.eval_baselines -h
python visualize.py -h
```

## Acknowledgements
This repository is an adaptation of
[wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) for the TOP. The baseline
algorithms (ACO, PSO, and GA) were implemented following the next repositories:
[robin-shaun/Multi-UAV-Task-Assignment-Benchmark](https://github.com/robin-shaun/Multi-UAV-Task-Assignment-Benchmark)
and [dietmarwo/Multi-UAV-Task-Assignment-Benchmark](https://github.com/dietmarwo/Multi-UAV-Task-Assignment-Benchmark)



