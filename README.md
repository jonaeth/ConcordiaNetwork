# Concordia - a neuro-symbolic framework

Concordia is a neuro-symbolci architecture that combines lifted graphical models and neural networks. The goal is to distill additional knowledge into a neural network in the form of rules. Concordia consists of three main parts, which are coded into classes:

1. [Teacher](#Teacher) - the lifted graphical model. So far, we have implemented only Probabilistic Soft Lofic (PSL) models, however, it can easily extended to other lifted graphical models such as Markov Logic Networks (MLNs).
2. [Student](#Student) - the neural model. This can be any type of Deep Neural Network and the architecture has been developed for easy integration of different models. So far, only classification tasks has been tested, but regression is an easy extension.
3. [Concordia Network](#Concordia-Network) - the main class that takes the outputs of each individual solver backpropagates their output respectively to the other and combines the outputs through a mixture-of-experts approach to give a unified prediction.

## Content

1. [Repo Structure](#repo-structure)
2. [Teacher](#teacher)
3. [Student](#student)
4. [Concordia Network](#concordia-network)
5. [Additional Files](#additional-files)
6. [Installation and Set-Up](#installation)
7. [Experiments](#experiments)

## Repo Structure

The Repo is split into two folders: `Concordia` and `Experiments`. `Concordia` contains all the necessary files to use this architecture. The `Experiments` folder contains all the experiments run in this paper. Running new datasets would simply consist in inserting the data, the desired neural model, and the rules in the corresponding folders.

All files in Concordia are described below. The Experiments folder is set up the following way:

Each experiment has its own folder, e.g. `CollectiveActivity`. Inside, Concordia expects a `data` folder, with the data being split between the data for the `student` and the `teacher` and inside between training and testing data respectively. 

Inside the `teacher` folder, in addition, you will find a `model` folder, which contains a `model.psl` file containing the rules, and if trained ahead of time their respective weights, and a `predicates.psl` describing all predicates and their arguments in the model, i.e.

```
/data
	/student
		/test
		/train
	/teacher
		/model
			.model.psl
			.predicates.psl
		/test
		/train
```

In addition, inside the respective experiment folde, you will find a `data_processing` folder. This folder is not necessary, but can be used to set up the data pre-processing.

Finally, there is a `NeuralNetworkModels` folder containing all the neural models that you might want to use for the student.



## Teacher

The `Teacher` class is a wrapper class for a lifted graphical model it is an abstract class allowing in the future to implement several instances of teachers. So far, we have implemented a `PSLTeacher`. 

The `Teacher` has the following form

```
Teacher(knowledge_base_factory, predicates=None, predicates_to_infer=None, **config)
```

where 

- `knowledge_base_factory` is an object that, from a list of predicates and raw data, grounds all possible atoms,
- `predicates` is a list of all predicates in the knowledge base,
- `predicates_to_infer` are the predicates that the overarching architecture is interested in inferring, and
- `**config` is a configuration file, which typically is the same as for the whole architecture.


## Concordia-Network

The $ConcordiaNetwork class combines the two solvers. Firstly, it takes the outputs of the 