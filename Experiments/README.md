# Experiments - Overview

This file gives a general overview on how the experiments have been set up and how you can create your own experiments.
So far, we have implemented 4 experiments:
- Collective Activity Detection (Classification, Supervised)
- Entity Linking (Classification, Unsupervised and Semisupervised)
- MovieLens Recommendation (Regression, Supervised)
- Yelp Recommendation (Regression, Supervised)

Each experiment contains a `NeuralNetworkModels` folder, which contains one or several classes of neural models, a `teacher` folder, which contains the different files implementing the logic solver, and several other files, such as main files to run the experiments, metric files to measure the performance of the experiments, logging files, and others. A `README.md` file is provided for each experiment to explain the useage of each file.

In this file, we will just touch on the necessary files and their formats.

## NeuralNetworkModels

The `NeuralNetworkModels` folders contain one or several neural network classes. You can use any neural network model you want and import it here. Concordia has been successfully tested on an array of neural models for both regression and classification tasks, in supervised, semi-supervised and unsupervised tasks. The tested neural models include:
- TODO Modestas add list of neural models here

TODO need to discuss if at all any particular constraints are put on the neural models (e.g. on the forward or backward function, the losses or anything else)

## Teacher

The `Teacher`, as explained in the main [`README.md` file](../README.md), is a wrapper for logical solvers. So far, only PSL implementations have been tested, but an adaption to Markov Logic Networks or ProbLog should work in the same way.

The `teacher` folder in each experiment contains two folders - the `model` folder and the `train` folder.

### Model

The `model` folder contains two files: `model.psl` and `predicates.psl`.

`model.psl` describes the structure of the rules and their weights:

```text
[predicate name]([list of logic variables and constants]) & ... & [predicate name]([list of logic variables and constants]) >> [predicate name]([list of logic variables and constants])	[initial weight of rule]
.
.
.
[predicate name]([list of logic variables and constants]) & ... & [predicate name]([list of logic variables and constants]) >> [predicate name]([list of logic variables and constants])	[initial weight of rule]
```
**Example**

For example, consider the [`model.psl` file](CollectiveActivity/teacher/model/model.psl) of the Collective Activity Task, where we have in total 33 rules.
- In the first rule, 
  - we propagate the neural priors (denoted by `Local`) meaning that if the neural model predicts activity `'0'` to be true that it is likely to be true.
  - `B` is a logical variable, which can take on any value of a bounding box, while `'0'` is a logical constant
  - the predicate names are `Local` and `Doing`
  - the initial weight is `'1'`
- In the final rule, the weight is `'-1'` implying that it is a hard rule that needs to be satisfied at all times.

`predicates.psl` describes the different predicates in the theory, whether we assume closed world assumptions for each of them and their respective arity.
```text
[predicate name(type string)][\t][closed world assumption(type Boolean)][\t][arity(type int)]
```
###Train
The train folder contains the training data in a format that is readable by PSL. The files inside are typically built by the `KnowledgeBaseFactory` in each Experiment.

The structure of the folders that is expected, yet also automatically built by the `KnowledgeBaseFactory` is
```text
\train
    \observations
    \targets
    \truths
```

- In `observations` all the observed ground atoms are stored.
- In targets the unobserved ground atoms without their truth assignments are stored. 
- In `truths` the unobserved ground variables ares stored with their truth assignments.

```text
[list of logical constants separated by tabs(type int)][\t][truth value(type float)]
```

**Examples**

For example, consider the [`Local.psl` file](CollectiveActivity/teacher/train/observations/Local.psl) of the Collective Activity Task, containing the ground atoms of the previously discussed `Local` atom.

The `predicates.psl` file states that it has arity 2, and the rules in the `model.psl` file shows that it expects an id for a bounding box and a constant representing a specific activity.
That is exactly, what you can find in the `Local.psl` file, as the first column, contains the ids for the bounding boxes, the second column the different activities, and the third column the predictions of the neural model for each bounding box and each activity.