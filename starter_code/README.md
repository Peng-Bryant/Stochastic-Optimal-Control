# UCSD ECE276B PR3

## Overview
In this assignment, We implement a controller for a car robot to track a trajectory.

## Dependencies

- python 3.10
- matplotlib 3.9.0
- numpy 1.26.4 
- joblib                    1.4.2
- scipy                     1.13.1
- tqdm                      4.66.4

## Project File Structure

This document provides an overview of the project file structure, explaining the purpose of each file and its contents.

### Part 1: Certainty Equivalent Control (CEC)

#### CEC.py
- **Description**: Implements the Certainty Equivalent Control (CEC) algorithm for Part 1 of the project.
- **Contents**: 
    - Implementation of the CEC algorithm.
    - Relevant functions and classes to support the CEC process.

#### cec.py
- **Description**: Entry point for testing the performance of the CEC algorithm.
- **Contents**:
    - Code to initialize and run the CEC algorithm.
    - Performance testing and evaluation.

### Part 2: Generalized Policy Iteration (GPI)

#### GPI_final.py
- **Description**: Contains the `GPI` class and implements the Generalized Policy Iteration (GPI) algorithm for Part 2 of the project.
- **Contents**:
    - `GPI` class implementation.
    - Policy evaluation and improvement methods.

#### stage_mm.py
- **Description**: Computes the stage cost matrix.
- **Contents**:
    - Functions to calculate and output the stage cost matrix.
    - Logic to handle various state and control variables.

#### trans_mm.py
- **Description**: Computes the transition matrix.
- **Contents**:
    - Functions to calculate transition probabilities and state transitions.
    - Methods to generate and store the transition matrix.

#### value_function.py
- **Description**: Implements the basic grid-based value function class.
- **Contents**:
    - `GridValueFunction` class.
    - Methods for value function initialization, updating, and querying.

#### gpi_main.py
- **Description**: Entry point for testing the performance of the GPI algorithm.
- **Contents**:
    - Code to initialize and run the GPI algorithm.
    - Performance testing and evaluation.

### Utilities

#### utils.py
- **Description**: Basic utility functions and classes used throughout the project.
- **Contents**:
    - General-purpose functions (e.g., logging, timing).
    - Helper functions for mathematical and computational tasks.

## Examples and Testing

### Testing
#### gpi_main.py

```bash
python gpi_main.py --exp 0 --cp 50
```
- `exp` is the experiment name
- `cp`  is policy check point
- the policy is stored in `output/exp`

#### cec_main.py

```bash
python cec_main.py
```
### GPI Training

#### trans_mm.py
```bash
python trans_mm.py --exp 0
```
- `exp` is the experiment name
- compute the transition matrix and save in `./data/exp`
#### stage_mm.py
```bash
python stage_mm.py --exp 0
```
- `exp` is the experiment name
- compute the stage_cost matrix and save in `./data/exp`

#### GPI_final.py
```bash
python GPI_final.py --exp 0
```
- `exp` is the experiment name
- compute the policy and save in `./output/exp`