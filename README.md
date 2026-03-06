# DemystifyActionSpace
## Overview

This repository contains code and datasets used to study how different **action space designs** affect robotic manipulation policies.

We provide:

* teleoperated manipulation datasets collected on AgileX robots
* training and evaluation code for policy learning
* tools for running experiments across different action space parameterizations
* a cross embodiment dataset for transfer experiments

The goal is to provide a simple benchmark for analyzing how action representations influence policy performance in robot manipulation.

## Build the Environment

`conda create -n das python=3.10`

`pip install -e .`

## Train

`bash train.sh`

## Evaluate

`bash server.sh`

On AgileX-PiPER:

`python agilex/client_air_eef6d_align_init.py`