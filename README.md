# CNN-based Solution of the ELPV dataset

[![license](https://img.shields.io/badge/license-by--nc--sa%204.0%20Int.-EF9421?logo=creative-commons)](LICENSE.md)
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg)](https://github.com/RichardLitt/standard-readme)

> CNN-based identification of defective solar cells in electroluminescence imagery.


## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [License](#license)


## Background

The life span is an important aspect of photovoltaic (PV) modules.
Electroluminescence (EL) imaging is an established technique for the visual
inspection of PV modules. It enables identification of defects in
solar cells that may impede the life span of the module. However, manual
inspection of EL images is tedious and requires expert knowledge. Therefore,
an automated pipeline may enable high-throughput quality assessment of
PV modules.

In this work, we use a convolutional neural network (CNN) architecture to
predict the defect probability and their structural type (mono- vs.
polycrystalline) of PV modules using EL images. We compare the performance of
our approach to existing solutions with respect to defect probability.
Furthermore, we also employ techniques from the field of explainable artificial
intelligence (AI) to visualize how the CNN reached its decision.


## Install

This project uses [`pipenv`](https://github.com/pypa/pipenv) to manage its
dependencies. You may install it via `pip install pipenv`.

```
git clone --recurse-submodules https://github.com/a-ludi/cnn-elpv-dataset.git
pipenv install
```


## Usage

To be completed...


## License

[![Creative Commons License][cc-by-nc-sa-4.0-logo]][cc-by-nc-sa-4.0]

This work is licensed under a [Creative Commons
Attribution-NonCommercial-ShareAlike 4.0
International License][cc-by-nc-sa-4.0].

[cc-by-nc-sa-4.0-logo]: https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-4.0]: http://creativecommons.org/licenses/by-nc-sa/4.0/
