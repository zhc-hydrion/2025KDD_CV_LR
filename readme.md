## Project Overview

This project implements the NYM algorithm based on the `causallearn` package. Below is an overview of the directories and scripts included in this project:

- **data/**: Contains the Sachs and Child real datasets.
- **data_generate.py**: Used to generate synthetic data.
- **GES.py**: Contains the implementation of various algorithms including CV and NYM.
- **main.py**: Used to conduct experiments. The following parameters can be specified:
  - `method`: Selects the method to use.
  - `dataset`: Chooses different datasets.
  - `graph_density`: Selects the graph density for synthetic data in experiments.
  - `generate`: Chooses whether to generate new synthetic data or use existing ones.
  - `epoch`: Specifies the number of experiment iterations.
## Usage

To use this project, simply run the following command:

```sh
python main.py