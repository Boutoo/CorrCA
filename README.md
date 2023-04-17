# CorrCA for Python

The CorrCA for Python project is an implementation of the Correlated Component Analysis (CorrCA) method [(Parra, 2018)](https://arxiv.org/abs/1801.08881) for Python. CorrCA is a multivariate analysis technique designed to find maximally correlated components between repetitions. It has been suggested as a method for finding relationships between neural activity and stimuli or behavior, as well as for data dimensionality-reduction. This implementation is designed to work with MNE.Epochs() objects, a commonly used data structure for handling EEG data in Python.

## Features

- A `CorrCA()` class that accepts a list of MNE.Epochs() objects and returns an object containing the maximally correlated components for each subject.
- F-Statistics to indicate component significativity.
- Functions to visualize and analyze the resulting components.

## Dependencies

- mne
- scipy
- numpy
- matplotlib

## Installation

To install the required libraries, you can use either `conda` or `pip`.

```bash
# Using conda
conda install mne scipy numpy matplotlib

# Using pip
pip install mne scipy numpy matplotlib
```

## Usage
Here's an example of how to use CorrCA for Python:
```python
from corrca import CorrCA

# list_of_epochs = [mne.Epochs(...), ...]

corrca = CorrCA(list_of_epochs)
ydata = cca.get_comps()
subs, comps, times = np.shape(ydata)
fstats = cca.f_stats[cca.f_stats>5] # Above 5 indicates high reproducibility

fig, axs = plt.subplots(comps, 1, dpi=250, layout='constrained')
for i, ax in enumerate(axs):
    for sub in range(subs):
        ax.plot(ydata[sub,i,:], 'gray')
    ax.plot(np.mean(ydata[:,i,:], axis=0),'red')
    ax.set_title(f'Component {i} ({fstats[i]:.2f})')
```

## Licence
This project is licensed under the MIT License.

## References
* Parra, Lucas C., et al. “Correlated Components Analysis - Extracting Reliable Dimensions in Multivariate Data”. Neurons, Behavior, Data Analysis, and Theory, vol. 2, no 1, janeiro de 2019. DOI.org (Crossref), https://doi.org/10.51628/001c.7125.