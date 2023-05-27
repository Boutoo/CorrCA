# ---------------------------------------------------------------------------
# CorrCA for EEG Event-Related Potentials
# File: gtrca.py
# Version: 1.0.0
# Author: Couto, B.A.N. (2023). Member of the Neuroengineering Lab from the Federal University of São Paulo.
# Date: 2023-05-27
# Description: Correlated Components Analysis (CorrCA) for EEG Event-Related Potentials. Works with mne.Epochs objects.
# ---------------------------------------------------------------------------

# %% Imports
import numpy as np
import mne

# %% CorrCA Class
class CorrCA():
    """ CorrCA class.
    Implemented according to the paper:
    Parra, L. C., Haufe, S., & Dmochowski, J. P. (2018). Correlated Components Analysis—Extracting Reliable Dimensions in Multivariate Data.
    DOI: https://doi.org/10.48550/ARXIV.1801.08881

    This Python implementation was made by Couto, B.A.N. (2023)
    Member of the Neuroengineering Lab from the Federal University of São Paulo.

    For more information, see the documentation of the fit method.

    Attributes:
        data (list of mne.Epochs or mne.Epochs): List of mne.Epochs objects or a single mne.Epochs object. It can also be a numpy.ndarray.
        Rb (numpy.ndarray): Between-subject covariance matrix of the data.
        Rw (numpy.ndarray): Within-subject covariance matrix of the data.
        eigenvalues (numpy.ndarray): Eigenvalues of the generalized eigenvalue problem.
        eigenvectors (numpy.ndarray): Eigenvectors of the generalized eigenvalue problem.
    """
    def __init__(self):
        self.data = None
        self.Rb = None
        self.Rw = None
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, data, group_analysis='trials', verbose=True):
        """Fit CorrCA to data.
        
         Args:
            data (list of mne.Epochs or mne.Epochs): List of mne.Epochs objects or a single mne.Epochs object. It can also be a numpy.ndarray.
            group_analysis (str, optional): Group analysis to be performed. Only when data is a list of mne.Epochs objects.
                Defaults to 'trials'.
                If 'trials', concatenates all epochs and performs CorrCA on the concatenated data.
                If 'evoked', performs CorrCA as if the evoked response of each epoch was a repetition.

        Raises:
            Exception: Data must be a list of mne.Epochs or a single mne.Epochs object.
        """
        if verbose:
            print('Fitting CorrCA to data...')
        if type(data) == list:
            repetitions = len(data)
            if type(data[0]) != np.ndarray:
                if group_analysis == 'trials':
                    data = mne.concatenate_epochs(data)
                    data = data.get_data()
                elif group_analysis == 'evoked':
                    data = [ep.average() for ep in data]
                    data = np.array([evk.get_data() for evk in data])
                else:
                    raise Exception('Group analysis must be "trials" or "evoked".')
        else:
            try:
                data = data.get_data()
            except:
                raise Exception('Data must be a list of mne.Epochs or a single mne.Epochs object.')

        self.data = data
        repetitions, dimension, exemplars = np.shape(data)
        
        # Normalizing Data (Zero Mean Exemplar-wise)
        if verbose:
            print('Normalizing data...')
        norm_data = np.array([repetition-np.mean(repetition, axis=1)[:,None] for repetition in data])

        # Calculating Between Subject Covariance Matrix (Rb)
        if verbose:
            print('Calculating Rb...')
        self.Rb = self._calculate_rb(norm_data, verbose=verbose)

        # Calculating Within Subject Covariance Matrix (Rw)
        if verbose:
            print('Calculating Rw...')
        self.Rw = self._calculate_rw(norm_data, verbose=verbose)

        # Eigenvalues and Eigenvectors
        if verbose:
            print('Solving for Eigenvalues and Eigenvectors...')
        self.eigenvalues, self.eigenvectors = np.linalg.eig(np.linalg.inv(self.Rw) @ self.Rb)

        # Sorting
        idx = self.eigenvalues.argsort()[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:,idx]
        
        if np.any(np.imag(self.eigenvalues)>0):
            print('Warning: Complex eigenvalues and eigenvectors found. They were converted to real numbers.')
        
        self.eigenvalues = np.real(self.eigenvalues)
        self.eigenvectors = np.real(self.eigenvectors)
        
        if verbose:
            print('Done! ✅')

    def get_projection(self, component='all'):
        """Get projections of data on the components of CorrCA.

        Args:
            component (int or str, optional): Component to get the projection. Defaults to 'all'.
                If 'all', returns the projections on all components.
                If int, returns the projection on the specified component.

        Raises:
            Exception: Component must be an integer or "all".

        Returns:
            numpy.ndarray: Projections of data on the components of CorrCA.
        """

        repetitions, dimension, exemplars = np.shape(self.data)
        # Calculating Maps
        inverse_matrix, _ = self._apply_inverse_regularization(self.eigenvectors.T @ self.Rw @ self.eigenvectors)
        maps = self.Rw @ self.eigenvectors @ inverse_matrix

        # Making Projections
        projections = []
        if component == 'all':
            for a in range(repetitions):
                projections.append([self.data[a].T @ self.eigenvectors[:,i] for i in range(dimension)])
            projections=np.array(projections)
        else:
            try:
                projections = self.data.T @ self.eigenvectors[:,component]
                maps = maps[:,component]
            except:
                raise Exception('Component must be an integer or "all".')
        return projections, maps

    def _calculate_rb(self, data, verbose=True):
        """Calculates the Between Subject Covariance Matrix (Rb).
        
        Args:
            data (list of numpy.ndarray): List of numpy.ndarray objects or a single numpy.ndarray object. It can also be a mne.Epochs object.

        Returns:
            Rb (numpy.ndarray): Between Subject Covariance Matrix (Rb).
        """
        repetitions, dimension, exemplars = np.shape(data)
        # Rb
        if verbose:
            self._print_progress_bar(0, exemplars, prefix='Rb:', suffix='Complete')
        Rb = np.zeros((dimension, dimension))
        for i in range(exemplars):
            if verbose:
                self._print_progress_bar(i+1, exemplars, prefix='Rb:', suffix='Complete')
            for a in range(repetitions):
                for b in range(repetitions):
                    if a != b:
                        Rb+= data[a][:,i][:,None] @ data[b][:,i][None,:]
        return Rb
    
    def _calculate_rw(self, data, verbose=True):
        """Calculates the Within Subject Covariance Matrix (Rw).

        Args:
            data (list of numpy.ndarray): List of numpy.ndarray objects or a single numpy.ndarray object. It can also be a mne.Epochs object.

        Returns:
            Rw (numpy.ndarray): Within Subject Covariance Matrix (Rw).
        """
        repetitions, dimension, exemplars = np.shape(data)
        # Rw
        if verbose:
            self._print_progress_bar(0, exemplars, prefix='Rw:', suffix='Complete')
        Rw = np.zeros((dimension, dimension))
        for i in range(exemplars):
            if verbose:
                self._print_progress_bar(i+1, exemplars, prefix='Rw:', suffix='Complete')
            for a in range(repetitions):
                Rw += data[a][:,i][:,None] @ data[a][:,i][None,:]
        return Rw

    def _print_progress_bar(self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 15, fill = '█'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
        # Print New Line on Complete
        if iteration == total: 
            print()

    def _apply_regularization(self, S, reg):
        """ This function is used to regularize the S matrix (from SVD).
        It is useful for further matrix inverse operations.
        Args:
            S (np.ndarray): S matrix.
            reg (float): Regularization parameter.
        Returns:
            reg_num (int): Number of components to be used for regularization.
        """

        eps = np.finfo(float).eps
        ix1 = np.where(np.abs(S)<1000*np.finfo(float).eps)[0] # Removing null components
        ix2 = np.where((S[0:-1]/(eps+S[1:]))>reg)[0] # Cut-off based on eingenvalue ratios
        ix = np.union1d(ix1,ix2)
        if len(ix)==0:
            reg_num=len(S)
        else:
            reg_num=np.min(ix)
        return reg_num

    def _apply_inverse_regularization(self, data, reg=10**5, verbose=False):
        """ This function regularizes a given matrix and calculates it's inverse.
        Args:
            data (np.array): List of matrix to be regularized and inverted.
            reg (float): Regularization parameter.
        Returns:
            data_inverse (np.ndarray): Inverse of the Q matrix.
            S (np.ndarray): S from SVD matrix.
        """
        if verbose:
            print('Regularizing via Ratio Eig > ('+str(reg)+')...')

        U, S, V = np.linalg.svd(data)
        reg_num = self._apply_regularization(S,reg)
        data_inverse = (V[0:reg_num, :].T* (1./S[0:reg_num])) @ U[:, 0:reg_num].T
        return data_inverse, S

