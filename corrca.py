# %% Imports
import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import mne

# %% Functions and Classes
def print_progress_bar(iteration, total, fill = '•', length=40, autosize = False):
    """
    Call in a loop to create terminal progress bar (Original Fill █)

    Args:
        iteration  (Int): current iteration (Required)
        total (int): total iterations (Required)
        fill (str): bar fill character (Optional)
        length (100): character length of bar (Optional)
        autosize (bool): automatically resize the length of the progress bar
        to the terminal window (Optional)

    Examples:
        >>> print_progres_bar(0,10)
        >>> for i in range(10):
        >>>     print_progress_bar(i,10)
        ∙ |••••••••••∙∙∙∙∙∙∙∙| 50% ∙

    """
    percent = ("{0:." + "0" + "f}").format(100 * (iteration / float(total)))
    styling = '%s |%s| %s%% %s' % ('∙', fill, percent, '∙')
    if autosize:
        cols, _ = shutil.get_terminal_size(fallback = (length, 1))
        length = cols - len(styling)
    filled_length = int(length * iteration // total)
    progress_bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s' % styling.replace(fill, progress_bar), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

class CorrCA() :
    """
    This class holds an instance of the Correlated Component Analysis.
    It holds information over its aplications over a given group and a set of useful methods.

    Args:
        data (list of objects): List with subjects
        reg (float): Percentage of explained variance when using SVD for inverse estimation.
        f_stat (int): F-statistics value for chosing significative components.
        project_over (CorrCA Object): Previous calculated CorrCA Object to project new data upon.
    
    Methods:
        calculate_rb (X): Calculates the between subject correlation
        calculate_rw (X): Calculates the within subject correlation

    """
    def __init__(self, data, reg = .99, f_stat=5, project_over=None, reference_ch='Cz'):
        if type(data) is not list:
            data = [data]
        dims = np.array([np.shape(sub) for sub in data])
        self.ntrials = dims[:,0]
        self.nsubjects = np.shape(dims)[0]
        self.nchannels = dims[:,1]
        self.nsamples = dims[:,2]
        self.mne_info = [sub.info for sub in data]
        self.times = data[0].times # If Group: Assume all subjects have the same time window
        for sub in data:
            if np.any(sub.times!=self.times):
                raise Exception('All subjects must have the same time window')
        self.tmin = self.times[0]
        self.montage = [ep.get_montage() for ep in data]
        self.ch_labels = [ep.info['ch_names'] for ep in data]
        self.reference_ch=reference_ch
        if len(data) != 1:
            data = self.group_to_array(data)
        else:
            data = data[0].get_data()
        if project_over is None:
            self.rwithin = self.calculate_rw(data)
            self.rbetween = self.calculate_rb(data, self.rwithin)
            self.smatrix, self.vecs, self.rho, self.isc, self.reg_num, self.a_fwd = self.apply_eigen_decomposition(self.rwithin, self.rbetween, reg)
            self.ydata = self.project_results(data, self.rho, self.vecs)
            self.ncomponents = np.shape(self.ydata)[1]
            self.f_stats = self.apply_f_statistics()
            self.n_sig_comps = len(np.where(self.f_stats >= f_stat)[0])
        else:
            self.rwithin, self.rbetween = [project_over.rwithin, project_over.rbetween]
            self.smatrix, self.vecs, self.rho, self.isc, self.reg_num, self.a_fwd  = [project_over.smatrix,project_over.vecs, project_over.rho, project_over.isc, project_over.reg_num]
            self.ydata= project_over.project_results(data, self.rho, self.vecs, self.rwithin)
            self.ncomponents = np.shape(self.ydata)[1]
            self.f_stats = project_over.f_stats     ### Corrigir F-Stat pós projeção aqui!
            #self.f_stats = self.apply_f_statistics()
            self.n_sig_comps = len(np.where(self.f_stats >= f_stat)[0])
        print('Done applying CorrCA!')
    def group_to_array(self, data):
        """group_to_array
        This function is used to convert a list of Subjects into a numpy array.
        Args:
            data (list of Epochs): List of Subject Epochs to apply CorrCA

        Returns:
            data (np.ndarray): Array of Subjects Data in np.ndarray format
        """
        data = [sub.average().get_data() for sub in data]
        return data
    def subject_to_array(self, data):
        print(np.shape(data))
        return data
    def calculate_rw(self, data):
        """ Calculates the within subject correlation.

        Args:
            data (list of np.ndarray): List of subjects evoked arrays [Subjects or Trials, Channels, Samples]
        
        Returns:
            rwithin (np.Array): Array containing the within subject correlation

        """
        print('Calculating Rwithin...')
        nsubs = np.shape(data)[0]
        nchs = np.shape(data)[1]
        nsamples = np.shape(data)[2]
        rwithin = np.zeros([np.shape(data)[1],np.shape(data)[1]])
        print_progress_bar(0, nsubs-1)
        for i in range(nsubs):
            print_progress_bar(i, nsubs-1)
            dist = np.zeros([nchs, nsamples])
            for t in range(nsamples):
                    # DxD = (Sample([n x D x time]) - T_Mean([n x D x T]))
                    dist[:, t] = (data[i][:, t] - np.mean(data[i], axis=1))
            rwithin += (dist @ dist.T) / (nsamples - 1)
        return rwithin
    def calculate_rb(self, data, rwithin):
        """ Calculates the within subject correlation via total correlation.
        This way, it is possible to reduce the processing time needed to calculate.

        Args:
            data (list of np.ndarray): 
        
        Returns:
            rbetween (np.ndarray): Between subjects correlation
        """
        print('Calculating Rbetween ...')
        nsubs = np.shape(data)[0] # Or trials
        nchs = np.shape(data)[1]
        nsamples = np.shape(data)[2]
        rtotal = np.zeros([nchs, nchs])
        subavg = np.mean(data, axis=0)
        dist = np.zeros([nchs, nsamples])
        for t in range(nsamples):
            # DxD = ([D x t] - T_Mean([D x T]))*Same.T
            dist[:, t] = (subavg[:, t] - np.mean(subavg, axis=1)) 
        rtotal = dist @ dist.T / (nsamples-1)
        rtotal = (nsubs**2)*rtotal
        rbetween = (rtotal - rwithin)/(nsubs-1)
        return rbetween
    def apply_eigen_decomposition(self, rwithin, rbetween, reg):
        U, S, V = np.linalg.svd(rwithin)
        reg_num = np.where(np.cumsum(S)/np.sum(S)>=reg)[0][0]
        reg_num = min(np.linalg.matrix_rank(rwithin), reg_num)
        inv_rwithin = U[:, 0:reg_num] @ np.diag(1./S[0: reg_num]) @ V[0:reg_num, :]
        rho, vecs = scipy.sparse.linalg.eigs(inv_rwithin @ rbetween, reg_num)
        # Sorting ISC Values:
        vecs = np.real(vecs)
        a_fwd = rwithin @ vecs @ np.linalg.inv(vecs.T @ rwithin @ vecs)
        isc = np.diag(vecs.T @ rbetween @ vecs) / np.diag(vecs.T @ rwithin @ vecs)
        return S, vecs, rho, isc, reg_num, a_fwd
    def project_results(self, data, rho, vecs):
        nsubs = len(data)
        nchs = np.shape(data)[1]
        nsamples = np.shape(data)[2]
        y_data = np.zeros([nsubs, len(rho), nsamples])
        for i, sub in enumerate(data):
            for comp in range(len(rho)):
                for t in range(nsamples):
                    y_data[i, comp, t] = vecs[:, comp].T @ data[i][:, t]            
        return y_data
    def get_comps(self, sig=True):
        if sig:
            return self.ydata[:,np.where(self.f_stats>5)[0],:]
        else:
            return self.ydata
    def plot_f_statistics(self, title='CorrCA: F-Statistics'):
        a = self.apply_f_statistics()
        plt.figure()
        plt.bar(range(len(a)),a)
        plt.hlines(5, 0, len(a), colors='gray', linestyles='dashed')
        plt.title(title)
    def apply_f_statistics(self):
        a = np.zeros(len(self.isc))
        T = np.shape(self.ydata)[2]
        N = np.shape(self.ydata)[0]
        for i,_ in enumerate(a):
            a[i] = ((T*(N-1)*self.isc[i]) + T)/((T-1)*(1-self.isc[i]))
        return a
    def get_component_power(self, components=None, time_window=None):
        """ Gets the component projection strength for each subject.

        Args:
            components ('significative' or int or list): Wich components to use when computing the projection strength
            time_window (list or None): Wich time window to use when computing the projection strength (e.g. [t,t1])
        
        Returns:
            component_power (np.ndarray): Array containing the projection strength for each subject

        """
        component_power = np.zeros(np.shape(self.ydata)[0])
        if time_window == None:
            time_window = [0, []]
            time_window[1] = np.shape(self.ydata)[2]
        if components == None:
            components = [0, []]
            components[1] = self.n_sig_comps
        if type(components) == int:
            components=[components,components+1]
        for i in range(len(component_power)):
                component_power[i] = np.sqrt(np.mean((self.ydata[i,components[0]:components[1],time_window[0]:time_window[1]])**2))
        return component_power

# %% Testing
from scipy.io import loadmat

PATH = "C:/Users/bruno/Documents/NNCLab/Data/SHAM/"
FILES = ["AM_Session_4_PremotorL_TMS_EVOKED.mat",
         "AP_Session_2_PremotorL_TMS_EVOKED.mat"]

file = PATH+FILES[0]

def mat_to_epochs(file, time_window=None, downsample=False, ch_labels=None, montage='easycap-M1'):
    """ This function reads EEG data with .mat format.
        Specially made to deal with Milan SHAM Protocol data.

    Args:
        file (str): Full file path
        time_window (list or None, optional): Time to segment epochs in seconds (e.g. [-.2, .5]). Defaults to None.
        downsample (bool, optional): Downsample data (True or False). Defaults to False.
        ch_labels (_type_, optional): Label of channels defaults to Milano SHAM Protocol format. Defaults to None.
        montage (str, optional): MNE's built-in montage to use. Defaults to 'easycap-M1'.
    """
    data_mat = loadmat(file)
    data_mat['times'] = data_mat['times'][0]
    data = np.array(data_mat['Y'])
    data = np.transpose(data, [2,0,1]) # Mne default is (ntrials, nchannels, nsamples)
    times = np.array(data_mat['times'])/1000 #in s        
    if downsample!=False:
        data=data[:,:,range(0,len(times),downsample)]
        times=times[range(0,len(times),downsample)]
    sfreq = 1/(times[1]-times[0])
    tmin = times[0]
    if 'origem' in data_mat:
        origem = data_mat['origem']
    else:
        origem='none'           
    # Creating MNE Instances:
    if ch_labels is None: # Sets to default
        ch_labels = ['Fp1','Fpz','Fp2','AF3','AFz','AF4','F7','F5',
                            'F3','F1','Fz','F2','F4','F6','F8','FT7','FC5',
                            'FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7',
                            'C5','C3','C1','Cz','C2', 'C4','C6','T8','TP9',
                            'TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6',
                            'TP8','TP10','P7','P5','P3','P1','Pz','P2','P4',
                            'P6','P8','PO7','PO3','POz','PO4','PO8','O1',
                            'Oz','O2','Iz']
    else:
        ch_labels=ch_labels

    if 'channels' in data_mat:
        ix=[ch_labels[z-1] for z in data_mat['channels'][0]]
        ch_labels=ix

    # Creating Info
    mne_info = mne.create_info(ch_names=ch_labels, sfreq=sfreq,
                                ch_types='eeg')
    ntrials = np.shape(data)[0]
    if time_window is None:
        idx = [0,-1]
        tmin = tmin
    else:
        idx = [np.abs(np.asarray(times)-time_window[i]).argmin() for i in range(len(time_window))]
        tmin = times[idx[0]]
    epochs = mne.EpochsArray(data[:,:,idx[0]:idx[1]], mne_info, tmin=tmin, verbose=False);
    if montage != None:          
        epochs.set_montage(montage)
    return epochs
epochs = []
for file in FILES:
    epochs+=[mat_to_epochs(PATH+file, [-.2,.5])]

cca = CorrCA(epochs)

# %% Analysis
ydata = cca.get_comps()
fstats = cca.f_stats[cca.f_stats>5]
subs, comps, times = np.shape(ydata)
fig, axs = plt.subplots(comps,1, dpi=250, layout='constrained')
for i, ax in enumerate(axs):
    for sub in range(subs):
        ax.plot(ydata[sub,i,:], 'gray')
    ax.plot(np.mean(ydata[:,i,:], axis=0),'red')
    ax.set_title(f'Component {i} ({fstats[i]:.2f})')

# %%
