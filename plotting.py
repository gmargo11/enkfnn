import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

class TrainingLogger:
    def __init__(self):
        self.train_acc = []
        self.train_acc_stdevs = []
        self.train_acc_times = []
        
        self.val_acc = []
        self.val_acc_stdevs = []
        self.val_acc_times = []

        self.eigenvals = []
        self.eigenvals_times = []

        self.max_weight = []
        self.max_weight_times = []

    def log_train_acc(self, train_acc, t):
        self.train_acc += [train_acc]
        self.train_acc_stdevs += [0]
        self.train_acc_times += [t]

    def log_val_acc(self, val_acc, t):
        self.val_acc += [val_acc]
        self.val_acc_stdevs = [0]
        self.val_acc_times += [t]

    def log_eigenvals(self, eigenvals, t):
        self.eigenvals += [eigenvals]
        self.eigenvals_times += [t]

    def log_max_weight(self, max_weight, t):
        self.max_weight += [max_weight]
        self.max_weight_times += [t]

    def plot_errors(self, ax, backprop_times=None, backprop_val_acc=None, invert=True):
        if backprop_times:
            print("plotting with backprop result!")
            #ax.plot(self.train_acc_times, self.train_acc)
            if invert:
                ax.plot(self.val_acc_times+backprop_times, 1 - np.array(self.val_acc+backprop_val_acc), linewidth=3)
            else:
                ax.plot(self.val_acc_times+backprop_times, np.array(self.val_acc+backprop_val_acc), linewidth=3)
        else:
            #ax.plot(self.train_acc_times, self.train_acc, linewidth=3)
            if invert:
                ax.plot(self.val_acc_times, 1 - np.array(self.val_acc), linewidth=3)
            else:
                ax.plot(self.val_acc_times, np.array(self.val_acc), linewidth=3)
        #ax.legend(['training accuracy', 'validation accuracy'])
        ax.set_xlabel('# weight updates', fontsize=18)
        ax.set_ylabel('Accuracy', fontsize=18)

    def plot_stdevs(self, ax, backprop_times=None, backprop_val_acc=None, invert=True):
        
        if invert:
            ax.fill_between(self.val_acc_times, 
                            1 - np.array(self.val_acc) - np.array(self.val_acc_stdevs),
                            1 - np.array(self.val_acc) + np.array(self.val_acc_stdevs), 
                            linewidth=1, facecolor='blue', alpha=0.5)
        else:
            ax.fill_between(self.val_acc_times, 
                            np.array(self.val_acc) - np.array(self.val_acc_stdevs),
                            np.array(self.val_acc) + np.array(self.val_acc_stdevs), 
                            linewidth=1, facecolor='blue', alpha=0.5)
        #ax.legend(['training accuracy', 'validation accuracy'])
        ax.set_xlabel('# weight updates', fontsize=18)
        ax.set_ylabel('Accuracy', fontsize=18)

    def plot_eigenvals(self, ax):
        eigenvals_np = np.array(self.eigenvals)
        print('eigenvals shape: ', eigenvals_np.shape)

        for i in range(eigenvals_np.shape[1]):
            ax.plot(self.eigenvals_times, eigenvals_np[:, i])

        ax.set_xlabel('# weight updates')
        ax.set_ylabel('Eigenvalue magnitude')
        #ax.title('Eigenvalues of Output Covariance')

    def update_eigenvals_curve(self, ax, i):
        eigenvals_np = np.array(self.eigenvals)
        ax.plot(range(eigenvals_np.shape[1]), eigenvals_np[i, :])
        return ax

    def save_log(self, filename):
        pickle.dump(self, open(filename, "wb"))


def average_runs(loggers):
    tl = TrainingLogger()
    
    tl.train_acc = np.mean(np.array([logger.train_acc for logger in loggers]), axis=0).tolist()
    tl.train_acc_stdevs = np.std(np.array([logger.train_acc for logger in loggers]), axis=0).tolist()
    tl.train_acc_times = loggers[0].train_acc_times
    
    tl.val_acc = np.mean(np.array([logger.val_acc for logger in loggers]), axis=0).tolist()
    tl.val_acc_stdevs = np.std(np.array([logger.val_acc for logger in loggers]), axis=0).tolist()
    tl.val_acc_times = loggers[0].val_acc_times

    return tl
    #t1.eigenvals = []
    #t1.eigenvals_times = []

    #t1.max_weight = []
    #t1.max_weight_times = []