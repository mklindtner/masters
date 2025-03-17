from models import plot_distribution
import matplotlib.pyplot as plt

def plotter(w, algo2D, w_MAP, w_MLE):
    _, axes = plt.subplots(1, 3, figsize=(18,6))
    axes[2].plot(w[:,1],w[:,0], "ro", label="estimated weights")
    # axes[2].plot(w[:,0],w[:,1], "yo", label="estimated weights opposite")

    # axes[2].plot(w_MAP[1],w_MAP[0], "bo")

    plot_distribution(axes[0],density_fun=algo2D.log_prior, color='b', label='Prior', title='Prior', visibility=0.25)
    plot_distribution(axes[1],density_fun=algo2D.log_likelihood, color='r', label='likelihood', title='Likelihood', visibility=0.25)
    plot_distribution(axes[2],density_fun=algo2D.log_joint, color='g', label='Posterior', title='Posterior', visibility=0.25)
    axes[1].plot(w_MLE[1], w_MLE[0], 'mo', label='MLE estimate')
    axes[1].legend(loc='lower right')
    
    axes[2].plot(w_MAP[1], w_MAP[0], 'bo', label='MAP/Posterior mean')
    axes[2].legend(loc='lower right')
    plt.show()

