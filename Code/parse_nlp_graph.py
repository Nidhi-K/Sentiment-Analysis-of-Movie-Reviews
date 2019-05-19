import argparse
import re
import matplotlib.pyplot as plt
import pylab

def _parse_args():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--file', type=str, default='nlp mini2')
    args = parser.parse_args()
    return args

def plot_graph(to_plot, graph_title, ylabel, graph_legends=None):
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    colors = ['r', 'b', 'g', 'y', 'm']

    if graph_legends is None:
        x_vals = [x for x in range(len(to_plot))]
        ax.plot(x_vals, to_plot, color = 'm')
        pylab.savefig(graph_title, bbox_inches='tight')
        return

    x_vals = [x for x in range(len(to_plot[0]))]
    for i in range(len(to_plot)):
        ax.plot(x_vals, to_plot[i], color = colors[i], label = graph_legends[i])

    ax.legend()
    pylab.savefig(graph_title, bbox_inches='tight')

args = _parse_args()
print(args)
# Use either 50-dim or 300-dim vectors
file_name = args.file

train_accs = []
dev_accs = []
losses = []
print(file_name)
with open (file_name) as f:
    for line in f:
        if 'train set = ' in line:
            train_accs.append(re.search(r'[0-9]+\.[0-9]+',line).group())
        if 'dev set =' in line:
            dev_accs.append(re.search(r'[0-9]+\.[0-9]+',line).group())
        if 'Loss' in line:
            losses.append(re.search(r'[0-9]+\.[0-9]+',line).group())
#print(train_accs)
#print(dev_accs)
#print(losses)
with open ('values nlp.txt','w') as f:
    f.write('Epochs')
    for a in range(len(train_accs)):
        f.write('\n'+str(a+1))
    f.write('\ntrain_accs')
    for a in train_accs:
        f.write("\n" + "{0:.2f}".format(float(a)))
    f.write('\ndev accs')
    for a in dev_accs:
        f.write("\n" + "{0:.2f}".format(float(a)))
    f.write('\nlosses')
    for a in losses:
        f.write("\n" + "{0:.2f}".format(float(a)))
    
to_plot = []
to_plot.append(train_accs)
to_plot.append(dev_accs)
plot_graph(to_plot, file_name + "accs.pdf", "Accuracy (%)", ["Train", "Dev"])
plot_graph(losses, file_name + "loss.pdf", "Loss")
#print(len(to_plot))
#print(to_plot)
