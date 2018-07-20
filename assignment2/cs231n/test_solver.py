from solver import Solver
from classifiers.fc_net import TwoLayerNet
from classifiers.fc_net import FullyConnectedNet
from classifiers.cnn import ThreeLayerConvNet
from gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
import numpy as np
from cs231n.data_utils import get_CIFAR10_data
import matplotlib.pyplot as plt

data = get_CIFAR10_data()

num_train = 50
mini_data = {}
mini_data['X_train'] = data['X_train'][0:num_train]
mini_data['y_train'] = data['y_train'][0:num_train]
mini_data['X_val'] = data['X_val'][0:num_train]
mini_data['y_val'] = data['y_val'][0:num_train]

#############ThreeLayerConvNet overfit small dataset############
np.random.seed(231)

num_train = 100
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

model = ThreeLayerConvNet(weight_scale=1e-2)

solver = Solver(model, mini_data,
                num_epochs=15, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-4,
                },
                verbose=True, print_every=1)
solver.train()

plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()






'''
#model = TwoLayerNet()
model = FullyConnectedNet([100,50])
loss,grad = model.loss(mini_data['X_val'],mini_data['y_val'])
print (loss)
'''


''' 
##testing initial loss and gradient
def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

np.random.seed(231)
N, D, H1, H2, C = 2, 15, 20, 30, 10
X = np.random.randn(N, D)
y = np.random.randint(C, size=(N,))

for reg in [0, 3.14]:
  print('Running check with reg = ', reg)
  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                            reg=reg, weight_scale=5e-2, dtype=np.float64)

  loss, grads = model.loss(X, y)
  print('Initial loss: ', loss)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
    print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))

'''



'''
solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                    'learning_rate': 8e-3,
                },
                lr_decay=0.95,
                num_epochs=10, batch_size=100,
                print_every=100
                )
solver.train()
'''

  



'''
print(len(solver.train_acc_history))
#plt.figure()
plt.subplot(3,1,1)
plt.title("training loss")
plt.plot(solver.loss_history,'o',label="eeeee")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.subplot(3,1,3)
plt.title("accuracy")
plt.plot(solver.train_acc_history,'-',label='train')
plt.plot(solver.val_acc_history,'-',label='val')
plt.plot([0.3],'--')
plt.xlabel("epoch")
plt.ylabel("acc")
plt.show()
'''


'''
solvers = {}
dropout_choices = [0, 0.75]
for dropout in dropout_choices:
  model = FullyConnectedNet([500], dropout=dropout)
  print(dropout)

  solver = Solver(model, mini_data,
                  num_epochs=25, batch_size=100,
                  update_rule='sgd',
                  optim_config={
                    'learning_rate': 5e-4,
                  },
                  verbose=True, print_every=100)
  solver.train()
  solvers[dropout] = solver


# Plot train and validation accuracies of the two models

train_accs = []
val_accs = []
for dropout in dropout_choices:
  solver = solvers[dropout]
  train_accs.append(solver.train_acc_history[-1])
  val_accs.append(solver.val_acc_history[-1])

plt.subplot(3, 1, 1)
for dropout in dropout_choices:
  plt.plot(solvers[dropout].train_acc_history, 'o', label='%.2f dropout' % dropout)
plt.title('Train accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(ncol=1, loc='lower right')
  
plt.subplot(3, 1, 3)
for dropout in dropout_choices:
  plt.plot(solvers[dropout].val_acc_history, 'o', label='%.2f dropout' % dropout)
plt.title('Val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')

#plt.gcf().set_size_inches(15, 15)
plt.show()
'''
