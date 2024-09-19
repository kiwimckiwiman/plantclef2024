import math

true = [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.]
with_sig = [0.9994, 0.9995, 0.9995, 0.0017, 0.0018, 0.0017, 0.0017, 0.0017, 0.0015, 0.0017]

calc_avg_val_loss = 5.16 
loss = 0
scce_loss = 0
for i in range(len(with_sig)):
    loss += -(true[i]*math.log(with_sig[i]))

soft_denom = 0

for i in range(len(with_sig)):
    soft_denom += math.exp(with_sig[i])
    
for i in range(len(with_sig)):
    scce_loss += -(math.log(math.exp(with_sig[i])/soft_denom)*true[i])
print(loss)
print(scce_loss)