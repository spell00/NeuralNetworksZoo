import torch
def balance_relu(x, hyper_balance=1., overall_mean=False, all0=False):
    if overall_mean:
        sum_a = len(x[x > 0.])
        tot = len(x)
        #sum_b = len(x[x == 0.])
        ratio = tot / (sum_a+1e-8)
        mu_x = torch.mean(x)
        x[x == 0.] = -mu_x*ratio
        #x = torch.Tensor([x_i if x_i > 0. else -mu_x*ratio for x_i in tmp]).cuda().reshape(x.shape)
    else:
        mu_x = torch.mean(x, 1)
        for i, x_i in enumerate(x):
            sum_a = len(x_i[x_i == 0.])
            sum_b = len(x_i[x_i > 0.])
            tot = len(x_i)
            #if true, the total of all neurons will be true. Otherwise, each neuron's sum of a batch should be 0
            if all0:
                ratio = tot / (sum_a+1e-8)
            else:
                ratio = sum_a / (sum_b+1e-8)

            x_i[x_i == 0.] = -mu_x[i]*ratio*hyper_balance
            x[i] = x_i

    return x

