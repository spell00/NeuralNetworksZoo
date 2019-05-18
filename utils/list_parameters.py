h_dims = [64, 32]
h_dims_classifier = [128]
betas=(0.9, 0.999)
z_dims = [20]
a_dims = [20]
num_elements = 2

labels_per_class = -1
early_stopping = 100

automatic_attribute_list = ["GB_ACC"]


hebb_values = [0., 0]
hebb_values_neurites = [0., 0.]
hebb_rates = [0., 0.]
hebb_rates_inputs = [0., 0.]
hebb_rates_neurites = [0., 0.]
#Ns = [512, 512]
new_ns = [32, 32]
hebb_rates_multiplier = [0., 0.]
hebb_rates_inputs_multiplier = 0
hebb_rates_neurites_multiplier = [0., 0.]
gt = [0., 0.]
gt_neurites = [0., 0.]
gt_convs = [-10, -10]
planes = [16, 32, 64, 128, 256, 512]
kernels = [3, 3, 3, 3, 3, 3]
pooling_layers = [1, 1, 1, 1, 1, 1]
hebb_rates_conv_multiplier = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
new_ns_convs = [16, 32, 64, 128, 256, 512]
ladder = False
import_geo = False
has_unlabelled = False
import_local_file = True
import_dessins = False
is_example = False
local_folder = "./data/kaggle_dessins/"
