from data_preparation.GeoParser import GeoParser
from dimension_reduction.ordination import ordination2d
import pandas as pd
import numpy as np
from numpy import genfromtxt


def adapt_datasets(df1, df2):
    empty_dataframe_indices1 = pd.DataFrame(index=df1.index)
    empty_dataframe_indices2 = pd.DataFrame(index=df2.index)
    df1 = df1.add(empty_dataframe_indices2, fill_value=0).fillna(0)
    df2 = df2.add(empty_dataframe_indices1, fill_value=0).fillna(0)
    print("LEN adapted", len(df1.index))
    return df1, df2


def print_infos():
    pass


def get_num_parameters():
    pass


def __main__():
    train_ds = None
    valid_ds = None
    meta_df = None
    unlabelled_meta_df = None
    from torchvision import transforms, datasets

    from models.semi_supervised.deep_generative_models.models.auxiliary_dgm import AuxiliaryDeepGenerativeModel
    from models.semi_supervised.deep_generative_models.models.ladder_dgm import LadderDeepGenerativeModel
    from models.semi_supervised.deep_generative_models.models.dgm import DeepGenerativeModel
    from utils.files_destinations import home_path, destination_folder, data_folder, results_folder, plots_folder_path
    examples_list = ["mnist", "cifar10", "cifar100"]
    if example not in examples_list and not import_local_file and import_geo:
        g = GeoParser(home_path=home_path, geo_ids=geo_ids, unlabelled_geo_ids=unlabelled_geo_ids, bad_geo_ids=bad_geo_ids)
        g.get_geo(load_from_disk=load_from_disk)
        meta_df = g.merge_datasets(load_from_disk=load_merge, labelled=True)
        unlabelled_meta_df = g.merge_datasets(load_from_disk=load_merge, labelled=False)
        if translate is "y":
            for geo_id in geo_ids:
                g.translate_indices_df(geo_id, labelled=True)
            for geo_id in unlabelled_geo_ids:
                g.translate_indices_df(geo_id, labelled=False)
        meta_df, unlabelled_meta_df = adapt_datasets(meta_df, unlabelled_meta_df)
    elif import_local_file:
        train_arrays = np.load(local_folder + train_images_fname, encoding="latin1")
        train_dataset = np.vstack(train_arrays[:, 1])
        train_labels = genfromtxt(local_folder + train_labels_fname, delimiter=",", dtype=str, skip_header=True)[:, 1]
        test_dataset = np.vstack(np.load(local_folder + "test_images.npy", encoding="latin1")[:, 1])

        meta_df = pd.DataFrame(train_dataset, columns=train_labels)
        img_shape = [1, 100, 100]
    elif import_dessins:
        train_labels = genfromtxt("data/kaggle_dessins/" + train_labels_fname, delimiter=",", dtype=str, skip_header=True)[:, 1]
        data_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        dessins_train_data = datasets.ImageFolder(root='data/kaggle_dessins/train/',
                                                   transform=data_transform)
        dessins_valid_data = datasets.ImageFolder(root='data/kaggle_dessins/valid/',
                                                   transform=data_transform)
        train_ds = torch.utils.data.DataLoader(dessins_train_data,
                                                     batch_size=batch_size, shuffle=True,
                                                     num_workers=0)
        valid_ds = torch.utils.data.DataLoader(dessins_valid_data,
                                                     batch_size=batch_size, shuffle=True,
                                                     num_workers=0)

    if ords == "pca" or ords == "both" or ords == "b" or ords == "p":
        print("PCA saved at: ", plots_folder_path)
        ordination2d(meta_df, epoch="pre", dataset_name=dataset_name, ord_type="pca",
                     images_folder_path=plots_folder_path, info=str(geo_ids)+str(unlabelled_geo_ids)+str(bad_geo_ids))
    if ords is "tsne" or ords is "both" or ords is "b" or ords is "t":
        ordination2d(g.meta_df, dataset_name, "tsne", dataset_name, plots_folder_path + "tsne/",
                     info=str(geo_ids)+str(unlabelled_geo_ids)+str(bad_geo_ids))




    if "dgm" in nets or "DGM" in nets or all_automatic:

        is_example = False
        from utils.list_parameters import z_dims, h_dims, num_elements, a_dims
        if auxiliary:
            dgm = AuxiliaryDeepGenerativeModel(vae_flavour, z_dims, h_dims, n_flows=number_of_flows,a_dim=a_dim,
                                               num_elements=num_elements, use_conv=use_conv)

            dgm.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                            destination_folder=destination_folder, dataset_name=dataset_name, lr=initial_lr,
                            meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers")

        elif ladder:
            dgm = LadderDeepGenerativeModel(vae_flavour, z_dims, h_dims, n_flows=number_of_flows, auxiliary=False)

            dgm.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                            destination_folder=destination_folder, dataset_name=dataset_name, lr=initial_lr,
                            meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers")
        else:
            print(vae_flavour)
            dgm = DeepGenerativeModel(vae_flavour, z_dims, h_dims, n_flows=number_of_flows, a_dim=None, auxiliary=False,
                                      num_elements=num_elements)

            dgm.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                            destination_folder=destination_folder, dataset_name=dataset_name, lr=initial_lr,
                            meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers")
        if meta_df is not None:
            dgm.import_dataframe(meta_df, batch_size, labelled=True)
            if has_unlabelled:
                dgm.import_dataframe(unlabelled_meta_df, batch_size, labelled=False)
            else:
                dgm.import_dataframe(meta_df, batch_size, labelled=False)
        elif example in examples_list:
            dgm.load_example_dataset(dataset=example, batch_size=batch_size, labels_per_class=400)
            is_example = True
            print("PCA saved at: ", plots_folder_path)
            #meta_df = pd.DataFrame(dgm.train_ds)
            #ordination2d(meta_df, epoch="pre", dataset_name=dataset_name, ord_type="pca",
            #             images_folder_path=plots_folder_path, info=str(geo_ids)+str(unlabelled_geo_ids)+str(bad_geo_ids))
        elif import_dessins:
            import os
            dgm.batch_size = batch_size
            dgm.input_shape = [1, 100, 100]
            dgm.input_size = 10000
            dgm.labels_set = os.listdir('data/kaggle_dessins/train/')
            dgm.num_classes = len(dgm.labels_set)
            dgm.make_loaders(train_ds=train_ds, valid_ds=valid_ds, test_ds=None, labels_per_class=labels_per_class,
                             unlabelled_train_ds=None, unlabelled_samples=True)

        dgm.define_configurations(early_stopping=early_stopping, warmup=warmup, flavour=vae_flavour)
        if import_local_file:
            dgm.labels = train_labels
            dgm.labels_set = list(set(train_labels))
        if not import_dessins:
            dgm.set_data(labels_per_class=labels_per_class, is_example=is_example)
        dgm.cuda()

        if auxiliary:
            if use_conv:
                dgm.set_conv_adgm_layers(is_hebb_layers=False, input_shape=img_shape)
            else:
                dgm.set_adgm_layers(h_dims=h_dims_classifier)
        elif ladder:
            dgm.set_ldgm_layers()
        else:
            if use_conv:
                dgm.set_conv_dgm_layers(input_shape=img_shape)
            else:
                dgm.set_dgm_layers()

        # import the M1 in the M1+M2 model (Kingma et al, 2014)
        if load_vae:
            print("Importing the model: ", dgm.model_file_name)
            if use_conv:
                dgm.import_cvae()
            else:
                dgm.load_ae(load_history=False)
            #dgm.set_dgm_layers_pretrained()

        if resume:
            print("Resuming training")
            dgm.load_model()

        dgm.cuda()
        # dgm.vae.generate_random(False, batch_size, z1_size, [1, 28, 28])
        dgm.run(n_epochs, auxiliary, mc, iw, lambda1=l1, lambda2=l2 )


if __name__ == "__main__":
    from utils.parameters import *
    from utils.list_parameters import *
    __main__()
