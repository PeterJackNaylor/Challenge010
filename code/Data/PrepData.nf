#!/usr/bin/env nextflow

process fuse_images {
    publishDir "../../intermediary_files/Data/labels", overwrite:true

    output:
    file "*_mask.png" into Labels
    file "description_table.csv" into DESC_TAB
    """
    #!/usr/bin/env python
    from Data.patch_img import image_files, masks_dic, fuse, meta_data
    from skimage.io import imsave
    from skimage.measure import label

    from os.path import basename
    from pandas import DataFrame

    tab = DataFrame()
    for file in image_files:
        bin_lab = fuse(file, masks_dic)
        label_name = basename(file.replace('.png', '_mask.png'))
        imsave(label_name, bin_lab)
        meta_data(file, label_name, tab)
    tab.to_csv('description_table.csv')
    """
}

process fuse_test_images {
    publishDir "../../intermediary_files/Data/test/tables", overwrite:true

    output:
    file "description_test_table.csv" into DESC_TEST_TAB
    """
    #!/usr/bin/env python
    from Data.patch_img import image_test_files, meta_data_test

    from os.path import basename
    from pandas import DataFrame

    tab = DataFrame()
    for file in image_test_files:
        meta_data_test(file, tab)
    tab.to_csv('description_test_table.csv')
    """
}

process dispatch_into_domaine {
    publishDir "../../intermediary_files/Data/Groups", overwrite:true
    input:
    file tab from DESC_TAB
    output:
    file "domain_*" into DOMAINED
    """
    #!/usr/bin/env python
    from pandas import read_csv
    from Data.patch_img import split_into_domain
    tab = read_csv('$tab', index_col=0)
    split_into_domain(tab)
    """
}

process dispatch_into_test_domaine {
    publishDir "../../intermediary_files/Data/test/Groups", overwrite:true
    input:
    file tab from DESC_TEST_TAB
    output:
    file "domain_*" into DOMAINED_TEST
    """
    #!/usr/bin/env python
    from pandas import read_csv
    from Data.patch_img import split_into_domain
    tab = read_csv('$tab', index_col=0)
    split_into_domain(tab)
    """
}

process FuseTable {
    publishDir "../../intermediary_files/Data/", overwrite: true
    input:
    file train_tab from DESC_TAB
    file test_tab from DESC_TEST_TAB
    output:
    file "train_test.csv" into TAB
    """
    #!/usr/bin/env python
    from pandas import read_csv, concat

    train = read_csv('$train_tab', index_col=0)
    test = read_csv('$test_tab', index_col=0)
    train['train'] = 1
    test['train'] = 0
    all = concat([train, test], axis=0)
    all.to_csv('train_test.csv')
    """
}

process OverlayRGB_GT {
    publishDir "../../intermediary_files/Data/Overlay", overwrite: true
    input:
    file train_tab from DESC_TAB
    output:
    file "*.png" into Overlay
    """
    #!/usr/bin/env python
    from pandas import read_csv, concat
    from Data.patch_img import Overlay
    from skimage.io import imsave
    from os.path import basename

    train = read_csv('$train_tab', index_col=0)
    def f(row):
        path_rgb = row['path_to_image']
        mask_name = row['path_to_label']
        if row['BlackBackGround'] == 1:
            black = False
        else:
            black = True
        overlay_rgb = basename(mask_name).replace('mask', 'overlay')
        imsave(overlay_rgb, Overlay(path_rgb, mask_name, black))

    train.apply(lambda row: f(row), axis=1)
    """
}


UNET_MAKE_DATA = file("UNet_Data_Prep.py")
SPLITS = 3

process MakeUNetData {
    publishDir "../../intermediary_files/Data/UNetData", overwrite: true
    input:
    file tab from TAB
    file UNET_MAKE_DATA
    val SPLITS
    output:
    file "data_unet" into TAB2
    """
    python $UNET_MAKE_DATA --input $tab --output data_unet --splits $SPLITS
    """
}