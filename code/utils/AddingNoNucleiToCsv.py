
from optparse import OptionParser
from glob import glob
import pandas as pd
import pdb
import numpy as np

if __name__== "__main__":
    parser = OptionParser()
    parser.add_option('--folder', dest="folder", type="str")
    parser.add_option('--csv', dest="csv", type="str")
    parser.add_option('--out_csv', dest="out_csv", type="str")
    (options, args) = parser.parse_args()

    # table = pd.read_csv(options.csv, index_=0)
    # pdb.set_trace()
    # for f in glob(options.folder+'/*'):
    #     name = f.split('/')[-1]
    #     if name not in table.index:
    #         table.loc[name, "EncodedPixels"] = ""
    # table.to_csv(options.out_csv)




    with open(options.csv) as f:
        lines = f.readlines()
        pdb.set_trace()
        with open(options.out_csv, "w") as f1:
            f1.writelines(lines)
        f1.close()
    f.close()
    del lines[0] # get ride of header
    line_names = [l.split(',')[0] for l in lines]
    reduce_line_names = list(np.unique(line_names))

    with open(options.out_csv, 'a') as f:
        for fi in glob(options.folder+'/*'):
            if '.csv' not in fi:
                name_img = fi.split('/')[-1]
                if name_img not in reduce_line_names:
                    f.write('{},\n'.format(name_img))
    f.close()