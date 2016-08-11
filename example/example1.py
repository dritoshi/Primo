#!/usr/bin/python3

import sys

sys.path.append("../")

if __name__ == '__main__':

    from primo import Primo

    output_dir = "../results"

    p = Primo()
    (p.load_scRNAseq_data("../data/St13_1st_dge.txt.gz",
                          num_stamp=1200).
     remove_gene_toohigh(500).
     remove_outlier_cells(2.0).
     normalize().
     filter_variable_genes(z_cutoff=1.1, max_count=3,
                           bin_num=2000, stack=True).
     plot_cv(output_dir).
     tsne(plot=True, output_dir=output_dir, init="pca", random_state=12345)
     )
