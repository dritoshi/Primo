#!/usr/bin/python3

import sys

sys.path.append("../")

if __name__ == '__main__':

    from primo import Primo

    p = Primo()
    (p.load_scRNAseq_data("../data/St13_1st_dge.txt.gz",
                          num_stamp=1200).
     remove_gene_toohigh(500).
     remove_outlier_cells(2.0).
     normalize().
     filter_variable_genes(z_cutoff=1.2, max_count=5,
                           bin_num=2000, stack=True).
     plot_cv("../results/")
     )
