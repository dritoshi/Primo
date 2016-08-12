#!/usr/bin/python3

import sys

sys.path.append("../")

if __name__ == '__main__':

    from primo import RNAseq, Wish, Position

    output_dir = "../results"

    r = (RNAseq().
         load_scRNAseq_data("../data/St13_1st_dge.txt.gz",
                            num_stamp=1200,
                            annotation_type="uid").
         remove_gene_toohigh(500).
         remove_outlier_cells(2.0).
         normalize().
         filter_variable_genes(z_cutoff=1.1, max_count=3,
                               bin_num=2000, stack=True).
         plot_cv(output_dir).
         tsne(plot=True, output_dir=output_dir, init="pca", random_state=12345)
         )

    w = (Wish().
         load_WISH_images("../data/wish",
                          annotation_type="symbol").
         symbol_to_uid("../data/uid_symbol.tsv").
         filter_images(pixel=80).
         plot_wish(output_dir)
         )

    p = (Position().
         load_inputs(r, w).
         calc_position().
         plot_position(output_dir, num_cells=100)
         )
