import sys

sys.path.append("..")

if __name__ == '__main__':

    from primo.rnaseq import RNAseq
    from primo.wish import Wish
    from primo.position import Position
    from primo.spatial_expression import SpatialExpression

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
         infer_position().
         plot_position(output_dir, num_cells=40)
         )

    s = (SpatialExpression().
         load_inputs(r, p).
         predict()
         )

    # It takes long time. Do not run
    # s.plot_spatial_variable(output_dir, is_uid=True,
    #                                 conversion_table_file="../data/uid_symbol.tsv")

    # Image show of interest genes
    gene_list = ['Xl.1685', 'Xl.16508', 'Xl.1588']
    s.plot_spatial_interest(output_dir, gene_list=gene_list, is_uid=True,
                            conversion_table_file="../data/uid_symbol.tsv")

    # LOOCV
    p.calc_position_loocv()
