

class PortfolioOptimConfig:

    def __init__(self,training_sample_width, out_of_sample_width):

        self.training_sample_width = training_sample_width
        self.out_of_sample_width = out_of_sample_width


class POETConfig:

    def __init__(self, cross_validation_fold_number, C_grid_precision, cross_validation_precision):
        self.cross_validation_fold_number = cross_validation_fold_number
        self.C_grid_precision = C_grid_precision
        self.cross_validation_precision = cross_validation_precision

