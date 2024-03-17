import numpy as np
from stroke.bids_loader import BIDSLoader
from rampwf.score_types import BaseScoreType


class DiceCoeff(BaseScoreType):
    def __init__(self, name="Sørensen–Dice Coefficient", precision=3):
        """
        Scoring class for RAMP workflows. When called, returns the Sørensen–Dice coefficient. Note that this
        implementation allows for continuous values in the prediction.
        Parameters
        ----------
        name : str
            Name of the score; used for creating column headers.
        precision : str
            Numerical precision.
        """
        # RAMP-related convention
        self.name = name
        self.precision = precision
        self.is_lower_the_better = False
        self.minimum = 0
        self.maximum = 1
        return

    def __call__(self, y_true: tuple, y_pred: np.array):
        return self.score_function(Y_true=y_true, Y_pred=y_pred)

    def score_function(self, Y_true: np.array, Y_pred: np.array):
        """
        Returns the Sørensen–Dice coefficient for the input images. If multiple samples are given, the mean score
        is returned.
        Parameters
        ----------
        true_bids : BIDSPrediction
            BIDSPrediction with true_bids.y_true set as an array.
        pred : np.array
            Array containing the predicted labels of an image.

        Returns
        -------
        dice_coefficient : float
            Sørensen–Dice coefficient.
        """
        y_true = np.array(Y_true.y_true)
        if len(Y_pred.y_pred) != 0:
            assert len(y_true) == len(Y_pred.y_pred)
        if len(Y_pred.y_pred) == 0:
            return 0
        estimator = Y_pred.y_pred[0].estimator

        fscore = 0
        for y_true_i, prediction_object in zip(y_true, Y_pred.y_pred):
            dat = estimator.predict(
                BIDSLoader.load_image_tuple(prediction_object.pred)[None]
            )[0]

            # Using proxy of y_true.shape != y_pred.shape to indicate that data needs to be unpacked
            if y_true_i.shape != dat.shape:
                y_true_i = np.array(
                    self.unpack_data(y_true_i, dat.shape), dtype=dat.dtype
                )
            fscore += self.calc_score(dat, y_true_i)

        # Return the mean score
        return fscore / len(y_true)

    @staticmethod
    def unpack_data(array_0: np.array, output_shape: np.array):
        """
        Unpacks boolean data packed via np.packbits into appropriate shape, discarding excess bytes
        Parameters
        ----------
        array_0 : np.array
            np.uint8 array to unpack
        output_shape : tuple
            Expected shape of output.

        Returns
        -------
        np.array
            Unpacked, reshape array
        """
        unpack_shape = np.prod(array_0.shape) * 8
        extra_entries = unpack_shape - np.prod(output_shape)
        return np.unpackbits(array_0)[:-extra_entries].reshape(output_shape)

    @staticmethod
    def calc_score(array_0: np.array, array_1: np.array):
        """
        Performs the calculation to get the Sørensen–Dice coefficient.
        Parameters
        ----------
        array_0 : np.array
            First array to score.
        array_1 : np.array
            Second array to score.

        Returns
        -------
        float
            Sørensen–Dice coefficient
        """
        # Reshape to use dot product
        # NOTE: This computation of the coefficient allows for continuous
        # values in the prediction.
        overlap, sum0, sum1 = DiceCoeff.calc_score_parts(array_0, array_1)
        sorenson = overlap / (sum0 + sum1)
        return sorenson

    @staticmethod
    def calc_score_parts(array_0: np.array, array_1: np.array):
        """
        Computes the three parts of the Sørensen–Dice coefficient: overlap and 2 positives
        Parameters
        ----------
        array_0
        array_1

        Returns
        -------
        tuple
            Tuple containing (overlap, sum(array_0), sum(array_1)
        """
        overlap = 2 * array_0.ravel() @ array_1.ravel()
        return (overlap, np.sum(array_0), np.sum(array_1))

    @staticmethod
    def check_y_pred_dimensions(array_0: np.array, array_1: np.array):
        """
        Checks that the dimensions of the inputs are consistent.
        Parameters
        ----------
        array_0 : np.array
            First array to check.
        array_1 : np.array
            Second array to check

        Returns
        -------
        bool
        """
        if array_0.shape != array_1.shape:
            return False
        else:
            return True
