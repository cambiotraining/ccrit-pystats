import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.figure import Figure

from ccrit_pystats import dgplots


class DiagnosticPlotsTest(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_regression_uses_fitted_values_and_cooks_distance(self):
        x = np.arange(8)
        y = [1.0, 2.0, 4.0, 6.0, 7.0, 8.0, 11.0, 18.0]
        result = sm.OLS(y, sm.add_constant(x)).fit()

        fig = dgplots(result)
        fig.canvas.draw()

        self.assertIsInstance(fig, Figure)
        self.assertTrue(plt.fignum_exists(fig.number))
        self.assertEqual(
            [(ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) for ax in fig.axes],
            [
                ("Residual Plot", "Predicted Values", "Residuals"),
                ("Q-Q Plot", "Theoretical Quantiles", "Sample Quantiles"),
                (
                    "Location-Scale Plot",
                    "Predicted Values",
                    r"$\sqrt{|\mathrm{Standardised\ Residuals}|}$",
                ),
                ("Cook's D versus Index Plot", "Observation Number", "Cook's D"),
            ],
        )
        np.testing.assert_allclose(
            fig.axes[0].collections[0].get_offsets(),
            np.column_stack((result.fittedvalues, result.resid)),
        )
        np.testing.assert_allclose(
            fig.axes[3].collections[1].get_offsets()[:, 1],
            result.get_influence().cooks_distance[0],
        )

    def test_inline_figure_is_returned_without_queuing_it_for_display(self):
        other_figure = plt.figure()
        result = sm.OLS([1, 2, 4, 8], sm.add_constant([1, 2, 3, 4])).fit()

        with patch.object(
            plt, "get_backend", return_value="module://matplotlib_inline.backend_inline"
        ):
            fig = dgplots(result)

        self.assertFalse(plt.fignum_exists(fig.number))
        self.assertTrue(plt.fignum_exists(other_figure.number))
        fig.canvas.draw()
        self.assertEqual(len(fig.axes), 4)

    def test_anova_uses_labels_from_the_rows_that_were_fitted(self):
        data = pd.DataFrame(
            {
                "group": ["a", "a", "a", None, "b", "b", "b"],
                "y": [1.0, 1.2, 1.1, 30.0, 2.0, 2.1, 4.0],
            },
            index=["a1", "a2", "a3", "excluded", "b1", "b2", "b3"],
        )
        result = smf.ols("y ~ C(group)", data=data).fit()

        fig = dgplots(result)
        fig.canvas.draw()

        self.assertEqual(len(fig.axes[0].collections[0].get_offsets()), 6)
        labels = [annotation.get_text() for annotation in fig.axes[3].texts]
        self.assertIn("b3", labels)
        self.assertNotIn("excluded", labels)
        self.assertEqual(len(fig.axes[2].lines[0].get_xdata()), 2)

    def test_anova_with_a_singleton_keeps_its_raw_residual(self):
        data = pd.DataFrame({"group": ["a", "a", "a", "b"], "y": [1, 2, 4, 6]})
        result = smf.ols("y ~ C(group)", data=data).fit()

        with self.assertWarnsRegex(RuntimeWarning, "Excluded 1 fitted observation"):
            fig = dgplots(result)

        self.assertEqual(len(fig.axes[0].collections[0].get_offsets()), 4)
        self.assertEqual(len(fig.axes[2].collections[0].get_offsets()), 3)

    def test_requires_fitted_ols(self):
        model = sm.OLS([1, 2, 3], sm.add_constant([1, 2, 3]))
        with self.assertRaisesRegex(TypeError, "fitted result"):
            dgplots(model)

    def test_requires_residual_degrees_of_freedom(self):
        result = sm.OLS([1, 2], sm.add_constant([1, 2])).fit()
        with self.assertRaisesRegex(ValueError, "degrees of freedom"):
            dgplots(result)
