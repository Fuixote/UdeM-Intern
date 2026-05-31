import unittest

import numpy as np

from surrogate_experiment_results.SPO_validation.step1c_vs_pyepo import validation_core


class Step1cVsPyepoValidationCoreTests(unittest.TestCase):
    def test_linear_predictions_match_pytorch_weight_convention(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        weight = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.5]])
        bias = np.array([0.1, -0.2, 0.3])

        pred = validation_core.linear_predict(x, weight, bias)

        expected = x @ weight + bias
        np.testing.assert_allclose(pred, expected)

    def test_mean_parameter_gradients_include_bias(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        grad_pred = np.array([[0.2, -0.4, 0.6], [0.8, 1.0, -1.2]])

        grad_weight, grad_bias = validation_core.mean_linear_parameter_gradients(
            x, grad_pred
        )

        np.testing.assert_allclose(grad_weight, x.T @ grad_pred / 2.0)
        np.testing.assert_allclose(grad_bias, grad_pred.mean(axis=0))

    def test_cost_min_gradient_is_scaled_for_batch_mean(self):
        y_opt = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        y_adv = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])

        grad = validation_core.step1c_cost_min_grad_pred(y_opt, y_adv, mean=True)

        expected = 2.0 * (y_opt - y_adv) / 2.0
        np.testing.assert_allclose(grad, expected)

    def test_sgd_step_updates_weight_and_bias(self):
        weight = np.array([[1.0, -1.0], [0.5, 0.25]])
        bias = np.array([0.1, -0.2])
        grad_weight = np.array([[0.2, 0.4], [-0.3, 0.5]])
        grad_bias = np.array([0.7, -0.9])

        next_weight, next_bias = validation_core.sgd_step(
            weight, bias, grad_weight, grad_bias, lr=0.05
        )

        np.testing.assert_allclose(next_weight, weight - 0.05 * grad_weight)
        np.testing.assert_allclose(next_bias, bias - 0.05 * grad_bias)


if __name__ == "__main__":
    unittest.main()
