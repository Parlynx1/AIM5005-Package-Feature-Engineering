from aim5005.features import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
import unittest
from unittest.case import TestCase


class TestFeatures(TestCase):
    def test_initialize_min_max_scaler(self):
        scaler = MinMaxScaler()
        assert isinstance(scaler, MinMaxScaler), "scaler is not a MinMaxScaler object"
        
    def test_min_max_fit(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler.fit(data)
        assert (scaler.maximum == np.array([1., 18.])).all(), "scaler fit does not return maximum values [1., 18.]"
        assert (scaler.minimum == np.array([-1., 2.])).all(), "scaler fit does not return minimum values [-1., 2.]"
        
    def test_min_max_scaler(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. Got: {}".format(result)
        
    def test_min_max_scaler_single_value(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[1.5, 0.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Got: {}".format(result)
        
    def test_standard_scaler_init(self):
        scaler = StandardScaler()
        assert isinstance(scaler, StandardScaler), "scaler is not a StandardScaler object"
        
    def test_standard_scaler_get_mean(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.mean == expected).all(), "scaler fit does not return expected mean. Got {}".format(scaler.mean)
        
    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. Got: {}".format(result)
        
    def test_standard_scaler_single_value(self):
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[3., 3.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Got: {}".format(result)

    # Additional test case for StandardScaler: Edge case when all values are the same
    def test_scalers_with_constant_values(self):
        data = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
        minmax_scaler = MinMaxScaler()
        standard_scaler = StandardScaler()
        standard_scaler.fit(data)
        standard_result = standard_scaler.transform(data)
        expected_standard = np.array([[0., 0.], [0., 0.], [0., 0.], [0., 0.]])
        assert (standard_result == expected_standard).all(), f"StandardScaler failed for constant values 0.5. Got: {standard_result}"

# Test cases for Label encoder
    def test_label_encoder_fit(self):
        encoder = LabelEncoder()
        labels = ['paris', 'tokyo', 'amsterdam', 'tokyo', 'paris']
        encoder.fit(labels)
        
        expected_classes = ['amsterdam', 'paris', 'tokyo']
        assert (encoder.classes_ == expected_classes).all(), f"Expected classes: {expected_classes}, but got: {encoder.classes_}"

    def test_label_encoder_transform(self):
        encoder = LabelEncoder()
        labels = ['paris', 'tokyo', 'amsterdam', 'tokyo', 'paris']
        encoder.fit(labels)
        
        transformed_labels = encoder.transform(labels)
        expected_transformed = np.array([1, 2, 0, 2, 1])
        
        assert (transformed_labels == expected_transformed).all(), f"Expected transformed labels: {expected_transformed}, but got: {transformed_labels}"

    def test_label_encoder_fit_transform(self):
        encoder = LabelEncoder()
        labels = ['paris', 'tokyo', 'amsterdam', 'tokyo', 'paris']
        transformed_labels = encoder.fit_transform(labels)
        expected_transformed = np.array([1, 2, 0, 2, 1])
        
        assert (transformed_labels == expected_transformed).all(), f"Expected transformed labels: {expected_transformed}, but got: {transformed_labels}"


if __name__ == '__main__':
    unittest.main()

