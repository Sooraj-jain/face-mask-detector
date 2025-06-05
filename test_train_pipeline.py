"""
Unit tests for the Face Mask Detection training pipeline.
Tests data validation, model building, and evaluation components.
"""

import unittest
import numpy as np
import os
import shutil
import cv2
from pathlib import Path
from train_mask_model import DataValidator, ModelValidator, build_model, CONFIG

class TestDataValidator(unittest.TestCase):
    """Test cases for data validation functionality."""
    
    def setUp(self):
        """Set up test environment with mock dataset."""
        self.test_data_path = Path("test_dataset")
        self.test_config = CONFIG.copy()
        self.test_config["dataset_path"] = str(self.test_data_path)
        self.test_config["min_samples_per_class"] = 2  # Lower for testing
        
        # Create mock dataset structure
        self._create_mock_dataset()
        
    def _create_mock_dataset(self):
        """Creates a mock dataset for testing."""
        # Create directories
        for class_name in ["with_mask", "without_mask"]:
            class_dir = self.test_data_path / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Create dummy images
            for i in range(3):
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                img_path = class_dir / f"test_image_{i}.jpg"
                cv2.imwrite(str(img_path), img)
    
    def tearDown(self):
        """Clean up test environment."""
        if self.test_data_path.exists():
            shutil.rmtree(str(self.test_data_path))
    
    def test_dataset_structure_validation(self):
        """Test dataset structure validation."""
        validator = DataValidator(self.test_config)
        results = validator.validate_dataset_structure()
        
        self.assertIn("with_mask_samples", results)
        self.assertIn("without_mask_samples", results)
        self.assertGreaterEqual(results["with_mask_samples"], 
                              self.test_config["min_samples_per_class"])

class TestModelValidator(unittest.TestCase):
    """Test cases for model validation functionality."""
    
    def setUp(self):
        """Set up test data for model validation."""
        self.X = np.random.random((100, 100, 100, 3))
        self.y = np.eye(2)[np.random.randint(0, 2, 100)]
        self.test_config = CONFIG.copy()
        self.test_config["cross_validation_folds"] = 2  # Reduce folds for testing
        
    def test_cross_validation(self):
        """Test cross-validation functionality."""
        validator = ModelValidator(self.test_config)
        metrics = validator.cross_validate_model(self.X, self.y, build_model)
        
        self.assertIn('cross_validation', metrics)
        self.assertIn('mean_accuracy', metrics['cross_validation'])
        self.assertIn('std_accuracy', metrics['cross_validation'])
        
    def test_robustness_evaluation(self):
        """Test model robustness evaluation."""
        model = build_model()
        validator = ModelValidator(self.test_config)
        
        # Train model briefly for testing
        model.fit(self.X, self.y, epochs=1, verbose=0)
        
        metrics = validator.evaluate_model_robustness(model, self.X[:10], self.y[:10])
        
        self.assertIn('base_accuracy', metrics)
        self.assertIn('noise_0.1_accuracy', metrics)
        self.assertIn('noise_0.2_accuracy', metrics)

class TestModelArchitecture(unittest.TestCase):
    """Test cases for model architecture."""
    
    def test_model_building(self):
        """Test model architecture building."""
        model = build_model()
        
        # Check model structure
        self.assertEqual(len(model.layers), 14)  # Updated number of layers
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, CONFIG["image_size"], 
                                          CONFIG["image_size"], 3))
        
        # Check output shape
        self.assertEqual(model.output_shape, (None, 2))  # Binary classification

if __name__ == '__main__':
    unittest.main() 