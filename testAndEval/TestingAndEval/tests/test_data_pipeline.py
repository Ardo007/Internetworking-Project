import sys
from pathlib import Path

# Add project root to sys.path so direct execution works seamlessly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.evaluation.metrics import calculate_eval_metrics

class TestEvaluationMetrics(unittest.TestCase):

    def test_calculate_eval_metrics(self):
        # Sample ground truth and predicted labels
        y_true = [0, 1, 0, 1, 0, 1, 0, 0]
        y_pred = [0, 1, 0, 0, 0, 1, 0, 1]  # 1 FN, 1 FP

        metrics = calculate_eval_metrics(y_true, y_pred)

        # Assertions to verify metric output format and math
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertEqual(metrics['true_positive'], 2)
        self.assertEqual(metrics['false_positive'], 1)

if __name__ == '__main__':
    unittest.main()