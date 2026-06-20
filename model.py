import pandas as pd
import joblib


class UFCModel:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.is_loaded = False  # Add this flag

    def load(self):
        """Load pre-trained model"""
        try:
            self.model = joblib.load('ufc_win_predictor.pkl')
            self.feature_columns = joblib.load('ufc_features.pkl')
            self.is_loaded = True
            print("✅ UFC model loaded successfully")
            print(f"   Features: {len(self.feature_columns)}")
            return True
        except Exception as e:
            print(f"⚠️ UFC model loading error: {e}")
            self.is_loaded = False
            return False

    def get_feature_columns(self):
        """Return the list of feature columns used by the model"""
        if self.feature_columns is None:
            return []
        return self.feature_columns

    def predict(self, metrics_dict):
        return self.predict_win_probability(
            height_cm=metrics_dict.get('height_cm', 180),
            weight_in_kg=metrics_dict.get('weight_in_kg', 77),
            reach_in_cm=metrics_dict.get('reach_in_cm', 185),
            strikes_per_minute=metrics_dict.get('significant_strikes_landed_per_minute', 4.5),
            striking_accuracy=metrics_dict.get('significant_striking_accuracy', 55),
            strikes_absorbed=metrics_dict.get('significant_strikes_absorbed_per_minute', 3.0),
            strike_defence=metrics_dict.get('significant_strike_defence', 60),
            takedowns_per_15min=metrics_dict.get('average_takedowns_landed_per_15_minutes', 2.5),
            takedown_accuracy=metrics_dict.get('takedown_accuracy', 50),
            takedown_defense=metrics_dict.get('takedown_defense', 70),
            submissions_per_15min=metrics_dict.get('average_submissions_attempted_per_15_minutes', 1.0),
            stance=metrics_dict.get('stance', 'Orthodox')
        )

    def predict_win_probability(self,
                                height_cm,
                                weight_in_kg,
                                reach_in_cm,
                                strikes_per_minute,
                                striking_accuracy,
                                strikes_absorbed,
                                strike_defence,
                                takedowns_per_15min,
                                takedown_accuracy,
                                takedown_defense,
                                submissions_per_15min,
                                stance='Orthodox'):
        """
        Predict win probability for a fighter with given metrics
        """
        # Create feature dictionary with numerical values
        features = {
            'height_cm': float(height_cm),
            'weight_in_kg': float(weight_in_kg),
            'reach_in_cm': float(reach_in_cm),
            'significant_strikes_landed_per_minute': float(strikes_per_minute),
            'significant_striking_accuracy': float(striking_accuracy),
            'significant_strikes_absorbed_per_minute': float(strikes_absorbed),
            'significant_strike_defence': float(strike_defence),
            'average_takedowns_landed_per_15_minutes': float(takedowns_per_15min),
            'takedown_accuracy': float(takedown_accuracy),
            'takedown_defense': float(takedown_defense),
            'average_submissions_attempted_per_15_minutes': float(submissions_per_15min),
        }

        # --- FIX: Handle stance dynamically ---
        # Get all stance columns from training data
        stance_columns = [col for col in self.feature_columns if col.startswith('stance_')]

        # Initialize all stance columns to 0
        for col in stance_columns:
            features[col] = 0

        # Set the correct stance to 1
        stance_col = f'stance_{stance}'
        if stance_col in features:
            features[stance_col] = 1
        else:
            # If stance not found, default to Orthodox if available
            if 'Stance_Orthodox' in features:
                features['stance_Orthodox'] = 1
                print(f"⚠️ Stance '{stance}' not found. Defaulting to Orthodox.")
            else:
                # If no stance columns exist, use the first one found
                for col in stance_columns:
                    features[col] = 1
                    print(f"⚠️ Stance '{stance}' not found. Defaulting to {col}.")
                    break

        # Create DataFrame with proper column order
        player_data = pd.DataFrame([features])[self.feature_columns]

        # Scale and predict
        lose_prob = self.model.predict(player_data)[0]

        # Calculate confidence based on prediction strength
        confidence = lose_prob

        return {
            'lose_probability': lose_prob,
            'confidence': round(confidence, 1),
            'metrics': {
                'height_cm': height_cm,
                'weight_in_kg': weight_in_kg,
                'reach_in_cm': reach_in_cm,
                'strikes_per_minute': strikes_per_minute,
                'striking_accuracy': striking_accuracy,
                'strikes_absorbed': strikes_absorbed,
                'strike_defence': strike_defence,
                'takedowns_per_15min': takedowns_per_15min,
                'takedown_accuracy': takedown_accuracy,
                'takedown_defense': takedown_defense,
                'submissions_per_15min': submissions_per_15min,
                'stance': stance
            }
        }

# Singleton instance
ufc_model = UFCModel()