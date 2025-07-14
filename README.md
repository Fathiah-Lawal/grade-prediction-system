# Grade Prediction System

A machine learning-based grade prediction system built with Python and Streamlit for deployment.

## Overview

This system predicts student grades based on various input features using machine learning models. The application provides an interactive web interface for users to input student data and get grade predictions.

## Features

- **Machine Learning Models**: Trained models for accurate grade prediction
- **Interactive Web Interface**: Built with Streamlit for easy user interaction
- **Data Visualization**: Charts and graphs to visualize prediction results
- **Model Performance**: Evaluation metrics and model comparison
- **User-Friendly**: Simple interface for inputting student information

## Project Structure

```
grade_prediction_streamlit/
├── models/                 # Trained ML models
├── assets/                # Static assets
├── venv/                  # Virtual environment
├── __pycache__/          # Python cache files
├── admin.py              # Admin functionality
├── app2.py               # Secondary application file
├── app3.py               # Tertiary application file
├── augment_data.py       # Data augmentation script
├── auth.py               # Authentication module
├── balanced_grade_model.pkl  # Balanced ML model
├── comprehensive_augmented_data.xlsx  # Augmented dataset
├── comprehensive_balanced_grade_model.pkl  # Comprehensive model
├── data.xlsx             # Original dataset
├── data_utils.py         # Data utility functions
├── db.py                 # Database operations
├── feedback_utils.py     # Feedback handling
├── main.py               # Main application file
├── model_utils.py        # Model utility functions
├── password_reset.py     # Password reset functionality
├── results.xlsx          # Results data
├── train_model.py        # Model training script
├── unsupervised_learning.py  # Unsupervised learning module
└── requirements.txt      # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Fathiah-Lawal/grade_prediction_streamlit.git
cd grade_prediction_streamlit
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

1. Start the Streamlit application:
```bash
streamlit run app2.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Use the interface to:
   - Input student information
   - Get grade predictions
   - View prediction confidence
   - Analyze results

### Training Models

To retrain the models with new data:
```bash
python train_model.py
```

### Data Augmentation

To augment the training data:
```bash
python augment_data.py
```

## Model Information

The system uses multiple machine learning models:
- **Balanced Grade Model**: Handles class imbalance in grade distribution
- **Comprehensive Model**: Uses comprehensive feature set for prediction
- **Unsupervised Learning**: For data exploration and pattern discovery

## Data Features

The system considers various factors for grade prediction:
- Student demographic information
- Academic history
- Attendance records
- Assignment scores
- Participation metrics
- Other relevant academic indicators

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy the app

### Local Deployment

For local deployment, simply run:
```bash
streamlit run app2.py.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please contact [fatiaholajumoke@gmail.com]

## Acknowledgments

- Thanks to the academic institutions providing data insights
- Built with Streamlit for web interface
- Machine learning models powered by scikit-learn