import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

class UnsupervisedLearningAnalyzer:
    """
    Comprehensive unsupervised learning analyzer for grade prediction system
    """
    
    def __init__(self, data_path="comprehensive_augmented_data.csv"):
        self.data_path = data_path
        self.df = None
        self.scaled_features = None
        self.feature_columns = None
        self.clusters = None
        self.anomalies = None
        self.student_profiles = None
        
    def load_and_prepare_data(self):
        """Load and prepare data for unsupervised analysis"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.df)} records")
            
            # Create grade classification
            self.df['Grade_Class'] = self.df['Exam_Score'].apply(self.classify_grade)
            
            # Select numerical features for clustering
            numerical_features = [
                'Hours_Studied', 'Previous_Scores', 'Attendance', 'Tutoring_Sessions',
                'Sleep_Hours', 'Physical_Activity', 'Study_Attendance_Interaction'
            ]
            
            # Add encoded categorical features
            categorical_features = [
                'School_Type', 'Teacher_Quality', 'Gender', 'Internet_Access',
                'Learning_Disabilities', 'Parental_Involvement', 'Family_Income',
                'Access_to_Resources', 'Motivation_Level', 'Peer_Influence'
            ]
            
            # One-hot encode categorical features
            df_encoded = pd.get_dummies(self.df[categorical_features], drop_first=True)
            
            # Combine numerical and encoded categorical features
            self.feature_columns = numerical_features + list(df_encoded.columns)
            feature_data = pd.concat([
                self.df[numerical_features], 
                df_encoded
            ], axis=1)
            
            # Scale features
            scaler = StandardScaler()
            self.scaled_features = scaler.fit_transform(feature_data)
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def classify_grade(self, score):
        """Grade classification function"""
        if score >= 70:
            return 'First Class'
        elif score >= 60:
            return 'Second Class Upper'
        elif score >= 50:
            return 'Second Class Lower'
        elif score >= 45:
            return 'Third Class'
        elif score >= 40:
            return 'Pass'
        else:
            return 'Fail'
    
    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        print("🔍 Finding optimal number of clusters...")
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(self.scaled_features)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_features, clusters))
        
        # Find optimal k (highest silhouette score)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"✅ Optimal number of clusters: {optimal_k}")
        print(f"📊 Best silhouette score: {max(silhouette_scores):.3f}")
        
        return optimal_k, inertias, silhouette_scores
    
    def perform_student_clustering(self, n_clusters=None):
        """Perform K-means clustering to identify student profiles"""
        print("👥 Performing student clustering analysis...")
        
        if n_clusters is None:
            n_clusters, _, _ = self.find_optimal_clusters()
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(self.scaled_features)
        
        # Add cluster labels to dataframe
        self.df['Student_Cluster'] = self.clusters
        
        # Analyze cluster characteristics
        self.analyze_cluster_profiles()
        
        return self.clusters
    
    def analyze_cluster_profiles(self):
        """Analyze and name cluster profiles based on characteristics"""
        print("📋 Analyzing cluster profiles...")
        
        cluster_profiles = {}
        
        for cluster_id in range(len(np.unique(self.clusters))):
            cluster_data = self.df[self.df['Student_Cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'avg_exam_score': cluster_data['Exam_Score'].mean(),
                'avg_hours_studied': cluster_data['Hours_Studied'].mean(),
                'avg_attendance': cluster_data['Attendance'].mean(),
                'avg_previous_scores': cluster_data['Previous_Scores'].mean(),
                'common_grade': cluster_data['Grade_Class'].mode().iloc[0],
                'grade_distribution': cluster_data['Grade_Class'].value_counts().to_dict()
            }
            
            # Assign meaningful names based on characteristics
            if profile['avg_exam_score'] >= 70:
                profile['name'] = "High Achievers"
                profile['description'] = "Students with excellent performance and strong study habits"
            elif profile['avg_exam_score'] >= 60:
                profile['name'] = "Consistent Performers"
                profile['description'] = "Students with good performance and regular study patterns"
            elif profile['avg_exam_score'] >= 45:
                profile['name'] = "Developing Students"
                profile['description'] = "Students with potential who need targeted support"
            else:
                profile['name'] = "At-Risk Students"
                profile['description'] = "Students requiring immediate intervention and support"
            
            cluster_profiles[cluster_id] = profile
        
        self.student_profiles = cluster_profiles
        
        # Print cluster summary
        for cluster_id, profile in cluster_profiles.items():
            print(f"\n🏷️  Cluster {cluster_id}: {profile['name']}")
            print(f"   📊 Size: {profile['size']} students")
            print(f"   📈 Avg Score: {profile['avg_exam_score']:.1f}%")
            print(f"   📚 Avg Study Hours: {profile['avg_hours_studied']:.1f}")
            print(f"   🎯 Avg Attendance: {profile['avg_attendance']:.1f}%")
            print(f"   📝 {profile['description']}")
    
    def detect_anomalies(self, contamination=0.1):
        """Detect anomalous students using Isolation Forest"""
        print("🚨 Detecting anomalous student patterns...")
        
        # Use Isolation Forest for anomaly detection
        isolation_forest = IsolationForest(
            contamination=contamination, 
            random_state=42
        )
        
        self.anomalies = isolation_forest.fit_predict(self.scaled_features)
        self.df['Is_Anomaly'] = self.anomalies == -1
        
        anomaly_count = sum(self.anomalies == -1)
        print(f"🔍 Detected {anomaly_count} anomalous students ({anomaly_count/len(self.df)*100:.1f}%)")
        
        # Analyze anomalous patterns
        self.analyze_anomalies()
        
        return self.anomalies
    
    def analyze_anomalies(self):
        """Analyze characteristics of anomalous students"""
        anomalous_students = self.df[self.df['Is_Anomaly'] == True]
        normal_students = self.df[self.df['Is_Anomaly'] == False]
        
        print("🔬 Anomaly Analysis:")
        print(f"   📊 Anomalous students: {len(anomalous_students)}")
        print(f"   📈 Avg anomaly score: {anomalous_students['Exam_Score'].mean():.1f}%")
        print(f"   📚 Avg study hours: {anomalous_students['Hours_Studied'].mean():.1f}")
        print(f"   🎯 Avg attendance: {anomalous_students['Attendance'].mean():.1f}%")
        
        # Find common characteristics of anomalies
        print("\n🎯 Common anomaly patterns:")
        for col in ['Motivation_Level', 'Family_Income', 'Access_to_Resources']:
            if col in anomalous_students.columns:
                most_common = anomalous_students[col].mode().iloc[0] if len(anomalous_students[col].mode()) > 0 else "N/A"
                print(f"   • Most common {col}: {most_common}")
    
    def perform_pca_analysis(self, n_components=2):
        """Perform PCA for dimensionality reduction and visualization"""
        print("🔄 Performing PCA analysis...")
        
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(self.scaled_features)
        
        # Add PCA components to dataframe
        for i in range(n_components):
            self.df[f'PCA_Component_{i+1}'] = pca_features[:, i]
        
        # Print explained variance
        print(f"📊 PCA Explained Variance:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"   Component {i+1}: {var:.3f} ({var*100:.1f}%)")
        print(f"   Total: {sum(pca.explained_variance_ratio_):.3f} ({sum(pca.explained_variance_ratio_)*100:.1f}%)")
        
        return pca_features, pca
    
    def generate_student_recommendations(self, student_data):
        """Generate personalized recommendations based on cluster analysis"""
        if self.student_profiles is None:
            return "Please run clustering analysis first."
        
        # Predict cluster for new student
        student_df = pd.DataFrame([student_data])
        
        # Apply same preprocessing
        numerical_features = [
            'Hours_Studied', 'Previous_Scores', 'Attendance', 'Tutoring_Sessions',
            'Sleep_Hours', 'Physical_Activity'
        ]
        
        # Create feature vector (simplified for demo)
        feature_vector = [student_data.get(col, 0) for col in numerical_features]
        
        # Find closest cluster (simplified approach)
        min_distance = float('inf')
        closest_cluster = 0
        
        for cluster_id, profile in self.student_profiles.items():
            cluster_data = self.df[self.df['Student_Cluster'] == cluster_id]
            cluster_center = [
                cluster_data['Hours_Studied'].mean(),
                cluster_data['Previous_Scores'].mean(),
                cluster_data['Attendance'].mean(),
                cluster_data['Tutoring_Sessions'].mean(),
                cluster_data['Sleep_Hours'].mean(),
                cluster_data['Physical_Activity'].mean()
            ]
            
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(feature_vector, cluster_center)))
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster_id
        
        # Generate recommendations based on cluster
        profile = self.student_profiles[closest_cluster]
        
        recommendations = {
            'cluster_name': profile['name'],
            'cluster_description': profile['description'],
            'targeted_advice': self.get_cluster_specific_advice(closest_cluster),
            'success_probability': self.calculate_success_probability(closest_cluster)
        }
        
        return recommendations
    
    def get_cluster_specific_advice(self, cluster_id):
        """Get specific advice based on cluster characteristics"""
        profile = self.student_profiles[cluster_id]
        
        if profile['name'] == "High Achievers":
            return [
                "Maintain your excellent study routine",
                "Consider peer tutoring to help others",
                "Focus on advanced concepts and critical thinking",
                "Prepare for leadership roles in group projects"
            ]
        elif profile['name'] == "Consistent Performers":
            return [
                "Aim to increase study hours slightly for better performance",
                "Focus on understanding weak subjects",
                "Join study groups for collaborative learning",
                "Set specific grade improvement targets"
            ]
        elif profile['name'] == "Developing Students":
            return [
                "Increase study hours gradually",
                "Improve attendance to at least 80%",
                "Seek additional tutoring support",
                "Break down complex topics into smaller parts"
            ]
        else:  # At-Risk Students
            return [
                "Immediate intervention needed - speak with counselor",
                "Start with basic study habit formation",
                "Improve attendance as top priority",
                "Consider one-on-one tutoring",
                "Address any underlying learning difficulties"
            ]
    
    def calculate_success_probability(self, cluster_id):
        """Calculate success probability based on cluster historical data"""
        cluster_data = self.df[self.df['Student_Cluster'] == cluster_id]
        success_grades = ['First Class', 'Second Class Upper', 'Second Class Lower']
        success_count = sum(cluster_data['Grade_Class'].isin(success_grades))
        success_rate = success_count / len(cluster_data) * 100
        
        return f"{success_rate:.1f}%"
    
    def create_visualization_data(self):
        """Create data for visualization in Streamlit"""
        if self.df is None:
            return None
        
        # PCA visualization data
        pca_data = self.df[['PCA_Component_1', 'PCA_Component_2', 'Grade_Class', 'Student_Cluster']].copy()
        
        # Cluster summary data
        cluster_summary = []
        for cluster_id, profile in self.student_profiles.items():
            cluster_summary.append({
                'Cluster': profile['name'],
                'Size': profile['size'],
                'Avg_Score': profile['avg_exam_score'],
                'Success_Rate': float(self.calculate_success_probability(cluster_id).replace('%', ''))
            })
        
        return {
            'pca_data': pca_data,
            'cluster_summary': pd.DataFrame(cluster_summary),
            'anomaly_count': sum(self.df['Is_Anomaly']),
            'total_students': len(self.df)
        }
    
    def run_complete_analysis(self):
        """Run complete unsupervised learning analysis"""
        print("🚀 RUNNING COMPLETE UNSUPERVISED LEARNING ANALYSIS")
        print("=" * 60)
        
        # Step 1: Load and prepare data
        if not self.load_and_prepare_data():
            return False
        
        # Step 2: Perform clustering
        self.perform_student_clustering()
        
        # Step 3: Detect anomalies
        self.detect_anomalies()
        
        # Step 4: PCA analysis
        self.perform_pca_analysis()
        
        print("\n✅ Analysis complete! Ready for integration with Streamlit app.")
        return True
    
    def save_analysis_results(self, output_path="unsupervised_analysis_results.pkl"):
        """Save analysis results for use in main app"""
        results = {
            'student_profiles': self.student_profiles,
            'df_with_clusters': self.df,
            'scaled_features': self.scaled_features,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(results, output_path)
        print(f"💾 Analysis results saved to {output_path}")

# Integration function for your main app
def integrate_unsupervised_learning(student_input):
    """
    Function to integrate with your existing app.py
    Call this from your prediction section
    """
    
    # Load pre-computed analysis results
    try:
        analyzer = UnsupervisedLearningAnalyzer()
        if analyzer.load_and_prepare_data():
            analyzer.perform_student_clustering()
            
            # Generate recommendations for the current student
            recommendations = analyzer.generate_student_recommendations(student_input)
            
            return recommendations
        else:
            return {"error": "Could not load unsupervised learning analysis"}
            
    except Exception as e:
        return {"error": f"Unsupervised learning error: {str(e)}"}

# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = UnsupervisedLearningAnalyzer()
    
    # Run complete analysis
    if analyzer.run_complete_analysis():
        # Save results
        analyzer.save_analysis_results()
        
        # Test with sample student data
        sample_student = {
            'Hours_Studied': 15,
            'Previous_Scores': 75,
            'Attendance': 85,
            'Tutoring_Sessions': 3,
            'Sleep_Hours': 7,
            'Physical_Activity': 3
        }
        
        recommendations = analyzer.generate_student_recommendations(sample_student)
        print("\n🎯 SAMPLE RECOMMENDATIONS:")
        print(f"Cluster: {recommendations['cluster_name']}")
        print(f"Description: {recommendations['cluster_description']}")
        print(f"Success Probability: {recommendations['success_probability']}")
        print("Advice:")
        for advice in recommendations['targeted_advice']:
            print(f"  • {advice}")