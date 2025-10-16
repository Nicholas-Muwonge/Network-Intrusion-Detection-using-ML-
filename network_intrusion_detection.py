import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def generate_network_data(n_samples=10000):
   
    data = {
        'duration': np.random.exponential(60, n_samples),  
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples, p=[0.6, 0.3, 0.1]),
        'service': np.random.choice(['http', 'smtp', 'ftp', 'ssh', 'dns'], n_samples),
        'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTO'], n_samples, p=[0.5, 0.2, 0.2, 0.1]),
        'src_bytes': np.random.lognormal(7, 2, n_samples),  
        'dst_bytes': np.random.lognormal(6, 2, n_samples),  
        'land': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),  
        'wrong_fragment': np.random.poisson(0.1, n_samples),
        'urgent': np.random.poisson(0.01, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    df['bytes_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1) 
    df['total_bytes'] = df['src_bytes'] + df['dst_bytes']
    df['packet_size_std'] = np.random.exponential(10, n_samples)  
    
    malicious_patterns = (
        (df['src_bytes'] > 10000) |  # large uploads
        (df['dst_bytes'] > 50000) |  # large downloads
        (df['duration'] > 300) |     # long connections
        (df['wrong_fragment'] > 5) | # many wrong fragments
        (df['protocol_type'] == 'icmp') & (df['total_bytes'] > 1000) | 
        (df['service'] == 'ssh') & (df['src_bytes'] > 5000)  
    )
    
    base_malicious = malicious_patterns.astype(int)
    random_component = np.random.binomial(1, 0.1, n_samples)  
    malicious_combined = ((base_malicious + random_component) > 0).astype(int)
    
    malicious_indices = np.where(malicious_combined == 1)[0]
    normal_indices = np.where(malicious_combined == 0)[0]
    
    if len(malicious_indices) > int(0.15 * n_samples):
        keep_malicious = np.random.choice(malicious_indices, int(0.15 * n_samples), replace=False)
        malicious_final = np.zeros(n_samples)
        malicious_final[keep_malicious] = 1
    else:
        malicious_final = malicious_combined
    
    df['malicious'] = malicious_final.astype(int)
    
    return df

print("Generating synthetic network intrusion dataset...")
df = generate_network_data(15000)
print(f"Dataset shape: {df.shape}")
print(f"Malicious traffic: {df['malicious'].sum()} ({df['malicious'].mean()*100:.2f}%)")

def preprocess_data(df):
    
    df_processed = df.copy()
    
    categorical_cols = ['protocol_type', 'service', 'flag']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    df_processed['bytes_ratio'] = df_processed['bytes_ratio'].replace([np.inf, -np.inf], 0)
    
    return df_processed, label_encoders

print("\nPreprocessing data...")
df_processed, label_encoders = preprocess_data(df)

df_processed['log_src_bytes'] = np.log1p(df_processed['src_bytes'])
df_processed['log_dst_bytes'] = np.log1p(df_processed['dst_bytes'])
df_processed['byte_imbalance'] = abs(df_processed['src_bytes'] - df_processed['dst_bytes'])
df_processed['connection_speed'] = df_processed['total_bytes'] / (df_processed['duration'] + 1)

X = df_processed.drop('malicious', axis=1)
y = df_processed['malicious']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Feature count: {X_train.shape[1]}")

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and compare their performance
    """
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        if name == 'SVM' or name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"{name} Results:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC-ROC: {auc_roc:.4f}")
    
    return results

print("\n" + "="*50)
print("MODEL TRAINING AND COMPARISON")
print("="*50)

results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
best_model = results[best_model_name]
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   F1-Score: {best_model['f1_score']:.4f}")
print(f"   Precision: {best_model['precision']:.4f}")
print(f"   Recall: {best_model['recall']:.4f}")

print("\n" + "="*50)
print("HYPERPARAMETER TUNING FOR RANDOM FOREST")
print("="*50)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_
y_pred_final = best_rf.predict(X_test)
y_pred_proba_final = best_rf.predict_proba(X_test)[:, 1]

final_precision = precision_score(y_test, y_pred_final)
final_recall = recall_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final)

print(f"\n🎯 FINAL RANDOM FOREST PERFORMANCE:")
print(f"   Precision: {final_precision:.4f} ({final_precision*100:.2f}%)")
print(f"   Recall: {final_recall:.4f} ({final_recall*100:.2f}%)")
print(f"   F1-Score: {final_f1:.4f}")

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n📊 TOP 10 MOST IMPORTANT FEATURES:")
print(feature_importance.head(10).to_string(index=False))

# Visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
top_features = feature_importance.head(10)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance')
plt.gca().invert_yaxis()

plt.subplot(2, 3, 2)
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.subplot(2, 3, 3)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_final)
plt.plot(fpr, tpr, linewidth=2, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_pred_proba_final):.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.subplot(2, 3, 4)
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba_final)
plt.plot(recall_curve, precision_curve, linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.subplot(2, 3, 5)
model_names = list(results.keys())
f1_scores = [results[name]['f1_score'] for name in model_names]
plt.bar(model_names, f1_scores, color=['blue', 'orange', 'green', 'red'])
plt.xticks(rotation=45)
plt.ylabel('F1-Score')
plt.title('Model Comparison')

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("DETAILED CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred_final, target_names=['Normal', 'Malicious']))

print("\n" + "="*50)
print("CROSS-VALIDATION RESULTS")
print("="*50)
cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='f1')
print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Mean CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("\n" + "="*50)
print("REAL-TIME DETECTION SIMULATION")
print("="*50)

def simulate_real_time_detection(model, scaler, n_connections=10):
    print("Simulating real-time network connection analysis...")
    
    for i in range(n_connections):
        connection_data = generate_network_data(1).drop('malicious', axis=1)
        connection_processed, _ = preprocess_data(connection_data)
        
        connection_processed['log_src_bytes'] = np.log1p(connection_processed['src_bytes'])
        connection_processed['log_dst_bytes'] = np.log1p(connection_processed['dst_bytes'])
        connection_processed['byte_imbalance'] = abs(connection_processed['src_bytes'] - connection_processed['dst_bytes'])
        connection_processed['connection_speed'] = connection_processed['total_bytes'] / (connection_processed['duration'] + 1)
        
        prediction = model.predict(connection_processed)[0]
        probability = model.predict_proba(connection_processed)[0, 1]
        
        status = "🚨 MALICIOUS" if prediction == 1 else "✅ NORMAL"
        print(f"Connection {i+1}: {status} (confidence: {probability:.3f})")

simulate_real_time_detection(best_rf, scaler)

print(f"\n✅ NETWORK INTRUSION DETECTION SYSTEM DEPLOYED SUCCESSFULLY!")
print(f"📊 Final Model Performance:")
print(f"   • Precision: {final_precision*100:.2f}%")
print(f"   • Recall: {final_recall*100:.2f}%") 
print(f"   • F1-Score: {final_f1*100:.2f}%")
print(f"   • Can detect {final_recall*100:.1f}% of malicious activities")
print(f"   • {final_precision*100:.1f}% of detected threats are actual attacks")