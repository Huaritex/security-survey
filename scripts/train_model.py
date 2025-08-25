import pandas as pd
import numpy as np
import json
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class SecurityRiskPredictor:
    """
    Modelo avanzado para predecir riesgos de seguridad basado en respuestas de encuesta
    """
    
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
    def load_and_prepare_data(self, data_path='data/survey_responses.json'):
        """
        Carga y prepara los datos para entrenamiento
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            print(f"Datos cargados: {len(df)} respuestas")
            
            # Crear características numéricas
            features_df = self._create_features(df)
            
            # Crear variable objetivo (score de seguridad categorizado)
            target = self._create_target_variable(df)
            
            return features_df, target
            
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            return None, None
    
    def _create_features(self, df):
        """
        Crea características numéricas a partir de los datos de la encuesta
        """
        features = pd.DataFrame()
        
        # Características demográficas (ordinales)
        age_mapping = {'18-25': 1, '26-35': 2, '36-45': 3, '46+': 4}
        features['age_score'] = df['age'].map(age_mapping).fillna(2)
        
        education_mapping = {'secundaria': 1, 'tecnico': 2, 'universitario': 3, 'posgrado': 4}
        features['education_score'] = df['education'].map(education_mapping).fillna(2)
        
        tech_exp_mapping = {'principiante': 1, 'intermedio': 2, 'avanzado': 3, 'experto': 4}
        features['tech_experience_score'] = df['techExperience'].map(tech_exp_mapping).fillna(2)
        
        # Características de dispositivos
        features['device_count'] = df['devices'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Características de seguridad de red
        router_age_mapping = {'menos-1': 4, '1-3': 3, '3-5': 2, 'mas-5': 1}
        features['router_security_score'] = df['routerAge'].map(router_age_mapping).fillna(2)
        
        wifi_password_mapping = {'compleja': 4, 'simple': 2, 'predeterminada': 1, 'no-se': 1}
        features['wifi_password_score'] = df['wifiPassword'].map(wifi_password_mapping).fillna(2)
        
        # Características de comportamiento de seguridad
        password_manager_mapping = {'si-siempre': 4, 'a-veces': 2, 'no': 1, 'no-se-que-es': 1}
        features['password_manager_score'] = df['passwordManager'].map(password_manager_mapping).fillna(1)
        
        twofa_mapping = {'si-siempre': 4, 'algunas': 2, 'no': 1, 'no-se-que-es': 1}
        features['twofa_score'] = df['twoFactor'].map(twofa_mapping).fillna(1)
        
        updates_mapping = {'inmediatamente': 4, 'semanalmente': 3, 'mensualmente': 2, 'raramente': 1}
        features['updates_score'] = df['softwareUpdates'].map(updates_mapping).fillna(2)
        
        public_wifi_mapping = {'nunca': 4, 'con-vpn': 4, 'navegacion-basica': 2, 'todo': 1}
        features['public_wifi_score'] = df['publicWifi'].map(public_wifi_mapping).fillna(2)
        
        # Características de privacidad
        privacy_concern_mapping = {'muy-preocupado': 4, 'algo-preocupado': 3, 'poco-preocupado': 2, 'nada-preocupado': 1}
        features['privacy_concern_score'] = df['privacyConcern'].map(privacy_concern_mapping).fillna(2)
        
        data_sharing_mapping = {'nunca': 4, 'raramente': 3, 'a-veces': 2, 'frecuentemente': 1}
        features['data_sharing_score'] = df['dataSharing'].map(data_sharing_mapping).fillna(2)
        
        trust_mapping = {'nada': 4, 'poco': 3, 'algo': 2, 'mucho': 1}
        features['trust_score'] = df['trustInTech'].map(trust_mapping).fillna(2)
        
        # Características de conocimiento
        security_knowledge_mapping = {'experto': 4, 'avanzado': 3, 'intermedio': 2, 'principiante': 1}
        features['security_knowledge_score'] = df['securityKnowledge'].map(security_knowledge_mapping).fillna(2)
        
        features['threat_awareness_count'] = df['threatAwareness'].apply(
            lambda x: len([t for t in x if t != 'Ninguna']) if isinstance(x, list) else 0
        )
        
        features['security_measures_count'] = df['securityMeasures'].apply(
            lambda x: len([m for m in x if m != 'Ninguna']) if isinstance(x, list) else 0
        )
        
        # Características derivadas
        features['security_behavior_score'] = (
            features['password_manager_score'] + 
            features['twofa_score'] + 
            features['updates_score'] + 
            features['public_wifi_score']
        ) / 4
        
        features['privacy_awareness_score'] = (
            features['privacy_concern_score'] + 
            features['data_sharing_score'] + 
            features['trust_score']
        ) / 3
        
        features['technical_competence_score'] = (
            features['tech_experience_score'] + 
            features['security_knowledge_score'] + 
            features['threat_awareness_count'] / 7 * 4  # Normalizar a escala 1-4
        ) / 3
        
        # Paradoja de privacidad (alta preocupación, bajo comportamiento seguro)
        features['privacy_paradox_score'] = np.where(
            (features['privacy_concern_score'] >= 3) & (features['security_behavior_score'] <= 2),
            1, 0
        )
        
        self.feature_names = features.columns.tolist()
        return features
    
    def _create_target_variable(self, df):
        """
        Crea la variable objetivo basada en un score de seguridad compuesto
        """
        # Calcular score de seguridad (0-100)
        security_score = np.zeros(len(df))
        
        # Componentes del score
        password_manager_scores = {'si-siempre': 20, 'a-veces': 10, 'no': 0, 'no-se-que-es': 0}
        security_score += df['passwordManager'].map(password_manager_scores).fillna(0)
        
        twofa_scores = {'si-siempre': 20, 'algunas': 10, 'no': 0, 'no-se-que-es': 0}
        security_score += df['twoFactor'].map(twofa_scores).fillna(0)
        
        update_scores = {'inmediatamente': 15, 'semanalmente': 10, 'mensualmente': 5, 'raramente': 0}
        security_score += df['softwareUpdates'].map(update_scores).fillna(0)
        
        wifi_scores = {'nunca': 15, 'con-vpn': 15, 'navegacion-basica': 5, 'todo': 0}
        security_score += df['publicWifi'].map(wifi_scores).fillna(0)
        
        wifi_pass_scores = {'compleja': 10, 'simple': 5, 'predeterminada': 0, 'no-se': 0}
        security_score += df['wifiPassword'].map(wifi_pass_scores).fillna(0)
        
        # Medidas de seguridad implementadas
        security_measures_count = df['securityMeasures'].apply(
            lambda x: len([m for m in x if m != 'Ninguna']) if isinstance(x, list) else 0
        )
        security_score += security_measures_count * 2
        
        # Conocimiento de amenazas
        threat_awareness_count = df['threatAwareness'].apply(
            lambda x: len([t for t in x if t != 'Ninguna']) if isinstance(x, list) else 0
        )
        security_score += threat_awareness_count * 1.5
        
        # Categorizar en niveles de riesgo
        risk_categories = pd.cut(
            security_score, 
            bins=[0, 30, 60, 100], 
            labels=['Alto Riesgo', 'Riesgo Medio', 'Bajo Riesgo']
        )
        
        return risk_categories
    
    def train_models(self, X, y):
        """
        Entrena múltiples modelos y selecciona el mejor
        """
        # Codificar etiquetas
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        self.encoders['target'] = le
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Definir modelos a entrenar
        models_to_train = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        print("Entrenando modelos...")
        
        for name, model in models_to_train.items():
            # Entrenar modelo
            model.fit(X_train_scaled, y_train)
            
            # Validación cruzada
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            mean_cv_score = cv_scores.mean()
            
            # Evaluación en conjunto de prueba
            test_score = model.score(X_test_scaled, y_test)
            
            print(f"{name}:")
            print(f"  CV Score: {mean_cv_score:.3f} (+/- {cv_scores.std() * 2:.3f})")
            print(f"  Test Score: {test_score:.3f}")
            
            # Guardar modelo
            self.models[name] = model
            
            # Actualizar mejor modelo
            if mean_cv_score > best_score:
                best_score = mean_cv_score
                best_model = model
                best_name = name
        
        # Guardar mejor modelo
        self.models['best'] = best_model
        self.models['best_name'] = best_name
        
        print(f"\nMejor modelo: {best_name} (CV Score: {best_score:.3f})")
        
        # Reporte detallado del mejor modelo
        y_pred = best_model.predict(X_test_scaled)
        print(f"\nReporte de clasificación ({best_name}):")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        
        # Importancia de características (si está disponible)
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 características más importantes:")
            for _, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
        
        self.is_trained = True
        return True
    
    def predict_risk(self, user_features):
        """
        Predice el riesgo para un usuario específico
        """
        if not self.is_trained:
            print("El modelo no ha sido entrenado")
            return None
        
        # Preparar características
        if isinstance(user_features, dict):
            # Convertir diccionario a DataFrame
            features_df = pd.DataFrame([user_features])
            # Asegurar que todas las características estén presentes
            for feature in self.feature_names:
                if feature not in features_df.columns:
                    features_df[feature] = 0
            features_df = features_df[self.feature_names]
        else:
            features_df = user_features
        
        # Escalar características
        features_scaled = self.scaler.transform(features_df)
        
        # Hacer predicción
        best_model = self.models['best']
        prediction = best_model.predict(features_scaled)[0]
        probabilities = best_model.predict_proba(features_scaled)[0]
        
        # Decodificar resultado
        risk_category = self.encoders['target'].inverse_transform([prediction])[0]
        
        return {
            'risk_category': risk_category,
            'probabilities': dict(zip(self.encoders['target'].classes_, probabilities)),
            'confidence': max(probabilities)
        }
    
    def save_model(self, filepath='models/security_risk_model.pkl'):
        """
        Guarda el modelo entrenado
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'models': self.models,
            'encoders': self.encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modelo guardado en: {filepath}")
    
    def load_model(self, filepath='models/security_risk_model.pkl'):
        """
        Carga un modelo previamente entrenado
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.encoders = model_data['encoders']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            
            print(f"Modelo cargado desde: {filepath}")
            return True
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            return False

def main():
    """
    Función principal para entrenar el modelo
    """
    print("Iniciando entrenamiento del modelo de predicción de riesgo...")
    
    # Crear predictor
    predictor = SecurityRiskPredictor()
    
    # Cargar y preparar datos
    X, y = predictor.load_and_prepare_data()
    
    if X is None or y is None:
        print("No se pudieron cargar los datos")
        return
    
    print(f"Características creadas: {X.shape[1]}")
    print(f"Distribución de clases:")
    print(y.value_counts())
    
    # Entrenar modelos
    if predictor.train_models(X, y):
        # Guardar modelo
        predictor.save_model()
        
        # Ejemplo de predicción
        print("\n=== EJEMPLO DE PREDICCIÓN ===")
        example_features = {
            'age_score': 2,
            'education_score': 3,
            'tech_experience_score': 2,
            'device_count': 5,
            'router_security_score': 3,
            'wifi_password_score': 3,
            'password_manager_score': 2,
            'twofa_score': 1,
            'updates_score': 3,
            'public_wifi_score': 2,
            'privacy_concern_score': 3,
            'data_sharing_score': 2,
            'trust_score': 2,
            'security_knowledge_score': 2,
            'threat_awareness_count': 3,
            'security_measures_count': 2,
            'security_behavior_score': 2.0,
            'privacy_awareness_score': 2.3,
            'technical_competence_score': 2.4,
            'privacy_paradox_score': 0
        }
        
        prediction = predictor.predict_risk(example_features)
        if prediction:
            print(f"Categoría de riesgo: {prediction['risk_category']}")
            print(f"Confianza: {prediction['confidence']:.3f}")
            print("Probabilidades por categoría:")
            for category, prob in prediction['probabilities'].items():
                print(f"  {category}: {prob:.3f}")
    
    print("\n¡Entrenamiento completado!")

if __name__ == "__main__":
    main()
