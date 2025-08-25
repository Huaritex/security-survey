import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class HomeSurveyAnalyzer:
    def __init__(self, data_path='data/survey_responses.json'):
        """
        Inicializa el analizador de encuestas de seguridad del hogar
        """
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.models = {}
        self.encoders = {}
        
    def load_data(self):
        """
        Carga los datos de la encuesta desde el archivo JSON
        """
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.df = pd.DataFrame(data)
            print(f"Datos cargados exitosamente: {len(self.df)} respuestas")
            return True
        except FileNotFoundError:
            print(f"Archivo no encontrado: {self.data_path}")
            return False
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            return False
    
    def preprocess_data(self):
        """
        Preprocesa los datos para análisis y modelado
        """
        if self.df is None:
            print("Primero debe cargar los datos")
            return False
        
        # Crear copia para procesamiento
        self.processed_df = self.df.copy()
        
        # Convertir arrays de strings a conteos
        array_columns = ['devices', 'threatAwareness', 'securityMeasures']
        for col in array_columns:
            if col in self.processed_df.columns:
                self.processed_df[f'{col}_count'] = self.processed_df[col].apply(
                    lambda x: len(x) if isinstance(x, list) else 0
                )
        
        # Crear variables categóricas ordinales
        ordinal_mappings = {
            'age': {'18-25': 1, '26-35': 2, '36-45': 3, '46+': 4},
            'education': {'secundaria': 1, 'tecnico': 2, 'universitario': 3, 'posgrado': 4},
            'techExperience': {'principiante': 1, 'intermedio': 2, 'avanzado': 3, 'experto': 4},
            'securityKnowledge': {'principiante': 1, 'intermedio': 2, 'avanzado': 3, 'experto': 4},
            'privacyConcern': {'nada-preocupado': 1, 'poco-preocupado': 2, 'algo-preocupado': 3, 'muy-preocupado': 4},
            'trustInTech': {'nada': 1, 'poco': 2, 'algo': 3, 'mucho': 4}
        }
        
        for col, mapping in ordinal_mappings.items():
            if col in self.processed_df.columns:
                self.processed_df[f'{col}_ordinal'] = self.processed_df[col].map(mapping)
        
        # Crear score de seguridad
        self.processed_df['security_score'] = self._calculate_security_score()
        
        # Crear categorías de riesgo
        self.processed_df['risk_category'] = pd.cut(
            self.processed_df['security_score'], 
            bins=[0, 30, 60, 100], 
            labels=['Alto Riesgo', 'Riesgo Medio', 'Bajo Riesgo']
        )
        
        print("Datos preprocesados exitosamente")
        return True
    
    def _calculate_security_score(self):
        """
        Calcula un score de seguridad basado en las respuestas
        """
        score = np.zeros(len(self.processed_df))
        
        # Puntuación por uso de gestor de contraseñas
        password_manager_scores = {'si-siempre': 20, 'a-veces': 10, 'no': 0, 'no-se-que-es': 0}
        score += self.processed_df['passwordManager'].map(password_manager_scores).fillna(0)
        
        # Puntuación por 2FA
        twofa_scores = {'si-siempre': 20, 'algunas': 10, 'no': 0, 'no-se-que-es': 0}
        score += self.processed_df['twoFactor'].map(twofa_scores).fillna(0)
        
        # Puntuación por actualizaciones
        update_scores = {'inmediatamente': 15, 'semanalmente': 10, 'mensualmente': 5, 'raramente': 0}
        score += self.processed_df['softwareUpdates'].map(update_scores).fillna(0)
        
        # Puntuación por WiFi público
        wifi_scores = {'nunca': 15, 'con-vpn': 15, 'navegacion-basica': 5, 'todo': 0}
        score += self.processed_df['publicWifi'].map(wifi_scores).fillna(0)
        
        # Puntuación por contraseña WiFi
        wifi_pass_scores = {'compleja': 10, 'simple': 5, 'predeterminada': 0, 'no-se': 0}
        score += self.processed_df['wifiPassword'].map(wifi_pass_scores).fillna(0)
        
        # Puntuación por medidas de seguridad implementadas
        score += self.processed_df['securityMeasures_count'] * 2
        
        # Puntuación por conocimiento de amenazas
        score += self.processed_df['threatAwareness_count'] * 1.5
        
        return np.clip(score, 0, 100)
    
    def analyze_privacy_paradox(self):
        """
        Analiza la paradoja de la privacidad en los datos
        """
        if self.processed_df is None:
            print("Primero debe preprocesar los datos")
            return
        
        # Crear índice de preocupación por privacidad
        privacy_concern_map = {'nada-preocupado': 1, 'poco-preocupado': 2, 'algo-preocupado': 3, 'muy-preocupado': 4}
        self.processed_df['privacy_concern_score'] = self.processed_df['privacyConcern'].map(privacy_concern_map)
        
        # Crear índice de comportamiento de compartir datos
        data_sharing_map = {'nunca': 1, 'raramente': 2, 'a-veces': 3, 'frecuentemente': 4}
        self.processed_df['data_sharing_score'] = self.processed_df['dataSharing'].map(data_sharing_map)
        
        # Identificar paradoja: alta preocupación pero alto compartir datos
        paradox_condition = (
            (self.processed_df['privacy_concern_score'] >= 3) & 
            (self.processed_df['data_sharing_score'] >= 3)
        )
        
        self.processed_df['privacy_paradox'] = paradox_condition
        
        # Estadísticas de la paradoja
        paradox_percentage = (self.processed_df['privacy_paradox'].sum() / len(self.processed_df)) * 100
        
        print(f"\n=== ANÁLISIS DE LA PARADOJA DE LA PRIVACIDAD ===")
        print(f"Porcentaje de usuarios con paradoja de privacidad: {paradox_percentage:.1f}%")
        
        # Análisis por grupos demográficos
        print(f"\nParadoja por grupo de edad:")
        age_paradox = self.processed_df.groupby('age')['privacy_paradox'].mean() * 100
        for age, percentage in age_paradox.items():
            print(f"  {age}: {percentage:.1f}%")
        
        print(f"\nParadoja por nivel educativo:")
        edu_paradox = self.processed_df.groupby('education')['privacy_paradox'].mean() * 100
        for edu, percentage in edu_paradox.items():
            print(f"  {edu}: {percentage:.1f}%")
        
        return paradox_percentage
    
    def train_risk_prediction_model(self):
        """
        Entrena un modelo para predecir el nivel de riesgo de seguridad
        """
        if self.processed_df is None:
            print("Primero debe preprocesar los datos")
            return False
        
        # Seleccionar características para el modelo
        feature_columns = [
            'age_ordinal', 'education_ordinal', 'techExperience_ordinal',
            'securityKnowledge_ordinal', 'devices_count', 'threatAwareness_count',
            'securityMeasures_count', 'privacy_concern_score', 'data_sharing_score'
        ]
        
        # Filtrar columnas que existen
        available_features = [col for col in feature_columns if col in self.processed_df.columns]
        
        if len(available_features) < 3:
            print("No hay suficientes características para entrenar el modelo")
            return False
        
        # Preparar datos
        X = self.processed_df[available_features].fillna(0)
        y = self.processed_df['risk_category'].fillna('Riesgo Medio')
        
        # Codificar etiquetas
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.encoders['risk_category'] = le
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.encoders['scaler'] = scaler
        
        # Entrenar Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Entrenar Regresión Logística
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        
        # Guardar modelos
        self.models['random_forest'] = rf_model
        self.models['logistic_regression'] = lr_model
        self.models['features'] = available_features
        
        # Evaluar modelos
        rf_pred = rf_model.predict(X_test_scaled)
        lr_pred = lr_model.predict(X_test_scaled)
        
        print(f"\n=== EVALUACIÓN DE MODELOS ===")
        print(f"\nRandom Forest:")
        print(classification_report(y_test, rf_pred, target_names=le.classes_))
        
        print(f"\nRegresión Logística:")
        print(classification_report(y_test, lr_pred, target_names=le.classes_))
        
        # Importancia de características (Random Forest)
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nImportancia de características:")
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return True
    
    def predict_risk(self, user_data):
        """
        Predice el riesgo de seguridad para un usuario específico
        """
        if 'random_forest' not in self.models:
            print("Primero debe entrenar el modelo")
            return None
        
        # Preparar datos del usuario
        features = []
        for feature in self.models['features']:
            if feature in user_data:
                features.append(user_data[feature])
            else:
                features.append(0)  # Valor por defecto
        
        # Escalar características
        features_scaled = self.encoders['scaler'].transform([features])
        
        # Hacer predicción
        prediction = self.models['random_forest'].predict(features_scaled)[0]
        probability = self.models['random_forest'].predict_proba(features_scaled)[0]
        
        # Decodificar resultado
        risk_category = self.encoders['risk_category'].inverse_transform([prediction])[0]
        
        return {
            'risk_category': risk_category,
            'probabilities': dict(zip(self.encoders['risk_category'].classes_, probability))
        }
    
    def generate_insights(self):
        """
        Genera insights y recomendaciones basadas en el análisis
        """
        if self.processed_df is None:
            print("Primero debe preprocesar los datos")
            return
        
        print(f"\n=== INSIGHTS Y RECOMENDACIONES ===")
        
        # Estadísticas generales
        total_responses = len(self.processed_df)
        avg_security_score = self.processed_df['security_score'].mean()
        
        print(f"Total de respuestas: {total_responses}")
        print(f"Score promedio de seguridad: {avg_security_score:.1f}/100")
        
        # Distribución de riesgo
        risk_distribution = self.processed_df['risk_category'].value_counts(normalize=True) * 100
        print(f"\nDistribución de riesgo:")
        for category, percentage in risk_distribution.items():
            print(f"  {category}: {percentage:.1f}%")
        
        # Correlaciones importantes
        numeric_cols = ['security_score', 'privacy_concern_score', 'data_sharing_score', 
                       'devices_count', 'threatAwareness_count', 'securityMeasures_count']
        available_numeric = [col for col in numeric_cols if col in self.processed_df.columns]
        
        if len(available_numeric) > 1:
            correlations = self.processed_df[available_numeric].corr()['security_score'].sort_values(ascending=False)
            print(f"\nCorrelaciones con score de seguridad:")
            for var, corr in correlations.items():
                if var != 'security_score':
                    print(f"  {var}: {corr:.3f}")
        
        # Recomendaciones
        print(f"\n=== RECOMENDACIONES ===")
        
        # Usuarios de alto riesgo
        high_risk_count = (self.processed_df['risk_category'] == 'Alto Riesgo').sum()
        if high_risk_count > 0:
            high_risk_percentage = (high_risk_count / total_responses) * 100
            print(f"• {high_risk_percentage:.1f}% de usuarios están en alto riesgo")
            print(f"  - Implementar programas de educación en seguridad")
            print(f"  - Promover el uso de gestores de contraseñas")
            print(f"  - Fomentar la activación de 2FA")
        
        # Paradoja de privacidad
        if 'privacy_paradox' in self.processed_df.columns:
            paradox_count = self.processed_df['privacy_paradox'].sum()
            if paradox_count > 0:
                paradox_percentage = (paradox_count / total_responses) * 100
                print(f"• {paradox_percentage:.1f}% muestran paradoja de privacidad")
                print(f"  - Desarrollar interfaces que faciliten decisiones de privacidad")
                print(f"  - Crear herramientas de transparencia de datos")
        
        return True
    
    def save_analysis_report(self, filename='analysis_report.txt'):
        """
        Guarda un reporte completo del análisis
        """
        if self.processed_df is None:
            print("Primero debe preprocesar los datos")
            return False
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE ANÁLISIS - DIAGNÓSTICO DE SEGURIDAD DEL HOGAR\n")
            f.write("=" * 60 + "\n\n")
            
            # Estadísticas básicas
            f.write(f"Total de respuestas: {len(self.processed_df)}\n")
            f.write(f"Fecha de análisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Distribución demográfica
            f.write("DISTRIBUCIÓN DEMOGRÁFICA\n")
            f.write("-" * 25 + "\n")
            for col in ['age', 'education', 'techExperience']:
                if col in self.processed_df.columns:
                    f.write(f"\n{col.upper()}:\n")
                    distribution = self.processed_df[col].value_counts(normalize=True) * 100
                    for value, percentage in distribution.items():
                        f.write(f"  {value}: {percentage:.1f}%\n")
            
            # Estadísticas de seguridad
            f.write(f"\nESTADÍSTICAS DE SEGURIDAD\n")
            f.write("-" * 25 + "\n")
            f.write(f"Score promedio de seguridad: {self.processed_df['security_score'].mean():.1f}/100\n")
            
            if 'risk_category' in self.processed_df.columns:
                f.write(f"\nDistribución de riesgo:\n")
                risk_dist = self.processed_df['risk_category'].value_counts(normalize=True) * 100
                for category, percentage in risk_dist.items():
                    f.write(f"  {category}: {percentage:.1f}%\n")
        
        print(f"Reporte guardado en: {filename}")
        return True

# Función principal para ejecutar el análisis
def main():
    """
    Función principal que ejecuta todo el pipeline de análisis
    """
    print("Iniciando análisis de encuesta de seguridad del hogar...")
    
    # Crear analizador
    analyzer = HomeSurveyAnalyzer()
    
    # Cargar y procesar datos
    if not analyzer.load_data():
        print("No se pudieron cargar los datos. Asegúrate de que existan respuestas en la encuesta.")
        return
    
    if not analyzer.preprocess_data():
        print("Error en el preprocesamiento de datos")
        return
    
    # Realizar análisis
    analyzer.analyze_privacy_paradox()
    analyzer.train_risk_prediction_model()
    analyzer.generate_insights()
    analyzer.save_analysis_report()
    
    print("\n¡Análisis completado exitosamente!")
    
    # Ejemplo de predicción para un usuario
    print("\n=== EJEMPLO DE PREDICCIÓN ===")
    example_user = {
        'age_ordinal': 2,  # 26-35 años
        'education_ordinal': 3,  # universitario
        'techExperience_ordinal': 2,  # intermedio
        'securityKnowledge_ordinal': 2,  # intermedio
        'devices_count': 5,
        'threatAwareness_count': 3,
        'securityMeasures_count': 2,
        'privacy_concern_score': 3,  # algo preocupado
        'data_sharing_score': 2  # raramente
    }
    
    prediction = analyzer.predict_risk(example_user)
    if prediction:
        print(f"Categoría de riesgo predicha: {prediction['risk_category']}")
        print("Probabilidades:")
        for category, prob in prediction['probabilities'].items():
            print(f"  {category}: {prob:.3f}")

if __name__ == "__main__":
    main()
