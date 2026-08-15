"""
Data Preparation Module
Limpa, processa e prepara dados para o modelo
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os
import json

# Nome do arquivo exatamente como vem no ZIP do Kaggle
# (henriqueyamahata/bank-marketing contém bank-additional-full.csv e
# bank-additional-names.txt), para que o download descrito no README funcione
# sem nenhum passo manual de renomear.
RAW_DATA_PATH = 'data/raw/bank-additional-full.csv'

class DataPreparation:
    def __init__(self, data_path=RAW_DATA_PATH):
        self.data_path = data_path
        self.df = None
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.scaler = None
        self.encoders = {}
        
    def load_data(self):
        """Carregar dados"""
        print("📥 Carregando dados...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"{self.data_path} não encontrado. Baixe a base do Kaggle antes de treinar:\n"
                "  kaggle datasets download -d henriqueyamahata/bank-marketing -p data/raw/\n"
                "  unzip -o data/raw/bank-marketing.zip -d data/raw/"
            )
        self.df = pd.read_csv(self.data_path, sep=';')
        print(f"   ✅ {len(self.df):,} registros carregados")
        return self
    
    def remove_temporal_leakage(self):
        """Remover coluna 'duration' que é vazamento temporal"""
        print("🗑️ Removendo vazamento temporal...")
        if 'duration' in self.df.columns:
            self.df = self.df.drop('duration', axis=1)
            print("   ✅ Coluna 'duration' removida")
        return self
    
    def clean_data(self):
        """Limpeza básica de dados"""
        print("🧹 Limpando dados...")
        
        # Verificar valores faltantes
        missing = self.df.isnull().sum().sum()
        if missing > 0:
            print(f"   ⚠️ Encontrados {missing} valores faltantes")
            self.df = self.df.dropna()
            print(f"   ✅ Removidas linhas com NaN")
        else:
            print("   ✅ Nenhum valor faltante")
        
        return self
    
    def encode_categorical(self):
        """Codificar variáveis categóricas"""
        print("🔤 Codificando variáveis categóricas...")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Separar target
        if 'y' in categorical_cols:
            categorical_cols.remove('y')
        
        print(f"   Encontradas {len(categorical_cols)} colunas categóricas")
        
        # Codificar cada uma
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.encoders[col] = le
            print(f"   ✅ {col}: {len(le.classes_)} classes codificadas")
        
        # Codificar target
        self.df['y'] = (self.df['y'] == 'yes').astype(int)
        print(f"   ✅ Target 'y' convertido para binário (0/1)")
        
        return self
    
    def split_train_test(self, test_size=0.3, random_state=42):
        """Separar em treino/teste com stratified split"""
        print(f"\n📊 Separando em treino/teste ({100-int(test_size*100)}/{int(test_size*100)})...")
        
        # Preparar X e y
        X = self.df.drop('y', axis=1)
        y = self.df['y']
        
        # Stratified split para manter proporção de classes
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"   Treino: {len(self.train_X):,} registros ({self.train_y.mean()*100:.1f}% Yes)")
        print(f"   Teste:  {len(self.test_X):,} registros ({self.test_y.mean()*100:.1f}% Yes)")
        
        return self
    
    def normalize_features(self):
        """Normalizar features numéricas"""
        print("\n📈 Normalizando features...")
        
        self.scaler = StandardScaler()
        
        # Ajustar no treino
        self.train_X_scaled = self.scaler.fit_transform(self.train_X)
        self.train_X_scaled = pd.DataFrame(self.train_X_scaled, columns=self.train_X.columns)
        
        # Transformar teste
        self.test_X_scaled = self.scaler.transform(self.test_X)
        self.test_X_scaled = pd.DataFrame(self.test_X_scaled, columns=self.test_X.columns)
        
        print(f"   ✅ Features normalizadas com StandardScaler")
        print(f"   Treino shape: {self.train_X_scaled.shape}")
        print(f"   Teste shape:  {self.test_X_scaled.shape}")
        
        return self
    
    def save_datasets(self, output_dir='data/processed'):
        """Salvar datasets processados e encoders"""
        print(f"\n💾 Salvando datasets em {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Salvar em CSV
        train_data = self.train_X_scaled.copy()
        train_data['y'] = self.train_y.values
        train_data.to_csv(f'{output_dir}/train_clean.csv', index=False)
        print(f"   ✅ {output_dir}/train_clean.csv ({len(train_data):,} x {train_data.shape[1]})")
        
        test_data = self.test_X_scaled.copy()
        test_data['y'] = self.test_y.values
        test_data.to_csv(f'{output_dir}/test_clean.csv', index=False)
        print(f"   ✅ {output_dir}/test_clean.csv ({len(test_data):,} x {test_data.shape[1]})")
        
        # Salvar scaler
        with open(f'{output_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"   ✅ {output_dir}/scaler.pkl")

        # Salvar encoders como JSON (classes por coluna)
        encoders_simple = {col: le.classes_.tolist() for col, le in self.encoders.items()}
        with open(f'{output_dir}/encoders.json', 'w') as f:
            json.dump(encoders_simple, f, ensure_ascii=False, indent=2)
        print(f"   ✅ {output_dir}/encoders.json")

        # Salvar lista de features esperadas (após encoding e normalização)
        feature_names = self.get_feature_names()
        with open(f'{output_dir}/feature_names.json', 'w') as f:
            json.dump(feature_names, f, ensure_ascii=False)
        print(f"   ✅ {output_dir}/feature_names.json ({len(feature_names)} features)")
        
        return self
    
    def get_feature_names(self):
        """Retornar nomes das features"""
        return self.train_X_scaled.columns.tolist()
    
    def get_metadata(self):
        """Retornar metadados do dataset"""
        return {
            'n_features': self.train_X_scaled.shape[1],
            'n_train': len(self.train_X_scaled),
            'n_test': len(self.test_X_scaled),
            'train_yes_rate': self.train_y.mean(),
            'test_yes_rate': self.test_y.mean(),
            'feature_names': self.get_feature_names()
        }

def prepare_data(data_path=RAW_DATA_PATH, test_size=0.3):
    """Pipeline completo de preparação"""
    prep = DataPreparation(data_path)
    prep.load_data() \
        .remove_temporal_leakage() \
        .clean_data() \
        .encode_categorical() \
        .split_train_test(test_size=test_size) \
        .normalize_features() \
        .save_datasets()
    
    print("\n" + "="*70)
    print("✅ PREPARAÇÃO CONCLUÍDA")
    print("="*70)
    print(f"\nMetadados:")
    for key, value in prep.get_metadata().items():
        print(f"  {key}: {value}")
    
    return prep

if __name__ == '__main__':
    prep = prepare_data()
