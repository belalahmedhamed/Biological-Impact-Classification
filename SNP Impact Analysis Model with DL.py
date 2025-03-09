# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 09:06:27 2025
@author: NoteBook
"""

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from layers import GraphConvolution  # تأكد من أن ملف `layers.py` موجود
import torch.nn.functional as F
import numpy as np  # لدعم العمليات الحسابية
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score  # لتقييم النماذج
import matplotlib.pyplot as plt

# تعديل النموذج لاستقبال السمات المدمجة
# تعديل النموذج لاستقبال السمات المدمجة
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, snp_feat_dim):
        super(GCN, self).__init__()
        self.gcn1 = nn.Linear(nfeat, nhid)
        self.gcn2 = nn.Linear(nhid, nhid)
        self.snp_fc = nn.Linear(snp_feat_dim, nhid)
        self.fc = nn.Linear(nhid * 2, nclass)
        self.dropout = dropout

    def forward(self, x, adj, snp_features):
        x_gcn = F.relu(self.gcn1(x))
        x_gcn = F.dropout(x_gcn, self.dropout, training=self.training)
        x_gcn = F.relu(self.gcn2(x_gcn))
        x_gcn = F.dropout(x_gcn, self.dropout, training=self.training)

        x_snp = F.relu(self.snp_fc(snp_features))
        x_snp = F.dropout(x_snp, self.dropout, training=self.training)

        if x_gcn.size(0) != x_snp.size(0):
            min_size = min(x_gcn.size(0), x_snp.size(0))
            x_gcn = x_gcn[:min_size]
            x_snp = x_snp[:min_size]

        x_combined = torch.cat([x_gcn, x_snp], dim=1)
        output = self.fc(x_combined)
        return F.log_softmax(output, dim=1)


class SNPAnalyzer:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.le = LabelEncoder()
        self.scaler = StandardScaler()

    def analyze_impact(self):
        # طباعة القيم الفريدة في عمود التأثير البيولوجي للتحقق
        #print("القيم الفريدة في عمود التأثير البيولوجي:")
        #print(self.df['Biological_Impact'].unique())

        # 1. تحليل التأثير البيولوجي المباشر
        impact_counts = self.df['Biological_Impact'].value_counts()
        #print("\nتوزيع التأثيرات البيولوجية:")
        #print(impact_counts)

        # 2. تحليل المواقع الوظيفية مع معالجة القيم المفقودة
        functional_analysis = pd.crosstab(
            self.df['Variant_Location_Category'],
            self.df['Biological_Impact']
        )

        # 3. تحليل تأثير الجين مع معالجة محسنة للقيم
        gene_impact = {}
        for idx, row in self.df.iterrows():
            try:
                genes = eval(row['Mapped_Genes_List'])
                impact = row['Biological_Impact'].lower()

                # تحويل التأثير إلى فئات قياسية
                if 'high' in impact:
                    impact_category = 'high'
                elif 'moderate' in impact:
                    impact_category = 'moderate'
                elif 'low' in impact:
                    impact_category = 'low'
                else:
                    impact_category = 'unknown'

                for gene in genes:
                    gene = gene.strip("'")
                    if gene not in gene_impact:
                        gene_impact[gene] = {
                            'high': 0,
                            'moderate': 0,
                            'low': 0,
                            'unknown': 0
                        }
                    gene_impact[gene][impact_category] += 1
            except Exception as e:
                print(f"خطأ في معالجة الصف {idx}: {e}")
                continue

        return {
            'impact_counts': impact_counts,
            'functional_analysis': functional_analysis,
            'gene_impact': gene_impact
        }

    def prepare_graph_features(self):
        # التحقق من وجود الأعمدة الأساسية
        required_columns = ['Biological_Impact', 'Variant_Location_Category', 'Consequence_Category', 'chromosomePosition']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"الأعمدة المطلوبة غير موجودة: {missing_columns}")

        # تحويل البيانات النصية إلى رقمية
        print("تحويل البيانات النصية إلى رقمية...")
        self.df['impact_encoded'] = self.le.fit_transform(self.df['Biological_Impact'].fillna('Unknown'))
        self.df['location_encoded'] = self.le.fit_transform(self.df['Variant_Location_Category'].fillna('Unknown'))
        self.df['consequence_encoded'] = self.le.fit_transform(self.df['Consequence_Category'].fillna('Unknown'))

        # إنشاء الميزات
        print("إنشاء الميزات...")
        features = pd.DataFrame({
            'chromosome_position': self.df['chromosomePosition'].fillna(self.df['chromosomePosition'].mean()),
            'location_type': self.df['location_encoded'],
            'consequence': self.df['consequence_encoded'],
            'is_high_impact': (self.df['Biological_Impact'] == 'High Impact').astype(int),
            'is_intronic': (self.df['Variant_Location_Category'] == 'Intron').astype(int),
            'is_exonic': (self.df['Variant_Location_Category'].isin(['Missense', 'Synonymous'])).astype(int)
        })

        # تطبيع الميزات
        print("تطبيع الميزات...")
        features = self.scaler.fit_transform(features)

        # بناء مصفوفة adjacency باستخدام cosine similarity
        print("بناء مصفوفة adjacency...")
        from sklearn.metrics.pairwise import cosine_similarity
        adj = cosine_similarity(features)  # مصفوفة التشابه
        adj = torch.tensor(adj, dtype=torch.float32)

        return torch.tensor(features, dtype=torch.float32), adj

    def build_gcn_model(self, nfeat, nhid, nclass, dropout=0.6, snp_feat_dim=6):
        model = GCN(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=dropout, snp_feat_dim=snp_feat_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        return model, optimizer

    def train_gcn(self, X, adj, y, epochs=500, snp_features=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X, adj = X.to(device), adj.to(device)
        y = y.clone().detach().to(device)

        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y.cpu().numpy()),
            y=y.cpu().numpy()
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        if snp_features is not None:
            snp_features = torch.tensor(snp_features, dtype=torch.float32).to(device)

        model, optimizer = self.build_gcn_model(
            nfeat=X.shape[1],
            nhid=32,
            nclass=len(torch.unique(y)),
            dropout=0.6,
            snp_feat_dim=snp_features.shape[1] if snp_features is not None else 6
        )

        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

        model.train()
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X, adj, snp_features)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

        model.eval()
        with torch.no_grad():
            output = model(X, adj, snp_features)
            _, preds = torch.max(output, dim=1)
            accuracy = torch.sum(preds == y).item() / len(y)
            print(f"Training Accuracy: {accuracy * 100:.2f}%")

        return model, preds.cpu().numpy()

    def evaluate_gcn(self, model, X, adj, y_true, snp_features=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X, adj = X.to(device), adj.to(device)
        y_true = y_true.to(device)
    
        if snp_features is not None:
            snp_features = snp_features.to(device)
    
        model.eval()
        with torch.no_grad():
            output = model(X, adj, snp_features)
            probs = torch.exp(output).cpu()  # نقل الاحتمالات إلى CPU
            _, preds = torch.max(output, dim=1)
            accuracy = torch.sum(preds == y_true).item() / len(y_true)
    
            # حساب F1-Score
            f1 = f1_score(y_true.cpu().numpy(), preds.cpu().numpy(), average='weighted') * 100
    
            # حساب AUC-ROC
            try:
                auc = roc_auc_score(y_true.cpu().numpy(), probs.detach().numpy(), multi_class='ovr')
            except ValueError:
                auc = 0.0  # استبدال القيم المفقودة بـ 0.0
    
            print(f"Test Accuracy: {accuracy * 100:.2f}%")
            print(f"Test F1-Score: {f1:.2f}%")
            print(f"Test AUC-ROC: {auc:.4f}")
    
        return preds.cpu().numpy(), accuracy * 100, f1, auc


def plot_results(train_accuracy, test_accuracy, f1_score, auc_roc):
    """
    رسم النتائج لكل من Training Accuracy, Test Accuracy, F1-Score, و AUC-ROC.
    
    :param train_accuracy: دقة التدريب
    :param test_accuracy: دقة الاختبار
    :param f1_score: قيمة F1-Score
    :param auc_roc: قيمة AUC-ROC
    """

    # رسم Training Accuracy
    plt.figure(figsize=(8, 6))
    plt.bar(['Training Accuracy'], [train_accuracy], color='blue')
    plt.title('Training Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y')
    plt.show()

    # رسم Test Accuracy
    plt.figure(figsize=(8, 6))
    plt.bar(['Test Accuracy'], [test_accuracy], color='green')
    plt.title('Test Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y')
    plt.show()

    # رسم F1-Score
    plt.figure(figsize=(8, 6))
    plt.bar(['F1-Score'], [f1_score], color='orange')
    plt.title('F1-Score')
    plt.ylabel('F1-Score (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y')
    plt.show()

    # رسم AUC-ROC
    plt.figure(figsize=(8, 6))
    plt.bar(['AUC-ROC'], [auc_roc], color='red')
    plt.title('AUC-ROC')
    plt.ylabel('AUC-ROC')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.show()

def main():
    try:
        print("بدء تحليل البيانات...")
        analyzer = SNPAnalyzer('output/Classified_Positive_SNPs_Data.csv')

        # تقليل حجم البيانات للاختبار
        analyzer.df = analyzer.df.sample(frac=0.1, random_state=42)

        # تحليل التأثير
        print("\nتحليل تأثير الطفرات...")
        impact_results = analyzer.analyze_impact()

        if impact_results:
            print("\nتحليل توزيع التأثيرات البيولوجية:")
            print(impact_results['impact_counts'])

        # إعداد البيانات للنموذج GCN
        print("\nإعداد البيانات للنموذج GCN...")
        X, adj = analyzer.prepare_graph_features()
        y = torch.tensor(analyzer.df['impact_encoded'].values, dtype=torch.long)

        # إعداد السمات الإضافية (snp_features)
        features_df = pd.DataFrame({
            'chromosome_position': analyzer.df['chromosomePosition'].fillna(analyzer.df['chromosomePosition'].mean()),
            'location_type': analyzer.df['location_encoded'],
            'consequence': analyzer.df['consequence_encoded'],
            'is_high_impact': (analyzer.df['Biological_Impact'] == 'High Impact').astype(int),
            'is_intronic': (analyzer.df['Variant_Location_Category'] == 'Intron').astype(int),
            'is_exonic': (analyzer.df['Variant_Location_Category'].isin(['Missense', 'Synonymous'])).astype(int)
        })
        snp_features = torch.tensor(features_df.values, dtype=torch.float32)

        # تقسيم البيانات
        X_train, X_test, adj_train, adj_test, y_train, y_test, snp_train, snp_test = train_test_split(
            X, adj, y, snp_features, test_size=0.2, random_state=42, stratify=y
        )

        # تدريب النموذج GCN
        print("\nتدريب النموذج GCN...")
        gcn_model, train_preds = analyzer.train_gcn(X_train, adj_train, y_train, epochs=500, snp_features=snp_train)
        train_accuracy = np.mean(train_preds == y_train.numpy()) * 100

        # تقييم النموذج GCN
        print("\nتقييم النموذج GCN...")
        gcn_preds, test_accuracy, f1_score_value, auc_roc = analyzer.evaluate_gcn(gcn_model, X_test, adj_test, y_test, snp_features=snp_test)

        # طباعة تقرير التصنيف
        impact_labels = analyzer.le.inverse_transform(np.unique(y_test))
        print("\nنتائج نموذج GCN:")
        print(classification_report(y_test.numpy(), gcn_preds, target_names=impact_labels, zero_division=0))

        # رسم النتائج
        print("\nرسم النتائج...")
        plot_results(train_accuracy, test_accuracy, f1_score_value, auc_roc)

    except Exception as e:
        print(f"حدث خطأ أثناء التحليل: {e}")
        return None, None

if __name__ == "__main__":
    main()
