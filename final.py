import os
from pathlib import Path
import re
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
import torch

# -------------------------------
# دوال مساعدة
# -------------------------------
def preprocess_arabic_text(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', '', str(text))
    text = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def heuristic_risk_level(text):
    high_keywords = ["عنف", "عنيف", "تنمر", "تهديد", "اكتئاب", "انتحار", "مخدرات", "اعتداء", "إيذاء",
                     "self-harm", "bullying", "violence"]
    medium_keywords = ["قلق", "توتر", "انعزال", "مشاكل أسرية", "مشاكل عائلية", "حزن", "تراجع", "عدوانية"]
    if any(kw in text for kw in high_keywords):
        return "high"
    if any(kw in text for kw in medium_keywords):
        return "medium"
    return "low"

def get_embeddings(texts, tokenizer, model, max_len=128):
    inputs = tokenizer(
        texts.tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    )
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

def assign_action_with_recommendations(row):
    if row['AttendanceRate'] < 65 or row['AcademicPerformance'] < 57 or row['RiskLevel'] == 'high':
        الإجراء = "تدخل فوري"
        التوصيات = {
            "المؤشرات / الأسباب": [
                "تم رصد مؤشرات خطورة عالية في الحضور أو الأداء أو السلوك.",
            ],
            "الإجراء الموصى به": [
                "إخطار الأخصائي النفسي/الاجتماعي فورًا.",
                "التواصل مع ولي الأمر وفق سياسات المدرسة.",
                "تأمين بيئة آمنة للطالب ومتابعة حالته عن قرب.",
            ],
            "متابعة لاحقة": [
                "إحالة الطالب إلى جهة مختصة إذا لزم الأمر.",
                "خطة دعم فردية وتقييم أسبوعي للتقدم.",
            ],
        }
        
    elif row['CounselingSessions'] > 2:
        الإجراء = "إحالة إلى جلسات إرشادية"
        التوصيات = {
            "المؤشرات / الأسباب": [
                "الطالب يحتاج إلى متابعة متخصصة عبر جلسات إرشاد متعددة.",
            ],
            "الإجراء الموصى به": [
                "جدولة جلسة إرشادية مع الأخصائي.",
            ],
            "متابعة لاحقة": [
                "تقييم فعالية جلسات الإرشاد ومتابعة تقدم الطالب.",
            ],
        }
        
    elif row['AttendanceRate'] >= 90 and row['AcademicPerformance'] >= 88 and row['RiskLevel'] == 'low':
        الإجراء = "متابعة منتظمة"
        التوصيات = {
            "المؤشرات / الأسباب": [
                "الطالب يظهر مؤشرات جيدة في الأداء والحضور والسلوك.",
            ],
            "الإجراء الموصى به": [
                "استمرار المتابعة الدورية وتشجيع الطالب.",
            ],
            "متابعة لاحقة": [
                "تقديم مواد تعليمية وتوعوية لتعزيز المهارات.",
            ],
        }
        
    else:
        الإجراء = "متابعة المعلم + مواد توعوية"
        التوصيات = {
            "المؤشرات / الأسباب": [
                "الطالب يحتاج متابعة منتظمة مع توفير دعم تعليمي وسلوكي.",
            ],
            "الإجراء الموصى به": [
                "تشجيع الطالب والمتابعة مع المعلمين.",
                "تقديم مواد توعوية داعمة داخل الصف.",
            ],
            "متابعة لاحقة": [
                "مراجعة دورية لأداء الطالب وتحديث خطة الدعم عند الحاجة.",
            ],
        }
    
    return {"الإجراء": الإجراء, "التوصيات": التوصيات}

# -------------------------------
# 1- قراءة بيانات الملاحظات
# -------------------------------
df_notes = pd.read_csv("Data/synthetic_students_500_with_notes.csv")
df_notes['teacher_note_clean'] = df_notes['teacher_note'].apply(preprocess_arabic_text)

# -------------------------------
# 2- تحميل AraBERT
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2")
model.eval()

# -------------------------------
# 3- تحويل النصوص إلى embeddings
# -------------------------------
X_embeddings = get_embeddings(df_notes['teacher_note_clean'], tokenizer, model)
y = df_notes['label']

# -------------------------------
# 4- تقسيم البيانات لتدريب classifier
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 5- تدريب Classifier على embeddings
# -------------------------------
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)

# -------------------------------
# 6- تقييم AraBERT (KPI)
# -------------------------------
y_pred_test_arabert = clf.predict(X_test)
kpi_arabert = accuracy_score(y_test, y_pred_test_arabert)
print("KPI AraBERT (Accuracy on test set):", kpi_arabert)

# -------------------------------
# 7- التنبؤ لكل الطلاب باستخدام AraBERT + fallback بالكلمات المفتاحية
# -------------------------------
risk_levels = []
for i, row in df_notes.iterrows():
    note = row['teacher_note_clean']
    heuristic = heuristic_risk_level(note)
    if heuristic != 'low':
        risk_levels.append(heuristic)
    else:
        emb = get_embeddings(pd.Series([note]), tokenizer, model)
        pred = clf.predict(emb)[0]
        risk_levels.append(pred)

df_notes['RiskLevel'] = risk_levels
df_notes.to_csv("Data/synthetic_students_500_with_risk.csv", index=False)
joblib.dump(clf, "arabert_classifier.pkl")

# KPI AraBERT على البيانات الكاملة
kpi_arabert_full = accuracy_score(df_notes['label'], df_notes['RiskLevel'])
print("KPI AraBERT (Full dataset):", kpi_arabert_full)

# -------------------------------
# Decision Tree
# -------------------------------
df = pd.read_csv("./Data/synthetic_students_500_with_risk.csv", dtype={"AI_RecommendedAction": "object"})

num_cols = ['Age', 'AttendanceRate', 'AcademicPerformance', 'CounselingSessions', 'HotlineCalls']
df[num_cols] = df[num_cols].fillna(-1)

text_cols = ['Name', 'BehaviorIncidents', 'RiskLevel']
df[text_cols] = df[text_cols].fillna("unknown")

df.drop_duplicates(subset=['StudentID'], inplace=True)

df['GenderNum'] = df['Gender'].map({'M':0, 'F':1})
df['BehaviorIncidentsNum'] = df['BehaviorIncidents'].factorize()[0]
risk_mapping = {"low": 0, "moderate": 1, "high": 2}
df['RiskLvlNum'] = df['RiskLevel'].map(risk_mapping)

df['AttendanceRate'] = df['AttendanceRate'].clip(0, 100)
df['AcademicPerformance'] = df['AcademicPerformance'].clip(0, 100)
df['CounselingSessions'] = df['CounselingSessions'].clip(0)

df['AI_RecommendedAction_Label'] = df.apply(assign_action_with_recommendations, axis=1)
df['AI_RecommendedAction'] = df['AI_RecommendedAction_Label'].apply(lambda x: x['الإجراء'])

features = ['StudentID', 'Age', 'GenderNum', 'AttendanceRate', 'AcademicPerformance',
            'CounselingSessions', 'HotlineCalls', 'RiskLvlNum']
X = df[features]
y = df['AI_RecommendedAction']

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# KPI Decision Tree
kpi_dt = accuracy_score(y_test, y_pred)
print("KPI Decision Tree (Accuracy on test set):", kpi_dt)

# Adding Prediction to dataframe
df_test = X_test.copy()
df_test['AI_RecommendedAction'] = y_test
df_test['AI_PredictedAction'] = y_pred

# KPI Whole Model (AraBERT + Decision Tree)
# استخدام RiskLevel المحسوب لكل طالب كمدخل Decision Tree
kpi_whole = accuracy_score(y_test, y_pred)
print("KPI Whole Model (Accuracy):", kpi_whole)

# save results and models
df_test.to_csv("./students_test_predictions.csv", index=False)
joblib.dump(model, "decision_tree_model.pkl")
