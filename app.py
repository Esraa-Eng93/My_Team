import streamlit as st
import pandas as pd
import joblib
import re
from transformers import AutoTokenizer, AutoModel
import torch

# ---- تحميل نموذج Decision Tree ----
dt_model = joblib.load("decision_tree_model.pkl")

# ---- تحميل نموذج AraBERT ----
arabert_clf = joblib.load("arabert_classifier.pkl")
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
arabert_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2")
arabert_model.eval()

# ---- دوال مساعدة ----
def preprocess_arabic_text(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', '', str(text))
    text = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def heuristic_risk_level(text):
    high_keywords = ["عنف","عنيف","تنمر","تهديد","اكتئاب","انتحار","مخدرات","اعتداء","إيذاء"]
    medium_keywords = ["قلق","توتر","انعزال","مشاكل أسرية","مشاكل عائلية","حزن","تراجع","عدوانية"]
    if any(kw in text for kw in high_keywords):
        return "high"
    if any(kw in text for kw in medium_keywords):
        return "moderate"
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

def get_risk_level(note):
    note_clean = preprocess_arabic_text(note)
    heuristic = heuristic_risk_level(note_clean)
    if heuristic != "low":
        return heuristic
    emb = get_embeddings(pd.Series([note_clean]), tokenizer, arabert_model)
    return arabert_clf.predict(emb)[0]

# ---- تحميل بيانات الطلاب السابقين أو إنشاء DataFrame فارغ ----
try:
    df_students = pd.read_csv("students_data.csv", dtype=str)
except FileNotFoundError:
    df_students = pd.DataFrame(columns=[
        "الاسم", "الجنس", "الرقم_الوطني","العمر","الحضور",
        "المعدل_الأكاديمي","عدد_الجلسات","ملاحظات",
        "مستوى_الخطورة","RiskLvlNum","الإجراء","التوصيات"
    ])

# ---- إعداد الصفحة ----
st.set_page_config(
    page_title="معاً لتعزيز الصحة النفسية للطلاب",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 معاً لتعزيز الصحة النفسية للطلاب ")

# ---- إدخال الرقم الوطني للبحث ----
national_id = st.text_input("أدخل الرقم الوطني للطالب للبحث:", key="national_id", max_chars=10)

if national_id:
    student = df_students[df_students["الرقم_الوطني"] == national_id]
    if not student.empty:
        # إذا وجد الطالب
        st.success("تم العثور على بيانات الطالب!")
        st.markdown(f"**الاسم:** {student.iloc[0]['الاسم']}")
        st.markdown(f"**الجنس:** {student.iloc[0]['الجنس']}")
        st.markdown(f"**مستوى الخطورة:** {student.iloc[0]['مستوى_الخطورة']}")
        st.markdown(f"**الإجراء الموصى به:** {student.iloc[0]['الإجراء']}")
        st.markdown(f"**التوصيات:** {student.iloc[0]['التوصيات']}")
    else:
        st.warning("الرقم الوطني غير موجود. الرجاء إدخال بيانات الطالب:")
        
        # ---- إدخال بيانات الطالب الجديدة ----
        full_name = st.text_input("الاسم الرباعي للطالب")
        gender = st.selectbox("الجنس", options=["ذكر", "أنثى"])
        age = st.number_input("العمر", min_value=5, max_value=25, step=1)
        attendance = st.number_input("نسبة الحضور (%)", min_value=0, max_value=100, step=1)
        academic = st.number_input("المعدل الأكاديمي", min_value=0, max_value=100, step=1)
        counseling = st.number_input("عدد الجلسات الاستشارية", min_value=0, max_value=20, step=1)
        teacher_notes = st.text_area("ملاحظات المعلم")

        # ---- زر التنبؤ ----
        if st.button("حساب الإجراء"):

            # تحويل الجنس إلى رقم
            gender_num = 0 if gender == "ذكر" else 1

            # ---- حساب RiskLevel ----
            risk_level_str = get_risk_level(teacher_notes)
            risk_mapping = {"low": 0, "moderate": 1, "high": 2}
            risk_level_num = risk_mapping[risk_level_str]

            # ---- تحويل البيانات الجديدة إلى DataFrame مطابق للـ features ----
            X_new = pd.DataFrame([{
                'StudentID': national_id,
                'Age': age,
                'GenderNum': gender_num,
                'AttendanceRate': attendance,
                'AcademicPerformance': academic,
                'CounselingSessions': counseling,
                'HotlineCalls': 0,
                'RiskLvlNum': risk_level_num
            }])

            # ---- التنبؤ بالإجراء باستخدام نموذج Decision Tree ----
            action = dt_model.predict(X_new)[0]

            # ---- توليد توصيات عامة بناءً على الإجراء ----
            recommendations_dict = {
                "تدخل فوري": "- إخطار الأخصائي النفسي/الاجتماعي فورًا.\n- التواصل مع ولي الأمر.\n- متابعة مستمرة للطالب.",
                "إحالة إلى جلسات إرشادية": "- جدولة جلسة إرشادية مع الأخصائي.\n- متابعة تقدم الطالب.",
                "متابعة منتظمة": "- الاستمرار بالمتابعة الدورية وتشجيع الطالب.",
                "متابعة المعلم + مواد توعوية": "- تشجيع الطالب والمتابعة مع المعلمين.\n- تقديم مواد داعمة داخل الصف."
            }

            recommendations = recommendations_dict.get(action, "")

            st.success(f"الإجراء: {action}")
            st.markdown(f"التوصيات:\n{recommendations}")

            # ---- إضافة الطالب الجديد إلى البيانات وحفظه ----
            new_row = {
                "الاسم": full_name,
                "الرقم_الوطني": national_id,
                "الجنس": gender,
                "العمر": age,
                "الحضور": attendance,
                "المعدل_الأكاديمي": academic,
                "عدد_الجلسات": counseling,
                "ملاحظات": teacher_notes,
                "مستوى_الخطورة": risk_level_str,  # النص
                "RiskLvlNum": risk_level_num,      # الرقم
                "الإجراء": action,
                "التوصيات": recommendations
            }
            df_students = pd.concat([df_students, pd.DataFrame([new_row])], ignore_index=True)
            df_students.to_csv("students_data.csv", index=False)
            st.success("تم حفظ بيانات الطالب بنجاح!")
