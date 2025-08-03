import pandas as pd
from sklearn.preprocessing import LabelEncoder

class PatientDataIntegration:
    def __init__(self, patient_data_path, biomarker_data_path):
        self.patient_data = pd.read_csv(patient_data_path, dtype={"PatientId": str})
        self.biomarkers = pd.read_csv(biomarker_data_path, dtype={"Biomarker": str})
        self._validate_data()
        self._encode_data()
    
    def _validate_data(self):
        if "PatientId" not in self.patient_data.columns:
            raise KeyError("Missing 'PatientId' column in patient data.")
        if "Biomarker" not in self.biomarkers.columns:
            raise KeyError("Missing 'Biomarker' column in biomarker data.")
        if "Recommended_Treatment" not in self.patient_data.columns:
            raise KeyError("Missing 'Recommended_Treatment' column in patient data.")
        
    def _encode_data(self):
        categorical_cols = ["Tumor_Stage", "Hormone_Receptor_Status", "HER2_Status"]
        self.patient_data = pd.get_dummies(self.patient_data, columns=categorical_cols, drop_first=True)
        self.label_encoder = LabelEncoder()
        self.patient_data["Recommended_Treatment"] = self.label_encoder.fit_transform(
            self.patient_data["Recommended_Treatment"]
        )
    
    def integrate_patient_data(self):
        merged_data = self.patient_data.merge(
            self.biomarkers, left_on="PatientId", right_on="Biomarker", how="left"
        )
        merged_data.fillna(0, inplace=True)
        return merged_data.infer_objects(copy=False)