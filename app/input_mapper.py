"""
Regional Input Mapper for Student Dropout Prediction

This module provides functionality to convert user-friendly, region-specific
academic inputs into the standardized format required by the ML models.

Supports academic systems from:
- USA (GPA 0-4.0, credit hours, etc.)
- UK (percentage grades, modules, etc.)
- Türkiye (0-100 grades, AKTS credits, etc.)
"""

import pandas as pd
from typing import Dict, Any, Tuple
import numpy as np
import joblib
import os


class RegionalInputMapper:
    """
    Maps region-specific academic inputs to model-ready format.
    
    The ML models expect 9 specific features with certain ranges:
    1. Age at enrollment (15-100)
    2. Admission grade (0-200)
    3. Tuition fees up to date (0/1)
    4. Previous qualification (grade) (0-200)
    5. Curricular units 1st sem (approved) (0-10)
    6. Curricular units 2nd sem (approved) (0-10)
    7. Curricular units 1st sem (grade) (0-20)
    8. Curricular units 2nd sem (grade) (0-20)
    9. Curricular units 2nd sem (evaluations) (0-20)
    """
    
    def __init__(self, scaler_path: str = None):
        self.supported_regions = ["USA", "UK", "Türkiye"]
        self.model_features = [
            "Age at enrollment",
            "Admission grade",
            "Tuition fees up to date",
            "Previous qualification (grade)",
            "Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (grade)",
            "Curricular units 2nd sem (evaluations)"
        ]
        
        # Load the scaler if path is provided
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"Scaler loaded successfully from {scaler_path}")
            except Exception as e:
                print(f"Warning: Could not load scaler from {scaler_path}: {e}")
    
    def get_region_input_schema(self, region: str) -> Dict[str, Dict[str, Any]]:
        """
        Returns the input schema for a specific region.
        
        Args:
            region (str): One of 'USA', 'UK', 'Türkiye'
            
        Returns:
            Dict containing input field definitions with types, ranges, and descriptions
        """
        schemas = {
            "USA": {
                "age": {
                    "type": "slider",
                    "min": 15, "max": 100, "default": 18,
                    "label": "🎂 Age at Enrollment",
                    "help": "Student's age when starting college/university"
                },
                "sat_score": {
                    "type": "slider",
                    "min": 400, "max": 1600, "default": 1100,
                    "label": "📝 SAT Score (or equivalent)",
                    "help": "Combined SAT score (Math + Evidence-Based Reading)"
                },
                "tuition_paid": {
                    "type": "selectbox",
                    "options": ["No", "Yes"],
                    "default": "Yes",
                    "label": "💰 Tuition Fees Current?",
                    "help": "Are tuition fees paid up to date?"
                },
                "high_school_gpa": {
                    "type": "slider",
                    "min": 0.0, "max": 4.0, "default": 3.0, "step": 0.1,
                    "label": "📚 High School GPA",
                    "help": "High school GPA on 4.0 scale"
                },
                "credits_completed_fall": {
                    "type": "slider",
                    "min": 0, "max": 18, "default": 15,
                    "label": "✅ Credits Completed - Fall Semester",
                    "help": "Number of credit hours successfully completed in fall"
                },
                "credits_completed_spring": {
                    "type": "slider",
                    "min": 0, "max": 18, "default": 15,
                    "label": "✅ Credits Completed - Spring Semester",
                    "help": "Number of credit hours successfully completed in spring"
                },
                "fall_semester_gpa": {
                    "type": "slider",
                    "min": 0.0, "max": 4.0, "default": 3.0, "step": 0.1,
                    "label": "📊 Fall Semester GPA",
                    "help": "GPA achieved in fall semester"
                },
                "spring_semester_gpa": {
                    "type": "slider",
                    "min": 0.0, "max": 4.0, "default": 3.0, "step": 0.1,
                    "label": "📊 Spring Semester GPA",
                    "help": "GPA achieved in spring semester"
                },
                "spring_exams_taken": {
                    "type": "slider",
                    "min": 0, "max": 12, "default": 6,
                    "label": "🧪 Spring Semester Exams Taken",
                    "help": "Number of final exams taken in spring semester"
                }
            },
            
            "UK": {
                "age": {
                    "type": "slider",
                    "min": 15, "max": 100, "default": 18,
                    "label": "🎂 Age at Enrollment",
                    "help": "Student's age when starting university"
                },
                "a_level_points": {
                    "type": "slider",
                    "min": 0, "max": 420, "default": 240,
                    "label": "📝 A-Level Points (UCAS)",
                    "help": "Total UCAS points from A-levels or equivalent"
                },
                "tuition_paid": {
                    "type": "selectbox",
                    "options": ["No", "Yes"],
                    "default": "Yes",
                    "label": "💰 Tuition Fees Current?",
                    "help": "Are tuition fees paid up to date?"
                },
                "previous_qualification_grade": {
                    "type": "slider",
                    "min": 0, "max": 100, "default": 70,
                    "label": "📚 Previous Qualification Grade (%)",
                    "help": "Percentage grade from previous qualification (A-levels, etc.)"
                },
                "modules_passed_year1_sem1": {
                    "type": "slider",
                    "min": 0, "max": 8, "default": 4,
                    "label": "✅ Modules Passed - Year 1, Semester 1",
                    "help": "Number of modules successfully passed"
                },
                "modules_passed_year1_sem2": {
                    "type": "slider",
                    "min": 0, "max": 8, "default": 4,
                    "label": "✅ Modules Passed - Year 1, Semester 2",
                    "help": "Number of modules successfully passed"
                },
                "average_grade_sem1": {
                    "type": "slider",
                    "min": 0, "max": 100, "default": 65,
                    "label": "📊 Average Grade - Semester 1 (%)",
                    "help": "Average percentage grade achieved in semester 1"
                },
                "average_grade_sem2": {
                    "type": "slider",
                    "min": 0, "max": 100, "default": 65,
                    "label": "📊 Average Grade - Semester 2 (%)",
                    "help": "Average percentage grade achieved in semester 2"
                },
                "assessments_sem2": {
                    "type": "slider",
                    "min": 0, "max": 16, "default": 8,
                    "label": "🧪 Assessments - Semester 2",
                    "help": "Number of assessments (exams, coursework) in semester 2"
                }
            },
            
            "Türkiye": {
                "age": {
                    "type": "slider",
                    "min": 15, "max": 100, "default": 18,
                    "label": "🎂 Kayıt Yaşı",
                    "help": "Üniversiteye kayıt olduğu andaki yaş"
                },
                "yks_score": {
                    "type": "slider",
                    "min": 150, "max": 560, "default": 300,
                    "label": "📝 YKS Puanı",
                    "help": "YKS sınav puanı (TYT + AYT)"
                },
                "tuition_paid": {
                    "type": "selectbox",
                    "options": ["Hayır", "Evet"],
                    "default": "Evet",
                    "label": "💰 Harç Ödemesi Güncel?",
                    "help": "Harç ödemeleri güncel mi?"
                },
                "lise_ortalama": {
                    "type": "slider",
                    "min": 0, "max": 100, "default": 70,
                    "label": "📚 Lise Diploma Ortalaması",
                    "help": "Lise diploma not ortalaması (0-100)"
                },
                "akts_guz": {
                    "type": "slider",
                    "min": 0, "max": 45, "default": 30,
                    "label": "✅ Tamamlanan AKTS - Güz Dönemi",
                    "help": "Güz döneminde başarıyla tamamlanan AKTS kredisi"
                },
                "akts_bahar": {
                    "type": "slider",
                    "min": 0, "max": 45, "default": 30,
                    "label": "✅ Tamamlanan AKTS - Bahar Dönemi",
                    "help": "Bahar döneminde başarıyla tamamlanan AKTS kredisi"
                },
                "guz_ortalama": {
                    "type": "slider",
                    "min": 0, "max": 100, "default": 65,
                    "label": "📊 Güz Dönemi Not Ortalaması",
                    "help": "Güz döneminde alınan not ortalaması (0-100)"
                },
                "bahar_ortalama": {
                    "type": "slider",
                    "min": 0, "max": 100, "default": 65,
                    "label": "📊 Bahar Dönemi Not Ortalaması",
                    "help": "Bahar döneminde alınan not ortalaması (0-100)"
                },
                "bahar_sinav_sayisi": {
                    "type": "slider",
                    "min": 0, "max": 15, "default": 8,
                    "label": "🧪 Bahar Dönemi Sınav Sayısı",
                    "help": "Bahar döneminde girilen sınav sayısı"
                }
            }
        }
        
        if region not in schemas:
            raise ValueError(f"Region '{region}' not supported. Use one of: {self.supported_regions}")
        
        return schemas[region]
    
    def map_to_model_format(self, region: str, user_inputs: Dict[str, Any], apply_scaling: bool = True) -> pd.DataFrame:
        """
        Converts region-specific user inputs to the standardized model format.
        
        Args:
            region (str): One of 'USA', 'UK', 'Türkiye'
            user_inputs (dict): Dictionary of user inputs with region-specific keys
            apply_scaling (bool): Whether to apply StandardScaler to the output
            
        Returns:
            pd.DataFrame: Single-row DataFrame with model-ready features
        """
        # First, convert to unscaled model format
        if region == "USA":
            model_data = self._map_usa_inputs(user_inputs)
        elif region == "UK":
            model_data = self._map_uk_inputs(user_inputs)
        elif region == "Türkiye":
            model_data = self._map_turkey_inputs(user_inputs)
        else:
            raise ValueError(f"Region '{region}' not supported")
        
        # Apply scaling if requested and scaler is available
        if apply_scaling and self.scaler is not None:
            try:
                # Convert to numpy array for scaling
                data_array = model_data.values
                # Apply the same scaling used during training
                scaled_data = self.scaler.transform(data_array)
                # Convert back to DataFrame with original columns
                model_data = pd.DataFrame(scaled_data, columns=model_data.columns)
                print("✅ Scaling applied successfully")
            except Exception as e:
                print(f"⚠️ Warning: Could not apply scaling: {e}")
        elif apply_scaling and self.scaler is None:
            print("⚠️ Warning: Scaling requested but scaler not loaded")
        
        return model_data
    
    def _map_usa_inputs(self, inputs: Dict[str, Any]) -> pd.DataFrame:
        """Map USA-specific inputs to model format"""
        model_data = {
            "Age at enrollment": inputs["age"],
            "Admission grade": self._convert_sat_to_admission_grade(inputs["sat_score"]),
            "Tuition fees up to date": 1 if inputs["tuition_paid"] == "Yes" else 0,
            "Previous qualification (grade)": self._convert_gpa_to_qualification_grade(inputs["high_school_gpa"]),
            "Curricular units 1st sem (approved)": self._convert_credits_to_units(inputs["credits_completed_fall"]),
            "Curricular units 2nd sem (approved)": self._convert_credits_to_units(inputs["credits_completed_spring"]),
            "Curricular units 1st sem (grade)": self._convert_gpa_to_grade_scale(inputs["fall_semester_gpa"]),
            "Curricular units 2nd sem (grade)": self._convert_gpa_to_grade_scale(inputs["spring_semester_gpa"]),
            "Curricular units 2nd sem (evaluations)": min(inputs["spring_exams_taken"], 20)
        }
        
        return pd.DataFrame([model_data])
    
    def _map_uk_inputs(self, inputs: Dict[str, Any]) -> pd.DataFrame:
        """Map UK-specific inputs to model format"""
        model_data = {
            "Age at enrollment": inputs["age"],
            "Admission grade": self._convert_ucas_to_admission_grade(inputs["a_level_points"]),
            "Tuition fees up to date": 1 if inputs["tuition_paid"] == "Yes" else 0,
            "Previous qualification (grade)": self._convert_percentage_to_qualification_grade(inputs["previous_qualification_grade"]),
            "Curricular units 1st sem (approved)": min(inputs["modules_passed_year1_sem1"], 10),
            "Curricular units 2nd sem (approved)": min(inputs["modules_passed_year1_sem2"], 10),
            "Curricular units 1st sem (grade)": self._convert_percentage_to_grade_scale(inputs["average_grade_sem1"]),
            "Curricular units 2nd sem (grade)": self._convert_percentage_to_grade_scale(inputs["average_grade_sem2"]),
            "Curricular units 2nd sem (evaluations)": min(inputs["assessments_sem2"], 20)
        }
        
        return pd.DataFrame([model_data])
    
    def _map_turkey_inputs(self, inputs: Dict[str, Any]) -> pd.DataFrame:
        """Map Türkiye-specific inputs to model format"""
        model_data = {
            "Age at enrollment": inputs["age"],
            "Admission grade": self._convert_yks_to_admission_grade(inputs["yks_score"]),
            "Tuition fees up to date": 1 if inputs["tuition_paid"] == "Evet" else 0,
            "Previous qualification (grade)": self._convert_percentage_to_qualification_grade(inputs["lise_ortalama"]),
            "Curricular units 1st sem (approved)": self._convert_ects_to_units(inputs["akts_guz"]),
            "Curricular units 2nd sem (approved)": self._convert_ects_to_units(inputs["akts_bahar"]),
            "Curricular units 1st sem (grade)": self._convert_percentage_to_grade_scale(inputs["guz_ortalama"]),
            "Curricular units 2nd sem (grade)": self._convert_percentage_to_grade_scale(inputs["bahar_ortalama"]),
            "Curricular units 2nd sem (evaluations)": min(inputs["bahar_sinav_sayisi"], 20)
        }
        
        return pd.DataFrame([model_data])
    
    # Conversion helper methods
    def _convert_sat_to_admission_grade(self, sat_score: int) -> float:
        """Convert SAT score (400-1600) to admission grade (0-200)"""
        # Linear mapping: 400->80, 1600->200
        return 80 + ((sat_score - 400) / 1200) * 120
    
    def _convert_ucas_to_admission_grade(self, ucas_points: int) -> float:
        """Convert UCAS points (0-420) to admission grade (0-200)"""
        # Linear mapping with reasonable bounds
        return min(200, max(0, (ucas_points / 420) * 200))
    
    def _convert_yks_to_admission_grade(self, yks_score: float) -> float:
        """Convert YKS score (150-560) to admission grade (0-200)"""
        # Linear mapping: 150->60, 560->200
        return 60 + ((yks_score - 150) / 410) * 140
    
    def _convert_gpa_to_qualification_grade(self, gpa: float) -> float:
        """Convert GPA (0-4.0) to qualification grade (0-200)"""
        # Linear mapping: 0->0, 4.0->200
        return (gpa / 4.0) * 200
    
    def _convert_percentage_to_qualification_grade(self, percentage: float) -> float:
        """Convert percentage (0-100) to qualification grade (0-200)"""
        # Linear mapping: 0->0, 100->200
        return (percentage / 100) * 200
    
    def _convert_credits_to_units(self, credits: int) -> int:
        """Convert US credit hours to units (0-10)"""
        # Typical semester: 12-18 credits -> map to 0-10 units
        return min(10, max(0, round(credits * 10 / 18)))
    
    def _convert_ects_to_units(self, ects: int) -> int:
        """Convert ECTS credits to units (0-10)"""
        # Typical semester: 30 ECTS -> map to 0-10 units
        return min(10, max(0, round(ects * 10 / 30)))
    
    def _convert_gpa_to_grade_scale(self, gpa: float) -> float:
        """Convert GPA (0-4.0) to grade scale (0-20)"""
        # Linear mapping: 0->0, 4.0->20
        return (gpa / 4.0) * 20
    
    def _convert_percentage_to_grade_scale(self, percentage: float) -> float:
        """Convert percentage (0-100) to grade scale (0-20)"""
        # Linear mapping: 0->0, 100->20
        return (percentage / 100) * 20
    
    def get_input_summary(self, region: str, user_inputs: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a human-readable summary of user inputs.
        
        Args:
            region (str): Selected region
            user_inputs (dict): User input values
            
        Returns:
            dict: Summary of inputs with descriptions
        """
        schema = self.get_region_input_schema(region)
        summary = {}
        
        for key, value in user_inputs.items():
            if key in schema:
                field_info = schema[key]
                summary[field_info["label"]] = str(value)
        
        return summary


# Factory function for easy instantiation
def create_input_mapper(scaler_path: str = None) -> RegionalInputMapper:
    """Create and return a RegionalInputMapper instance"""
    if scaler_path is None:
        # Try to find the scaler in the default location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        scaler_path = os.path.join(project_root, "models", "scaler.pkl")
    
    return RegionalInputMapper(scaler_path=scaler_path)


# Example usage and testing
if __name__ == "__main__":
    # Test the mapper
    mapper = create_input_mapper()
    
    # Test Türkiye inputs
    turkey_inputs = {
        "age": 22,
        "yks_score": 289,
        "tuition_paid": "Hayır",
        "lise_ortalama": 43,
        "akts_guz": 12,
        "akts_bahar": 11,
        "guz_ortalama": 20,
        "bahar_ortalama": 15,
        "bahar_sinav_sayisi": 3
    }
    
    print("=== Testing Türkiye Inputs ===")
    print("User inputs:", turkey_inputs)
    
    # Test without scaling
    turkey_model_data_unscaled = mapper.map_to_model_format("Türkiye", turkey_inputs, apply_scaling=False)
    print("\nUnscaled Model Data:")
    print(turkey_model_data_unscaled)
    
    # Test with scaling
    turkey_model_data_scaled = mapper.map_to_model_format("Türkiye", turkey_inputs, apply_scaling=True)
    print("\nScaled Model Data:")
    print(turkey_model_data_scaled)
    
    # Show the difference
    print("\n=== Data Comparison ===")
    print("Feature ranges after conversion:")
    for col in turkey_model_data_unscaled.columns:
        unscaled_val = turkey_model_data_unscaled[col].iloc[0]
        scaled_val = turkey_model_data_scaled[col].iloc[0] if mapper.scaler else "N/A"
        print(f"{col}: {unscaled_val:.2f} -> {scaled_val}")
    
    print("\n=== Quick test with other regions ===")
    
    # Test USA inputs
    usa_inputs = {
        "age": 19,
        "sat_score": 1200,
        "tuition_paid": "Yes",
        "high_school_gpa": 3.5,
        "credits_completed_fall": 15,
        "credits_completed_spring": 12,
        "fall_semester_gpa": 3.2,
        "spring_semester_gpa": 3.8,
        "spring_exams_taken": 5
    }
    
    usa_model_data = mapper.map_to_model_format("USA", usa_inputs)
    print("USA Model Data (scaled):")
    print(usa_model_data)
    
    # Test UK inputs
    uk_inputs = {
        "age": 18,
        "a_level_points": 300,
        "tuition_paid": "Yes",
        "previous_qualification_grade": 85,
        "modules_passed_year1_sem1": 4,
        "modules_passed_year1_sem2": 3,
        "average_grade_sem1": 72,
        "average_grade_sem2": 68,
        "assessments_sem2": 6
    }
    
    uk_model_data = mapper.map_to_model_format("UK", uk_inputs)
    print("UK Model Data (scaled):")
    print(uk_model_data) 
