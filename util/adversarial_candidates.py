ADVERSARIAL_CANDIDATES = {
    "bmi": {
        "target_column": "bmi_group_code",
        "observed_column": "bmi_observed",
        "adversarial_num_classes": 3,
        "description": "BMI group code",
    },
    "psa_value": {
        "target_column": "psa_group_code",
        "observed_column": "psa_value_observed",
        "adversarial_num_classes": 4,
        "description": "PSA group code",
    },
    "scan_prostate_volume_ml": {
        "target_column": "prostate_volume_group_code",
        "observed_column": "scan_prostate_volume_ml_observed",
        "adversarial_num_classes": 3,
        "description": "Prostate volume group code",
    },
    "max_pirads": {
        "target_column": "max_pirads",
        "observed_column": "max_pirads_observed",
        "adversarial_num_classes": 6,
        "allow_raw_class_ids": True,
        "description": "Original max PI-RADS class id; class 0 is unused",
    },
    "pirads_high": {
        "target_column": "pirads_high",
        "observed_column": "pirads_high_observed",
        "adversarial_num_classes": 2,
        "description": "Binned PI-RADS high indicator",
    },
    "age": {
        "target_column": "age_group_code",
        "observed_column": "age_observed",
        "adversarial_num_classes": 4,
        "description": "Age group code",
    },
    "hypertension": {
        "target_column": "hypertension",
        "observed_column": "hypertension_observed",
        "adversarial_num_classes": 2,
        "description": "Hypertension indicator",
    },
    "cardiovascular_disease": {
        "target_column": "cardiovascular_disease",
        "observed_column": "cardiovascular_disease_observed",
        "adversarial_num_classes": 2,
        "description": "Cardiovascular disease indicator",
    },
    "chronic_kidney_disease": {
        "target_column": "chronic_kidney_disease",
        "observed_column": "chronic_kidney_disease_observed",
        "adversarial_num_classes": 2,
        "description": "Chronic kidney disease indicator",
    },
    "copd": {
        "target_column": "copd",
        "observed_column": "copd_observed",
        "adversarial_num_classes": 2,
        "description": "COPD indicator",
    },
    "asthma": {
        "target_column": "asthma",
        "observed_column": "asthma_observed",
        "adversarial_num_classes": 2,
        "description": "Asthma indicator",
    },
    "diabetes": {
        "target_column": "diabetes",
        "observed_column": "diabetes_observed",
        "adversarial_num_classes": 2,
        "description": "Diabetes indicator",
    },
    "dre_abnormal": {
        "target_column": "dre_abnormal",
        "observed_column": "dre_abnormal_observed",
        "adversarial_num_classes": 2,
        "description": "Abnormal DRE indicator",
    },
    "smoking_encoded": {
        "target_column": "smoking_encoded",
        "observed_column": "smoking_encoded_observed",
        "adversarial_num_classes": 3,
        "description": "Smoking encoded category",
    },
    "alcohol_encoded": {
        "target_column": "alcohol_encoded",
        "observed_column": "alcohol_encoded_observed",
        "adversarial_num_classes": 7,
        "description": "Alcohol encoded category",
    },
    "cardio_any": {
        "target_column": "cardio_any",
        "observed_column": "cardio_any_observed",
        "adversarial_num_classes": 2,
        "description": "Hypertension or cardiovascular disease",
    },
    "respiratory_any": {
        "target_column": "respiratory_any",
        "observed_column": "respiratory_any_observed",
        "adversarial_num_classes": 2,
        "description": "COPD or asthma",
    },
    "renal_metabolic_any": {
        "target_column": "renal_metabolic_any",
        "observed_column": "renal_metabolic_any_observed",
        "adversarial_num_classes": 2,
        "description": "Diabetes or chronic kidney disease",
    },
}


def adversarial_variable_choices():
    return list(ADVERSARIAL_CANDIDATES)


def get_adversarial_candidate(variable):
    return ADVERSARIAL_CANDIDATES.get(variable, {})
