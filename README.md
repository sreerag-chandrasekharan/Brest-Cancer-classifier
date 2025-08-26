# Multiomic Model for Cancer Diagnosis

Cancer is a complex disease driven by changes across multiple molecular layers. Multiomic data—including genomics, transcriptomics, proteomics, and metabolomics—provides a comprehensive view of tumor biology. Integrating these datasets can improve early detection, prognosis, and treatment strategies.

## Motivation

Analyzing a single omics layer often fails to capture the complexity of cancer. Multiomic integration allows us to uncover hidden patterns, identify biomarkers, and make more accurate predictions of disease outcomes. This approach can guide personalized medicine by providing insights into patient-specific molecular mechanisms.

## Application

In this project, we developed machine learning models to classify cancer samples (benign or malignant) using mammography image and cytosis lab data. Our workflow includes:

1. **Data Preprocessing:**  
   - Normalization and scaling of different omics layers.
   - Handling missing data and batch effects.
   - Feature selection to reduce dimensionality.

2. **Modeling Approaches:**  
   - **Traditional Machine Learning:** Random Forest, XGBoost, and Support Vector Machines for individual omics and combined datasets.  
   - **Deep Learning:** Fully connected neural networks to capture nonlinear interactions between features.  
   - **Integration Strategies:**  
     - Early integration (concatenating multiomic features).  
     - Late integration (ensembling predictions from individual omics models).  

3. **Evaluation Metrics:**  
   We evaluated model performance using multiple metrics:
   - Accuracy, precision, recall, F1-score
   - Confusion matrix
   - ROC-AUC
   - Feature importance and interpretability analysis

## Results

Our models demonstrate that multiomic integration improves classification accuracy compared to single-omics approaches. Feature analysis highlighted key molecular signatures associated with malignancy, which could serve as potential biomarkers for further clinical research.

## Conclusion

This project highlights the power of multiomic data integration for cancer diagnosis. By combining diverse molecular data layers, we can achieve more robust predictions and gain deeper insights into tumor biology. The project also strengthened our skills in data preprocessing, machine learning, deep learning, and multiomic analysis techniques.

## Future Work

- Incorporate additional omics layers (e.g., epigenomics, metabolomics).  
- Explore graph-based models to capture interactions between molecular features.  
- Validate models on independent clinical datasets for potential translational use.
