import streamlit as st
import pandas as pd

from sklearn.metrics import classification_report

from constants import total_labels, total_predictions, class_names

def classification_report_page():
    st.header("Classification Report")
    st.markdown(
        """
        This is confusion matrix that results from the data recorded from the evaluating against test set
        of this model.
        """
    )
    
    with st.spinner("Loading..."):
        report = classification_report(total_labels, total_predictions, target_names=class_names, output_dict=True, digits=2)

        accuracy = report.pop('accuracy')
        support = report['macro avg']['support']

        dataframe_report = pd.DataFrame(report).transpose()
        # dataframe_report = dataframe_report.applymap(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

        for col in ['precision', 'recall', 'f1-score']:
            if col in dataframe_report.columns:
                dataframe_report[col] = dataframe_report[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")

        if 'support' in dataframe_report.columns:
            dataframe_report['support'] = dataframe_report['support'].astype(int)

        accuracy_dataframe = pd.DataFrame([["", "", round(accuracy, 2), int(support)]], columns=dataframe_report.columns, index=['accuracy'])
    
        dataframe_report = pd.concat([dataframe_report[:len(dataframe_report)-2], accuracy_dataframe, dataframe_report[len(dataframe_report)-2:]], ignore_index=False)
        
        st.table(dataframe_report)
