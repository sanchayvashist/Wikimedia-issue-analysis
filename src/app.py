import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from st_pages import Page, show_pages

show_pages([Page(path = "src/app.py")])

show_pages([ Page(path = "src/app.py"),Page(path = "src/pages/chatbot.py")])
# Set the title of the Streamlit app
st.title("Wikimedia Incident Report Analysis")

# Load the extracted data from the CSV file
data_df = pd.read_csv("src/data/categorised.csv", usecols=["Summary", "Detection", "Conclusion", "Actionable", "Time", "cause_of_incident", "severity_level", "major_impact"])
# Remove empty lists from specified columns
columns_to_clean = ["cause_of_incident", "severity_level", "major_impact"]
for column in columns_to_clean:
    data_df[column] = data_df[column].apply(lambda x: x if pd.notna(x) and x != '[]' else None)

# Display the dataframe in the Streamlit app
st.dataframe(data_df, hide_index=True)

st.write("Number of incident over the years")
# Convert the 'Time' column to datetime format if it's not already
data_df['Time'] = pd.to_datetime(data_df['Time'])
data_df.sort_values(by='Time', inplace=True)

# Extract the year and month from the 'Time' column
data_df['Year-Month'] = data_df['Time'].apply(lambda x: f"{x.strftime('%B')} {x.year}")

# Create a bar chart for the value counts of 'Year-Month'
year_month_counts = data_df['Year-Month'].value_counts()

# Plot the bar chart using Streamlit
st.bar_chart(year_month_counts)

# Group the data by 'cause_of_incident' and 'severity_level' and count the occurrences
incident_counts = data_df.groupby(['cause_of_incident', 'severity_level']).size().unstack(fill_value=0)

# Plot the bar chart using Streamlit
st.write("Number of Incidents by Cause and Severity Level")
st.bar_chart(incident_counts, use_container_width=True)

# Group the data by 'cause_of_incident' and 'severity_level' and count the occurrences
incident_counts = data_df.groupby(['major_impact', 'severity_level']).size().unstack(fill_value=0)

# Plot the bar chart using Streamlit
st.write("Number of Incidents by Major_impact and Severity Level")
st.bar_chart(incident_counts)

