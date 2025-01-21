import streamlit as st

#This function is used inside the eda_component
def Dataset_overviewHelperF(dataset):
    
    st.dataframe(dataset)
    
    # Text-specific statistics
    dataset['resume_length'] = dataset['resume'].apply(len)
    st.write("Resume Length Statistics:")
    st.write(dataset['resume_length'].describe())

    # Longest and shortest resumes
    dataset['resume_length'] = dataset['resume'].apply(len)
    longest_resume = dataset.loc[dataset['resume_length'].idxmax()]
    shortest_resume = dataset.loc[dataset['resume_length'].idxmin()]

    st.subheader("Longest Resume")
    with st.expander("Click to View Longest Resume"):
        st.text_area("Longest Resume Content", longest_resume['resume'], height=200)
    st.write(f"**Character Count:** {longest_resume['resume_length']}")

    # Shortest Resume
    st.subheader("Shortest Resume")
    with st.expander("Click to View Shortest Resume"):
        st.text_area("Shortest Resume Content", shortest_resume['resume'], height=100)
    st.write(f"**Character Count:** {shortest_resume['resume_length']}")

    # Search functionality for Resume
    st.subheader("Search in Resumes")
    search_term = st.text_input("Search Term", "")

    if search_term:
        # Highlight search term in resumes
        highlighted_resumes = dataset['resume'].apply(
            lambda text: text.replace(
                search_term, f"<span style='background-color: lightgreen; color: black;'>{search_term}</span>"
            )
        )
        
        # Filter top 10 resumes containing the search term
        filtered_resumes = highlighted_resumes[highlighted_resumes.str.contains(search_term, case=False)].head(10)

        if not filtered_resumes.empty:
            # Let user select a specific resume
            options = ["None"] + [f"Resume {i+1}" for i in range(len(filtered_resumes))]
            selected_resume = st.selectbox("Select a Resume to View", options=options)
            
            # Display selected resume
            if selected_resume != "None" :
                resume_index = int(selected_resume.split(" ")[1]) - 1  # Extract the index
                st.markdown(f"**{selected_resume} Content:**", unsafe_allow_html=True)
                st.markdown(filtered_resumes.iloc[resume_index], unsafe_allow_html=True)
        else:
            st.write("No resumes found containing the search term.")
    else:
        st.write("Enter a search term to view relevant resumes.")

###