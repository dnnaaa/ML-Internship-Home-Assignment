# dashboard.py
import streamlit as st
from Component.component_manager import ComponentManager

def main():
    st.title("Resume Classification Dashboard")
    st.sidebar.title("Dashboard Modes")

    # Create component manager
    manager = ComponentManager()

    # Select component
    sidebar_options = st.sidebar.selectbox("Options", manager.components.keys())

    # Render selected component
    component = manager.get_component(sidebar_options)
    if component:
        component.render()
    else:
        st.error("Invalid component selected!")

if __name__ == "__main__":
    main()

