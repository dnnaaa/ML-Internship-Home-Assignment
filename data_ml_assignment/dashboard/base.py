from abc import ABC, abstractmethod
import streamlit as st

class DashboardComponent(ABC):
    """Base class for dashboard components."""
    
    @abstractmethod
    def render(self):
        """Render the component in the dashboard."""
        pass

    def show_error(self, message: str, exception: Exception = None):
        """Display error message and optionally the exception."""
        st.error(message)
        if exception:
            st.exception(exception) 