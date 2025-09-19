"""
User Feedback and Notification System for Enterprise Flex Property Intelligence Platform

This module provides comprehensive notification, toast messages, error handling,
and user feedback components with animations and contextual guidance.
"""

import streamlit as st
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import time
import uuid
import json
from datetime import datetime
import traceback


class NotificationType(Enum):
    """Notification type enumeration"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class NotificationPosition(Enum):
    """Notification position on screen"""
    TOP_RIGHT = "top-right"
    TOP_LEFT = "top-left"
    BOTTOM_RIGHT = "bottom-right"
    BOTTOM_LEFT = "bottom-left"
    TOP_CENTER = "top-center"
    BOTTOM_CENTER = "bottom-center"


@dataclass
class Notification:
    """Notification data structure"""
    id: str
    type: NotificationType
    title: str
    message: str
    timestamp: datetime
    duration: int = 5000  # milliseconds
    position: NotificationPosition = NotificationPosition.TOP_RIGHT
    icon: Optional[str] = None
    action_label: Optional[str] = None
    action_callback: Optional[callable] = None
    dismissible: bool = True


class NotificationManager:
    """Manages application notifications"""

    def __init__(self):
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
        if 'notification_history' not in st.session_state:
            st.session_state.notification_history = []

    def show(
        self,
        message: str,
        title: Optional[str] = None,
        type: NotificationType = NotificationType.INFO,
        duration: int = 5000,
        position: NotificationPosition = NotificationPosition.TOP_RIGHT,
        icon: Optional[str] = None,
        action_label: Optional[str] = None,
        action_callback: Optional[callable] = None,
        dismissible: bool = True
    ) -> str:
        """Show a notification"""
        notification_id = str(uuid.uuid4())

        # Determine default title and icon if not provided
        if not title:
            title = self._get_default_title(type)
        if not icon:
            icon = self._get_default_icon(type)

        notification = Notification(
            id=notification_id,
            type=type,
            title=title,
            message=message,
            timestamp=datetime.now(),
            duration=duration,
            position=position,
            icon=icon,
            action_label=action_label,
            action_callback=action_callback,
            dismissible=dismissible
        )

        st.session_state.notifications.append(notification)
        st.session_state.notification_history.append(notification)

        # Render the notification
        self._render_notification(notification)

        return notification_id

    def success(self, message: str, title: Optional[str] = None, **kwargs):
        """Show success notification"""
        return self.show(message, title, NotificationType.SUCCESS, **kwargs)

    def error(self, message: str, title: Optional[str] = None, **kwargs):
        """Show error notification"""
        return self.show(message, title, NotificationType.ERROR, **kwargs)

    def warning(self, message: str, title: Optional[str] = None, **kwargs):
        """Show warning notification"""
        return self.show(message, title, NotificationType.WARNING, **kwargs)

    def info(self, message: str, title: Optional[str] = None, **kwargs):
        """Show info notification"""
        return self.show(message, title, NotificationType.INFO, **kwargs)

    def dismiss(self, notification_id: str):
        """Dismiss a notification"""
        st.session_state.notifications = [
            n for n in st.session_state.notifications
            if n.id != notification_id
        ]

    def dismiss_all(self):
        """Dismiss all notifications"""
        st.session_state.notifications = []

    def get_active_notifications(self) -> List[Notification]:
        """Get all active notifications"""
        return st.session_state.notifications

    def get_notification_history(self, limit: int = 50) -> List[Notification]:
        """Get notification history"""
        return st.session_state.notification_history[-limit:]

    @staticmethod
    def _get_default_title(type: NotificationType) -> str:
        """Get default title for notification type"""
        titles = {
            NotificationType.SUCCESS: "Success",
            NotificationType.ERROR: "Error",
            NotificationType.WARNING: "Warning",
            NotificationType.INFO: "Information"
        }
        return titles.get(type, "Notification")

    @staticmethod
    def _get_default_icon(type: NotificationType) -> str:
        """Get default icon for notification type"""
        icons = {
            NotificationType.SUCCESS: "✅",
            NotificationType.ERROR: "❌",
            NotificationType.WARNING: "⚠️",
            NotificationType.INFO: "ℹ️"
        }
        return icons.get(type, "📢")

    def _render_notification(self, notification: Notification):
        """Render a notification using Streamlit"""
        # Map notification type to Streamlit alert type
        alert_type_map = {
            NotificationType.SUCCESS: "success",
            NotificationType.ERROR: "error",
            NotificationType.WARNING: "warning",
            NotificationType.INFO: "info"
        }

        # Create notification container
        with st.container():
            if notification.type == NotificationType.SUCCESS:
                st.success(f"{notification.icon} **{notification.title}**: {notification.message}")
            elif notification.type == NotificationType.ERROR:
                st.error(f"{notification.icon} **{notification.title}**: {notification.message}")
            elif notification.type == NotificationType.WARNING:
                st.warning(f"{notification.icon} **{notification.title}**: {notification.message}")
            else:
                st.info(f"{notification.icon} **{notification.title}**: {notification.message}")

            # Add action button if provided
            if notification.action_label and notification.action_callback:
                if st.button(notification.action_label, key=f"action_{notification.id}"):
                    notification.action_callback()


class ToastNotification:
    """Toast notification system with HTML/CSS"""

    @staticmethod
    def show(
        message: str,
        type: NotificationType = NotificationType.INFO,
        duration: int = 3000,
        position: NotificationPosition = NotificationPosition.TOP_RIGHT
    ):
        """Show a toast notification"""
        toast_id = f"toast_{uuid.uuid4().hex[:8]}"

        # Get styling based on type
        styles = ToastNotification._get_toast_styles(type, position)

        # Render toast HTML
        toast_html = f"""
        <div id="{toast_id}" class="toast-notification toast-{type.value} toast-{position.value}">
            <div class="toast-content">
                <span class="toast-icon">{ToastNotification._get_toast_icon(type)}</span>
                <span class="toast-message">{message}</span>
            </div>
        </div>
        <style>
            #{toast_id} {{
                {styles}
                animation: slideIn 0.3s ease-out, fadeOut 0.3s ease-out {duration/1000 - 0.3}s forwards;
            }}
        </style>
        <script>
            setTimeout(function() {{
                document.getElementById('{toast_id}').style.display = 'none';
            }}, {duration});
        </script>
        """

        st.markdown(toast_html, unsafe_allow_html=True)

    @staticmethod
    def _get_toast_styles(type: NotificationType, position: NotificationPosition) -> str:
        """Get CSS styles for toast notification"""
        # Base styles
        styles = """
            position: fixed;
            z-index: 9999;
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
            max-width: 350px;
        """

        # Position styles
        position_styles = {
            NotificationPosition.TOP_RIGHT: "top: 20px; right: 20px;",
            NotificationPosition.TOP_LEFT: "top: 20px; left: 20px;",
            NotificationPosition.BOTTOM_RIGHT: "bottom: 20px; right: 20px;",
            NotificationPosition.BOTTOM_LEFT: "bottom: 20px; left: 20px;",
            NotificationPosition.TOP_CENTER: "top: 20px; left: 50%; transform: translateX(-50%);",
            NotificationPosition.BOTTOM_CENTER: "bottom: 20px; left: 50%; transform: translateX(-50%);"
        }

        # Type styles
        type_styles = {
            NotificationType.SUCCESS: "background-color: #10b981; color: white;",
            NotificationType.ERROR: "background-color: #ef4444; color: white;",
            NotificationType.WARNING: "background-color: #f59e0b; color: white;",
            NotificationType.INFO: "background-color: #3b82f6; color: white;"
        }

        styles += position_styles.get(position, "")
        styles += type_styles.get(type, "")

        return styles

    @staticmethod
    def _get_toast_icon(type: NotificationType) -> str:
        """Get icon for toast notification"""
        icons = {
            NotificationType.SUCCESS: "✓",
            NotificationType.ERROR: "✕",
            NotificationType.WARNING: "!",
            NotificationType.INFO: "i"
        }
        return icons.get(type, "•")


class LoadingIndicator:
    """Loading indicator component"""

    @staticmethod
    def show(message: str = "Loading...", spinner_type: str = "default"):
        """Show loading indicator"""
        with st.spinner(message):
            time.sleep(0.1)  # Brief pause to ensure spinner shows

    @staticmethod
    def show_progress(
        message: str,
        current: int,
        total: int,
        show_percentage: bool = True,
        show_time_remaining: bool = False,
        start_time: Optional[float] = None
    ):
        """Show progress bar with message"""
        progress = current / total if total > 0 else 0

        # Create progress bar
        progress_bar = st.progress(progress)

        # Build status message
        status_parts = [message]
        if show_percentage:
            status_parts.append(f"{int(progress * 100)}%")
        if show_time_remaining and start_time:
            elapsed = time.time() - start_time
            if current > 0:
                estimated_total = elapsed * total / current
                remaining = estimated_total - elapsed
                status_parts.append(f"~{int(remaining)}s remaining")

        # Display status
        st.text(" | ".join(status_parts))

        return progress_bar


class ErrorHandler:
    """Comprehensive error handling and user feedback"""

    @staticmethod
    def handle_error(
        error: Exception,
        context: str = "",
        show_traceback: bool = False,
        suggestions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Handle an error with user-friendly feedback"""
        error_id = str(uuid.uuid4())
        error_type = type(error).__name__
        error_message = str(error)

        # Log error details
        error_details = {
            'id': error_id,
            'type': error_type,
            'message': error_message,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc() if show_traceback else None
        }

        # Store in session state for debugging
        if 'error_log' not in st.session_state:
            st.session_state.error_log = []
        st.session_state.error_log.append(error_details)

        # Display user-friendly error message
        st.error(f"❌ **Error**: {ErrorHandler._get_friendly_message(error)}")

        # Show error details in expander
        with st.expander("Error Details", expanded=False):
            st.code(f"Error ID: {error_id}")
            st.code(f"Type: {error_type}")
            if context:
                st.text(f"Context: {context}")
            if show_traceback:
                st.code(traceback.format_exc())

        # Show suggestions if provided
        if suggestions:
            st.info("💡 **Suggestions:**")
            for suggestion in suggestions:
                st.write(f"• {suggestion}")
        else:
            # Auto-generate suggestions based on error type
            auto_suggestions = ErrorHandler._generate_suggestions(error)
            if auto_suggestions:
                st.info("💡 **Possible Solutions:**")
                for suggestion in auto_suggestions:
                    st.write(f"• {suggestion}")

        return error_details

    @staticmethod
    def _get_friendly_message(error: Exception) -> str:
        """Get user-friendly error message"""
        error_messages = {
            'FileNotFoundError': "The specified file could not be found.",
            'PermissionError': "Permission denied. Please check file permissions.",
            'ValueError': "Invalid value provided. Please check your input.",
            'KeyError': "Required data field is missing.",
            'TypeError': "Data type mismatch. Please check your data format.",
            'MemoryError': "Not enough memory to complete the operation.",
            'TimeoutError': "The operation timed out. Please try again.",
            'ConnectionError': "Connection failed. Please check your network.",
        }

        error_type = type(error).__name__
        return error_messages.get(error_type, str(error))

    @staticmethod
    def _generate_suggestions(error: Exception) -> List[str]:
        """Generate suggestions based on error type"""
        suggestions_map = {
            'FileNotFoundError': [
                "Check if the file path is correct",
                "Ensure the file exists in the specified location",
                "Verify you have the right file extension"
            ],
            'PermissionError': [
                "Check if you have read/write permissions",
                "Try running with administrator privileges",
                "Check if the file is not locked by another process"
            ],
            'ValueError': [
                "Verify the input data format",
                "Check for missing or invalid values",
                "Ensure all required fields are provided"
            ],
            'MemoryError': [
                "Try processing smaller batches of data",
                "Close other applications to free up memory",
                "Consider upgrading system memory"
            ],
            'ConnectionError': [
                "Check your internet connection",
                "Verify the API endpoint is correct",
                "Check if any firewall is blocking the connection"
            ]
        }

        error_type = type(error).__name__
        return suggestions_map.get(error_type, [])


class FeedbackCollector:
    """Collect and display user feedback"""

    @staticmethod
    def show_feedback_form(context: str = "general"):
        """Show feedback collection form"""
        with st.expander("📝 Provide Feedback", expanded=False):
            feedback_type = st.selectbox(
                "Feedback Type",
                ["Bug Report", "Feature Request", "General Feedback", "Performance Issue"]
            )

            feedback_text = st.text_area(
                "Your Feedback",
                placeholder="Please describe your feedback in detail..."
            )

            email = st.text_input(
                "Email (optional)",
                placeholder="your.email@example.com"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit Feedback", type="primary"):
                    if feedback_text:
                        FeedbackCollector._save_feedback(
                            feedback_type,
                            feedback_text,
                            email,
                            context
                        )
                        st.success("✅ Thank you for your feedback!")
                        st.balloons()
                    else:
                        st.warning("Please enter your feedback before submitting.")

            with col2:
                if st.button("Cancel"):
                    st.rerun()

    @staticmethod
    def _save_feedback(feedback_type: str, text: str, email: str, context: str):
        """Save feedback to session state"""
        if 'feedback_log' not in st.session_state:
            st.session_state.feedback_log = []

        feedback = {
            'id': str(uuid.uuid4()),
            'type': feedback_type,
            'text': text,
            'email': email,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }

        st.session_state.feedback_log.append(feedback)
        return feedback


# Inject notification styles
def inject_notification_styles():
    """Inject custom CSS for notifications"""
    st.markdown(
        """
        <style>
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }

        @keyframes fadeOut {
            from {
                opacity: 1;
            }
            to {
                opacity: 0;
            }
        }

        .toast-notification {
            min-width: 250px;
            max-width: 400px;
        }

        .toast-content {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .toast-icon {
            font-size: 18px;
            font-weight: bold;
        }

        .toast-message {
            flex: 1;
            line-height: 1.4;
        }

        .notification-container {
            position: fixed;
            z-index: 10000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )