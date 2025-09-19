"""
Workflow Navigation Component for Enterprise Flex Property Intelligence Platform

This module provides comprehensive workflow navigation and progress tracking
with state management, validation gates, and progress indicators.
"""

import streamlit as st
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from datetime import datetime


class WorkflowStep(Enum):
    """Workflow step enumeration"""
    UPLOAD = "upload"
    VALIDATION = "validation"
    PROCESSING = "processing"
    FILTERING = "filtering"
    ANALYSIS = "analysis"
    EXPORT = "export"
    COMPLETE = "complete"


@dataclass
class WorkflowStepConfig:
    """Configuration for a workflow step"""
    id: WorkflowStep
    title: str
    description: str
    icon: str
    required: bool = True
    validation_func: Optional[Callable] = None
    next_step: Optional[WorkflowStep] = None
    previous_step: Optional[WorkflowStep] = None


class WorkflowState:
    """Manages workflow state across sessions"""

    def __init__(self):
        if 'workflow_state' not in st.session_state:
            st.session_state.workflow_state = {
                'current_step': WorkflowStep.UPLOAD,
                'completed_steps': [],
                'step_data': {},
                'workflow_id': self._generate_workflow_id(),
                'started_at': datetime.now().isoformat(),
                'last_activity': datetime.now().isoformat()
            }

    @staticmethod
    def _generate_workflow_id() -> str:
        """Generate unique workflow ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]

    def get_current_step(self) -> WorkflowStep:
        """Get current workflow step"""
        return st.session_state.workflow_state['current_step']

    def set_current_step(self, step: WorkflowStep):
        """Set current workflow step"""
        st.session_state.workflow_state['current_step'] = step
        st.session_state.workflow_state['last_activity'] = datetime.now().isoformat()

    def mark_step_completed(self, step: WorkflowStep):
        """Mark a step as completed"""
        if step not in st.session_state.workflow_state['completed_steps']:
            st.session_state.workflow_state['completed_steps'].append(step)

    def is_step_completed(self, step: WorkflowStep) -> bool:
        """Check if a step is completed"""
        return step in st.session_state.workflow_state['completed_steps']

    def can_access_step(self, step: WorkflowStep, step_configs: Dict[WorkflowStep, WorkflowStepConfig]) -> bool:
        """Check if a step can be accessed based on dependencies"""
        config = step_configs.get(step)
        if not config:
            return False

        if config.previous_step:
            return self.is_step_completed(config.previous_step)
        return True

    def save_step_data(self, step: WorkflowStep, data: Any):
        """Save data for a specific step"""
        st.session_state.workflow_state['step_data'][step.value] = data

    def get_step_data(self, step: WorkflowStep) -> Any:
        """Get data for a specific step"""
        return st.session_state.workflow_state['step_data'].get(step.value)

    def reset_workflow(self):
        """Reset workflow to initial state"""
        st.session_state.workflow_state = {
            'current_step': WorkflowStep.UPLOAD,
            'completed_steps': [],
            'step_data': {},
            'workflow_id': self._generate_workflow_id(),
            'started_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat()
        }

    def get_progress_percentage(self, total_steps: int) -> float:
        """Calculate workflow progress percentage"""
        completed = len(st.session_state.workflow_state['completed_steps'])
        return (completed / total_steps) * 100 if total_steps > 0 else 0


class WorkflowNavigator:
    """Main workflow navigation component"""

    def __init__(self):
        self.state = WorkflowState()
        self.step_configs = self._initialize_step_configs()

    def _initialize_step_configs(self) -> Dict[WorkflowStep, WorkflowStepConfig]:
        """Initialize workflow step configurations"""
        return {
            WorkflowStep.UPLOAD: WorkflowStepConfig(
                id=WorkflowStep.UPLOAD,
                title="Upload Data",
                description="Upload your property data files",
                icon="📁",
                next_step=WorkflowStep.VALIDATION
            ),
            WorkflowStep.VALIDATION: WorkflowStepConfig(
                id=WorkflowStep.VALIDATION,
                title="Validate Data",
                description="Validate and clean your data",
                icon="✅",
                previous_step=WorkflowStep.UPLOAD,
                next_step=WorkflowStep.PROCESSING
            ),
            WorkflowStep.PROCESSING: WorkflowStepConfig(
                id=WorkflowStep.PROCESSING,
                title="Process Data",
                description="Apply transformations and enhancements",
                icon="⚙️",
                previous_step=WorkflowStep.VALIDATION,
                next_step=WorkflowStep.FILTERING
            ),
            WorkflowStep.FILTERING: WorkflowStepConfig(
                id=WorkflowStep.FILTERING,
                title="Filter Properties",
                description="Apply filters and ML analysis",
                icon="🔍",
                previous_step=WorkflowStep.PROCESSING,
                next_step=WorkflowStep.ANALYSIS
            ),
            WorkflowStep.ANALYSIS: WorkflowStepConfig(
                id=WorkflowStep.ANALYSIS,
                title="Analyze Results",
                description="View analytics and insights",
                icon="📊",
                previous_step=WorkflowStep.FILTERING,
                next_step=WorkflowStep.EXPORT
            ),
            WorkflowStep.EXPORT: WorkflowStepConfig(
                id=WorkflowStep.EXPORT,
                title="Export Data",
                description="Export your filtered results",
                icon="💾",
                previous_step=WorkflowStep.ANALYSIS,
                next_step=WorkflowStep.COMPLETE
            ),
            WorkflowStep.COMPLETE: WorkflowStepConfig(
                id=WorkflowStep.COMPLETE,
                title="Complete",
                description="Workflow completed successfully",
                icon="🎉",
                previous_step=WorkflowStep.EXPORT
            )
        }

    def render_progress_bar(self):
        """Render workflow progress bar"""
        steps = list(self.step_configs.values())
        current_step = self.state.get_current_step()
        progress = self.state.get_progress_percentage(len(steps) - 1)  # Exclude COMPLETE step

        # Progress bar container
        st.markdown(
            f"""
            <div class="progress-container">
                <div class="progress progress-lg">
                    <div class="progress-bar" style="width: {progress}%"></div>
                </div>
                <div class="progress-label">
                    <span>Progress</span>
                    <span>{int(progress)}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Step indicators
        cols = st.columns(len(steps))
        for idx, (col, step_config) in enumerate(zip(cols, steps)):
            with col:
                is_current = step_config.id == current_step
                is_completed = self.state.is_step_completed(step_config.id)
                is_accessible = self.state.can_access_step(step_config.id, self.step_configs)

                # Determine step state CSS class
                if is_current:
                    step_class = "step-current"
                elif is_completed:
                    step_class = "step-completed"
                elif is_accessible:
                    step_class = "step-accessible"
                else:
                    step_class = "step-disabled"

                # Render step indicator
                st.markdown(
                    f"""
                    <div class="workflow-step {step_class}">
                        <div class="step-icon">{step_config.icon}</div>
                        <div class="step-title">{step_config.title}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    def render_navigation_controls(self):
        """Render navigation control buttons"""
        current_step = self.state.get_current_step()
        current_config = self.step_configs[current_step]

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if current_config.previous_step:
                if st.button("⬅️ Previous", key="nav_prev", use_container_width=True):
                    self.navigate_to_step(current_config.previous_step)

        with col2:
            # Current step info
            st.markdown(
                f"""
                <div class="text-center">
                    <h4>{current_config.icon} {current_config.title}</h4>
                    <p class="text-muted">{current_config.description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            if current_config.next_step and current_config.next_step != WorkflowStep.COMPLETE:
                # Check if current step is validated
                can_proceed = self.validate_current_step()
                if st.button(
                    "Next ➡️",
                    key="nav_next",
                    use_container_width=True,
                    disabled=not can_proceed
                ):
                    self.complete_current_step()
                    self.navigate_to_step(current_config.next_step)

    def navigate_to_step(self, step: WorkflowStep):
        """Navigate to a specific workflow step"""
        if self.state.can_access_step(step, self.step_configs):
            self.state.set_current_step(step)
            st.rerun()

    def complete_current_step(self):
        """Mark current step as completed"""
        current_step = self.state.get_current_step()
        self.state.mark_step_completed(current_step)

    def validate_current_step(self) -> bool:
        """Validate if current step can be completed"""
        current_step = self.state.get_current_step()
        current_config = self.step_configs[current_step]

        # Custom validation function if provided
        if current_config.validation_func:
            return current_config.validation_func(self.state)

        # Default validation: check if step has data
        step_data = self.state.get_step_data(current_step)
        return step_data is not None

    def render_step_navigation_sidebar(self):
        """Render step navigation in sidebar"""
        with st.sidebar:
            st.markdown("### 🧭 Navigation")

            for step_config in self.step_configs.values():
                is_current = step_config.id == self.state.get_current_step()
                is_completed = self.state.is_step_completed(step_config.id)
                is_accessible = self.state.can_access_step(step_config.id, self.step_configs)

                # Style based on state
                if is_current:
                    button_style = "🔵"
                elif is_completed:
                    button_style = "✅"
                elif is_accessible:
                    button_style = "⚪"
                else:
                    button_style = "🔒"

                # Create navigation button
                button_label = f"{button_style} {step_config.title}"
                if is_accessible and st.sidebar.button(
                    button_label,
                    key=f"sidebar_nav_{step_config.id.value}",
                    use_container_width=True
                ):
                    self.navigate_to_step(step_config.id)

    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get workflow summary information"""
        return {
            'workflow_id': st.session_state.workflow_state['workflow_id'],
            'current_step': self.state.get_current_step().value,
            'completed_steps': [s.value for s in st.session_state.workflow_state['completed_steps']],
            'progress_percentage': self.state.get_progress_percentage(len(self.step_configs) - 1),
            'started_at': st.session_state.workflow_state['started_at'],
            'last_activity': st.session_state.workflow_state['last_activity']
        }


class StepValidator:
    """Validation helpers for workflow steps"""

    @staticmethod
    def validate_upload_step(state: WorkflowState) -> bool:
        """Validate upload step completion"""
        data = state.get_step_data(WorkflowStep.UPLOAD)
        return data is not None and 'dataframe' in data

    @staticmethod
    def validate_validation_step(state: WorkflowState) -> bool:
        """Validate data validation step completion"""
        data = state.get_step_data(WorkflowStep.VALIDATION)
        return data is not None and data.get('validation_passed', False)

    @staticmethod
    def validate_processing_step(state: WorkflowState) -> bool:
        """Validate processing step completion"""
        data = state.get_step_data(WorkflowStep.PROCESSING)
        return data is not None and 'processed_dataframe' in data

    @staticmethod
    def validate_filtering_step(state: WorkflowState) -> bool:
        """Validate filtering step completion"""
        data = state.get_step_data(WorkflowStep.FILTERING)
        return data is not None and 'filtered_dataframe' in data

    @staticmethod
    def validate_analysis_step(state: WorkflowState) -> bool:
        """Validate analysis step completion"""
        data = state.get_step_data(WorkflowStep.ANALYSIS)
        return data is not None and 'analysis_results' in data

    @staticmethod
    def validate_export_step(state: WorkflowState) -> bool:
        """Validate export step completion"""
        data = state.get_step_data(WorkflowStep.EXPORT)
        return data is not None and data.get('export_completed', False)


# Add custom CSS for workflow navigation
def inject_workflow_styles():
    """Inject custom CSS for workflow navigation"""
    st.markdown(
        """
        <style>
        .workflow-step {
            text-align: center;
            padding: 10px;
            border-radius: 8px;
            transition: all 0.3s ease;
        }

        .workflow-step.step-completed {
            background-color: rgba(16, 185, 129, 0.1);
            color: #10b981;
        }

        .workflow-step.step-current {
            background-color: rgba(30, 64, 175, 0.1);
            color: #1e40af;
            border: 2px solid #1e40af;
        }

        .workflow-step.step-accessible {
            background-color: rgba(148, 163, 184, 0.1);
            color: #64748b;
            cursor: pointer;
        }

        .workflow-step.step-disabled {
            background-color: rgba(241, 245, 249, 0.5);
            color: #cbd5e1;
            opacity: 0.5;
        }

        .step-icon {
            font-size: 24px;
            margin-bottom: 5px;
        }

        .step-title {
            font-size: 12px;
            font-weight: 600;
        }

        .progress-container {
            margin: 20px 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )