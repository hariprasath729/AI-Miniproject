
# main.py
"""
Streamlit entry point for the AI-Based Personalized Study Planner
Sections: Input, CSP Schedule Generator, Feedback, RL Optimizer, Dashboard
"""
import streamlit as st
import pandas as pd
import json
import os
from csp_planner import generate_schedule
from rl_optimizer import QLearningOptimizer
from utils import calculate_schedule_metrics

# Initialize session state
if 'schedule' not in st.session_state:
    st.session_state.schedule = None
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}
if 'opt_schedule' not in st.session_state:
    st.session_state.opt_schedule = None
if 'rl_optimizer' not in st.session_state:
    st.session_state.rl_optimizer = None

# Load initial state from file
os.makedirs("data", exist_ok=True)
STATE_FILE = 'data/planner_state.json'
if os.path.exists(STATE_FILE):
    with open(STATE_FILE, 'r') as f:
        state_data = json.load(f)
    
    # Initialize RL optimizer with saved data if available
    if 'q_table' in state_data and state_data['q_table']:  # Only if q_table is not empty
        # The saved state might not have all the required fields, so we'll recreate it properly
        subjects = state_data.get('user_preferences', {}).get('subjects', ['Math', 'Science', 'English'])
        st.session_state.rl_optimizer = QLearningOptimizer(subjects=subjects)
        st.session_state.rl_optimizer.load(STATE_FILE)

st.title('AI-Based Personalized Study Planner')

# Sidebar for user inputs
st.sidebar.header('User Preferences')

# Input for subjects and difficulties
subject_count = st.sidebar.slider('Number of Subjects', min_value=1, max_value=10, value=3)
subjects = []
difficulties = {}

for i in range(subject_count):
    subj = st.sidebar.text_input(f'Subject {i+1}', value=f'Subject {i+1}', key=f'subj_{i}')
    difficulty = st.sidebar.selectbox(f'Difficulty for {subj}', ['low', 'medium', 'high'], key=f'diff_{i}')
    subjects.append(subj)
    difficulties[subj] = difficulty

hours_per_day = st.sidebar.slider('Hours per Day', min_value=1, max_value=12, value=4)
days = st.sidebar.slider('Study Days', min_value=1, max_value=7, value=5)

# Generate schedule button
if st.sidebar.button('Generate Schedule'):
    with st.spinner('Generating your personalized study schedule...'):
        # Initialize RL optimizer after subjects are known
        if st.session_state.rl_optimizer is None:
            st.session_state.rl_optimizer = QLearningOptimizer(subjects=subjects)
        
        # Generate initial schedule using CSP
        schedule = generate_schedule(
            subjects=subjects,
            difficulties=difficulties,
            hours_per_day=hours_per_day,
            days=days
        )
        
        st.session_state.schedule = schedule
        st.session_state.feedback = {subj: 'fair' for subj in subjects}  # Initialize feedback dict
    
    st.success('Schedule generated successfully!')

# Main content sections
st.subheader('Phase 1: CSP Schedule Generator')

if st.session_state.schedule:
    st.write('Your personalized study schedule:')
    
    # Convert schedule to a DataFrame for better display
    df_data = []
    for day, subjects_list in st.session_state.schedule.items():
        row = {'Day': day}
        for i, subject in enumerate(subjects_list):
            row[f'Hour {i+1}'] = subject
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    st.table(df)
    
    # Calculate and display schedule metrics
    metrics = calculate_schedule_metrics(st.session_state.schedule, subjects)
    st.write(f"**Total Study Hours:** {metrics['total_hours']}")
    st.write(f"**Schedule Balance Score:** {metrics['balance_score']:.2f} (higher is more balanced)")
    
    # Save schedule to state file
    user_preferences = {
        'subjects': subjects,
        'difficulties': difficulties,
        'hours_per_day': hours_per_day,
        'days': days
    }
    
    state_data = {
        'user_preferences': user_preferences,
        'generated_schedule': st.session_state.schedule
    }
    
    # Save to file
    with open(STATE_FILE, 'w') as f:
        json.dump(state_data, f)
else:
    st.info('Generate a schedule using the sidebar inputs.')

# Feedback and RL Optimization section
st.subheader('Phase 2: Feedback & RL Optimizer')

if st.session_state.schedule:
    st.write('Rate each subject in your schedule to improve future recommendations:')
    
    # Create feedback form
    feedback = {}
    for subject in subjects:
        rating = st.selectbox(
            f'How was your experience with {subject}?',
            ['excellent', 'good', 'fair', 'poor', 'very_poor'],
            key=f'feedback_{subject}',
            index=['excellent', 'good', 'fair', 'poor', 'very_poor'].index(st.session_state.feedback.get(subject, 'fair'))
        )
        feedback[subject] = rating
        st.session_state.feedback[subject] = rating

    if st.button('Optimize Schedule with RL'):
        with st.spinner('Optimizing schedule based on your feedback using Q-Learning...'):
            # Initialize RL optimizer if not already done
            if st.session_state.rl_optimizer is None:
                st.session_state.rl_optimizer = QLearningOptimizer(subjects=subjects)
            
            # Optimize schedule using RL
            optimized_schedule = st.session_state.rl_optimizer.get_optimized_schedule(
                st.session_state.schedule,
                feedback
            )
            
            st.session_state.opt_schedule = optimized_schedule
            
            # Save updated Q-table to state file
            st.session_state.rl_optimizer.save(STATE_FILE)
            
        st.success('Schedule optimized with Reinforcement Learning!')
        st.balloons()
else:
    st.info('Generate a schedule first to provide feedback.')

if st.session_state.opt_schedule:
    st.write('Your optimized study schedule:')
    
    # Convert optimized schedule to a DataFrame for better display
    df_data = []
    for day, subjects_list in st.session_state.opt_schedule.items():
        row = {'Day': day}
        for i, subject in enumerate(subjects_list):
            row[f'Hour {i+1}'] = subject
        df_data.append(row)
    
    df_opt = pd.DataFrame(df_data)
    st.table(df_opt)
    
    # Calculate and display metrics for optimized schedule
    opt_metrics = calculate_schedule_metrics(st.session_state.opt_schedule, subjects)
    st.write(f"**Total Study Hours:** {opt_metrics['total_hours']}")
    st.write(f"**Schedule Balance Score:** {opt_metrics['balance_score']:.2f} (higher is more balanced)")

# Visualization dashboard section
st.subheader('Phase 3: Visualization Dashboard')

if st.session_state.schedule or st.session_state.opt_schedule:
    import matplotlib.pyplot as plt
    import plotly.express as px
    
    # Prepare data for visualization
    all_schedule = st.session_state.opt_schedule if st.session_state.opt_schedule else st.session_state.schedule
    schedule_type = "Optimized" if st.session_state.opt_schedule else "Initial"
    
    st.write(f"Visualizing {schedule_type.lower()} schedule:")
    
    # Count occurrences of each subject in the schedule
    subject_counts = {}
    for day, subjects_list in all_schedule.items():
        for subject in subjects_list:
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    subjects_list = list(subject_counts.keys())
    counts = list(subject_counts.values())
    
    ax.bar(subjects_list, counts)
    ax.set_xlabel('Subjects')
    ax.set_ylabel('Hours Scheduled')
    ax.set_title(f'Distribution of Study Hours by Subject ({schedule_type})')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Also use Plotly for interactive visualization
    df_viz = pd.DataFrame({
        'Subject': subjects_list,
        'Hours': counts
    })
    
    fig_plotly = px.bar(df_viz, x='Subject', y='Hours', 
                        title=f'Interactive: Distribution of Study Hours by Subject ({schedule_type})',
                        color='Subject')
    st.plotly_chart(fig_plotly)
    
    # Show study schedule as heatmap
    if all_schedule:
        # Prepare heatmap data
        days = list(all_schedule.keys())
        schedule_matrix = []
        for day in days:
            schedule_matrix.append(all_schedule[day])
        
        # Create DataFrame for heatmap
        df_heatmap = pd.DataFrame(schedule_matrix, columns=[f'Hour {i+1}' for i in range(len(schedule_matrix[0]))], index=days)
        
        # Map subjects to numbers for heatmap visualization
        unique_subjects = list(set([item for sublist in schedule_matrix for item in sublist]))
        subject_mapping = {subj: i+1 for i, subj in enumerate(unique_subjects)}
        
        df_numeric = df_heatmap.replace(subject_mapping)
        
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 6))
        im = ax_heatmap.imshow(df_numeric.values, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        ax_heatmap.set_xticks(range(len(df_numeric.columns)))
        ax_heatmap.set_yticks(range(len(df_numeric.index)))
        ax_heatmap.set_xticklabels(df_numeric.columns)
        ax_heatmap.set_yticklabels(df_numeric.index)
        
        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Subject')
        
        # Add text annotations
        for i in range(len(df_numeric.index)):
            for j in range(len(df_numeric.columns)):
                text = ax_heatmap.text(j, i, df_heatmap.iloc[i, j],
                                       ha="center", va="center", color="white")
        
        ax_heatmap.set_title(f'Study Schedule Heatmap ({schedule_type})')
        st.pyplot(fig_heatmap)
else:
    st.info('Generate a schedule first to see visualizations.')
